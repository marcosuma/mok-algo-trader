"""Stable in-process proxy for CTraderBroker running in a subprocess.

``CTraderBrokerProxy`` is a long-lived object that looks like a normal
``CTraderBroker`` to the rest of the application.  Internally it manages a
``multiprocessing.Process`` that runs the real broker (with its own Twisted
reactor).  When the subprocess signals ``ReconnectExhausted`` the proxy
kills the old process and spawns a fresh one—giving a clean reactor without
restarting the entire trading service or closing any positions.

Architecture summary
--------------------
* Two queues cross the process boundary:
    - ``_cmd_queue``   (proxy → subprocess)  – commands
    - ``_event_queue`` (subprocess → proxy)  – results, ticks, events
* Python callbacks (market-data, order-status, execution) cannot be pickled,
  so they are stored locally in the proxy.  The subprocess sends serialisable
  data dicts; the proxy dispatches them to the right callback.
* ``ConnectionEventBus`` in the proxy is the authoritative bus that
  ``TelegramNotifier`` and other consumers subscribe to.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from live_trading.notifications.connection_event_bus import ConnectionEventBus
from live_trading.notifications.connection_events import (
    FullRestartAttempt,
    FullRestartFailed,
    ReconnectExhausted,
)

logger = logging.getLogger(__name__)

_RESTART_DELAY_SECONDS = 300  # 5 min between full-restart attempts (same as ConnectionManager)
_MP_CTX = multiprocessing.get_context("spawn")  # safe with asyncio + Twisted threads


class CTraderBrokerProxy:
    """Drop-in replacement for ``CTraderBroker`` that runs in a subprocess."""

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def __init__(self, restart_delay_seconds: int = _RESTART_DELAY_SECONDS) -> None:
        self._restart_delay = restart_delay_seconds

        # Public event bus – TelegramNotifier subscribes to this.
        self.event_bus = ConnectionEventBus()

        # IPC queues (created per-subprocess; recreated on restart).
        self._cmd_queue: Optional[multiprocessing.Queue] = None
        self._event_queue: Optional[multiprocessing.Queue] = None
        self._process: Optional[multiprocessing.Process] = None

        # Reader task that drains _event_queue.
        self._reader_task: Optional[asyncio.Task] = None

        # Shutdown flag.
        self._shutdown = False

        # Restart state.
        self._restart_count = 0
        self._restarting = False

        # ------------------------------------------------------------------ #
        # Connection state (updated from subprocess "state" messages)
        # ------------------------------------------------------------------ #
        self.connected: bool = False
        self.authenticated: bool = False
        self.account_id: Optional[int] = None

        # Spot-price cache forwarded from ticks (keyed by symbol_id).
        self._last_spot_prices: Dict[int, Dict] = {}
        # Symbol cache exposed for health endpoints (populated on tick).
        self._symbol_cache: Dict[str, int] = {}

        # ------------------------------------------------------------------ #
        # Pending async results: request_id → asyncio.Future
        # ------------------------------------------------------------------ #
        self._pending: Dict[str, asyncio.Future] = {}

        # ------------------------------------------------------------------ #
        # Locally-stored callbacks (cannot be pickled across process boundary)
        # ------------------------------------------------------------------ #
        # Market-data subscriptions  asset → [(callback, callback_id)]
        self._data_callbacks: Dict[str, List[tuple]] = {}
        # bar-size subscriptions for restoration after restart
        self._trendbar_subscriptions: Dict[str, List[str]] = {}

        # Order status callbacks  request_id → callable  (pre-order_id resolution)
        self._pending_order_callbacks: Dict[str, Callable] = {}
        # order_id → callable  (post-resolution)
        self._order_callbacks: Dict[str, Callable] = {}
        # position_id → callable  (broker-initiated closes, e.g. TP/SL)
        self._position_close_callbacks: Dict[str, Callable] = {}
        # Global execution listeners  (registered via add_execution_listener)
        self._execution_listeners: List[Callable] = []

        # ------------------------------------------------------------------ #
        # Compatibility attributes read directly by API endpoints / adapter
        # ------------------------------------------------------------------ #
        self._reconnect_attempts: int = 0

    # -----------------------------------------------------------------------
    # Subprocess lifecycle
    # -----------------------------------------------------------------------

    def _spawn_subprocess(self) -> None:
        """Create fresh IPC queues and start a new subprocess."""
        from live_trading.brokers.ctrader_subprocess_worker import run_worker

        self._cmd_queue = _MP_CTX.Queue()
        self._event_queue = _MP_CTX.Queue(maxsize=20_000)
        self._process = _MP_CTX.Process(
            target=run_worker,
            args=(self._cmd_queue, self._event_queue),
            daemon=True,
            name="CTraderWorker",
        )
        self._process.start()
        logger.info(f"[Proxy] Spawned cTrader subprocess PID={self._process.pid}")

    def _kill_subprocess(self) -> None:
        """Terminate the current subprocess and close its queues."""
        if self._process and self._process.is_alive():
            logger.info(f"[Proxy] Terminating cTrader subprocess PID={self._process.pid}")
            try:
                self._cmd_queue.put_nowait({"cmd": "shutdown"})
            except Exception:
                pass
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
        self._process = None

        # Close queues to release resources.
        for q in (self._cmd_queue, self._event_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self._cmd_queue = None
        self._event_queue = None

    # -----------------------------------------------------------------------
    # Event-queue reader
    # -----------------------------------------------------------------------

    def _start_reader(self) -> None:
        """Start the background asyncio task that drains the event queue."""
        loop = asyncio.get_event_loop()
        self._reader_task = loop.create_task(self._reader_loop())

    async def _stop_reader(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        self._reader_task = None

    async def _reader_loop(self) -> None:
        """Continuously drain _event_queue and dispatch messages."""
        loop = asyncio.get_event_loop()
        while not self._shutdown:
            eq = self._event_queue
            if eq is None:
                await asyncio.sleep(0.05)
                continue
            try:
                # Use an executor so the asyncio loop is never blocked.
                msg = await loop.run_in_executor(None, self._blocking_get, eq)
                if msg is None:
                    continue
                await self._dispatch(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[Proxy] Reader error: {exc}", exc_info=True)

    @staticmethod
    def _blocking_get(queue, timeout: float = 0.1):
        """Blocking queue get with timeout; returns None on timeout."""
        try:
            return queue.get(timeout=timeout)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Message dispatch
    # -----------------------------------------------------------------------

    async def _dispatch(self, msg: dict) -> None:
        msg_type = msg.get("type")

        if msg_type == "result":
            self._resolve_future(msg)

        elif msg_type == "state":
            self.connected = msg.get("connected", False)
            self.authenticated = msg.get("authenticated", False)
            self.account_id = msg.get("account_id")

        elif msg_type == "connection_event":
            event = msg.get("event")
            if event is None:
                return
            # Intercept ReconnectExhausted — handle subprocess restart here
            # instead of letting ConnectionManager create a stale new broker.
            if isinstance(event, ReconnectExhausted):
                self.connected = False
                self.authenticated = False
                self.event_bus.emit(event)  # notify Telegram etc.
                if not self._shutdown and not self._restarting:
                    asyncio.get_event_loop().create_task(self._full_restart())
            else:
                self.event_bus.emit(event)

        elif msg_type == "tick":
            self._handle_tick(msg)

        elif msg_type == "order_status":
            self._handle_order_status(msg)

        elif msg_type == "execution_event":
            self._handle_execution_event(msg.get("data", {}))

        elif msg_type == "fatal_error":
            logger.error(f"[Proxy] Subprocess fatal error: {msg.get('error')}")
            if not self._shutdown and not self._restarting:
                asyncio.get_event_loop().create_task(self._full_restart())

    def _resolve_future(self, msg: dict) -> None:
        request_id = msg.get("request_id")
        fut = self._pending.pop(request_id, None)
        if fut is None or fut.done():
            return
        error = msg.get("error")
        if error:
            fut.set_exception(RuntimeError(error))
        else:
            fut.set_result(msg.get("value"))

    def _handle_tick(self, msg: dict) -> None:
        asset = msg.get("asset")
        data = msg.get("data", {})
        # Update spot-price cache (keyed by asset for simplicity in proxy).
        if data.get("bid") and data.get("ask"):
            self._last_spot_prices[asset] = {
                "bid": data["bid"], "ask": data["ask"],
                "mid": (data["bid"] + data["ask"]) / 2.0,
            }
        # Dispatch to all registered callbacks for this asset.
        for cb, _ in self._data_callbacks.get(asset, []):
            try:
                cb(data)
            except Exception as exc:
                logger.error(f"[Proxy] Tick callback error for {asset}: {exc}")

    def _handle_order_status(self, msg: dict) -> None:
        rid = msg.get("order_request_id")
        order_id = msg.get("order_id")
        data = msg.get("data", {})

        # Try pending (pre-order_id) callback first.
        cb = self._pending_order_callbacks.get(rid)
        if cb:
            if order_id:
                # Promote to order_id-keyed callback for subsequent updates.
                self._order_callbacks[order_id] = cb
                del self._pending_order_callbacks[rid]
            try:
                cb(data)
            except Exception as exc:
                logger.error(f"[Proxy] Order status callback error (rid={rid}): {exc}")
            return

        # Fall back to order_id-keyed callback.
        if order_id:
            cb = self._order_callbacks.get(order_id)
            if cb:
                try:
                    cb(data)
                except Exception as exc:
                    logger.error(f"[Proxy] Order callback error (order_id={order_id}): {exc}")

    def _handle_execution_event(self, data: dict) -> None:
        status = data.get("status")
        order_id = data.get("order_id")
        broker_position_id = data.get("broker_position_id")

        # Broker-initiated position close (TP/SL triggered externally).
        if status == "FILLED" and broker_position_id:
            cb = self._position_close_callbacks.pop(broker_position_id, None)
            if cb:
                try:
                    cb({**data, "is_position_close": True})
                except Exception as exc:
                    logger.error(f"[Proxy] Position-close callback error: {exc}")

        # Global execution listeners (used by CTraderAdapter for reconciliation).
        for listener in self._execution_listeners:
            try:
                listener(data)
            except Exception as exc:
                logger.error(f"[Proxy] Execution listener error: {exc}")

    # -----------------------------------------------------------------------
    # Full-restart logic (mirrors ConnectionManager._full_restart)
    # -----------------------------------------------------------------------

    async def _full_restart(self) -> None:
        self._restarting = True
        try:
            self._restart_count += 1
            self.event_bus.emit(FullRestartAttempt(restart_count=self._restart_count))
            logger.critical(
                f"[Proxy] All fast reconnects exhausted — performing full subprocess restart "
                f"(#{self._restart_count}).  Killing old subprocess immediately, then waiting "
                f"{self._restart_delay}s before spawning a fresh one..."
            )

            # Kill the old subprocess NOW so it stops generating events during the wait.
            await self._stop_reader()
            self._kill_subprocess()
            self._clear_pending_futures(RuntimeError("Subprocess restarted"))

            attempt = 0
            while not self._shutdown:
                attempt += 1
                await asyncio.sleep(self._restart_delay)
                if self._shutdown:
                    break

                logger.info(f"[Proxy] Full restart #{self._restart_count} attempt {attempt}...")
                try:
                    # Fresh subprocess + reader.
                    self._spawn_subprocess()
                    self._start_reader()

                    # Re-connect (120 s matches proxy.connect() timeout).
                    ok = await self._send_cmd("connect", timeout=120.0)
                    if not ok:
                        raise RuntimeError("connect() returned False after restart")
                    await self._send_cmd("start_connection_monitor")

                    # Restore market-data subscriptions.
                    await self._restore_subscriptions()

                    logger.info(
                        f"[Proxy] Full restart #{self._restart_count} attempt {attempt} succeeded."
                    )
                    return

                except Exception as exc:
                    logger.error(
                        f"[Proxy] Full restart #{self._restart_count} attempt {attempt} failed: {exc}",
                        exc_info=True,
                    )
                    self.event_bus.emit(
                        FullRestartFailed(restart_count=self._restart_count, attempt=attempt)
                    )
                    # Kill the subprocess that was spawned for this failed attempt
                    # so we start clean on the next try.
                    await self._stop_reader()
                    self._kill_subprocess()
        finally:
            self._restarting = False

    async def _restore_subscriptions(self) -> None:
        """Re-subscribe all market-data callbacks in the fresh subprocess."""
        for asset, entries in list(self._data_callbacks.items()):
            bar_sizes = self._trendbar_subscriptions.get(asset)
            for cb, callback_id in entries:
                try:
                    await self._send_cmd(
                        "subscribe_market_data",
                        asset=asset,
                        callback_id=callback_id,
                        bar_sizes=bar_sizes,
                    )
                    logger.info(f"[Proxy] Restored subscription: {asset} ({callback_id})")
                except Exception as exc:
                    logger.error(f"[Proxy] Failed to restore subscription {asset}: {exc}")

    def _clear_pending_futures(self, exc: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # -----------------------------------------------------------------------
    # Command helpers
    # -----------------------------------------------------------------------

    async def _send_cmd(self, cmd: str, timeout: float = 60.0, **kwargs) -> Any:
        """Send a command to the subprocess and await the result."""
        if self._cmd_queue is None:
            raise RuntimeError("Subprocess not started")
        request_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut
        self._cmd_queue.put_nowait({"cmd": cmd, "request_id": request_id, **kwargs})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise RuntimeError(f"Command {cmd!r} timed out after {timeout}s")

    # -----------------------------------------------------------------------
    # CTraderBroker-compatible public interface
    # -----------------------------------------------------------------------

    async def connect(self) -> bool:
        self._spawn_subprocess()
        self._start_reader()
        ok = await self._send_cmd("connect", timeout=120.0)
        if ok:
            await self._send_cmd("start_connection_monitor", timeout=30.0)
        return bool(ok)

    async def disconnect(self) -> None:
        self._shutdown = True
        await self._stop_reader()
        self._kill_subprocess()

    async def start_connection_monitor(self) -> None:
        # Called by ConnectionManager; already handled in connect() but safe to call again.
        pass

    async def subscribe_market_data(
        self,
        asset: str,
        callback: Callable,
        callback_id: str = None,
        bar_sizes: list = None,
    ) -> bool:
        if callback_id is None:
            callback_id = f"cb_{id(callback)}"

        # Register callback locally.
        if asset not in self._data_callbacks:
            self._data_callbacks[asset] = []
        # Replace if same callback_id already exists.
        self._data_callbacks[asset] = [
            (cb, cid) for cb, cid in self._data_callbacks[asset] if cid != callback_id
        ]
        self._data_callbacks[asset].append((callback, callback_id))
        if bar_sizes:
            self._trendbar_subscriptions[asset] = bar_sizes

        return await self._send_cmd(
            "subscribe_market_data",
            asset=asset,
            callback_id=callback_id,
            bar_sizes=bar_sizes,
        )

    async def subscribe_live_trendbars(self, asset: str, bar_sizes: list = None) -> None:
        if bar_sizes:
            self._trendbar_subscriptions[asset] = bar_sizes
        await self._send_cmd("subscribe_live_trendbars", asset=asset, bar_sizes=bar_sizes or [])

    async def unsubscribe_market_data(self, asset: str, callback_id: str = None) -> None:
        if asset in self._data_callbacks:
            if callback_id:
                self._data_callbacks[asset] = [
                    (cb, cid) for cb, cid in self._data_callbacks[asset] if cid != callback_id
                ]
            else:
                del self._data_callbacks[asset]
                self._trendbar_subscriptions.pop(asset, None)
        await self._send_cmd("unsubscribe_market_data", asset=asset, callback_id=callback_id)

    async def place_order(
        self,
        asset: str,
        action: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_status_callback: Optional[Callable] = None,
    ) -> str:
        request_id = str(uuid.uuid4())
        if order_status_callback:
            self._pending_order_callbacks[request_id] = order_status_callback

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut
        self._cmd_queue.put_nowait({
            "cmd": "place_order",
            "request_id": request_id,
            "asset": asset,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })
        try:
            order_id = await asyncio.wait_for(fut, timeout=60.0)
            return order_id or ""
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            self._pending_order_callbacks.pop(request_id, None)
            logger.error(f"[Proxy] place_order timed out for {asset}")
            return ""

    async def cancel_order(self, broker_order_id: str) -> bool:
        return bool(await self._send_cmd("cancel_order", broker_order_id=broker_order_id))

    async def close_position_by_id(
        self,
        broker_position_id: str,
        volume: Optional[int] = None,
    ) -> str:
        result = await self._send_cmd(
            "close_position_by_id",
            broker_position_id=broker_position_id,
            volume=volume,
        )
        return result or ""

    async def get_positions(self) -> list:
        return await self._send_cmd("get_positions") or []

    async def get_account_info(self) -> dict:
        return await self._send_cmd("get_account_info") or {}

    async def get_deal_history(self, from_timestamp_ms: int, to_timestamp_ms: int) -> list:
        return await self._send_cmd(
            "get_deal_history",
            from_timestamp_ms=from_timestamp_ms,
            to_timestamp_ms=to_timestamp_ms,
        ) or []

    async def get_open_broker_orders(self) -> list:
        return await self._send_cmd("get_open_broker_orders") or []

    async def fetch_historical_data(
        self,
        asset: str,
        bar_size: str,
        interval: str,
        callback: Callable,
        context: Optional[dict] = None,
    ) -> bool:
        # The subprocess collects bars via a local callback and returns them in the
        # result dict.  We then call the real callback here in the main process.
        result = await self._send_cmd(
            "fetch_historical_data",
            asset=asset,
            bar_size=bar_size,
            interval=interval,
            context=context,
            timeout=120.0,
        )
        if isinstance(result, dict) and result.get("ok") and result.get("bars") is not None:
            try:
                callback(result["bars"], result.get("context"))
            except Exception as exc:
                logger.error(f"[Proxy] fetch_historical_data callback error: {exc}")
            return True
        return False

    # -----------------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------------

    def add_execution_listener(self, callback: Callable) -> None:
        if callback not in self._execution_listeners:
            self._execution_listeners.append(callback)

    def register_order_callback(self, broker_order_id: str, callback: Callable) -> None:
        self._order_callbacks[broker_order_id] = callback

    def register_position_close_callback(self, broker_position_id: str, callback: Callable) -> None:
        self._position_close_callbacks[broker_position_id] = callback

    # -----------------------------------------------------------------------
    # Calculation helpers (pure, no subprocess needed)
    # -----------------------------------------------------------------------

    def _convert_quantity_to_volume(self, quantity: float, symbol_id: Optional[int] = None) -> int:
        """Duplicated from CTraderBroker — pure arithmetic, no IO."""
        UNITS_PER_LOT = 100_000
        MAX_VOLUME = 10_000_000
        volume = int(round(quantity * UNITS_PER_LOT))
        if volume < 1000:
            volume = 1000
        if volume > MAX_VOLUME:
            volume = MAX_VOLUME
        return volume

    # -----------------------------------------------------------------------
    # Status / health
    # -----------------------------------------------------------------------

    def get_connection_status(self) -> dict:
        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "account_id": self.account_id,
            "subprocess_alive": self._process is not None and self._process.is_alive(),
            "restart_count": self._restart_count,
            "restarting": self._restarting,
        }

    async def _attempt_reconnect(self) -> None:
        """Force a full subprocess restart (used by the /reconnect API endpoint)."""
        self._reconnect_attempts = 0
        if not self._restarting:
            asyncio.get_event_loop().create_task(self._full_restart())
