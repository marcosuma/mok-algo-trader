"""Subprocess worker for CTraderBroker isolation.

This module is the entry point for the cTrader subprocess spawned by
``CTraderBrokerProxy``.  It runs a real ``CTraderBroker`` (with its own
Twisted reactor) in an isolated process so that the reactor can be cleanly
restarted by killing and re-spawning the process—something that is impossible
within a single Python process because ``reactor.run()`` raises
``ReactorNotRestartable`` on a second invocation.

IPC protocol
------------
Commands arrive from the main process via ``cmd_queue`` as plain dicts:
    {"cmd": <str>, "request_id": <str>, **kwargs}

Results and events are sent back via ``event_queue``:
    {"type": "result",           "request_id": str, "value": Any, "error": str|None}
    {"type": "connection_event", "event": <pickled dataclass>}
    {"type": "tick",             "asset": str, "data": dict}
    {"type": "order_status",     "order_request_id": str, "order_id": str, "data": dict}
    {"type": "execution_event",  "data": dict}
    {"type": "state",            "connected": bool, "authenticated": bool, "account_id": int|None}
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def run_worker(cmd_queue, event_queue) -> None:
    """Entry point called by ``multiprocessing.Process``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ctrader-worker] %(name)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(_worker_main(cmd_queue, event_queue))
    except Exception as exc:
        logger.error(f"[SubprocessWorker] Fatal error: {exc}", exc_info=True)
        try:
            event_queue.put_nowait({"type": "fatal_error", "error": str(exc)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put(event_queue, msg: dict) -> None:
    """Put a message onto the event queue, dropping silently on overflow."""
    try:
        event_queue.put_nowait(msg)
    except Exception:
        pass


def _emit_state(event_queue, broker) -> None:
    _put(event_queue, {
        "type": "state",
        "connected": broker.connected,
        "authenticated": broker.authenticated,
        "account_id": getattr(broker, "account_id", None),
    })


async def _worker_main(cmd_queue, event_queue) -> None:
    from live_trading.notifications.connection_event_bus import ConnectionEventBus
    from live_trading.brokers.ctrader_broker import CTraderBroker

    bus = ConnectionEventBus()
    broker = CTraderBroker(event_bus=bus)

    # -----------------------------------------------------------------------
    # Forward connection events to main process
    # -----------------------------------------------------------------------
    def _forward_connection_event(event: Any) -> None:
        _put(event_queue, {"type": "connection_event", "event": event})

    bus.subscribe(_forward_connection_event)

    # -----------------------------------------------------------------------
    # Forward global execution events (order fills, broker-initiated closes)
    # -----------------------------------------------------------------------
    def _forward_execution_event(event_data: dict) -> None:
        _put(event_queue, {"type": "execution_event", "data": event_data})

    broker.add_execution_listener(_forward_execution_event)

    # -----------------------------------------------------------------------
    # Bridge multiprocessing.Queue → asyncio.Queue for commands
    # -----------------------------------------------------------------------
    loop = asyncio.get_running_loop()
    asyncio_cmd_queue: asyncio.Queue = asyncio.Queue()

    def _cmd_reader() -> None:
        while True:
            try:
                msg = cmd_queue.get()  # blocks
                loop.call_soon_threadsafe(asyncio_cmd_queue.put_nowait, msg)
                if msg is None or msg.get("cmd") == "shutdown":
                    break
            except Exception as exc:
                logger.error(f"[SubprocessWorker] cmd_reader error: {exc}")
                break

    reader_thread = threading.Thread(target=_cmd_reader, daemon=True, name="CmdReader")
    reader_thread.start()

    # -----------------------------------------------------------------------
    # Command dispatch loop
    # -----------------------------------------------------------------------
    while True:
        msg = await asyncio_cmd_queue.get()
        if msg is None or msg.get("cmd") == "shutdown":
            break

        cmd: str = msg.get("cmd", "")
        request_id: str = msg.get("request_id", "")

        try:
            if cmd == "connect":
                value = await broker.connect()
                _emit_state(event_queue, broker)
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "disconnect":
                await broker.disconnect()
                _put(event_queue, {"type": "result", "request_id": request_id, "value": True, "error": None})

            elif cmd == "start_connection_monitor":
                await broker.start_connection_monitor()
                _put(event_queue, {"type": "result", "request_id": request_id, "value": True, "error": None})

            elif cmd == "subscribe_market_data":
                asset: str = msg["asset"]
                callback_id = msg.get("callback_id")
                bar_sizes = msg.get("bar_sizes")

                def _make_tick_fwd(a: str):
                    def _fwd(tick_data: dict) -> None:
                        _put(event_queue, {"type": "tick", "asset": a, "data": tick_data})
                    return _fwd

                value = await broker.subscribe_market_data(
                    asset, _make_tick_fwd(asset), callback_id, bar_sizes
                )
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "subscribe_live_trendbars":
                await broker.subscribe_live_trendbars(msg["asset"], msg.get("bar_sizes", []))
                _put(event_queue, {"type": "result", "request_id": request_id, "value": True, "error": None})

            elif cmd == "unsubscribe_market_data":
                await broker.unsubscribe_market_data(msg["asset"], msg.get("callback_id"))
                _put(event_queue, {"type": "result", "request_id": request_id, "value": True, "error": None})

            elif cmd == "place_order":
                rid = request_id

                def _make_order_status_fwd(r: str):
                    def _fwd(status_data: dict) -> None:
                        _put(event_queue, {
                            "type": "order_status",
                            "order_request_id": r,
                            "order_id": status_data.get("order_id"),
                            "data": status_data,
                        })
                    return _fwd

                value = await broker.place_order(
                    asset=msg["asset"],
                    action=msg["action"],
                    quantity=msg["quantity"],
                    order_type=msg.get("order_type", "MARKET"),
                    price=msg.get("price"),
                    stop_loss=msg.get("stop_loss"),
                    take_profit=msg.get("take_profit"),
                    order_status_callback=_make_order_status_fwd(rid),
                )
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "cancel_order":
                value = await broker.cancel_order(msg["broker_order_id"])
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "close_position_by_id":
                value = await broker.close_position_by_id(
                    broker_position_id=msg["broker_position_id"],
                    volume=msg.get("volume"),
                )
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "get_positions":
                value = await broker.get_positions()
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "get_account_info":
                value = await broker.get_account_info()
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "get_deal_history":
                value = await broker.get_deal_history(
                    from_timestamp_ms=msg["from_timestamp_ms"],
                    to_timestamp_ms=msg["to_timestamp_ms"],
                )
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "get_open_broker_orders":
                value = await broker.get_open_broker_orders()
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            elif cmd == "fetch_historical_data":
                # The caller's callback cannot cross the process boundary.
                # Collect bars locally here and forward them in the result.
                collected: dict = {"bars": None, "ctx": None}
                bars_ready = asyncio.Event()

                def _collect(bars, ctx, _collected=collected, _ev=bars_ready):
                    _collected["bars"] = bars
                    _collected["ctx"] = ctx
                    loop.call_soon_threadsafe(_ev.set)

                kwargs = {k: v for k, v in msg.items() if k not in ("cmd", "request_id")}
                kwargs["callback"] = _collect
                ok = await broker.fetch_historical_data(**kwargs)
                if ok:
                    try:
                        await asyncio.wait_for(bars_ready.wait(), timeout=90.0)
                    except asyncio.TimeoutError:
                        logger.warning("[SubprocessWorker] fetch_historical_data callback timed out")
                value = {"ok": bool(ok), "bars": collected["bars"], "context": collected["ctx"]}
                _put(event_queue, {"type": "result", "request_id": request_id, "value": value, "error": None})

            else:
                logger.warning(f"[SubprocessWorker] Unknown command: {cmd!r}")
                _put(event_queue, {
                    "type": "result", "request_id": request_id,
                    "value": None, "error": f"Unknown command: {cmd}",
                })

        except Exception as exc:
            logger.error(f"[SubprocessWorker] Error handling {cmd!r}: {exc}", exc_info=True)
            _put(event_queue, {
                "type": "result", "request_id": request_id,
                "value": None, "error": str(exc),
            })

    # Graceful shutdown
    try:
        await broker.disconnect()
    except Exception:
        pass
