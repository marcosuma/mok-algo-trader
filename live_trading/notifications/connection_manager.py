# live_trading/notifications/connection_manager.py
"""Orchestrates CTraderBroker lifecycle with indefinite full-restart recovery."""
import asyncio
import logging
from typing import Callable, Optional

from live_trading.notifications.connection_event_bus import ConnectionEventBus
from live_trading.notifications.connection_events import (
    FullRestartAttempt, ReconnectExhausted,
)

logger = logging.getLogger(__name__)

_RESTART_DELAY_SECONDS = 300  # 5 minutes between full restarts


class ConnectionManager:
    """
    Wraps a CTraderBroker.
    - On ReconnectExhausted: tears down broker, waits restart_delay_seconds, creates
      a fresh broker via broker_factory, and calls connect() again.
    - Loops indefinitely until disconnect() is called.
    """

    def __init__(
        self,
        broker_factory: Callable[["ConnectionEventBus"], object],
        restart_delay_seconds: int = _RESTART_DELAY_SECONDS,
    ) -> None:
        self._broker_factory = broker_factory
        self._restart_delay = restart_delay_seconds
        self._bus = ConnectionEventBus()
        self._broker = None
        self._shutdown = False
        self._restart_count = 0
        self._restart_task: Optional[asyncio.Task] = None

        self._bus.subscribe(self._on_event)

    def _on_event(self, event) -> None:
        if isinstance(event, ReconnectExhausted) and not self._shutdown:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._full_restart())
                    )
            except Exception:
                logger.exception("[ConnectionManager] Failed to schedule full restart")

    async def connect(self) -> bool:
        self._broker = self._broker_factory(self._bus)
        result = await self._broker.connect()
        if result:
            await self._broker.start_connection_monitor()
        return result

    async def disconnect(self) -> None:
        self._shutdown = True
        if self._broker:
            await self._broker.disconnect()

    async def _full_restart(self) -> None:
        if self._shutdown:
            return

        self._restart_count += 1
        self._bus.emit(FullRestartAttempt(restart_count=self._restart_count))

        logger.critical(
            f"[ConnectionManager] All fast reconnects failed. "
            f"Tearing down and restarting full connection in {self._restart_delay}s "
            f"(restart #{self._restart_count})..."
        )

        # Tear down current broker
        if self._broker:
            try:
                await self._broker.disconnect()
            except Exception:
                logger.exception("[ConnectionManager] Error during broker teardown before restart")
            self._broker = None

        if self._shutdown:
            return

        await asyncio.sleep(self._restart_delay)

        if self._shutdown:
            return

        logger.info(f"[ConnectionManager] Starting fresh connection (restart #{self._restart_count})...")
        try:
            self._broker = self._broker_factory(self._bus)
            success = await self._broker.connect()
            if success:
                await self._broker.start_connection_monitor()
                logger.info(f"[ConnectionManager] Full restart #{self._restart_count} succeeded.")
            else:
                logger.error(
                    f"[ConnectionManager] Full restart #{self._restart_count} connect() returned False. "
                    f"Will retry on next ReconnectExhausted event."
                )
        except Exception:
            logger.exception(
                f"[ConnectionManager] Exception during full restart #{self._restart_count}. "
                f"Will retry on next ReconnectExhausted event."
            )
