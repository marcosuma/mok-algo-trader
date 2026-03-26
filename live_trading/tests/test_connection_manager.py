# live_trading/tests/test_connection_manager.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call
from live_trading.notifications.connection_manager import ConnectionManager
from live_trading.notifications.connection_event_bus import ConnectionEventBus
from live_trading.notifications.connection_events import (
    ReconnectExhausted, FullRestartAttempt,
)


def _make_mock_broker(connect_return=True):
    broker = MagicMock()
    broker.connect = AsyncMock(return_value=connect_return)
    broker.disconnect = AsyncMock()
    broker.start_connection_monitor = AsyncMock()
    broker._event_bus = MagicMock()
    return broker


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_connect_delegates_to_broker(self):
        broker = _make_mock_broker(connect_return=True)
        manager = ConnectionManager(broker_factory=lambda bus: broker, restart_delay_seconds=0)
        result = await manager.connect()
        assert result is True
        broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_restart_triggered_on_reconnect_exhausted(self):
        call_count = 0
        brokers = []

        def factory(bus):
            nonlocal call_count
            call_count += 1
            b = _make_mock_broker(connect_return=True)
            b._event_bus = bus
            brokers.append(b)
            return b

        manager = ConnectionManager(broker_factory=factory, restart_delay_seconds=0)
        await manager.connect()

        restart_events = []
        manager._bus.subscribe(lambda e: restart_events.append(e) if isinstance(e, FullRestartAttempt) else None)

        # Simulate ReconnectExhausted
        manager._bus.emit(ReconnectExhausted(attempts=5))
        await asyncio.sleep(0.1)

        # A new broker should have been created
        assert call_count == 2
        assert any(isinstance(e, FullRestartAttempt) for e in restart_events)

    @pytest.mark.asyncio
    async def test_full_restart_count_increments(self):
        brokers_created = []

        def factory(bus):
            b = _make_mock_broker(connect_return=True)
            b._event_bus = bus
            brokers_created.append(b)
            return b

        manager = ConnectionManager(broker_factory=factory, restart_delay_seconds=0)
        await manager.connect()

        restart_events = []
        manager._bus.subscribe(lambda e: restart_events.append(e) if isinstance(e, FullRestartAttempt) else None)

        manager._bus.emit(ReconnectExhausted(attempts=5))
        await asyncio.sleep(0.1)
        manager._bus.emit(ReconnectExhausted(attempts=5))
        await asyncio.sleep(0.1)

        full_restarts = [e for e in restart_events if isinstance(e, FullRestartAttempt)]
        assert full_restarts[0].restart_count == 1
        assert full_restarts[1].restart_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_stops_restart_loop(self):
        broker = _make_mock_broker(connect_return=True)
        manager = ConnectionManager(broker_factory=lambda bus: broker, restart_delay_seconds=0)
        await manager.connect()
        await manager.disconnect()
        assert manager._shutdown is True
        broker.disconnect.assert_called_once()
