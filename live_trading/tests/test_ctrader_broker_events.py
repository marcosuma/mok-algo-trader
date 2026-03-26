"""Test that CTraderBroker emits the correct events at the correct points."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from live_trading.notifications.connection_event_bus import ConnectionEventBus
from live_trading.notifications.connection_events import (
    ConnectionDropped, ReconnectAttempt, ReconnectExhausted,
    AuthFailed, TokenRefreshFailed,
)


def _make_broker(bus):
    from live_trading.brokers.ctrader_broker import CTraderBroker
    broker = object.__new__(CTraderBroker)
    broker._event_bus = bus
    broker._shutdown_requested = False
    broker._reconnecting = False
    broker._reconnect_attempts = 0
    broker._max_reconnect_attempts = 5
    broker._reconnect_delay = 5
    broker.connected = False
    broker.authenticated = False
    broker._connection_error = None
    broker._auth_error = None
    broker._data_subscriptions = {}
    broker._data_callbacks = {}
    broker._data_callback_ids = {}
    broker._trendbar_subscriptions = {}
    broker._live_trendbar_volumes = {}
    broker._last_message_time = None
    return broker


class TestBrokerEmitsEvents:
    def test_on_disconnected_emits_connection_dropped(self):
        bus = ConnectionEventBus()
        received = []
        bus.subscribe(received.append)
        broker = _make_broker(bus)
        broker._on_disconnected(MagicMock(), "Connection refused")
        assert any(isinstance(e, ConnectionDropped) for e in received)
        dropped = next(e for e in received if isinstance(e, ConnectionDropped))
        assert "Connection refused" in dropped.reason

    def test_on_auth_error_emits_auth_failed(self):
        bus = ConnectionEventBus()
        received = []
        bus.subscribe(received.append)
        broker = _make_broker(bus)
        broker._on_auth_error("INVALID_REQUEST")
        assert any(isinstance(e, AuthFailed) for e in received)

    def test_handle_token_expired_failure_emits_token_refresh_failed(self):
        bus = ConnectionEventBus()
        received = []
        bus.subscribe(received.append)
        broker = _make_broker(bus)
        broker._token_manager = MagicMock()
        broker._token_manager.force_refresh.return_value = False
        broker._handle_token_expired()
        assert any(isinstance(e, TokenRefreshFailed) for e in received)

    @pytest.mark.asyncio
    async def test_attempt_reconnect_emits_reconnect_attempt(self):
        bus = ConnectionEventBus()
        received = []
        bus.subscribe(received.append)
        broker = _make_broker(bus)
        broker._stop_reactor = MagicMock()
        broker.connect = AsyncMock(return_value=False)
        await broker._attempt_reconnect()
        assert any(isinstance(e, ReconnectAttempt) for e in received)
        attempt_event = next(e for e in received if isinstance(e, ReconnectAttempt))
        assert attempt_event.attempt == 1
        assert attempt_event.max_attempts == 5

    @pytest.mark.asyncio
    async def test_attempt_reconnect_emits_exhausted_at_max(self):
        bus = ConnectionEventBus()
        received = []
        bus.subscribe(received.append)
        broker = _make_broker(bus)
        broker._reconnect_attempts = 5  # already at max
        await broker._attempt_reconnect()
        assert any(isinstance(e, ReconnectExhausted) for e in received)
