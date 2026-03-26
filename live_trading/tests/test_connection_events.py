from datetime import datetime
from live_trading.notifications.connection_events import (
    SystemStarted, SystemStopped,
    ConnectionDropped, ConnectionStale, ConnectionRestored,
    ReconnectAttempt, ReconnectExhausted, FullRestartAttempt,
    AuthFailed, TokenRefreshFailed,
)


class TestConnectionEvents:
    def test_system_started_fields(self):
        e = SystemStarted(environment="LIVE")
        assert e.environment == "LIVE"
        assert isinstance(e.timestamp, datetime)

    def test_system_stopped_fields(self):
        e = SystemStopped(reason="KeyboardInterrupt")
        assert e.reason == "KeyboardInterrupt"
        assert isinstance(e.timestamp, datetime)

    def test_connection_dropped_fields(self):
        e = ConnectionDropped(reason="Connection refused")
        assert e.reason == "Connection refused"
        assert isinstance(e.timestamp, datetime)

    def test_connection_stale_fields(self):
        e = ConnectionStale(seconds_since_last_message=95.3)
        assert e.seconds_since_last_message == 95.3
        assert isinstance(e.timestamp, datetime)

    def test_connection_restored_fields(self):
        e = ConnectionRestored(was_down_for_seconds=47.2)
        assert e.was_down_for_seconds == 47.2
        assert isinstance(e.timestamp, datetime)

    def test_reconnect_attempt_fields(self):
        e = ReconnectAttempt(attempt=3, max_attempts=5, delay_seconds=15)
        assert e.attempt == 3
        assert e.max_attempts == 5
        assert e.delay_seconds == 15
        assert isinstance(e.timestamp, datetime)

    def test_reconnect_exhausted_fields(self):
        e = ReconnectExhausted(attempts=5)
        assert e.attempts == 5
        assert isinstance(e.timestamp, datetime)

    def test_full_restart_attempt_fields(self):
        e = FullRestartAttempt(restart_count=2)
        assert e.restart_count == 2
        assert isinstance(e.timestamp, datetime)

    def test_auth_failed_fields(self):
        e = AuthFailed(reason="INVALID_REQUEST")
        assert e.reason == "INVALID_REQUEST"
        assert isinstance(e.timestamp, datetime)

    def test_token_refresh_failed_fields(self):
        e = TokenRefreshFailed(reason="HTTP 401")
        assert e.reason == "HTTP 401"
        assert isinstance(e.timestamp, datetime)
