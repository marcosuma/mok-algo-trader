# live_trading/tests/test_telegram_notifier.py
import time
import threading
from unittest.mock import MagicMock, patch
from live_trading.notifications.telegram_notifier import TelegramNotifier
from live_trading.notifications.connection_event_bus import ConnectionEventBus
from live_trading.notifications.connection_events import (
    ConnectionDropped, ConnectionRestored, ReconnectAttempt,
    ReconnectExhausted, AuthFailed, SystemStarted, SystemStopped,
    ConnectionStale, FullRestartAttempt, TokenRefreshFailed,
)


class TestTelegramNotifier:
    def _make_notifier(self, batch_window=0.05):
        """Returns a notifier with a fast batch window and a mock send function."""
        sent = []
        def fake_send(token, chat_id, text):
            sent.append(text)
        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123",
            environment="LIVE",
            batch_window_seconds=batch_window,
            _send_fn=fake_send,
        )
        return notifier, sent

    def test_single_event_produces_one_message(self):
        notifier, sent = self._make_notifier()
        notifier.on_event(ConnectionDropped(reason="Connection refused"))
        time.sleep(0.2)
        assert len(sent) == 1
        assert "Connection refused" in sent[0]

    def test_multiple_events_batched_into_one_message(self):
        notifier, sent = self._make_notifier()
        notifier.on_event(ConnectionDropped(reason="refused"))
        notifier.on_event(ReconnectAttempt(attempt=1, max_attempts=5, delay_seconds=5))
        notifier.on_event(ReconnectAttempt(attempt=2, max_attempts=5, delay_seconds=10))
        time.sleep(0.2)
        assert len(sent) == 1
        assert "refused" in sent[0]
        assert "1/5" in sent[0]
        assert "2/5" in sent[0]

    def test_events_after_flush_start_new_batch(self):
        notifier, sent = self._make_notifier(batch_window=0.05)
        notifier.on_event(ConnectionDropped(reason="first"))
        time.sleep(0.2)
        notifier.on_event(ConnectionRestored(was_down_for_seconds=10.0))
        time.sleep(0.2)
        assert len(sent) == 2

    def test_red_emoji_for_degraded_events(self):
        notifier, sent = self._make_notifier()
        notifier.on_event(ConnectionDropped(reason="refused"))
        time.sleep(0.2)
        assert "🔴" in sent[0]

    def test_green_emoji_for_recovery_events(self):
        notifier, sent = self._make_notifier()
        notifier.on_event(ConnectionRestored(was_down_for_seconds=30.0))
        time.sleep(0.2)
        assert "✅" in sent[0]

    def test_yellow_emoji_for_stale_event(self):
        notifier, sent = self._make_notifier()
        notifier.on_event(ConnectionStale(seconds_since_last_message=95.0))
        time.sleep(0.2)
        assert "🟡" in sent[0]

    def test_send_failure_does_not_raise(self):
        def failing_send(token, chat_id, text):
            raise ConnectionError("network error")
        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123",
            environment="DEMO",
            batch_window_seconds=0.05,
            _send_fn=failing_send,
        )
        notifier.on_event(ConnectionDropped(reason="test"))
        time.sleep(0.2)  # must not raise

    def test_disabled_when_no_token(self):
        sent = []
        def fake_send(token, chat_id, text):
            sent.append(text)
        notifier = TelegramNotifier(
            bot_token=None,
            chat_id="123",
            environment="LIVE",
            batch_window_seconds=0.05,
            _send_fn=fake_send,
        )
        notifier.on_event(ConnectionDropped(reason="test"))
        time.sleep(0.2)
        assert sent == []

    def test_disabled_when_no_chat_id(self):
        sent = []
        def fake_send(token, chat_id, text):
            sent.append(text)
        notifier = TelegramNotifier(
            bot_token="token",
            chat_id=None,
            environment="LIVE",
            batch_window_seconds=0.05,
            _send_fn=fake_send,
        )
        notifier.on_event(ConnectionDropped(reason="test"))
        time.sleep(0.2)
        assert sent == []
