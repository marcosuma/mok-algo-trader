# live_trading/notifications/telegram_notifier.py
"""Telegram notifier: batches connection events and sends summary messages."""
import logging
import threading
from datetime import datetime
from typing import Callable, List, Optional

import requests

from live_trading.notifications.connection_events import (
    AuthFailed, ConnectionDropped, ConnectionRestored, ConnectionStale,
    FullRestartAttempt, ReconnectAttempt, ReconnectExhausted,
    SystemStarted, SystemStopped, TokenRefreshFailed,
)

logger = logging.getLogger(__name__)

# Events that indicate degraded state (🔴)
_DEGRADED_TYPES = (
    ConnectionDropped, ReconnectAttempt, ReconnectExhausted,
    FullRestartAttempt, AuthFailed, TokenRefreshFailed, SystemStopped,
)
# Events that indicate recovery (✅)
_RECOVERY_TYPES = (ConnectionRestored, SystemStarted)
# Events that indicate a warning (🟡)
_WARNING_TYPES = (ConnectionStale,)


def _default_send(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    if not response.ok:
        raise RuntimeError(f"Telegram API error {response.status_code}: {response.text}")


def _format_event(event) -> str:
    if isinstance(event, SystemStarted):
        return f"System started (environment: {event.environment})"
    if isinstance(event, SystemStopped):
        return f"System stopped: {event.reason}"
    if isinstance(event, ConnectionDropped):
        return f"Connection dropped: {event.reason}"
    if isinstance(event, ConnectionStale):
        return f"Connection stale ({event.seconds_since_last_message:.0f}s since last message)"
    if isinstance(event, ConnectionRestored):
        return f"Connection restored (was down {event.was_down_for_seconds:.0f}s)"
    if isinstance(event, ReconnectAttempt):
        return f"Reconnect attempt {event.attempt}/{event.max_attempts} (delay: {event.delay_seconds}s)"
    if isinstance(event, ReconnectExhausted):
        return f"All {event.attempts} reconnect attempts failed — starting full restart"
    if isinstance(event, FullRestartAttempt):
        return f"Full restart #{event.restart_count} — tearing down and reconnecting from scratch"
    if isinstance(event, AuthFailed):
        return f"Authentication failed: {event.reason}"
    if isinstance(event, TokenRefreshFailed):
        return f"Token refresh failed: {event.reason} — manual regeneration may be required"
    return str(event)


def _classify_batch(events: list) -> str:
    """Return the emoji for the batch based on the most severe event type."""
    for event in events:
        if isinstance(event, _DEGRADED_TYPES):
            return "🔴"
    for event in events:
        if isinstance(event, _WARNING_TYPES):
            return "🟡"
    return "✅"


class TelegramNotifier:
    """Subscribes to a ConnectionEventBus, batches events for batch_window_seconds, sends one message."""

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        environment: str,
        batch_window_seconds: float = 30.0,
        _send_fn: Callable = _default_send,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._environment = environment
        self._batch_window = batch_window_seconds
        self._send_fn = _send_fn
        self._enabled = bool(bot_token and chat_id)

        self._buffer: List = []
        self._buffer_lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        if not self._enabled:
            logger.warning(
                "[Telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "Telegram notifications are disabled."
            )

    def on_event(self, event) -> None:
        if not self._enabled:
            return
        with self._buffer_lock:
            self._buffer.append(event)
            if self._timer is None:
                self._timer = threading.Timer(self._batch_window, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self) -> None:
        with self._buffer_lock:
            events = list(self._buffer)
            self._buffer.clear()
            self._timer = None

        if not events:
            return

        emoji = _classify_batch(events)
        timestamp = events[0].timestamp.strftime("%H:%M:%S")
        lines = [f"{emoji} cTrader Alert — {self._environment} [{timestamp}]", ""]
        for event in events:
            lines.append(f"• {_format_event(event)}")
        text = "\n".join(lines)

        try:
            self._send_fn(self._token, self._chat_id, text)
            logger.info(f"[Telegram] Sent alert ({len(events)} events)")
        except Exception:
            logger.warning("[Telegram] Failed to send alert — dropping batch", exc_info=True)
