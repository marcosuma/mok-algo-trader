"""Thread-safe publish/subscribe bus for connection events."""
import logging
import threading
from typing import Callable, List

logger = logging.getLogger(__name__)


class ConnectionEventBus:
    """Synchronous pub/sub bus. Thread-safe emit; exceptions in subscribers are caught."""

    def __init__(self) -> None:
        self._subscribers: List[Callable] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def emit(self, event) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    f"[EventBus] Subscriber {callback!r} raised an exception "
                    f"on event {type(event).__name__} — ignoring"
                )
