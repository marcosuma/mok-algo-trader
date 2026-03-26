"""Typed events emitted by the cTrader connection layer."""
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SystemStarted:
    environment: str
    timestamp: datetime = field(default_factory=_now)


@dataclass
class SystemStopped:
    reason: str
    timestamp: datetime = field(default_factory=_now)


@dataclass
class ConnectionDropped:
    reason: str
    timestamp: datetime = field(default_factory=_now)


@dataclass
class ConnectionStale:
    seconds_since_last_message: float
    timestamp: datetime = field(default_factory=_now)


@dataclass
class ConnectionRestored:
    was_down_for_seconds: float
    timestamp: datetime = field(default_factory=_now)


@dataclass
class ReconnectAttempt:
    attempt: int
    max_attempts: int
    delay_seconds: int
    timestamp: datetime = field(default_factory=_now)


@dataclass
class ReconnectExhausted:
    attempts: int
    timestamp: datetime = field(default_factory=_now)


@dataclass
class FullRestartAttempt:
    restart_count: int
    timestamp: datetime = field(default_factory=_now)


@dataclass
class AuthFailed:
    reason: str
    timestamp: datetime = field(default_factory=_now)


@dataclass
class TokenRefreshFailed:
    reason: str
    timestamp: datetime = field(default_factory=_now)
