"""Typed events for trading decisions."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SkippedDecision:
    """Strategy evaluation could not run (no data or error)."""
    operation_id: str
    asset: str
    strategy_name: str
    reason: str  # e.g. "insufficient_data", "strategy_error: missing RSI_14"
    timestamp: datetime = field(default_factory=_now)


@dataclass
class HoldDecision:
    """Strategy ran but produced no signal."""
    operation_id: str
    asset: str
    strategy_name: str
    price: float
    timestamp: datetime = field(default_factory=_now)


@dataclass
class BlockedDecision:
    """Signal generated but blocked before reaching the broker."""
    operation_id: str
    asset: str
    strategy_name: str
    signal_type: str  # "BUY" | "SELL"
    price: float
    reason: str  # "trend_filter" | "pyramiding"
    timestamp: datetime = field(default_factory=_now)


@dataclass
class OrderPlacedDecision:
    """Order submitted to broker and filled."""
    operation_id: str
    asset: str
    strategy_name: str
    signal_type: str  # "BUY" | "SELL"
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    lot_size: float
    timestamp: datetime = field(default_factory=_now)


@dataclass
class PositionClosedDecision:
    """Position closed (TP / SL / manual)."""
    operation_id: str
    asset: str
    strategy_name: str
    side: str  # "LONG" | "SHORT"
    close_price: float
    pnl: float
    close_reason: str
    timestamp: datetime = field(default_factory=_now)
