"""
Position model — covers the full lifecycle from OPEN to CLOSED.

A closed Position with ``status == CLOSED`` is the completed trade record.
No separate Transaction or Trade collection is needed.
"""
from datetime import datetime
from typing import Optional

from beanie import Document
from bson import ObjectId
from pydantic import ConfigDict, Field

from live_trading.brokers.middleware.types import (
    CloseReason,
    PositionSide,
    PositionStatus,
)


class Position(Document):
    """Position document — open and historical positions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation_id: ObjectId = Field(...)
    broker_position_id: str = Field(...)

    symbol: str = Field(...)
    side: PositionSide = Field(...)

    entry_price: float = Field(...)
    quantity: float = Field(..., description="Always positive, in lots")
    current_price: float = Field(...)
    unrealized_pnl: float = Field(default=0.0)
    unrealized_pnl_pct: float = Field(default=0.0)

    stop_loss: Optional[float] = Field(None)
    take_profit: Optional[float] = Field(None)
    swap: float = Field(default=0.0)
    commission: float = Field(default=0.0)

    status: PositionStatus = Field(default=PositionStatus.OPEN)

    close_price: Optional[float] = Field(None)
    close_reason: Optional[CloseReason] = Field(None)
    realized_pnl: Optional[float] = Field(None)
    realized_pnl_pct: Optional[float] = Field(None)
    total_commission: Optional[float] = Field(None)
    total_swap: Optional[float] = Field(None)
    duration_seconds: Optional[float] = Field(None)

    entry_order_id: Optional[ObjectId] = Field(None)
    exit_order_id: Optional[ObjectId] = Field(None)

    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = Field(None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "positions"
        indexes = [
            "operation_id",
            "broker_position_id",
            [("operation_id", 1), ("status", 1)],
            [("operation_id", 1), ("opened_at", -1)],
            [("operation_id", 1), ("closed_at", -1)],
        ]
