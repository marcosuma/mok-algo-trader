"""
Order model.
"""
from datetime import datetime
from typing import Optional

from beanie import Document
from bson import ObjectId
from bson.decimal128 import Decimal128
from pydantic import ConfigDict, Field, model_validator

from live_trading.brokers.middleware.types import (
    CloseReason,
    OrderIntent,
    OrderSide,
    OrderSource,
    OrderStatus,
    OrderType,
)


class Order(Document):
    """Order document — every order submitted to the broker."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation_id: ObjectId = Field(...)
    broker_order_id: Optional[str] = Field(None)
    broker_position_id: Optional[str] = Field(None)

    symbol: str = Field(...)
    side: OrderSide = Field(...)
    order_type: OrderType = Field(default=OrderType.MARKET)
    requested_quantity: float = Field(...)
    requested_price: Optional[float] = Field(None)
    stop_loss: Optional[float] = Field(None)
    take_profit: Optional[float] = Field(None)

    status: OrderStatus = Field(default=OrderStatus.PENDING_SUBMIT)
    filled_quantity: float = Field(default=0.0)
    avg_fill_price: Optional[float] = Field(None)
    commission: float = Field(default=0.0)

    intent: OrderIntent = Field(default=OrderIntent.OPEN)
    close_reason: Optional[CloseReason] = Field(None)
    source: OrderSource = Field(default=OrderSource.STRATEGY)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = Field(None)
    filled_at: Optional[datetime] = Field(None)
    cancelled_at: Optional[datetime] = Field(None)

    @model_validator(mode="before")
    @classmethod
    def convert_decimal128(cls, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, Decimal128):
                    data[key] = float(value.to_decimal())
        return data

    class Settings:
        name = "orders"
        indexes = [
            "operation_id",
            "broker_order_id",
            "broker_position_id",
            [("operation_id", 1), ("created_at", -1)],
            [("operation_id", 1), ("status", 1)],
        ]
