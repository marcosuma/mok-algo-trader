"""
Typed contracts for the broker middleware layer.

These Pydantic models and enums define the contract between the application
logic (OrderManager, Reconciler) and any broker adapter implementation.
All broker-specific data must be translated into these types before reaching
the application layer.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AccountMode(str, Enum):
    HEDGING = "HEDGING"
    NETTING = "NETTING"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class OrderIntent(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class OrderSource(str, Enum):
    STRATEGY = "STRATEGY"
    MANUAL = "MANUAL"
    BROKER_DETECTED = "BROKER_DETECTED"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class CloseReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"


class OrderParams(BaseModel):
    """Parameters for submitting an order to the broker."""

    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(..., gt=0, description="Quantity in lots")
    price: Optional[float] = Field(None, description="Limit/stop price")
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    broker_position_id: Optional[str] = Field(
        None,
        description="Target position for close orders (hedging mode)",
    )


class BrokerOrderResult(BaseModel):
    """Immediate response after submitting an order to the broker."""

    broker_order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    broker_position_id: Optional[str] = None
    commission: float = 0.0


class BrokerOrderUpdate(BaseModel):
    """Asynchronous status update for an order (fill, cancel, reject, etc.)."""

    broker_order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    broker_position_id: Optional[str] = None
    commission: float = 0.0
    is_position_close: bool = False


class BrokerPosition(BaseModel):
    """Snapshot of a single position as reported by the broker."""

    broker_position_id: str
    symbol: str
    side: PositionSide
    quantity: float = Field(..., gt=0, description="Absolute quantity in lots")
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    swap: float = 0.0
    commission: float = 0.0


class BrokerDeal(BaseModel):
    """A single execution/deal record from the broker's history."""

    deal_id: str
    broker_position_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: OrderSide
    execution_price: float
    quantity: float = Field(..., ge=0)
    is_close: bool = False
    commission: float = 0.0
    swap: float = 0.0
    timestamp: Optional[datetime] = None


class BrokerAccountInfo(BaseModel):
    """Account-level information from the broker."""

    balance: float
    equity: Optional[float] = None
    margin_used: Optional[float] = None
    margin_free: Optional[float] = None
    currency: str = "USD"
