"""
Position model.
"""
from datetime import datetime
from typing import Optional
from beanie import Document
from pydantic import Field, ConfigDict
from bson import ObjectId


class Position(Document):
    """Position document (current open positions per operation)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation_id: ObjectId = Field(..., description="Reference to trading_operations")
    contract_symbol: str = Field(..., description="Contract symbol")

    quantity: float = Field(..., description="Quantity (positive = long, negative = short)")
    entry_price: float = Field(..., description="Entry price")
    current_price: float = Field(..., description="Current market price")

    unrealized_pnl: float = Field(default=0.0, description="Unrealized profit/loss")
    unrealized_pnl_pct: float = Field(default=0.0, description="Unrealized profit/loss percentage")

    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")

    # Broker's internal position ID — used to match cTrader deal history on close detection.
    broker_position_id: Optional[str] = Field(None, description="Broker position ID (e.g. cTrader positionId)")

    opened_at: datetime = Field(default_factory=datetime.utcnow, description="When position was opened")
    closed_at: Optional[datetime] = None

    class Settings:
        name = "positions"
        indexes = [
            "operation_id",
            [("operation_id", 1), ("closed_at", 1)],
            [("operation_id", 1), ("opened_at", -1)],
        ]

