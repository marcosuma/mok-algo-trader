"""
Trading Operation model.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from beanie import Document
from pydantic import Field


class TradingOperation(Document):
    """Trading operation document"""

    asset: str = Field(..., description="Asset symbol (e.g., 'USD-CAD')")
    bar_sizes: List[str] = Field(..., description="List of bar sizes (e.g., ['1 hour', '15 mins', '1 day'])")
    primary_bar_size: str = Field(..., description="Primary timeframe for entry/exit")
    strategy_name: str = Field(..., description="Strategy class name (e.g., 'MomentumStrategy')")
    strategy_config: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    broker_type: str = Field(default="IBKR", description="Broker type: 'IBKR', 'OANDA', 'PEPPERSTONE'")

    status: str = Field(default="active", description="Operation status: 'active', 'paused', 'closed', 'error'")

    # Risk management config
    stop_loss_type: str = Field(default="ATR", description="Stop loss type: 'ATR', 'PERCENTAGE', 'FIXED'")
    stop_loss_value: float = Field(default=1.5, description="Stop loss value (multiplier for ATR, percentage for PERCENTAGE, absolute for FIXED)")
    take_profit_type: str = Field(default="RISK_REWARD", description="Take profit type: 'ATR', 'PERCENTAGE', 'FIXED', 'RISK_REWARD'")
    take_profit_value: float = Field(default=2.0, description="Take profit value (multiplier for ATR/RISK_REWARD, percentage for PERCENTAGE, absolute for FIXED)")

    # Pyramiding
    allow_pyramiding: bool = Field(default=False, description="Allow multiple same-direction entries")
    max_pyramid_entries: int = Field(default=1, description="Max concurrent entries in the same direction")

    # Trend direction filter (optional): only allow signals that align with the EMA trend.
    # SELL signals are blocked when close > EMA; BUY signals are blocked when close < EMA.
    trend_filter_enabled: bool = Field(default=False, description="Block counter-trend signals using EMA")
    trend_filter_ma_period: int = Field(default=200, description="EMA period used for trend direction filter")

    # Crash recovery config
    crash_recovery_mode: str = Field(default="CLOSE_ALL", description="Crash recovery mode: 'CLOSE_ALL', 'RESUME', 'EMERGENCY_EXIT'")
    emergency_stop_loss_pct: float = Field(default=0.05, description="Emergency exit threshold (e.g., 0.05 for 5%)")

    # Data retention
    data_retention_bars: int = Field(default=1000, description="Number of bars to keep per bar_size")

    # Capital tracking
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    current_capital: float = Field(default=10000.0, description="Current capital")
    total_pnl: float = Field(default=0.0, description="Total profit/loss")
    total_pnl_pct: float = Field(default=0.0, description="Total profit/loss percentage")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    class Settings:
        name = "trading_operations"
        indexes = [
            "status",
            "asset",
            "created_at",
            [("status", 1), ("created_at", -1)],
        ]

