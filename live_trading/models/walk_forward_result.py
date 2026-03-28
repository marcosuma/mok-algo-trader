"""
Walk Forward Result model.
"""
from datetime import datetime
from typing import Any, Dict

from beanie import Document
from pymongo import ASCENDING, DESCENDING
from pydantic import Field


class WalkForwardResult(Document):
    """Stores the best walk-forward parameters and performance stats for an (asset, bar_size, strategy) triple."""

    asset: str
    bar_size: str
    strategy_name: str
    best_params: Dict[str, Any]
    mean_sharpe: float
    sharpe_std: float
    beat_bh_pct: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "walk_forward_results"
        indexes = [
            [
                ("asset", ASCENDING),
                ("bar_size", ASCENDING),
                ("strategy_name", ASCENDING),
                ("generated_at", DESCENDING),
            ],
        ]
