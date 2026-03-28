"""
Performance monitor: daily circuit breaker for live trading operations.

Queries closed positions, computes realized Sharpe, compares to walk-forward
baselines, and sends Telegram alerts when performance degrades.

Run daily via cron:
    python -m scripts.performance_monitor
"""
import asyncio
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from live_trading.models import Position, TradingOperation, WalkForwardResult

logger = logging.getLogger(__name__)

# Threshold multipliers
WARN_SIGMA = 2.0
PAUSE_SIGMA = 3.0

# Minimum days of data before computing Sharpe
MIN_DAYS = 5


def compute_sharpe(daily_pnl: List[float], capital: float) -> Optional[float]:
    """
    Compute annualized Sharpe from a list of daily P&L values.

    Returns None if there are fewer than MIN_DAYS data points.
    Returns 0.0 if std dev is zero (flat returns — safe fallback).
    """
    if len(daily_pnl) < MIN_DAYS:
        return None

    daily_returns = [pnl / capital for pnl in daily_pnl]
    n = len(daily_returns)
    mean = sum(daily_returns) / n
    variance = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(variance)

    if std == 0:
        return 0.0

    return (mean / std) * math.sqrt(252)


def detect_drift(
    live_params: Dict[str, Any],
    optimal_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare live operation params against walk-forward optimal params.

    Returns a list of dicts with keys 'param', 'live', 'optimal' for each
    parameter that differs. Extra params in live (not in optimal) are not flagged.
    """
    drifted = []
    for param, optimal_value in optimal_params.items():
        live_value = live_params.get(param)
        if live_value != optimal_value:
            drifted.append({
                "param": param,
                "live": live_value,
                "optimal": optimal_value,
            })
    return drifted


def _send_telegram(message: str) -> None:
    """Send a Telegram message using env vars BOT_TOKEN and CHAT_ID."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.warning("Telegram env vars not set — skipping alert")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)


async def _pause_operation(operation_id: str) -> None:
    """Set operation status to 'paused' in MongoDB."""
    op = await TradingOperation.get(operation_id)
    if op and op.status != "paused":
        op.status = "paused"
        await op.save()
        logger.info("Auto-paused operation %s", operation_id)


async def _run_checks() -> None:
    """Execute all performance checks for active operations."""
    try:
        active_ops = await TradingOperation.find(
            TradingOperation.status == "active"
        ).to_list()
    except Exception as exc:
        logger.error("Failed to query active operations: %s", exc)
        return

    if not active_ops:
        logger.info("No active operations found")
        return

    since = datetime.now(timezone.utc) - timedelta(days=30)

    for op in active_ops:
        try:
            op_id = str(op.id)
            asset = op.asset
            bar_size = op.primary_bar_size
            strategy_name = op.strategy_name
            strategy_config = op.strategy_config or {}

            # Load walk-forward baseline
            wf = await WalkForwardResult.find(
                WalkForwardResult.asset == asset,
                WalkForwardResult.bar_size == bar_size,
                WalkForwardResult.strategy_name == strategy_name,
            ).sort(-WalkForwardResult.generated_at).first_or_none()

            if wf is None:
                logger.info("No walk-forward baseline for op %s (%s %s %s)", op_id, asset, bar_size, strategy_name)
                continue

            # Check parameter drift
            drifted = detect_drift(strategy_config, wf.best_params)
            if drifted:
                lines = "\n".join(
                    f"  {d['param']}: live={d['live']} optimal={d['optimal']}"
                    for d in drifted
                )
                _send_telegram(
                    f"⚠️ Param drift detected — {asset} {strategy_name}\n{lines}"
                )

            # Compute realized Sharpe from last 30 days of closed positions
            positions = await Position.find(
                Position.operation_id == op_id,
                Position.status == "closed",
                Position.closed_at >= since,
            ).to_list()

            if not positions:
                logger.info("No closed positions in last 30 days for op %s", op_id)
                continue

            # Group by calendar date, sum realized_pnl per day
            daily: Dict[str, float] = {}
            for pos in positions:
                if pos.closed_at is None:
                    continue
                day = pos.closed_at.strftime("%Y-%m-%d")
                daily[day] = daily.get(day, 0.0) + (pos.realized_pnl or 0.0)

            daily_pnl = list(daily.values())
            capital = op.initial_capital
            sharpe = compute_sharpe(daily_pnl, capital)

            if sharpe is None:
                logger.info("Insufficient data for Sharpe on op %s (%d days)", op_id, len(daily_pnl))
                continue

            mean_wf = wf.mean_sharpe
            std_wf = wf.sharpe_std

            warn_threshold = mean_wf - WARN_SIGMA * std_wf
            pause_threshold = mean_wf - PAUSE_SIGMA * std_wf

            logger.info(
                "Op %s Sharpe=%.2f  wf_mean=%.2f  warn<%.2f  pause<%.2f",
                op_id, sharpe, mean_wf, warn_threshold, pause_threshold,
            )

            if sharpe < pause_threshold:
                msg = (
                    f"🚨 AUTO-PAUSING {asset} {strategy_name} (op {op_id[-6:]})\n"
                    f"Realized Sharpe: {sharpe:.2f}\n"
                    f"Walk-forward baseline: {mean_wf:.2f} ± {std_wf:.2f}\n"
                    f"Threshold (3σ): {pause_threshold:.2f}"
                )
                _send_telegram(msg)
                await _pause_operation(op_id)

            elif sharpe < warn_threshold:
                msg = (
                    f"⚠️ Performance degradation — {asset} {strategy_name} (op {op_id[-6:]})\n"
                    f"Realized Sharpe: {sharpe:.2f}\n"
                    f"Walk-forward baseline: {mean_wf:.2f} ± {std_wf:.2f}\n"
                    f"Threshold (2σ): {warn_threshold:.2f}"
                )
                _send_telegram(msg)
        except Exception as exc:
            logger.error("Error monitoring operation %s: %s", str(op.id), exc)
            continue


async def monitor() -> None:
    """Main monitoring loop: check all active operations."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "trading_db")

    client = AsyncIOMotorClient(mongo_uri)
    try:
        await init_beanie(
            database=client[db_name],
            document_models=[TradingOperation, Position, WalkForwardResult],
        )
        await _run_checks()
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor())
