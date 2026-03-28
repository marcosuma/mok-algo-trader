"""
Scheduled walk-forward runner.

Runs weekly via cron to re-validate strategy parameters and update MongoDB
with the latest walk-forward results. Alerts if live operation params drift
from optimal.

Run:
    python -m scripts.scheduled_walkforward
"""
import asyncio
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from live_trading.models import TradingOperation, WalkForwardResult

logger = logging.getLogger(__name__)

# Map asset strings to data directories (relative to project root)
ASSET_DATA_DIRS = {
    "USD-CAD": "data/USD-CAD",
    "XAU-USD": "data/XAU-USD",
    "EUR-USD": "data/EUR-USD",
    "GBP-USD": "data/GBP-USD",
    "USD-JPY": "data/USD-JPY",
}


def parse_best_configs(markdown: str) -> List[Dict[str, Any]]:
    """
    Parse the top-ranked strategy configs from a walk-forward markdown report.

    Returns a list of dicts with keys:
      strategy_name, params_str, mean_sharpe, beat_bh_pct, windows
    """
    results = []
    # Match table rows: | rank | Strategy | Key Params | Mean Sharpe | Beat B&H | ... |
    row_pattern = re.compile(
        r"\|\s*\d+\s*\|"               # rank column
        r"\s*([^|]+?)\s*\|"            # strategy name
        r"\s*([^|]+?)\s*\|"            # params string
        r"\s*([-\d.]+)\s*\|"           # mean sharpe
        r"\s*([\d.]+)%\s*\|",          # beat b&h %
    )
    for m in row_pattern.finditer(markdown):
        strategy_name = m.group(1).strip()
        params_str = m.group(2).strip()
        try:
            mean_sharpe = float(m.group(3))
            beat_bh_pct = float(m.group(4)) / 100.0
        except ValueError:
            continue
        results.append({
            "strategy_name": strategy_name,
            "params_str": params_str,
            "mean_sharpe": mean_sharpe,
            "beat_bh_pct": beat_bh_pct,
        })
    return results


def parse_params_string(params_str: str) -> Dict[str, Any]:
    """
    Parse a params string like 'lookback=10, mult=1.5' into a dict.

    Handles int and float values.
    """
    params: Dict[str, Any] = {}
    for part in params_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        try:
            # Try int first, then float
            if "." in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            params[key] = value  # Keep as string
    return params


def _estimate_sharpe_std(markdown: str, strategy_name: str) -> float:
    """
    Estimate Sharpe std from per-window results if available in markdown.
    Falls back to 0.5 if not parseable.
    """
    # Look for a per-window Sharpe table or summary block for this strategy.
    # Format varies; use a conservative fallback if not found.
    pattern = re.compile(
        rf"{re.escape(strategy_name)}.*?\bstd\b[:\s]*([\d.]+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(markdown)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    logger.debug("Could not parse Sharpe std for %s; using fallback 0.5", strategy_name)
    return 0.5  # conservative fallback


async def _run_checks(active_ops: List[TradingOperation]) -> None:
    """Discover unique (asset, bar_size) combos and run walk-forward for each."""
    pairs: set = set()
    for op in active_ops:
        pairs.add((op.asset, op.primary_bar_size))

    output_dir = Path("strategy_test_results")
    output_dir.mkdir(exist_ok=True)

    for asset, bar_size in pairs:
        data_dir = ASSET_DATA_DIRS.get(asset)
        if data_dir is None:
            logger.warning("No data dir mapping for asset %s — skipping", asset)
            continue

        symbol = asset.replace("-", "")  # "USD-CAD" → "USDCAD"
        logger.info("Running walk-forward: %s %s", asset, bar_size)

        cmd = [
            "python", "walk_forward_analyzer.py",
            "--symbol", symbol,
            "--data", data_dir,
            "--output", str(output_dir),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max
            )
            if result.returncode != 0:
                logger.error("walk_forward_analyzer failed for %s: %s", asset, result.stderr[-500:])
                continue
        except subprocess.TimeoutExpired:
            logger.error("Walk-forward timed out for %s", asset)
            continue
        except Exception as exc:
            logger.error("Walk-forward error for %s: %s", asset, exc)
            continue

        # Find the most recently modified markdown for this symbol
        md_files = sorted(
            output_dir.glob(f"{symbol}_walkforward_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not md_files:
            logger.warning("No markdown output found for %s", symbol)
            continue

        markdown = md_files[0].read_text()
        configs = parse_best_configs(markdown)

        if not configs:
            logger.warning("Could not parse any configs from %s", md_files[0].name)
            continue

        # Keep only the top config per strategy_name
        best_by_strategy: Dict[str, Dict] = {}
        for cfg in configs:
            name = cfg["strategy_name"]
            if name not in best_by_strategy:
                best_by_strategy[name] = cfg

        for strategy_name, cfg in best_by_strategy.items():
            best_params = parse_params_string(cfg["params_str"])
            sharpe_std = _estimate_sharpe_std(markdown, strategy_name)

            # Upsert: find existing record or create new
            existing = await WalkForwardResult.find(
                WalkForwardResult.asset == asset,
                WalkForwardResult.bar_size == bar_size,
                WalkForwardResult.strategy_name == strategy_name,
            ).sort(-WalkForwardResult.generated_at).first_or_none()

            if existing is not None:
                existing.best_params = best_params
                existing.mean_sharpe = cfg["mean_sharpe"]
                existing.sharpe_std = sharpe_std
                existing.beat_bh_pct = cfg["beat_bh_pct"]
                existing.generated_at = datetime.utcnow()
                await existing.save()
                logger.info(
                    "Updated WalkForwardResult: %s %s %s Sharpe=%.2f",
                    asset, bar_size, strategy_name, cfg["mean_sharpe"],
                )
            else:
                record = WalkForwardResult(
                    asset=asset,
                    bar_size=bar_size,
                    strategy_name=strategy_name,
                    best_params=best_params,
                    mean_sharpe=cfg["mean_sharpe"],
                    sharpe_std=sharpe_std,
                    beat_bh_pct=cfg["beat_bh_pct"],
                    generated_at=datetime.utcnow(),
                )
                await record.insert()
                logger.info(
                    "Inserted WalkForwardResult: %s %s %s Sharpe=%.2f",
                    asset, bar_size, strategy_name, cfg["mean_sharpe"],
                )


async def run() -> None:
    """Main entry point."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "trading_db")

    client = AsyncIOMotorClient(mongo_uri)
    try:
        await init_beanie(
            database=client[db_name],
            document_models=[TradingOperation, WalkForwardResult],
        )

        active_ops = await TradingOperation.find(
            TradingOperation.status == "active"
        ).to_list()

        if not active_ops:
            logger.info("No active operations — nothing to run")
            return

        await _run_checks(active_ops)
    except Exception as exc:
        logger.error("Scheduled walk-forward failed: %s", exc)
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
