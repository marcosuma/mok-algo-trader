#!/usr/bin/env python3
"""
Walk-Forward Analyzer
=====================
Evaluates strategy/parameter configs across rolling independent test windows to
identify which configurations consistently beat Buy & Hold on a given instrument.

Usage:
    python walk_forward_analyzer.py --symbol XAUUSD --data data/XAU-USD/
    python walk_forward_analyzer.py --symbol XAUUSD --data-file data/XAU-USD/xauusd_h1.csv

Data requirements:
    Place an OHLCV CSV (columns: date, open, high, low, close, volume) in data/XAU-USD/.
    Technical indicators are auto-computed if missing.
"""

import argparse
import contextlib
import io
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Strategy imports
from forex_strategies.backtesting_strategy import ForexBacktestingStrategy
from forex_strategies.breakout_strategy import ATRBreakout
from forex_strategies.momentum_strategy import MomentumStrategy, TrendMomentumStrategy
from forex_strategies.market_structure_strategy import MarketStructureStrategy
from forex_strategies.institutional_flow_strategy import InstitutionalFlowStrategy
from forex_strategies.adaptive_multi_indicator_strategy import AdaptiveMultiIndicatorStrategy
from forex_strategies.multi_timeframe_strategy import AdaptiveMultiTimeframeStrategy
from forex_strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from technical_indicators.technical_indicators import TechnicalIndicators


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: str, bar_size: str = "1 hour") -> pd.DataFrame:
    """Load OHLCV data from a file or directory; compute indicators if missing.

    When data_path is a directory containing multiple bar-size files (e.g.
    data-AUD-CASH-IDEALPRO-CAD-1 Y-1 hour.csv), only the file whose name ends
    with '-{bar_size}.csv' is loaded.  If no match is found the first CSV is
    used and a warning is printed.
    """
    path = Path(data_path)

    if path.is_file():
        csv_file = path
    elif path.is_dir():
        all_csvs = sorted(path.glob("*.csv"))
        if not all_csvs:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        # Pick the file whose name ends with the requested bar size
        suffix = f"-{bar_size}.csv"
        matching = [f for f in all_csvs if f.name.endswith(suffix)]
        if matching:
            csv_file = matching[0]
        else:
            csv_file = all_csvs[0]
            print(
                f"WARNING: No file ending with '{suffix}' found in {data_path}. "
                f"Using: {csv_file.name}\n"
                f"Available files:\n" +
                "\n".join(f"  {f.name}" for f in all_csvs)
            )
        print(f"Selected: {csv_file.name}")
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    df = pd.read_csv(csv_file, index_col=0)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    # Parse datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Compute indicators if not already present
    indicator_cols = ["RSI_14", "macd", "atr", "adx", "EMA_10"]
    has_indicators = sum(1 for c in indicator_cols if c in df.columns) >= 3
    if not has_indicators:
        print("Computing technical indicators (this may take a moment)...")
        ti = TechnicalIndicators(candlestickData=None, fileToSave=None)
        df = ti.execute(df)
    else:
        if "shifted_open" not in df.columns:
            df["shifted_open"] = df["open"].shift(-1)

    # Merge daily close + SMA_200 for trend_filter backtesting.
    # Only when loading from a directory and the primary bar_size is not already daily.
    if path.is_dir() and bar_size != "1 day":
        daily_suffix = "-1 day.csv"
        daily_files = [f for f in path.glob("*.csv") if f.name.endswith(daily_suffix)]
        if daily_files:
            try:
                daily_raw = pd.read_csv(daily_files[0], index_col=[0])
                daily_raw.index = pd.to_datetime(daily_raw.index, errors="coerce")
                daily_raw = daily_raw[daily_raw.index.notna()].sort_index()
                if "SMA_200" not in daily_raw.columns:
                    daily_raw["SMA_200"] = daily_raw["close"].rolling(200, min_periods=1).mean()
                df["1 day_close"] = daily_raw["close"].reindex(df.index, method="ffill")
                df["1 day_SMA_200"] = daily_raw["SMA_200"].reindex(df.index, method="ffill")
                print(f"Loaded daily data for trend filter: {daily_files[0].name}")
            except Exception as exc:
                print(f"Warning: could not load daily data for trend filter: {exc}")

    print(f"Loaded {len(df):,} bars  [{df.index[0].date()} → {df.index[-1].date()}]")
    return df


# ---------------------------------------------------------------------------
# Strategy config grid
# ---------------------------------------------------------------------------

def build_configs(data_dir: str = "data", contract_name: str = "") -> list[tuple[str, type, dict]]:
    """Return list of (display_name, strategy_class, params_dict) for every config."""
    configs: list[tuple[str, type, dict]] = []

    # ATRBreakout: 3 × 3 = 9 configs
    for lookback, mult in itertools.product([10, 20, 30], [1.0, 1.5, 2.0]):
        configs.append((
            "ATRBreakout",
            ATRBreakout,
            {"lookback_period": lookback, "atr_multiplier": mult},
        ))

    # ATRBreakout with daily trend filter: 3 × 3 = 9 additional configs
    for lookback, mult in itertools.product([10, 20, 30], [1.0, 1.5, 2.0]):
        configs.append((
            "ATRBreakout+TrendFilter",
            ATRBreakout,
            {"lookback_period": lookback, "atr_multiplier": mult, "trend_filter": True},
        ))

    # MomentumStrategy: 2 × 2 × 2 = 8 configs
    for oversold, overbought, mult in itertools.product([25, 30], [70, 75], [1.5, 2.0]):
        configs.append((
            "MomentumStrategy",
            MomentumStrategy,
            {"rsi_oversold": oversold, "rsi_overbought": overbought, "atr_multiplier": mult},
        ))

    # TrendMomentumStrategy: 3 × 2 × 2 = 12 configs
    for adx_thresh, ema_fast, ema_slow in itertools.product([20, 25, 30], [8, 12], [21, 26]):
        configs.append((
            "TrendMomentumStrategy",
            TrendMomentumStrategy,
            {"adx_threshold": adx_thresh, "ema_fast": ema_fast, "ema_slow": ema_slow},
        ))

    # MarketStructureStrategy: 2 × 2 × 2 = 8 configs
    for atr_stop, min_rr, swing_left in itertools.product([1.5, 2.0], [1.5, 2.0], [5, 8]):
        configs.append((
            "MarketStructureStrategy",
            MarketStructureStrategy,
            {"atr_stop_multiplier": atr_stop, "min_rr_ratio": min_rr, "swing_left_bars": swing_left},
        ))

    # InstitutionalFlowStrategy: 2 × 2 × 2 = 8 configs
    for hurst, adx_min, min_rr in itertools.product([0.50, 0.52], [20, 25], [2.0, 2.5]):
        configs.append((
            "InstitutionalFlowStrategy",
            InstitutionalFlowStrategy,
            {"hurst_threshold": hurst, "adx_min": adx_min, "min_rr_ratio": min_rr},
        ))

    # AdaptiveMultiIndicatorStrategy: 2 × 2 = 4 configs
    for adx_thresh, atr_stop in itertools.product([20, 25], [1.5, 2.0]):
        configs.append((
            "AdaptiveMultiIndicatorStrategy",
            AdaptiveMultiIndicatorStrategy,
            {"adx_trend_threshold": adx_thresh, "atr_stop_multiplier": atr_stop},
        ))

    # AdaptiveMultiTimeframeStrategy: 2 × 2 = 4 configs
    # Loads higher-TF CSVs from data_dir/contract_name if available;
    # silently degrades to single-TF (AdaptiveMultiIndicatorStrategy) otherwise.
    for adx_thresh, atr_stop in itertools.product([20, 25], [1.5, 2.0]):
        configs.append((
            "AdaptiveMultiTimeframeStrategy",
            AdaptiveMultiTimeframeStrategy,
            {
                "adx_trend_threshold": adx_thresh,
                "atr_stop_multiplier": atr_stop,
                "data_dir": data_dir,
                "contract_name": contract_name,
            },
        ))

    # BuyAndHoldStrategy: 1 config (baseline — excluded from ranking)
    configs.append(("BuyAndHoldStrategy", BuyAndHoldStrategy, {}))

    return configs


def fmt_params(params: dict) -> str:
    """Compact representation of a params dict."""
    if not params:
        return "(baseline)"
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe from a decimal-return series with DatetimeIndex."""
    if len(returns) < 2:
        return float("nan")
    daily = returns.resample("D").sum()
    daily = daily[daily != 0]
    if len(daily) < 2 or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(252))


def bh_return_pct(test_df: pd.DataFrame) -> float:
    """Simple buy-and-hold return over the test window (%)."""
    if len(test_df) < 2:
        return 0.0
    entry = float(test_df["open"].iloc[0])
    exit_ = float(test_df["close"].iloc[-1])
    if entry == 0:
        return 0.0
    return (exit_ / entry - 1) * 100.0


# ---------------------------------------------------------------------------
# Single window runner
# ---------------------------------------------------------------------------

def run_config(
    strategy_class: type,
    params: dict,
    window_df: pd.DataFrame,
    test_start_dt,
) -> dict | None:
    """
    Backtest one (strategy, params) config on window_df; filter trades to test period.
    Returns metrics dict or None if insufficient trades / error.
    """
    try:
        strategy = strategy_class(**params)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            stats, _ = strategy.execute(window_df, ForexBacktestingStrategy)
    except Exception:
        return None

    if stats is None:
        return None

    trades = getattr(stats, "_trades", None)
    if trades is None or len(trades) == 0:
        return None

    # Normalise test_start_dt timezone to match EntryTime
    entry_times = trades["EntryTime"]
    if hasattr(entry_times.iloc[0], "tzinfo") and entry_times.iloc[0].tzinfo is not None:
        if not hasattr(test_start_dt, "tzinfo") or test_start_dt.tzinfo is None:
            test_start_dt = test_start_dt.tz_localize("UTC")
    else:
        if hasattr(test_start_dt, "tzinfo") and test_start_dt.tzinfo is not None:
            test_start_dt = test_start_dt.tz_localize(None)

    test_trades = trades[trades["EntryTime"] >= test_start_dt].copy()
    n_trades = len(test_trades)
    if n_trades < 3:
        return None

    win_rate = float((test_trades["ReturnPct"] > 0).mean() * 100)
    # ReturnPct is a decimal in backtesting.py (0.10 = 10%)
    total_return = float((test_trades["ReturnPct"] + 1).prod() - 1) * 100

    ret_series = test_trades.set_index("EntryTime")["ReturnPct"]
    sharpe = compute_sharpe(ret_series)

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    configs: list,
    warmup_bars: int,
    test_bars: int,
    step: int,
) -> pd.DataFrame:
    """Run all configs across all windows. Returns raw per-window results DataFrame."""
    n = len(df)

    # Build window index list
    windows: list[tuple[int, int, int]] = []
    i = 0
    while True:
        warmup_start = i * step
        test_start = warmup_start + warmup_bars
        test_end = test_start + test_bars
        if test_end > n:
            break
        windows.append((warmup_start, test_start, test_end))
        i += 1

    if not windows:
        print(
            f"\nERROR: Not enough data for one window. "
            f"Need {warmup_bars + test_bars} bars, have {n}."
        )
        return pd.DataFrame()

    n_windows = len(windows)
    n_configs = len(configs)
    print(
        f"\nWalk-forward: {n_windows} windows × {n_configs} configs = "
        f"{n_windows * n_configs:,} backtests"
    )
    print(f"Window layout: {warmup_bars} warmup + {test_bars} test bars, step {step}\n")

    records: list[dict] = []

    for w_idx, (warmup_start, test_start_idx, test_end) in enumerate(windows):
        window_df = df.iloc[warmup_start:test_end].copy()
        test_df = df.iloc[test_start_idx:test_end].copy()
        test_start_dt = test_df.index[0]
        bh = bh_return_pct(test_df)
        date_str = f"{test_df.index[0].date()} → {test_df.index[-1].date()}"

        print(f"Window {w_idx + 1}/{n_windows}: {date_str}  B&H={bh:+.1f}%")

        for cfg_idx, (strategy_name, strategy_class, params) in enumerate(configs):
            label = f"{strategy_name}({fmt_params(params)[:55]})"
            print(f"  [{cfg_idx + 1:2d}/{n_configs}] {label:<68s}", end="", flush=True)

            metrics = run_config(strategy_class, params, window_df, test_start_dt)

            if metrics is None:
                print("skip")
                continue

            excess = metrics["total_return"] - bh
            sharpe_str = f"{metrics['sharpe']:.2f}" if not np.isnan(metrics["sharpe"]) else " nan"
            print(
                f"WR={metrics['win_rate']:4.0f}%  ret={metrics['total_return']:+6.1f}%  "
                f"exc={excess:+6.1f}%  S={sharpe_str}  n={metrics['n_trades']}"
            )

            records.append({
                "window": w_idx + 1,
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                "strategy": strategy_name,
                "params": fmt_params(params),
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "total_return": metrics["total_return"],
                "bh_return": bh,
                "excess_return": excess,
                "sharpe": metrics["sharpe"],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cross-window aggregation
# ---------------------------------------------------------------------------

def aggregate(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-window rows to per-(strategy, params) summary."""
    if results_df.empty:
        return pd.DataFrame()

    # Exclude B&H baseline from ranking
    df = results_df[results_df["strategy"] != "BuyAndHoldStrategy"].copy()
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["strategy", "params"])
        .agg(
            mean_sharpe=("sharpe", "mean"),
            mean_excess_return=("excess_return", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_total_return=("total_return", "mean"),
            n_windows=("window", "count"),
        )
        .reset_index()
    )

    beat = (
        df.groupby(["strategy", "params"])
        .apply(lambda g: (g["excess_return"] > 0).mean() * 100, include_groups=False)
        .reset_index(name="pct_beat_bh")
    )
    agg = agg.merge(beat, on=["strategy", "params"])

    # Require at least 3 windows to be included
    agg = agg[agg["n_windows"] >= 3].copy()

    # Composite score: penalise configs with negative Sharpe
    agg["consistency"] = agg["pct_beat_bh"] * agg["mean_sharpe"].clip(lower=0)

    return agg.sort_values("consistency", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(
    results_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    symbol: str,
    warmup_bars: int,
    test_bars: int,
    full_df: pd.DataFrame,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    first_date = full_df.index[0].date()
    last_date = full_df.index[-1].date()
    n_windows = results_df["window"].nunique() if not results_df.empty else 0

    lines = [
        f"# Walk-Forward Analysis: {symbol}",
        f"Generated: {now}  |  Data: {first_date} → {last_date}",
        f"Windows: {n_windows} × {test_bars} bars  |  Warmup: {warmup_bars} bars",
        "",
    ]

    if agg_df.empty:
        lines.append("*No configs had n_windows ≥ 3. Try a longer dataset or smaller test_bars.*")
        return "\n".join(lines)

    # --- Top 10 ---
    lines += [
        "## Top 10 Configs (by Consistency Score)",
        "",
        "| Rank | Strategy | Key Params | Mean Sharpe | Beat B&H | Mean Excess Ret | Mean Win Rate | Windows |",
        "|------|----------|------------|-------------|----------|-----------------|---------------|---------|",
    ]
    for rank, row in enumerate(agg_df.head(10).itertuples(), 1):
        sharpe_str = f"{row.mean_sharpe:.2f}" if not np.isnan(row.mean_sharpe) else "N/A"
        lines.append(
            f"| {rank} | {row.strategy} | {row.params} | {sharpe_str} | "
            f"{row.pct_beat_bh:.0f}% | {row.mean_excess_return:+.1f}% | "
            f"{row.mean_win_rate:.0f}% | {row.n_windows} |"
        )
    lines.append("")

    # --- B&H baseline ---
    if not results_df.empty:
        bh_by_window = results_df.drop_duplicates("window")["bh_return"]
        avg_bh = bh_by_window.mean()
        lines += [
            "## B&H Baseline Summary",
            "",
            f"Avg B&H Return per window: {avg_bh:+.1f}%  |  Windows: {n_windows}",
            "",
        ]

    # --- Per-strategy summary ---
    lines += [
        "## Per-Strategy Summary",
        "",
        "| Strategy | Best Config | Best Sharpe | Avg Sharpe | Beat B&H % | Total Windows |",
        "|----------|-------------|-------------|------------|------------|---------------|",
    ]
    for strat_name in agg_df["strategy"].unique():
        strat_rows = agg_df[agg_df["strategy"] == strat_name]
        best = strat_rows.iloc[0]
        avg_sharpe = strat_rows["mean_sharpe"].mean()
        avg_beat = strat_rows["pct_beat_bh"].mean()
        best_s = f"{best.mean_sharpe:.2f}" if not np.isnan(best.mean_sharpe) else "N/A"
        avg_s = f"{avg_sharpe:.2f}" if not np.isnan(avg_sharpe) else "N/A"
        lines.append(
            f"| {strat_name} | {best.params} | {best_s} | {avg_s} | "
            f"{avg_beat:.0f}% | {int(strat_rows['n_windows'].sum())} |"
        )
    lines.append("")

    # --- Window-by-window detail for top 3 ---
    lines += ["## Window-by-Window Detail (Top 3 Configs)", ""]
    for _, cfg_row in agg_df.head(3).iterrows():
        strat = cfg_row["strategy"]
        params = cfg_row["params"]
        lines.append(f"### {strat} — {params}")
        lines += [
            "",
            "| Window | Test Period | Ret % | B&H % | Excess % | Win Rate | Trades |",
            "|--------|-------------|-------|-------|----------|----------|--------|",
        ]
        cfg_results = results_df[
            (results_df["strategy"] == strat) & (results_df["params"] == params)
        ].sort_values("window")
        for _, row in cfg_results.iterrows():
            period = f"{row.test_start.date()} → {row.test_end.date()}"
            lines.append(
                f"| {int(row.window)} | {period} | {row.total_return:+.1f}% | "
                f"{row.bh_return:+.1f}% | {row.excess_return:+.1f}% | "
                f"{row.win_rate:.0f}% | {int(row.n_trades)} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward strategy analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol label used in output filename")
    parser.add_argument("--data", default="data/XAU-USD/", help="Data directory containing CSV(s)")
    parser.add_argument("--data-file", help="Explicit CSV file path (overrides --data)")
    parser.add_argument("--bar-size", default="1 hour", help="Bar size to select from the data directory (e.g. '1 hour', '4 hours', '15 mins', '1 day')")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup bars per window")
    parser.add_argument("--test-bars", type=int, default=500, help="Test bars per window")
    parser.add_argument("--step", type=int, default=500, help="Step between window starts")
    parser.add_argument("--output", default="strategy_test_results/", help="Output directory")
    args = parser.parse_args()

    data_path = args.data_file if args.data_file else args.data

    print(f"Loading data from: {data_path}  [bar-size: {args.bar_size}]")
    df = load_data(data_path, bar_size=args.bar_size)

    # Derive data_dir / contract_name for multi-timeframe CSV loading
    p = Path(data_path).resolve()
    contract_dir = p.parent if p.is_file() else p
    data_dir = str(contract_dir.parent)
    contract_name = contract_dir.name

    print("\nBuilding strategy configs...")
    configs = build_configs(data_dir=data_dir, contract_name=contract_name)
    print(f"Total configs: {len(configs)}")

    results_df = walk_forward(df, configs, args.warmup, args.test_bars, args.step)

    if results_df.empty:
        print("\nNo results generated. Ensure data covers at least "
              f"{args.warmup + args.test_bars} bars.")
        sys.exit(1)

    agg_df = aggregate(results_df)

    report = build_report(results_df, agg_df, args.symbol, args.warmup, args.test_bars, df)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = output_dir / f"{args.symbol}_walkforward_{ts}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport written to: {report_path}")

    # Console summary
    if not agg_df.empty:
        print("\n" + "=" * 75)
        print("TOP 5 CONFIGS BY CONSISTENCY SCORE:")
        print("=" * 75)
        for rank, row in enumerate(agg_df.head(5).itertuples(), 1):
            sharpe_str = f"{row.mean_sharpe:.2f}" if not np.isnan(row.mean_sharpe) else " N/A"
            print(
                f"{rank}. {row.strategy:<35s} {row.params[:40]:<42s}"
                f"Sharpe={sharpe_str}  Beat={row.pct_beat_bh:4.0f}%  "
                f"ExRet={row.mean_excess_return:+5.1f}%  n={row.n_windows}"
            )


if __name__ == "__main__":
    main()
