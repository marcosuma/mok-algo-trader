# tests/test_atr_breakout_trend_filter.py
import pandas as pd
import numpy as np
import pytest
from forex_strategies.breakout_strategy import ATRBreakout


def _make_df(n: int = 50, price: float = 1.3000, atr: float = 0.0010) -> pd.DataFrame:
    """Minimal OHLCV + ATR DataFrame with stable prices (no accidental breakouts)."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": price, "high": price + 0.0002,
        "low": price - 0.0002, "close": price,
        "volume": 1000.0, "atr": atr,
    }, index=idx)


def _force_buy_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Push the last bar's close well above recent_high + atr*1.5 to trigger a buy."""
    df = df.copy()
    spike = df["high"].max() + df["atr"].iloc[-1] * 2.0
    df.iloc[-1, df.columns.get_loc("close")] = spike
    df.iloc[-1, df.columns.get_loc("high")] = spike
    return df


def _force_sell_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Push the last bar's close well below recent_low - atr*1.5 to trigger a sell."""
    df = df.copy()
    drop = df["low"].min() - df["atr"].iloc[-1] * 2.0
    df.iloc[-1, df.columns.get_loc("close")] = drop
    df.iloc[-1, df.columns.get_loc("low")] = drop
    return df


class TestTrendFilterDisabled:

    def test_buy_signal_fires_without_trend_filter(self):
        df = _force_buy_breakout(_make_df())
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=False)
        out = s.generate_signals(df)
        assert pd.notna(out["execute_buy"].iloc[-1])

    def test_sell_signal_fires_without_trend_filter(self):
        df = _force_sell_breakout(_make_df())
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=False)
        out = s.generate_signals(df)
        assert pd.notna(out["execute_sell"].iloc[-1])


class TestTrendFilterEnabled:

    def test_long_allowed_when_daily_close_above_sma200(self):
        df = _force_buy_breakout(_make_df())
        df["1 day_close"] = 1.3100   # above SMA_200
        df["1 day_SMA_200"] = 1.3000
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=True)
        out = s.generate_signals(df)
        assert pd.notna(out["execute_buy"].iloc[-1])

    def test_long_suppressed_when_daily_close_below_sma200(self):
        df = _force_buy_breakout(_make_df())
        df["1 day_close"] = 1.2800   # below SMA_200
        df["1 day_SMA_200"] = 1.3000
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=True)
        out = s.generate_signals(df)
        assert pd.isna(out["execute_buy"].iloc[-1])

    def test_short_allowed_when_daily_close_below_sma200(self):
        df = _force_sell_breakout(_make_df())
        df["1 day_close"] = 1.2700   # below SMA_200
        df["1 day_SMA_200"] = 1.3000
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=True)
        out = s.generate_signals(df)
        assert pd.notna(out["execute_sell"].iloc[-1])

    def test_short_suppressed_when_daily_close_above_sma200(self):
        df = _force_sell_breakout(_make_df())
        df["1 day_close"] = 1.3200   # above SMA_200
        df["1 day_SMA_200"] = 1.3000
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=True)
        out = s.generate_signals(df)
        assert pd.isna(out["execute_sell"].iloc[-1])

    def test_fallback_to_unfiltered_when_daily_columns_absent(self):
        """trend_filter=True with no 1 day_* columns must NOT suppress signals."""
        df = _force_buy_breakout(_make_df())
        # No 1 day_close / 1 day_SMA_200 columns
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=True)
        out = s.generate_signals(df)
        assert pd.notna(out["execute_buy"].iloc[-1])

    def test_long_allowed_when_close_equals_sma200(self):
        """Boundary: close == SMA_200 is treated as uptrend (>=), long allowed."""
        df = _force_buy_breakout(_make_df())
        df["1 day_close"] = 1.3000
        df["1 day_SMA_200"] = 1.3000
        s = ATRBreakout(lookback_period=10, atr_multiplier=1.5, trend_filter=True)
        out = s.generate_signals(df)
        assert pd.notna(out["execute_buy"].iloc[-1])
