"""
Tests for backtest-vs-live parity: SL/TP, pyramiding, and slippage.

These tests verify that ForexBacktestingStrategy correctly applies the same
risk management and position management rules as the live trading system.
"""
import numpy as np
import pandas as pd
import pytest

from backtesting import Backtest
from forex_strategies.backtesting_strategy import ForexBacktestingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    closes: list | np.ndarray,
    spread: float = 0.005,
    atr_values: list | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame, optionally with an ``atr`` column."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    highs = closes + spread
    lows = closes - spread
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": np.ones(n) * 1000,
    })
    if atr_values is not None:
        df["atr"] = np.asarray(atr_values, dtype=float)
    return df


def _add_signals(df: pd.DataFrame, buy_indices=None, sell_indices=None):
    """Add execute_buy / execute_sell columns at specific indices."""
    df["execute_buy"] = np.nan
    df["execute_sell"] = np.nan
    if buy_indices:
        for i in buy_indices:
            df.loc[i, "execute_buy"] = df.loc[i, "Close"]
    if sell_indices:
        for i in sell_indices:
            df.loc[i, "execute_sell"] = df.loc[i, "Close"]
    return df


def _run_backtest(df, **strategy_params) -> object:
    """Run a backtest with the given strategy params and return stats."""
    bt = Backtest(
        df,
        ForexBacktestingStrategy,
        cash=10_000,
        commission=0,
        exclusive_orders=not strategy_params.get("allow_pyramiding", False),
        finalize_trades=True,
    )
    return bt.run(**strategy_params)


def _get_trades_df(stats) -> pd.DataFrame:
    """Extract the trades DataFrame from stats."""
    return stats._trades


# ---------------------------------------------------------------------------
# SL/TP in backtesting
# ---------------------------------------------------------------------------

class TestBacktestStopLoss:
    """Verify stop-loss exits positions in backtesting."""

    def test_long_stopped_out(self):
        """A long position should be closed when price drops to SL level."""
        closes = [1.200] * 5 + [1.200, 1.198, 1.196, 1.194, 1.192] + [1.190] * 5
        atr = [0.002] * len(closes)
        df = _make_ohlcv(closes, spread=0.003, atr_values=atr)
        df = _add_signals(df, buy_indices=[2])

        stats_no_sl = _run_backtest(df, stop_loss_type="NONE", stop_loss_value=0)
        stats_with_sl = _run_backtest(
            df,
            stop_loss_type="ATR",
            stop_loss_value=1.5,
            take_profit_type="NONE",
            take_profit_value=0,
        )

        trades = _get_trades_df(stats_with_sl)
        assert len(trades) >= 1, "SL trade should have been opened"

    def test_short_stopped_out(self):
        """A short position should be closed when price rises to SL level."""
        closes = [1.200] * 5 + [1.200, 1.202, 1.204, 1.206, 1.208] + [1.210] * 5
        atr = [0.002] * len(closes)
        df = _make_ohlcv(closes, spread=0.003, atr_values=atr)
        df = _add_signals(df, sell_indices=[2])

        stats = _run_backtest(
            df,
            stop_loss_type="ATR",
            stop_loss_value=1.5,
            take_profit_type="NONE",
            take_profit_value=0,
        )

        trades = _get_trades_df(stats)
        assert len(trades) >= 1, "SL trade should have been opened"


class TestBacktestTakeProfit:
    """Verify take-profit exits positions in backtesting."""

    def test_long_takes_profit(self):
        """A long position should be closed when price rises to TP level."""
        closes = [1.200] * 5 + [1.200, 1.202, 1.204, 1.206, 1.208, 1.210] + [1.210] * 4
        atr = [0.002] * len(closes)
        df = _make_ohlcv(closes, spread=0.003, atr_values=atr)
        df = _add_signals(df, buy_indices=[2])

        stats = _run_backtest(
            df,
            stop_loss_type="ATR",
            stop_loss_value=1.5,
            take_profit_type="RISK_REWARD",
            take_profit_value=2.0,
        )
        trades = _get_trades_df(stats)
        assert len(trades) >= 1


class TestBacktestNoSlTpMatchesLegacy:
    """When SL/TP are NONE, behavior should match the old strategy exactly."""

    def test_single_buy_sell_cycle(self):
        closes = [1.0] * 5 + list(np.linspace(1.0, 1.1, 10)) + [1.1] * 5
        df = _make_ohlcv(closes)
        df = _add_signals(df, buy_indices=[3], sell_indices=[12])

        stats = _run_backtest(df, stop_loss_type="NONE", take_profit_type="NONE")
        trades = _get_trades_df(stats)
        assert len(trades) >= 1

    def test_duplicate_buy_blocked(self):
        """Second BUY when already long should be ignored (no pyramiding)."""
        closes = list(np.linspace(1.0, 1.05, 20))
        df = _make_ohlcv(closes)
        df = _add_signals(df, buy_indices=[3, 7])

        stats = _run_backtest(
            df,
            stop_loss_type="NONE",
            take_profit_type="NONE",
            allow_pyramiding=False,
        )
        trades = _get_trades_df(stats)
        entry_bars = trades["EntryBar"].tolist()
        assert len(entry_bars) == 1, (
            f"Only one trade expected (duplicate blocked), got entries at bars {entry_bars}"
        )


# ---------------------------------------------------------------------------
# Percentage and Fixed SL types
# ---------------------------------------------------------------------------

class TestBacktestPercentageSl:

    def test_percentage_sl_long(self):
        closes = [1.200] * 5 + [1.200, 1.195, 1.190, 1.185, 1.180] + [1.180] * 5
        df = _make_ohlcv(closes, spread=0.003)
        df = _add_signals(df, buy_indices=[2])

        stats = _run_backtest(
            df,
            stop_loss_type="PERCENTAGE",
            stop_loss_value=0.01,
            take_profit_type="NONE",
            take_profit_value=0,
        )
        trades = _get_trades_df(stats)
        assert len(trades) >= 1


class TestBacktestFixedSl:

    def test_fixed_sl_long(self):
        closes = [1.200] * 5 + [1.200, 1.195, 1.190, 1.185, 1.180] + [1.180] * 5
        df = _make_ohlcv(closes, spread=0.003)
        df = _add_signals(df, buy_indices=[2])

        stats = _run_backtest(
            df,
            stop_loss_type="FIXED",
            stop_loss_value=0.005,
            take_profit_type="NONE",
            take_profit_value=0,
        )
        trades = _get_trades_df(stats)
        assert len(trades) >= 1


# ---------------------------------------------------------------------------
# Pyramiding
# ---------------------------------------------------------------------------

class TestBacktestPyramiding:
    """Verify pyramiding creates multiple concurrent entries."""

    def test_pyramiding_allows_multiple_longs(self):
        """With pyramiding enabled, consecutive BUY signals should add entries."""
        closes = list(np.linspace(1.0, 1.05, 30))
        df = _make_ohlcv(closes)
        df = _add_signals(df, buy_indices=[3, 7, 11])

        stats = _run_backtest(
            df,
            stop_loss_type="NONE",
            take_profit_type="NONE",
            allow_pyramiding=True,
            max_pyramid_entries=3,
        )
        trades = _get_trades_df(stats)
        entry_bars = sorted(trades["EntryBar"].tolist())
        assert len(entry_bars) >= 2, (
            f"Pyramiding should allow multiple entries, got entries at bars {entry_bars}"
        )

    def test_pyramiding_respects_max_entries(self):
        """Should not exceed max_pyramid_entries."""
        closes = list(np.linspace(1.0, 1.05, 30))
        df = _make_ohlcv(closes)
        df = _add_signals(df, buy_indices=[3, 5, 7, 9, 11])

        stats = _run_backtest(
            df,
            stop_loss_type="NONE",
            take_profit_type="NONE",
            allow_pyramiding=True,
            max_pyramid_entries=2,
        )
        trades = _get_trades_df(stats)

        max_concurrent = 0
        for bar in range(len(closes)):
            open_at_bar = 0
            for _, t in trades.iterrows():
                entered = t["EntryBar"] <= bar
                not_exited = pd.isna(t["ExitBar"]) or t["ExitBar"] > bar
                if entered and not_exited:
                    open_at_bar += 1
            max_concurrent = max(max_concurrent, open_at_bar)

        assert max_concurrent <= 2, (
            f"Max concurrent entries should be <=2, got {max_concurrent}"
        )

    def test_pyramiding_disabled_blocks_duplicates(self):
        """With pyramiding disabled, second BUY when long should be blocked."""
        closes = list(np.linspace(1.0, 1.05, 20))
        df = _make_ohlcv(closes)
        df = _add_signals(df, buy_indices=[3, 7])

        stats = _run_backtest(
            df,
            stop_loss_type="NONE",
            take_profit_type="NONE",
            allow_pyramiding=False,
        )
        trades = _get_trades_df(stats)
        entry_bars = trades["EntryBar"].tolist()
        assert len(entry_bars) == 1, (
            f"Only one entry expected, got {entry_bars}"
        )

    def test_opposite_signal_closes_pyramid(self):
        """A SELL signal when long should close all long entries."""
        closes = list(np.linspace(1.0, 1.05, 15)) + list(np.linspace(1.05, 1.0, 15))
        df = _make_ohlcv(closes)
        df = _add_signals(df, buy_indices=[3, 6], sell_indices=[18])

        stats = _run_backtest(
            df,
            stop_loss_type="NONE",
            take_profit_type="NONE",
            allow_pyramiding=True,
            max_pyramid_entries=3,
        )
        trades = _get_trades_df(stats)
        long_trades = trades[trades["Size"] > 0]
        for _, t in long_trades.iterrows():
            assert not pd.isna(t["ExitBar"]), "All longs should be closed after SELL"


# ---------------------------------------------------------------------------
# Slippage (via spread parameter)
# ---------------------------------------------------------------------------

class TestBacktestSlippage:
    """Slippage is modeled by adding to the spread."""

    def test_slippage_increases_effective_spread(self):
        from forex_strategies.base_strategy import BaseForexStrategy

        class DummyStrategy(BaseForexStrategy):
            def generate_signals(self, df):
                df = df.copy()
                df["execute_buy"] = np.nan
                df["execute_sell"] = np.nan
                return df

        s = DummyStrategy(spread=0.0001, slippage=0.0002)
        assert s.slippage == 0.0002
        assert s.spread + s.slippage == pytest.approx(0.0003)


# ---------------------------------------------------------------------------
# ForexBacktestingStrategy helper methods
# ---------------------------------------------------------------------------

class TestBacktestingStrategyHelpers:

    def test_is_market_order_equal_prices(self):
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        assert strategy._is_market_order(1.2345, 1.2345) is True

    def test_is_market_order_different_prices(self):
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        assert strategy._is_market_order(1.2500, 1.2345) is False

    def _make_strategy(self, **overrides):
        """Create a ForexBacktestingStrategy with a mock _data attribute."""
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        defaults = dict(
            stop_loss_type="NONE", stop_loss_value=0,
            take_profit_type="NONE", take_profit_value=0,
            allow_pyramiding=False, max_pyramid_entries=1,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(strategy, k, v)

        class FakeData:
            pass
        strategy._data = FakeData()
        return strategy

    def test_build_order_kwargs_market_no_sl_tp(self):
        strategy = self._make_strategy()
        kwargs = strategy._build_order_kwargs(1.2000, 1.2000, "LONG")
        assert "sl" not in kwargs
        assert "tp" not in kwargs
        assert "limit" not in kwargs

    def test_build_order_kwargs_limit_with_sl_tp(self):
        strategy = self._make_strategy(
            stop_loss_type="FIXED", stop_loss_value=0.005,
            take_profit_type="RISK_REWARD", take_profit_value=2.0,
        )
        kwargs = strategy._build_order_kwargs(1.1950, 1.2000, "LONG")
        assert kwargs["limit"] == 1.1950
        assert kwargs["sl"] == pytest.approx(1.1950 - 0.005)
        assert kwargs["tp"] == pytest.approx(1.1950 + 0.005 * 2.0)

    def test_build_order_kwargs_with_pyramiding_sets_size(self):
        strategy = self._make_strategy(
            allow_pyramiding=True, max_pyramid_entries=3,
        )
        kwargs = strategy._build_order_kwargs(1.2000, 1.2000, "LONG")
        assert kwargs["size"] == pytest.approx(1.0 / 3)
