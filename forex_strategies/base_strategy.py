"""
Base strategy class for forex trading strategies.
All strategies should inherit from this and implement the execute method.
"""
import pandas as pd
import numpy as np
from backtesting import Backtest
from abc import ABC, abstractmethod


class BaseForexStrategy(ABC):
    """Base class for all forex trading strategies."""

    def __init__(self, initial_cash=10000, commission=0.0002, spread=0.0001):
        """
        Initialize strategy.

        Args:
            initial_cash: Starting capital
            commission: Commission rate (0.0002 = 0.02% for forex)
            spread: Spread in price units passed to the backtesting engine
                    (e.g. 0.0001 = 1 pip for forex). Set to 0 to disable.
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.spread = spread

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals for the strategy.
        Must add columns: 'execute_buy' and 'execute_sell' with entry prices or NaN.

        Args:
            df: DataFrame with OHLCV and technical indicators

        Returns:
            DataFrame with signals added
        """
        pass

    def execute(self, df: pd.DataFrame, backtest_strategy_class):
        """
        Execute strategy with backtesting.

        Args:
            df: DataFrame with OHLCV and technical indicators
            backtest_strategy_class: Backtesting Strategy class

        Returns:
            Backtest statistics and marker function for plotting
        """
        # Generate signals
        df = self.generate_signals(df)

        # Prepare data for backtesting
        clean_df = df.dropna(subset=["open", "high", "low", "close"]).copy()
        if len(clean_df) == 0:
            print("Warning: No valid data after cleaning")
            return None, None

        # Ensure required columns exist
        clean_df["Open"] = clean_df.open
        clean_df["Close"] = clean_df.close
        clean_df["High"] = clean_df.high
        clean_df["Low"] = clean_df.low

        # Ensure the index is suitable for Backtest statistics.
        # The `backtesting` library expects either a numeric index or a
        # `pd.DatetimeIndex` to compute drawdown durations correctly. Many of
        # our CSVs load with a plain object index of date strings, which can
        # cause "TypeError: unsupported operand type(s) for -: 'str' and 'str'".
        if not isinstance(clean_df.index, pd.DatetimeIndex):
            parsed_index = pd.to_datetime(clean_df.index, errors="coerce")
            if parsed_index.notna().all():
                clean_df.index = parsed_index
            else:
                clean_df = clean_df.reset_index(drop=True)

        # Run backtest
        backtest_kwargs = dict(
            cash=self.initial_cash,
            commission=self.commission,
            exclusive_orders=True,
            finalize_trades=True,
        )
        if self.spread:
            backtest_kwargs["spread"] = self.spread
        bt = Backtest(
            clean_df,
            backtest_strategy_class,
            **backtest_kwargs,
        )
        stats = bt.run()
        print("\n" + "=" * 60)
        print(f"Strategy: {self.__class__.__name__}")
        print("=" * 60)
        print(stats)
        print("=" * 60 + "\n")

        # Create marker function for plotting
        marker_fn = self._create_marker_function(df)

        return stats, marker_fn

    def _create_marker_function(self, df: pd.DataFrame):
        """Create a function that adds strategy markers to a plotly figure."""
        import plotly.graph_objects as go

        def print_strategy_markers(fig):
            # Buy signals
            buy_signals = df[df["execute_buy"].notna()]
            if len(buy_signals) > 0:
                fig.append_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals["execute_buy"],
                        name="Buy Signal",
                        mode="markers",
                        marker=dict(
                            color="green",
                            size=10,
                            symbol="triangle-down",
                            line=dict(color="darkgreen", width=2),
                        ),
                        legendgroup="signals",
                    ),
                    row=1,
                    col=1,
                )

            # Sell signals
            sell_signals = df[df["execute_sell"].notna()]
            if len(sell_signals) > 0:
                fig.append_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals["execute_sell"],
                        name="Sell Signal",
                        mode="markers",
                        marker=dict(
                            color="red",
                            size=10,
                            symbol="triangle-up",
                            line=dict(color="darkred", width=2),
                        ),
                        legendgroup="signals",
                    ),
                    row=1,
                    col=1,
                )

        return print_strategy_markers

