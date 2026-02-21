"""
Pattern-based trading strategy.

Detects chart patterns (Head & Shoulders, Triangles, Rectangles, etc.) and generates
trading signals based on pattern completion.
"""
import pandas as pd
import numpy as np
import collections
from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg
from forex_strategies.base_strategy import BaseForexStrategy


class PatternStrategy(BaseForexStrategy):
    """
    Strategy that trades based on chart pattern completion.

    Detects patterns using kernel regression and extrema analysis:
    - Bullish patterns (IHS, TBOT, RBOT): Buy signals
    - Bearish patterns (HS, TTOP, RTOP): Sell signals
    """

    def __init__(
        self,
        initial_cash=10000,
        commission=0.0002,
        max_bars=35,
        prominence_factor=0.5,
        min_distance=5,
    ):
        super().__init__(initial_cash, commission)
        self.max_bars = max_bars
        self.prominence_factor = prominence_factor
        self.min_distance = min_distance

    def _find_extrema(self, df: pd.DataFrame, price_column: str = "close"):
        """Find local extrema using kernel regression smoothing."""
        # Ensure numeric series
        series = pd.to_numeric(df[price_column], errors="coerce").copy()
        series = series.ffill().bfill()

        # KernelReg requires a numeric independent variable; a DatetimeIndex
        # causes a dtype promotion error between float64 and datetime64.
        x_numeric = np.arange(len(df), dtype=float)

        kr = KernelReg(
            pd.to_numeric(series), x_numeric, var_type="c", bw=[0.85]
        )
        f = kr.fit([x_numeric])
        smooth_prices = pd.Series(data=f[0], index=df.index)

        # Find extrema in smoothed prices
        smoothed_local_max = argrelextrema(smooth_prices.values, np.greater)[0]
        smoothed_local_min = argrelextrema(smooth_prices.values, np.less)[0]

        # Map back to original price extrema
        local_max_min = np.sort(
            np.concatenate([smoothed_local_max, smoothed_local_min])
        )
        smooth_extrema = smooth_prices.loc[local_max_min]

        # Get actual price extrema
        price_local_max_dt = []
        for i in smoothed_local_max:
            if (i > 1) and (i < len(df) - 1):
                price_local_max_dt.append(
                    pd.to_numeric(df[price_column]).iloc[i - 2 : i + 2].idxmax()
                )

        price_local_min_dt = []
        for i in smoothed_local_min:
            if (i > 1) and (i < len(df) - 1):
                price_local_min_dt.append(
                    pd.to_numeric(df[price_column]).iloc[i - 2 : i + 2].idxmin()
                )

        # Combine and sort
        max_min = pd.concat(
            [
                pd.to_numeric(df.loc[price_local_min_dt, price_column]),
                pd.to_numeric(df.loc[price_local_max_dt, price_column]),
            ]
        ).sort_index()

        return max_min

    def _find_patterns(self, extrema: pd.Series, df: pd.DataFrame = None):
        """Detect chart patterns from extrema.

        Args:
            extrema: Series of extrema values with datetime index
            df: Original DataFrame (optional, used for counting bars between extrema)
        """
        patterns = collections.defaultdict(list)

        # Need at least 5 extrema for pattern generation
        for i in range(5, len(extrema)):
            window = extrema.iloc[i - 5 : i]

            # Pattern must play out within max_bars
            if df is not None:
                # Count actual bars between first and last extrema in the original DataFrame
                start_idx = window.index[0]
                end_idx = window.index[-1]
                try:
                    # Find the positions of these indices in the original DataFrame
                    start_pos = df.index.get_loc(start_idx)
                    end_pos = df.index.get_loc(end_idx)
                    bar_count = abs(end_pos - start_pos)
                    if bar_count > self.max_bars:
                        continue
                except (KeyError, TypeError):
                    # If indices don't match, skip the check
                    pass
            else:
                # If DataFrame not provided, skip the max_bars check
                # This avoids dtype promotion issues with datetime index subtraction
                pass

            e1, e2, e3, e4, e5 = (
                window.iloc[0],
                window.iloc[1],
                window.iloc[2],
                window.iloc[3],
                window.iloc[4],
            )

            rtop_g1 = np.mean([e1, e3, e5])
            rtop_g2 = np.mean([e2, e4])

            # Head and Shoulders (bearish)
            if (
                (e1 > e2)
                and (e3 > e1)
                and (e3 > e5)
                and (abs(e1 - e5) <= 0.03 * np.mean([e1, e5]))
                and (abs(e2 - e4) <= 0.03 * np.mean([e1, e5]))
            ):
                patterns["HS"].append((window.index[0], window.index[-1]))

            # Inverse Head and Shoulders (bullish)
            elif (
                (e1 < e2)
                and (e3 < e1)
                and (e3 < e5)
                and (abs(e1 - e5) <= 0.03 * np.mean([e1, e5]))
                and (abs(e2 - e4) <= 0.03 * np.mean([e1, e5]))
            ):
                patterns["IHS"].append((window.index[0], window.index[-1]))

            # Triangle Top (bearish)
            elif (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
                patterns["TTOP"].append((window.index[0], window.index[-1]))

            # Triangle Bottom (bullish)
            elif (e1 < e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
                patterns["TBOT"].append((window.index[0], window.index[-1]))

            # Rectangle Top (bearish)
            elif (
                (e1 > e2)
                and (abs(e1 - rtop_g1) / rtop_g1 < 0.0075)
                and (abs(e3 - rtop_g1) / rtop_g1 < 0.0075)
                and (abs(e5 - rtop_g1) / rtop_g1 < 0.0075)
                and (abs(e2 - rtop_g2) / rtop_g2 < 0.0075)
                and (abs(e4 - rtop_g2) / rtop_g2 < 0.0075)
                and (min(e1, e3, e5) > max(e2, e4))
            ):
                patterns["RTOP"].append((window.index[0], window.index[-1]))

            # Rectangle Bottom (bullish)
            elif (
                (e1 < e2)
                and (abs(e1 - rtop_g1) / rtop_g1 < 0.0075)
                and (abs(e3 - rtop_g1) / rtop_g1 < 0.0075)
                and (abs(e5 - rtop_g1) / rtop_g1 < 0.0075)
                and (abs(e2 - rtop_g2) / rtop_g2 < 0.0075)
                and (abs(e4 - rtop_g2) / rtop_g2 < 0.0075)
                and (max(e1, e3, e5) > min(e2, e4))
            ):
                patterns["RBOT"].append((window.index[0], window.index[-1]))

        return patterns

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on pattern completion."""
        df = df.copy()

        # Initialize signal columns
        df["execute_buy"] = np.nan
        df["execute_sell"] = np.nan

        try:
            # Find extrema
            extrema = self._find_extrema(df, "close")

            if len(extrema) < 5:
                # Not enough extrema for pattern detection
                return df

            # Detect patterns (pass df to enable max_bars check)
            patterns = self._find_patterns(extrema, df)

            # Bullish patterns: Buy on completion
            bullish_patterns = ["IHS", "TBOT", "RBOT"]
            for pattern_type in bullish_patterns:
                if pattern_type in patterns:
                    for start_idx, end_idx in patterns[pattern_type]:
                        # Buy signal at pattern completion
                        if end_idx in df.index:
                            df.loc[end_idx, "execute_buy"] = df.loc[end_idx, "close"]

            # Bearish patterns: Sell on completion
            bearish_patterns = ["HS", "TTOP", "RTOP"]
            for pattern_type in bearish_patterns:
                if pattern_type in patterns:
                    for start_idx, end_idx in patterns[pattern_type]:
                        # Sell signal at pattern completion
                        if end_idx in df.index:
                            df.loc[end_idx, "execute_sell"] = df.loc[end_idx, "close"]

        except Exception as e:
            # If pattern detection fails, return empty signals
            print(f"Warning: Pattern detection failed: {e}")
            pass

        return df

