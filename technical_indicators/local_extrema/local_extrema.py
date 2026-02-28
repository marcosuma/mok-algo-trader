"""Local Extrema technical indicator.

This module detects and marks local minima and maxima in price series.
It uses scipy.signal.find_peaks to identify swing highs and lows based on
volatility-based prominence thresholds.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Import constants from the original module for consistency
from local_extrema.local_extrema import LOCAL_MAX, LOCAL_MIN


class LocalExtrema:
    """Detect and mark local minima and maxima in price series.

    This indicator adds a 'local_extrema' column to the DataFrame that marks
    each row as LOCAL_MAX (swing high), LOCAL_MIN (swing low), or None.

    The algorithm uses scipy.signal.find_peaks with:
    - distance: minimum number of bars between extrema
    - prominence: volatility-based threshold to ignore tiny wiggles
    """

    def __init__(
        self,
        price_column: str = "close",
        output_column: str = "local_extrema",
        min_distance: int = 5,
        prominence_factor: float = 0.5,
    ):
        """
        Initialize LocalExtrema indicator.

        Args:
            price_column: Column name to use for extrema detection (default: "close")
            output_column: Column name for the output (default: "local_extrema")
            min_distance: Minimum number of bars between extrema (default: 5)
            prominence_factor: Factor to multiply std dev for prominence threshold (default: 0.5)
        """
        self.price_column = price_column
        self.output_column = output_column
        self.min_distance = min_distance
        self.prominence_factor = prominence_factor

    def calculate(self, df: pd.DataFrame):
        """
        Calculate local extrema and add column to DataFrame.

        Args:
            df: DataFrame with OHLCV data. Must contain the price_column.

        The method modifies df in-place by adding the output_column.
        """
        if self.price_column not in df.columns:
            # Silently skip if price column doesn't exist
            return

        # Ensure numeric series on a copy; fill gaps to keep SciPy happy
        series = pd.to_numeric(df[self.price_column], errors="coerce").copy()
        series = series.ffill().bfill()

        values = series.to_numpy(dtype=float)

        # Volatility-based prominence threshold. This follows common practice
        # in peak detection on financial time series: require that a swing
        # stands out relative to recent volatility instead of using a fixed
        # price threshold.
        std = float(np.nanstd(values))
        prominence = std * float(self.prominence_factor) if std > 0 else None

        peak_kwargs = {"distance": max(int(self.min_distance), 1)}
        if prominence is not None and prominence > 0:
            peak_kwargs["prominence"] = prominence

        # Swing highs
        max_idx, _ = find_peaks(values, **peak_kwargs)
        # Swing lows (peaks on the inverted series)
        min_idx, _ = find_peaks(-values, **peak_kwargs)

        # Create an object-typed column with None by default
        result = pd.Series(index=df.index, dtype="object")
        if len(max_idx):
            result.iloc[max_idx] = LOCAL_MAX
        if len(min_idx):
            result.iloc[min_idx] = LOCAL_MIN

        # Shift forward by min_distance to eliminate look-ahead bias.
        # scipy find_peaks on the full series identifies a peak at bar N only
        # after observing min_distance subsequent bars — those bars are "in the
        # future" at the time bar N closes.  Shifting the labels forward makes
        # the signal available only once the confirmation bars have passed.
        df[self.output_column] = result.shift(self.min_distance)

