"""
Multi-Timeframe Strategy Base Class

This module provides a base class for strategies that analyze multiple bar sizes
simultaneously. Higher timeframes (e.g., 1 day) are used for trend confirmation,
while lower timeframes (e.g., 15 mins, 1 hour) are used for precise entry/exit signals.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from forex_strategies.base_strategy import BaseForexStrategy
from forex_strategies.adaptive_multi_indicator_strategy import AdaptiveMultiIndicatorStrategy

logger = logging.getLogger(__name__)


class MultiTimeframeStrategy(BaseForexStrategy):
    """
    Base class for multi-timeframe strategies.

    This strategy analyzes multiple bar sizes:
    - Higher timeframes (1 day, 4 hours) for trend confirmation
    - Lower timeframes (1 hour, 15 mins, 5 mins) for entry/exit signals

    The strategy aligns data from different timeframes and uses them together
    to make trading decisions.
    """

    def __init__(
        self,
        initial_cash=10000,
        commission=0.0002,
        higher_timeframes: List[str] = ["1 day", "4 hours"],
        lower_timeframes: List[str] = ["1 hour", "15 mins"],
        base_strategy_class=None,
    ):
        """
        Initialize multi-timeframe strategy.

        Args:
            initial_cash: Starting capital
            commission: Commission rate
            higher_timeframes: List of bar sizes to use for trend confirmation
            lower_timeframes: List of bar sizes to use for entry/exit signals
            base_strategy_class: The base strategy class to use for signal generation
        """
        super().__init__(initial_cash, commission)
        self.higher_timeframes = higher_timeframes
        self.lower_timeframes = lower_timeframes
        self.base_strategy_class = base_strategy_class or AdaptiveMultiIndicatorStrategy

    def _load_timeframe_data(
        self, base_df: pd.DataFrame, data_dir: str, contract_name: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for all required timeframes from CSV files on disk.

        Note: This method is used during backtesting. In live trading the
        StrategyAdapter provides multi-timeframe columns directly in the
        DataFrame (prefixed with ``{bar_size}_``).

        Args:
            base_df: The base DataFrame (from the input CSV)
            data_dir: Base directory containing contract folders
            contract_name: Contract name (e.g., "USD-CAD")

        Returns:
            Dictionary mapping bar size to DataFrame
        """
        import os
        import glob

        timeframe_data = {}

        if not os.path.isdir(data_dir):
            logger.debug(f"Data directory '{data_dir}' does not exist, skipping CSV loading")
            return timeframe_data

        # Extract contract info from base_df or contract_name
        # Try to infer from the CSV path or use contract_name
        contract_folder = os.path.join(data_dir, contract_name)

        if not os.path.exists(contract_folder):
            # Fallback: try to find contract folder
            try:
                possible_folders = [
                    os.path.join(data_dir, d)
                    for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
            except OSError as e:
                logger.debug(f"Could not list data directory '{data_dir}': {e}")
                return timeframe_data
            if possible_folders:
                contract_folder = possible_folders[0]

        if not os.path.exists(contract_folder):
            return timeframe_data

        # Bar size to file pattern mapping
        bar_size_patterns = {
            "1 day": "*1 day*.csv",
            "4 hours": "*4 hours*.csv",
            "1 hour": "*1 hour*.csv",
            "15 mins": "*15 mins*.csv",
            "5 mins": "*5 mins*.csv",
            "1 week": "*1 week*.csv",
        }

        all_timeframes = self.higher_timeframes + self.lower_timeframes

        for bar_size in all_timeframes:
            pattern = bar_size_patterns.get(bar_size, f"*{bar_size}*.csv")
            csv_files = glob.glob(os.path.join(contract_folder, pattern))

            if csv_files:
                # Use the first matching file
                try:
                    df = pd.read_csv(csv_files[0], index_col=[0])
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index, errors="coerce")
                    df = df[df.index.notna()]
                    if len(df) > 0:
                        timeframe_data[bar_size] = df
                except Exception as e:
                    logger.warning(f"Could not load {bar_size} data from {csv_files[0]}: {e}")
                    continue

        return timeframe_data

    def _align_timeframes(
        self, timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple timeframes to a common index (using the lowest timeframe).

        Args:
            timeframe_data: Dictionary mapping bar size to DataFrame

        Returns:
            Dictionary with aligned DataFrames
        """
        if not timeframe_data:
            return {}

        # Use the lowest timeframe as the base (most granular)
        if self.lower_timeframes:
            base_timeframe = self.lower_timeframes[0]
        else:
            base_timeframe = list(timeframe_data.keys())[0]

        if base_timeframe not in timeframe_data:
            return timeframe_data

        base_df = timeframe_data[base_timeframe]
        base_index = base_df.index

        aligned_data = {base_timeframe: base_df}

        # Align other timeframes to base index using forward fill
        for bar_size, df in timeframe_data.items():
            if bar_size == base_timeframe:
                continue

            # Reindex to base index and forward fill
            aligned_df = df.reindex(base_index, method="ffill")
            aligned_data[bar_size] = aligned_df

        return aligned_data

    def _get_trend_from_higher_timeframe(
        self, aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        Determine overall trend from higher timeframes.

        Args:
            aligned_data: Dictionary of aligned DataFrames

        Returns:
            Series with trend signals: 1 for uptrend, -1 for downtrend, 0 for unclear
        """
        trend_signals = pd.Series(0, index=list(aligned_data.values())[0].index)

        for bar_size in self.higher_timeframes:
            if bar_size not in aligned_data:
                continue

            df = aligned_data[bar_size]

            # Check for required indicators
            required_cols = ["adx", "plus_di", "minus_di", "SMA_50", "close"]
            if not all(col in df.columns for col in required_cols):
                continue

            # Trend conditions from higher timeframe
            strong_trend = df["adx"] > 25
            bullish = df["plus_di"] > df["minus_di"]
            bearish = df["minus_di"] > df["plus_di"]
            price_above_sma = df["close"] > df["SMA_50"]
            price_below_sma = df["close"] < df["SMA_50"]

            # Uptrend: strong trend + bullish + price above SMA
            uptrend = strong_trend & bullish & price_above_sma
            # Downtrend: strong trend + bearish + price below SMA
            downtrend = strong_trend & bearish & price_below_sma

            # Combine signals (higher timeframes have more weight)
            trend_signals = trend_signals + (uptrend.astype(int) - downtrend.astype(int))

        # Normalize: if multiple higher timeframes agree, signal is stronger
        # For now, just use sign: >0 = uptrend, <0 = downtrend
        return np.sign(trend_signals)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using multi-timeframe analysis.

        Args:
            df: Base DataFrame (from input CSV, typically the lowest timeframe)

        Returns:
            DataFrame with execute_buy and execute_sell signals
        """
        # This is a base implementation - subclasses should override
        # For now, we'll create a wrapper that uses the base strategy
        # but filters by higher timeframe trend

        # Try to infer contract name and data directory from the DataFrame
        # This is a limitation - we need the data directory and contract name
        # For now, assume they're passed via instance variables or we use defaults

        # Use the base strategy to generate signals on the lower timeframe
        base_strategy = self.base_strategy_class(
            initial_cash=self.initial_cash,
            commission=self.commission,
        )

        # Generate base signals
        df = base_strategy.generate_signals(df)

        # Note: Full multi-timeframe implementation would require:
        # 1. Loading data from multiple bar sizes
        # 2. Aligning them
        # 3. Using higher timeframe for trend filter
        # 4. Using lower timeframe for entry signals

        return df


class AdaptiveMultiTimeframeStrategy(MultiTimeframeStrategy):
    """
    Multi-timeframe version of AdaptiveMultiIndicatorStrategy.

    Uses higher timeframes (1 day, 4 hours) to confirm trend direction,
    and lower timeframes (1 hour, 15 mins) for precise entry/exit signals.
    """

    def __init__(
        self,
        initial_cash=10000,
        commission=0.0002,
        higher_timeframes: List[str] = ["1 day", "4 hours"],
        lower_timeframes: List[str] = ["1 hour", "15 mins"],
        # Parameters from AdaptiveMultiIndicatorStrategy
        adx_trend_threshold=25,
        adx_range_threshold=20,
        rsi_trend_min=40,
        rsi_trend_max=65,
        rsi_oversold=35,
        rsi_overbought=65,
        atr_stop_multiplier=2.0,
        atr_take_profit_multiplier=2.5,
        atr_extreme_multiplier=3.0,
        extrema_lookback=20,
        data_dir: str = "data",
        contract_name: Optional[str] = None,
    ):
        super().__init__(
            initial_cash=initial_cash,
            commission=commission,
            higher_timeframes=higher_timeframes,
            lower_timeframes=lower_timeframes,
        )
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.rsi_trend_min = rsi_trend_min
        self.rsi_trend_max = rsi_trend_max
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.atr_extreme_multiplier = atr_extreme_multiplier
        self.extrema_lookback = extrema_lookback
        self.data_dir = data_dir
        self.contract_name = contract_name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-timeframe trading signals.

        Strategy:
        1. Load data from higher and lower timeframes
        2. Use higher timeframes to determine overall trend
        3. Use lower timeframe for entry/exit signals
        4. Only take trades in the direction of the higher timeframe trend

        In live trading the StrategyAdapter pre-populates multi-timeframe
        columns with a ``{bar_size}_`` prefix (e.g. ``1 day_close``).  When
        these columns are detected the strategy uses them directly instead
        of trying to load CSV files from disk.
        """
        df = df.copy()

        # ----- Live-trading path: use inline multi-timeframe columns -----
        inline_higher_tf = self._detect_inline_timeframes(df)
        if inline_higher_tf:
            logger.info(
                f"[AMTS] Detected inline higher-timeframe columns for: "
                f"{list(inline_higher_tf.keys())}. Using live-trading path."
            )
            return self._generate_signals_live(df, inline_higher_tf)

        # ----- Backtesting path: load CSV files from disk -----
        contract_name = self.contract_name
        if contract_name is None:
            raise ValueError("contract_name must be provided for multi-timeframe strategy")

        logger.debug(f"[AMTS] No inline timeframe columns found, loading CSVs from {self.data_dir}/{contract_name}")
        timeframe_data = self._load_timeframe_data(df, self.data_dir, contract_name)

        if not timeframe_data:
            logger.info("[AMTS] Multi-timeframe CSV data not available, falling back to single timeframe")
            return self._generate_base_signals(df)

        # Align timeframes
        aligned_data = self._align_timeframes(timeframe_data)
        if not aligned_data:
            return self._generate_base_signals(df)

        return self._generate_signals_from_aligned(df, aligned_data)

    # ------------------------------------------------------------------
    # Live-trading signal generation
    # ------------------------------------------------------------------

    def _detect_inline_timeframes(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Check if the DataFrame already contains columns prefixed with higher
        timeframe bar sizes (e.g. ``1 day_close``).

        Returns:
            Dict mapping bar_size -> list of available column names, or
            empty dict if no inline timeframe columns found.
        """
        found: Dict[str, List[str]] = {}
        for bar_size in self.higher_timeframes:
            prefix = f"{bar_size}_"
            cols = [c for c in df.columns if c.startswith(prefix)]
            if cols:
                found[bar_size] = cols
        return found

    def _generate_signals_live(
        self, df: pd.DataFrame, inline_higher_tf: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Live-trading path: higher-timeframe data is already in the DataFrame
        as ``{bar_size}_column`` columns provided by StrategyAdapter.
        """
        # Compute higher-timeframe trend from inline columns
        higher_trend = self._get_inline_higher_trend(df, inline_higher_tf)

        # Generate base signals on the primary timeframe
        df = self._generate_base_signals(df)

        # Filter by higher-timeframe trend
        df = self._apply_trend_filter(df, higher_trend)

        return df

    def _get_inline_higher_trend(
        self, df: pd.DataFrame, inline_higher_tf: Dict[str, List[str]]
    ) -> pd.Series:
        """
        Determine overall trend from inline higher-timeframe columns.

        Uses the same logic as ``_get_trend_from_higher_timeframe`` but
        reads from ``{bar_size}_adx``, ``{bar_size}_plus_di``, etc.
        """
        trend_signals = pd.Series(0, index=df.index, dtype=float)

        for bar_size in self.higher_timeframes:
            prefix = f"{bar_size}_"
            required = ["adx", "plus_di", "minus_di", "SMA_50", "close"]
            col_map = {col: f"{prefix}{col}" for col in required}

            # Check all required columns exist
            if not all(col_map[c] in df.columns for c in required):
                missing = [c for c in required if col_map[c] not in df.columns]
                logger.debug(f"[AMTS] Skipping {bar_size} trend: missing {missing}")
                continue

            adx = df[col_map["adx"]]
            plus_di = df[col_map["plus_di"]]
            minus_di = df[col_map["minus_di"]]
            sma50 = df[col_map["SMA_50"]]
            close = df[col_map["close"]]

            strong_trend = adx > 25
            bullish = plus_di > minus_di
            bearish = minus_di > plus_di
            price_above_sma = close > sma50
            price_below_sma = close < sma50

            uptrend = strong_trend & bullish & price_above_sma
            downtrend = strong_trend & bearish & price_below_sma

            trend_signals = trend_signals + (uptrend.astype(int) - downtrend.astype(int))

        return np.sign(trend_signals)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _generate_base_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run AdaptiveMultiIndicatorStrategy on the primary timeframe data."""
        base_strategy = AdaptiveMultiIndicatorStrategy(
            initial_cash=self.initial_cash,
            commission=self.commission,
            adx_trend_threshold=self.adx_trend_threshold,
            adx_range_threshold=self.adx_range_threshold,
            rsi_trend_min=self.rsi_trend_min,
            rsi_trend_max=self.rsi_trend_max,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
            atr_stop_multiplier=self.atr_stop_multiplier,
            atr_take_profit_multiplier=self.atr_take_profit_multiplier,
            atr_extreme_multiplier=self.atr_extreme_multiplier,
            extrema_lookback=self.extrema_lookback,
        )
        return base_strategy.generate_signals(df)

    def _apply_trend_filter(
        self, df: pd.DataFrame, higher_trend: pd.Series
    ) -> pd.DataFrame:
        """Filter buy/sell signals by the higher-timeframe trend direction."""
        higher_trend_aligned = higher_trend.reindex(df.index, method="ffill").fillna(0)

        # Only buy when higher timeframe is uptrend or neutral
        buy_allowed = higher_trend_aligned >= 0
        df["execute_buy"] = np.where(
            buy_allowed & df["execute_buy"].notna(),
            df["execute_buy"],
            np.nan,
        )

        # Only sell when higher timeframe is downtrend or neutral
        sell_allowed = higher_trend_aligned <= 0
        df["execute_sell"] = np.where(
            sell_allowed & df["execute_sell"].notna(),
            df["execute_sell"],
            np.nan,
        )
        return df

    def _generate_signals_from_aligned(
        self, df: pd.DataFrame, aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Generate signals from aligned backtesting DataFrames (CSV path)."""
        # Get trend from higher timeframes
        higher_trend = self._get_trend_from_higher_timeframe(aligned_data)

        # Use the lowest timeframe for entry signals
        if self.lower_timeframes:
            signal_timeframe = self.lower_timeframes[0]
        else:
            signal_timeframe = list(aligned_data.keys())[0]

        if signal_timeframe not in aligned_data:
            return self._generate_base_signals(df)

        signal_df = aligned_data[signal_timeframe].copy()

        # Generate signals on the lower timeframe
        signal_df = self._generate_base_signals(signal_df)

        # Filter by higher-timeframe trend
        signal_df = self._apply_trend_filter(signal_df, higher_trend)

        # Map signals back to original df index
        if len(df) != len(signal_df) or not df.index.equals(signal_df.index):
            df["execute_buy"] = signal_df["execute_buy"].reindex(df.index, method="ffill")
            df["execute_sell"] = signal_df["execute_sell"].reindex(df.index, method="ffill")
        else:
            df["execute_buy"] = signal_df["execute_buy"]
            df["execute_sell"] = signal_df["execute_sell"]

        return df

