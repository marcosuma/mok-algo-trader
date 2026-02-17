"""
Adaptive Multi-Indicator Strategy (AMIS)

This strategy adapts to market conditions by switching between:
- Trend-following mode (when ADX >25)
- Mean reversion mode (when ADX <20)

It uses multiple technical indicators for signal confirmation to reduce false signals.
"""

import pandas as pd
import numpy as np
from forex_strategies.base_strategy import BaseForexStrategy


class AdaptiveMultiIndicatorStrategy(BaseForexStrategy):
    """
    Adaptive strategy that combines multiple indicators:
    - ADX for market regime detection
    - MACD for momentum
    - RSI for overbought/oversold
    - Bollinger Bands for mean reversion
    - Local Extrema for support/resistance
    - SMA for trend direction
    - ATR for volatility and risk management
    """

    def __init__(
        self,
        initial_cash=10000,
        commission=0.0002,
        # Trend-following parameters
        adx_trend_threshold=25,
        adx_range_threshold=20,
        rsi_trend_min=40,
        rsi_trend_max=65,
        # Mean reversion parameters
        rsi_oversold=35,
        rsi_overbought=65,
        # Risk management
        atr_stop_multiplier=2.0,
        atr_take_profit_multiplier=2.5,
        atr_extreme_multiplier=3.0,
        # Local extrema lookback
        extrema_lookback=20,
    ):
        super().__init__(initial_cash, commission)
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

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive multi-indicator trading signals."""
        df = df.copy()

        # Core required indicators (strategy cannot function without these)
        core_required_cols = [
            "adx",
            "plus_di",
            "minus_di",
            "RSI_14",
            "macd",
            "macd_s",
            "macd_h",
            "SMA_50",
            "bollinger_up",
            "bollinger_down",
            "atr",
            "close",
            "high",
            "low",
        ]
        
        # Optional indicators (strategy can work without these, with reduced functionality)
        optional_cols = ["local_extrema"]
        
        missing_core = [col for col in core_required_cols if col not in df.columns]
        if missing_core:
            raise ValueError(f"Missing required indicators: {missing_core}")
        
        # Check for optional indicators
        missing_optional = [col for col in optional_cols if col not in df.columns]
        has_local_extrema = "local_extrema" in df.columns
        
        # Also check if local_extrema has any valid (non-None) values
        if has_local_extrema:
            # Check if we have at least some valid local extrema values
            valid_extrema_count = df["local_extrema"].notna().sum()
            if valid_extrema_count == 0:
                has_local_extrema = False
        
        if missing_optional or not has_local_extrema:
            # Log warning but continue - strategy will work with reduced functionality
            import logging
            logger = logging.getLogger(__name__)
            if missing_optional:
                logger.warning(f"[AdaptiveMultiIndicatorStrategy] Optional indicators missing: {missing_optional}. "
                              f"Support/resistance features will be disabled.")
            elif not has_local_extrema:
                logger.warning(f"[AdaptiveMultiIndicatorStrategy] local_extrema has no valid values. "
                              f"Support/resistance features will be disabled.")

        # Calculate rolling average ATR for volatility filter
        df["atr_avg"] = df["atr"].rolling(window=20, min_periods=1).mean()
        df["atr_extreme"] = (df["atr"] > (df["atr_avg"] * self.atr_extreme_multiplier)).fillna(False)

        # Market regime detection
        # Fill NaN values with False to ensure boolean operations work correctly
        df["strong_trend"] = (df["adx"] > self.adx_trend_threshold).fillna(False)
        df["ranging_market"] = (df["adx"] < self.adx_range_threshold).fillna(False)
        df["unclear_market"] = (
            (df["adx"] >= self.adx_range_threshold).fillna(False)
            & (df["adx"] <= self.adx_trend_threshold).fillna(False)
        )

        # Trend direction indicators
        df["bullish_direction"] = (df["plus_di"] > df["minus_di"]).fillna(False)
        df["bearish_direction"] = (df["minus_di"] > df["plus_di"]).fillna(False)
        df["price_above_sma50"] = (df["close"] > df["SMA_50"]).fillna(False)
        df["price_below_sma50"] = (df["close"] < df["SMA_50"]).fillna(False)

        # MACD signals
        # Fill NaN values with False to ensure boolean operations work correctly
        df["macd_bullish"] = (df["macd"] > df["macd_s"]).fillna(False).astype(bool)
        df["macd_bearish"] = (df["macd"] < df["macd_s"]).fillna(False).astype(bool)
        macd_bullish_shifted = df["macd_bullish"].shift(1).infer_objects(copy=False).fillna(False).astype(bool)
        df["macd_cross_up"] = df["macd_bullish"] & (~macd_bullish_shifted)
        df["macd_cross_down"] = df["macd_bearish"] & (~macd_bullish_shifted)
        df["macd_histogram_turning_positive"] = (
            ((df["macd_h"] > 0).fillna(False)) & ((df["macd_h"].shift(1) <= 0).fillna(False))
        )
        df["macd_histogram_turning_negative"] = (
            ((df["macd_h"] < 0).fillna(False)) & ((df["macd_h"].shift(1) >= 0).fillna(False))
        )

        # RSI conditions
        df["rsi_oversold"] = (df["RSI_14"] < self.rsi_oversold).fillna(False)
        df["rsi_overbought"] = (df["RSI_14"] > self.rsi_overbought).fillna(False)
        df["rsi_trend_buy"] = (
            (df["RSI_14"] >= self.rsi_trend_min).fillna(False)
            & (df["RSI_14"] <= self.rsi_trend_max).fillna(False)
        )
        df["rsi_trend_sell"] = (
            (df["RSI_14"] >= self.rsi_trend_min).fillna(False)
            & (df["RSI_14"] <= self.rsi_trend_max).fillna(False)
        )

        # Bollinger Band conditions
        df["at_lower_band"] = (df["close"] <= df["bollinger_down"]).fillna(False)
        df["at_upper_band"] = (df["close"] >= df["bollinger_up"]).fillna(False)
        df["near_lower_band"] = (
            (df["close"] <= df["bollinger_down"] * 1.002).fillna(False)
        )  # Within 0.2% of lower band
        df["near_upper_band"] = (
            (df["close"] >= df["bollinger_up"] * 0.998).fillna(False)
        )  # Within 0.2% of upper band

        # Local extrema analysis (optional - provides support/resistance features)
        if has_local_extrema:
            from local_extrema.local_extrema import LOCAL_MAX, LOCAL_MIN

            df["is_local_max"] = (df["local_extrema"] == LOCAL_MAX).fillna(False)
            df["is_local_min"] = (df["local_extrema"] == LOCAL_MIN).fillna(False)

            # Find recent local extrema levels
            df["recent_local_max"] = df["high"].where(df["is_local_max"]).ffill()
            df["recent_local_min"] = df["low"].where(df["is_local_min"]).ffill()

            # Check if price is breaking above recent local max or below recent local min
            df["breaking_above_resistance"] = (
                (df["close"] > df["recent_local_max"].shift(1)).fillna(False)
            )
            df["breaking_below_support"] = (
                (df["close"] < df["recent_local_min"].shift(1)).fillna(False)
            )

            # Check if price is near support/resistance
            # Handle division by zero and NaN values
            price_diff_max = abs(df["close"] - df["recent_local_max"].shift(1))
            price_diff_min = abs(df["close"] - df["recent_local_min"].shift(1))
            df["near_support"] = (
                (price_diff_min / df["close"].replace(0, np.nan) < 0.005).fillna(False)
            )  # Within 0.5% of support
            df["near_resistance"] = (
                (price_diff_max / df["close"].replace(0, np.nan) < 0.005).fillna(False)
            )  # Within 0.5% of resistance
        else:
            # Fallback: disable support/resistance features
            # Use Bollinger bands as proxy for support/resistance instead
            df["is_local_max"] = False
            df["is_local_min"] = False
            df["recent_local_max"] = np.nan
            df["recent_local_min"] = np.nan
            df["breaking_above_resistance"] = False  # Will rely on other conditions
            df["breaking_below_support"] = False  # Will rely on other conditions
            # Use Bollinger bands proximity as proxy for near support/resistance
            df["near_support"] = df["near_lower_band"]
            df["near_resistance"] = df["near_upper_band"]

        # ===== TREND-FOLLOWING BUY SIGNALS =====
        trend_buy_conditions = (
            df["strong_trend"]  # Strong trend
            & df["bullish_direction"]  # Bullish direction
            & df["price_above_sma50"]  # Price above SMA50
            & df["macd_bullish"]  # MACD bullish
            & df["rsi_trend_buy"]  # RSI in trend range
            & (
                df["breaking_above_resistance"]  # Breaking above resistance
                | df["macd_cross_up"]  # Or MACD just crossed up
            )
            & ~df["atr_extreme"]  # Not extreme volatility
        )

        # ===== MEAN REVERSION BUY SIGNALS =====
        mean_reversion_buy_conditions = (
            df["ranging_market"]  # Ranging market
            & df["near_lower_band"]  # At or near lower Bollinger band
            & df["rsi_oversold"]  # RSI oversold
            & (
                df["near_support"]  # Near support level
                | df["at_lower_band"]
            )  # Or at lower band
            & (
                df["macd_histogram_turning_positive"]  # Momentum turning positive
                | df["macd_cross_up"]
            )
            & ~df["atr_extreme"]  # Not extreme volatility
        )

        # ===== TREND-FOLLOWING SELL SIGNALS =====
        trend_sell_conditions = (
            df["strong_trend"]  # Strong trend
            & df["bearish_direction"]  # Bearish direction
            & df["price_below_sma50"]  # Price below SMA50
            & df["macd_bearish"]  # MACD bearish
            & df["rsi_trend_sell"]  # RSI in trend range
            & (
                df["breaking_below_support"]  # Breaking below support
                | df["macd_cross_down"]
            )  # Or MACD just crossed down
            & ~df["atr_extreme"]  # Not extreme volatility
        )

        # ===== MEAN REVERSION SELL SIGNALS =====
        mean_reversion_sell_conditions = (
            df["ranging_market"]  # Ranging market
            & df["near_upper_band"]  # At or near upper Bollinger band
            & df["rsi_overbought"]  # RSI overbought
            & (
                df["near_resistance"]  # Near resistance level
                | df["at_upper_band"]
            )  # Or at upper band
            & (
                df["macd_histogram_turning_negative"]  # Momentum turning negative
                | df["macd_cross_down"]
            )
            & ~df["atr_extreme"]  # Not extreme volatility
        )

        # Combine all buy and sell conditions
        df["buy_signal"] = trend_buy_conditions | mean_reversion_buy_conditions
        df["sell_signal"] = trend_sell_conditions | mean_reversion_sell_conditions

        # Entry prices: Use close price with ATR-based buffer for trend-following,
        # close price for mean reversion
        df["execute_buy"] = np.where(
            df["buy_signal"],
            np.where(
                df["strong_trend"],
                df["close"] + df["atr"] * 0.5,  # Small buffer for trend-following
                df["close"],  # At market for mean reversion
            ),
            np.nan,
        )

        df["execute_sell"] = np.where(
            df["sell_signal"],
            np.where(
                df["strong_trend"],
                df["close"] - df["atr"] * 0.5,  # Small buffer for trend-following
                df["close"],  # At market for mean reversion
            ),
            np.nan,
        )

        return df

