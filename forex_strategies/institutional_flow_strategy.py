"""
Institutional Flow Strategy (IFS)
==================================

Enters at Fair Value Gap (FVG) zones -- imbalances left by institutional order
flow -- adapting its mode based on the Hurst exponent regime:

  TREND mode  (H > hurst_threshold AND structure = UPTREND / DOWNTREND)
      Enter FVG zones WITH the trend direction (classic SMC institutional entry).
      Long: price pulls back into a bullish FVG in an uptrend.
      Short: price rallies into a bearish FVG in a downtrend.

  MEAN-REVERSION mode  (H <= hurst_threshold OR structure = RANGING)
      Fade the imbalance -- enter bearish FVGs long (gap will fill) and bullish
      FVGs short. Works on short intraday timeframes where H < 0.5 is the norm.

Layers applied to all modes:
    Layer 1: FairValueGapDetector       3-candle price imbalance zones (ICT/SMC)
    Layer 2: VolatilityRegimeClassifier ATR z-score gate -- skip HIGH vol bars
    Layer 3: Rolling Hurst Exponent     H > 0.5 trending  /  H < 0.5 mean-rev
    Layer 4: MarketStructureClassifier  HH/HL swing sequence -> UPTREND/DOWNTREND
    Layer 5: Optional RSI filter        Confirms pullback / extension
    Layer 6: Optional ADX filter        Confirms trend strength (trend mode only)
    Layer 7: Risk management            ATR stop, R:R gate, cooldown, session filter

Research basis (2025):
  - FVG quantification: joshyattridge/smart-money-concepts (GitHub 2025)
  - Volatility regime:  2025 practitioner consensus -- HMM + ATR z-score method
  - Hurst exponent:     H-ETE-GNN, MDPI Fractals June 2025 -- Hurst improves
                        regime adaptability to structural market shifts
  - Trend + momentum:   "Forecast-to-Fill" arXiv:2511.08571 (Nov 2025),
                        Sharpe 2.88 walk-forward 2015-2025 on trend+momentum
  - Regime switching:   "Combining mean reversion and momentum in FX" (ScienceDirect)
                        -- 20.24% p.a., higher Sharpe than either strategy alone
  - RSI / ADX filters:  2025 feature engineering consensus for rule-based systems
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from forex_strategies.base_strategy import BaseForexStrategy
from forex_strategies.market_structure_strategy import (
    MarketStructureClassifier,
    SupportResistanceMapper,
    SwingPointDetector,
    TrendRegime,
)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class FVGZone:
    """A confirmed Fair Value Gap imbalance zone."""
    bar_idx: int        # Index of the 3rd candle (confirmation bar)
    top: float          # Upper boundary of the gap
    bottom: float       # Lower boundary of the gap
    is_bullish: bool    # True = gap was created by a bullish impulse move
    touched: bool = False      # True once price entered the zone
    invalidated: bool = False  # True once price closed through the zone


# -----------------------------------------------------------------------------
# Layer 1: Fair Value Gap Detector
# -----------------------------------------------------------------------------

class FairValueGapDetector:
    """
    Detects Fair Value Gaps: 3-candle patterns where the wick of candle 1 and
    the wick of candle 3 do not overlap, leaving an untraded price region.

        Bullish FVG: high[i-2] < low[i]    zone = [high[i-2], low[i]]
        Bearish FVG: low[i-2]  > high[i]   zone = [high[i],   low[i-2]]

    The gap is confirmed when candle i closes (bar_idx = i) and is available
    for trading from bar i+1 onwards (no look-ahead).
    A size filter (min_gap_atr_ratio) discards micro-gaps that are pure noise.
    """

    def __init__(self, min_gap_atr_ratio: float = 0.3):
        self.min_gap_atr_ratio = min_gap_atr_ratio

    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        atr: np.ndarray,
    ) -> List[FVGZone]:
        zones: List[FVGZone] = []
        n = len(highs)
        for i in range(2, n):
            min_gap = atr[i] * self.min_gap_atr_ratio if atr[i] > 0 else 0.0
            gap_bull = lows[i] - highs[i - 2]
            if gap_bull > 0 and gap_bull >= min_gap:
                zones.append(FVGZone(
                    bar_idx=i,
                    top=float(lows[i]),
                    bottom=float(highs[i - 2]),
                    is_bullish=True,
                ))
                continue
            gap_bear = lows[i - 2] - highs[i]
            if gap_bear > 0 and gap_bear >= min_gap:
                zones.append(FVGZone(
                    bar_idx=i,
                    top=float(lows[i - 2]),
                    bottom=float(highs[i]),
                    is_bullish=False,
                ))
        return zones


# -----------------------------------------------------------------------------
# Layer 2: Volatility Regime Classifier
# -----------------------------------------------------------------------------

class VolatilityRegimeClassifier:
    """
    Classifies each bar as LOW or HIGH volatility using ATR z-score.

    HIGH regime: ATR exceeds (rolling mean + threshold x rolling sigma).
    Strategy is suppressed in HIGH vol regimes: trends are unreliable,
    FVG entries are whipsawed, and risk-adjusted returns deteriorate.

    2025 practitioner consensus: HMM + ATR z-score is the gold standard
    for retail/semi-institutional quant regime detection.
    """

    def __init__(self, lookback: int = 60, threshold: float = 1.0):
        self.lookback = lookback
        self.threshold = threshold

    def classify(self, atr: np.ndarray) -> np.ndarray:
        """Returns bool array: True = HIGH vol (skip), False = LOW vol (ok)."""
        s = pd.Series(atr)
        min_p = max(self.lookback // 2, 10)
        roll_mean = s.rolling(self.lookback, min_periods=min_p).mean().values
        roll_std = s.rolling(self.lookback, min_periods=min_p).std().values
        upper = roll_mean + self.threshold * roll_std
        is_high = np.zeros(len(atr), dtype=bool)
        valid = ~np.isnan(upper)
        is_high[valid] = atr[valid] > upper[valid]
        return is_high


# -----------------------------------------------------------------------------
# Layer 3: Rolling Hurst Exponent
# -----------------------------------------------------------------------------

def compute_rolling_hurst(log_prices: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Rolling Hurst exponent via variance-time scaling.

    For fractional Brownian motion: Var(X_t+tau - X_t) ~ tau^(2H)
    => log(Var) ~ 2H * log(tau)  =>  H = slope / 2

    H > 0.5: trending / persistent   -> favour trend-following entries
    H ~ 0.5: random walk             -> no directional edge
    H < 0.5: mean-reverting          -> favour mean-reversion entries

    Research: H-ETE-GNN (MDPI Fractals, June 2025) -- Hurst flag improves
    adaptability to structural market shifts in FX.
    """
    n = len(log_prices)
    hurst = np.full(n, np.nan)
    lags = [2, 4, 8, 16]

    for i in range(window, n):
        ts = log_prices[i - window: i]
        log_vars: List[float] = []
        log_lags: List[float] = []
        for lag in lags:
            if lag >= window // 4:
                continue
            diffs = ts[lag:] - ts[:-lag]
            var = float(np.var(diffs))
            if var > 1e-20:
                log_vars.append(np.log(var))
                log_lags.append(np.log(lag))
        if len(log_lags) >= 2:
            slope = float(np.polyfit(log_lags, log_vars, 1)[0])
            hurst[i] = float(np.clip(slope / 2.0, 0.0, 1.0))

    return hurst


# -----------------------------------------------------------------------------
# Main Strategy
# -----------------------------------------------------------------------------

class InstitutionalFlowStrategy(BaseForexStrategy):
    """
    Institutional Flow Strategy -- regime-switching FVG entry system.

    TREND mode  (H > threshold, confirmed directional structure):
        Enters FVG zones in the direction of the trend. The FVG marks where
        institutional orders remain unfilled after an impulse move; price
        often returns to this zone before continuing in trend direction.

    MEAN-REVERSION mode  (H <= threshold or ranging):
        Fades the imbalance. After a bearish FVG price tends to fill the gap
        (mean-revert upward); after a bullish FVG it tends to fill downward.
        This is consistent with H < 0.5 mean-reverting market structure.

    Common filters (both modes):
        - Volatility regime: HIGH vol bars are skipped
        - R:R gate: trade rejected if reward/risk < min_rr_ratio
        - Cooldown: enforces minimum bars between signals
        - Session filter: intraday data only, default 07-17 UTC

    Optional confirmation (improves win rate):
        - RSI filter: requires RSI to confirm pullback direction
        - ADX filter: requires minimum trend strength (trend mode only)
    """

    def __init__(
        self,
        initial_cash: int = 10_000,
        commission: float = 0.0002,
        spread: float = 0.0001,
        # Backtesting risk management (used by the engine)
        stop_loss_type: str = "ATR",
        stop_loss_value: float = 1.5,
        take_profit_type: str = "RISK_REWARD",
        take_profit_value: float = 2.0,
        # Market structure (Layer 4)
        swing_left_bars: int = 8,   # raised: require more significant swing confirmation
        swing_right_bars: int = 8,  # raised: require more significant swing confirmation
        lookback_swings: int = 4,
        # Fair Value Gap (Layer 1)
        min_fvg_atr_ratio: float = 0.5,   # raised from 0.3: filter micro-gaps / noise
        max_active_fvgs: int = 10,
        # Volatility regime (Layer 2)
        vol_lookback: int = 60,
        vol_threshold: float = 1.5,   # raised from 1.0: allow moderate vol (trend moves need it)
        # Hurst exponent (Layer 3)
        hurst_window: int = 100,
        hurst_threshold: float = 0.52,   # raised from 0.5: require clearer trending signal
        # Regime switching
        use_mean_reversion_mode: bool = False,  # disabled: MR fires in wrong direction on trending pairs
        # RSI filter (Layer 5, optional)
        use_rsi_filter: bool = True,   # enabled: confirm pullback / extension
        rsi_long_max: float = 55.0,    # Long signal only if RSI <= this (55 allows mild momentum)
        rsi_short_min: float = 45.0,   # Short signal only if RSI >= this
        # ADX filter (Layer 6, optional -- trend mode only)
        use_adx_filter: bool = True,   # enabled: require confirmed trend strength
        adx_min: float = 25.0,         # raised from 20: stronger trend requirement
        # Signal quality
        atr_sl_multiplier: float = 1.5,
        min_rr_ratio: float = 2.0,     # raised from 1.5: require better reward/risk
        cooldown_bars: int = 8,        # raised from 5: reduce noise signals
        # Session filter (UTC hours, intraday only)
        session_start_hour: int = 7,
        session_end_hour: int = 17,
        enable_session_filter: bool = True,
    ):
        super().__init__(
            initial_cash=initial_cash,
            commission=commission,
            spread=spread,
            stop_loss_type=stop_loss_type,
            stop_loss_value=stop_loss_value,
            take_profit_type=take_profit_type,
            take_profit_value=take_profit_value,
        )

        self.swing_detector = SwingPointDetector(swing_left_bars, swing_right_bars)
        self.structure_classifier = MarketStructureClassifier(lookback_swings)
        self.sr_mapper = SupportResistanceMapper(zone_merge_pct=0.003, min_touches=2)
        self.fvg_detector = FairValueGapDetector(min_fvg_atr_ratio)
        self.vol_classifier = VolatilityRegimeClassifier(vol_lookback, vol_threshold)

        self.max_active_fvgs = max_active_fvgs
        self.hurst_window = hurst_window
        self.hurst_threshold = hurst_threshold
        self.use_mean_reversion_mode = use_mean_reversion_mode
        self.use_rsi_filter = use_rsi_filter
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.use_adx_filter = use_adx_filter
        self.adx_min = adx_min
        self.atr_sl_multiplier = atr_sl_multiplier
        self.min_rr_ratio = min_rr_ratio
        self.cooldown_bars = cooldown_bars
        self.session_start_hour = session_start_hour
        self.session_end_hour = session_end_hour
        self.enable_session_filter = enable_session_filter

    # -------------------------------------------------------------------------
    # generate_signals  (BaseForexStrategy interface)
    # -------------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals. No look-ahead bias:
          - FVGs confirmed at bar i are available from i+1 onwards.
          - S/R levels built incrementally from swings seen so far.
          - Hurst and vol regime use only past bars (rolling windows).
        """
        df = df.copy()
        df["execute_buy"] = np.nan
        df["execute_sell"] = np.nan

        n = len(df)
        if n < max(self.hurst_window + 20, 60):
            return df

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)

        # ATR
        atr = self._get_atr(df)

        # Optional RSI
        rsi = self._get_column(df, ("rsi_14", "RSI_14"), n)

        # Optional ADX
        adx = self._get_column(df, ("adx", "ADX_14"), n)

        # Layer 1: FVGs
        all_fvgs = self.fvg_detector.detect(highs, lows, atr)
        fvg_by_bar: Dict[int, List[FVGZone]] = {}
        for fvg in all_fvgs:
            fvg_by_bar.setdefault(fvg.bar_idx, []).append(fvg)

        # Layer 2: Volatility regime
        is_high_vol = self.vol_classifier.classify(atr)

        # Layer 3: Hurst exponent
        log_prices = np.log(np.maximum(closes, 1e-10))
        hurst = compute_rolling_hurst(log_prices, self.hurst_window)

        # Layer 4: Market structure
        swings = self.swing_detector.detect(highs, lows)
        if len(swings) < 4:
            return df
        regimes = self.structure_classifier.classify_at_each_bar(swings, n)

        # S/R for TP target estimation
        support_levels = np.full(n, np.nan)
        resistance_levels = np.full(n, np.nan)
        self._compute_sr_levels(swings, closes, support_levels, resistance_levels)

        # Session filter
        hours = self._extract_hours(df)

        # ── Main signal loop ─────────────────────────────────────────────────
        active_bull_fvgs: List[FVGZone] = []
        active_bear_fvgs: List[FVGZone] = []
        last_trade_bar = -self.cooldown_bars - 1

        for i in range(1, n):

            # Activate FVGs confirmed at bar i-1
            for fvg in fvg_by_bar.get(i - 1, []):
                if fvg.is_bullish:
                    active_bull_fvgs.append(fvg)
                else:
                    active_bear_fvgs.append(fvg)

            if len(active_bull_fvgs) > self.max_active_fvgs:
                active_bull_fvgs = active_bull_fvgs[-self.max_active_fvgs:]
            if len(active_bear_fvgs) > self.max_active_fvgs:
                active_bear_fvgs = active_bear_fvgs[-self.max_active_fvgs:]

            # Gate filters
            if (
                (i - last_trade_bar) < self.cooldown_bars
                or bool(is_high_vol[i])
                or atr[i] <= 0
                or (hours is not None
                    and not (self.session_start_hour <= int(hours[i]) < self.session_end_hour))
            ):
                self._update_fvg_state(i, highs, lows, closes, active_bull_fvgs, active_bear_fvgs)
                continue

            h = hurst[i]
            regime = regimes[i]
            close = closes[i]
            cur_atr = atr[i]
            risk = cur_atr * self.atr_sl_multiplier

            # Determine which mode applies
            h_available = not np.isnan(h)
            is_trending_hurst = h_available and h > self.hurst_threshold
            is_ranging_hurst = h_available and h <= self.hurst_threshold

            in_trend = regime in (TrendRegime.UPTREND.value, TrendRegime.DOWNTREND.value)
            in_range = regime == TrendRegime.RANGING.value

            # ADX value for trend-mode filter
            cur_adx = float(adx[i]) if adx is not None and not np.isnan(adx[i]) else None
            # RSI value for entry filter
            cur_rsi = float(rsi[i]) if rsi is not None and not np.isnan(rsi[i]) else None

            signal: Optional[tuple] = None

            # =================================================================
            # TREND MODE: H > threshold AND trending market structure
            # =================================================================
            if is_trending_hurst and in_trend:

                # Optional ADX gate
                if self.use_adx_filter and cur_adx is not None and cur_adx < self.adx_min:
                    self._update_fvg_state(i, highs, lows, closes, active_bull_fvgs, active_bear_fvgs)
                    continue

                if regime == TrendRegime.UPTREND.value:
                    # Long: price enters unmitigated bullish FVG
                    for fvg in active_bull_fvgs:
                        if fvg.touched or fvg.invalidated:
                            continue
                        if lows[i] <= fvg.top and closes[i] >= fvg.bottom:
                            # RSI confirmation: pullback should have RSI <= rsi_long_max
                            if self.use_rsi_filter and cur_rsi is not None and cur_rsi > self.rsi_long_max:
                                continue
                            res = resistance_levels[i]
                            reward = (
                                (res - close) if (not np.isnan(res) and res > close)
                                else risk * self.min_rr_ratio
                            )
                            if risk > 0 and reward / risk >= self.min_rr_ratio:
                                signal = ("buy", close)
                                fvg.touched = True
                                break

                elif regime == TrendRegime.DOWNTREND.value:
                    # Short: price enters unmitigated bearish FVG
                    for fvg in active_bear_fvgs:
                        if fvg.touched or fvg.invalidated:
                            continue
                        if highs[i] >= fvg.bottom and closes[i] <= fvg.top:
                            if self.use_rsi_filter and cur_rsi is not None and cur_rsi < self.rsi_short_min:
                                continue
                            sup = support_levels[i]
                            reward = (
                                (close - sup) if (not np.isnan(sup) and sup < close)
                                else risk * self.min_rr_ratio
                            )
                            if risk > 0 and reward / risk >= self.min_rr_ratio:
                                signal = ("sell", close)
                                fvg.touched = True
                                break

            # =================================================================
            # MEAN-REVERSION MODE: H <= threshold OR ranging structure
            # Fade the gap: bearish FVG -> long (gap will fill up)
            #               bullish FVG -> short (gap will fill down)
            # =================================================================
            elif self.use_mean_reversion_mode and (is_ranging_hurst or in_range):

                # Long: enter a BEARISH FVG from below (expect gap fill upward)
                for fvg in active_bear_fvgs:
                    if fvg.touched or fvg.invalidated:
                        continue
                    # Price rallied back into the bearish FVG from below
                    if highs[i] >= fvg.bottom and closes[i] >= fvg.bottom:
                        # TP is the gap fill (fvg.top); SL is ATR below entry
                        reward = fvg.top - close
                        if self.use_rsi_filter and cur_rsi is not None and cur_rsi > self.rsi_long_max:
                            continue
                        if risk > 0 and reward > 0 and reward / risk >= self.min_rr_ratio:
                            signal = ("buy", close)
                            fvg.touched = True
                            break

                # Short: enter a BULLISH FVG from above (expect gap fill downward)
                if signal is None:
                    for fvg in active_bull_fvgs:
                        if fvg.touched or fvg.invalidated:
                            continue
                        # Price pulled back into the bullish FVG from above
                        if lows[i] <= fvg.top and closes[i] <= fvg.top:
                            reward = close - fvg.bottom
                            if self.use_rsi_filter and cur_rsi is not None and cur_rsi < self.rsi_short_min:
                                continue
                            if risk > 0 and reward > 0 and reward / risk >= self.min_rr_ratio:
                                signal = ("sell", close)
                                fvg.touched = True
                                break

            # Apply signal
            if signal is not None:
                direction, price = signal
                col = "execute_buy" if direction == "buy" else "execute_sell"
                df.iloc[i, df.columns.get_loc(col)] = price
                last_trade_bar = i

            self._update_fvg_state(i, highs, lows, closes, active_bull_fvgs, active_bear_fvgs)

        return df

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _update_fvg_state(
        self,
        i: int,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        bull_fvgs: List[FVGZone],
        bear_fvgs: List[FVGZone],
    ) -> None:
        """Update touched/invalidated state for all active FVGs at bar i."""
        for fvg in bull_fvgs[:]:
            if not fvg.touched and lows[i] <= fvg.top and highs[i] >= fvg.bottom:
                fvg.touched = True
            if closes[i] < fvg.bottom:
                fvg.invalidated = True
                bull_fvgs.remove(fvg)

        for fvg in bear_fvgs[:]:
            if not fvg.touched and highs[i] >= fvg.bottom and lows[i] <= fvg.top:
                fvg.touched = True
            if closes[i] > fvg.top:
                fvg.invalidated = True
                bear_fvgs.remove(fvg)

    def _get_atr(self, df: pd.DataFrame) -> np.ndarray:
        for col in ("atr", "ATR_14"):
            if col in df.columns:
                atr = df[col].values.astype(float)
                first = float(np.nanmean(atr[:20])) if np.any(~np.isnan(atr)) else 0.0
                return np.where(np.isnan(atr), first, atr)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        return self._compute_atr(highs, lows, closes, period=14)

    @staticmethod
    def _get_column(
        df: pd.DataFrame,
        names: tuple,
        n: int,
    ) -> Optional[np.ndarray]:
        """Return first matching column as float array, or None if absent."""
        for name in names:
            if name in df.columns:
                arr = df[name].values.astype(float)
                return arr
        return None

    @staticmethod
    def _compute_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        n = len(highs)
        tr = np.empty(n)
        tr[0] = highs[0] - lows[0]
        for j in range(1, n):
            tr[j] = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j - 1]),
                abs(lows[j] - closes[j - 1]),
            )
        atr = np.zeros(n)
        if n >= period:
            atr[period - 1] = float(tr[:period].mean())
            for j in range(period, n):
                atr[j] = (atr[j - 1] * (period - 1) + tr[j]) / period
        first_valid = atr[period - 1] if n >= period else tr[0]
        return np.where(atr == 0.0, first_valid, atr)

    def _compute_sr_levels(
        self,
        swings: list,
        closes: np.ndarray,
        support_out: np.ndarray,
        resistance_out: np.ndarray,
    ) -> None:
        """Incremental S/R levels -- no look-ahead."""
        swing_ptr = 0
        visible: list = []
        for bar in range(len(closes)):
            while swing_ptr < len(swings) and swings[swing_ptr].index <= bar:
                visible.append(swings[swing_ptr])
                swing_ptr += 1
            if not visible:
                continue
            zones = self.sr_mapper.build_zones(visible)
            sup = self.sr_mapper.nearest_support(zones, float(closes[bar]))
            res = self.sr_mapper.nearest_resistance(zones, float(closes[bar]))
            if sup is not None:
                support_out[bar] = sup.level
            if res is not None:
                resistance_out[bar] = res.level

    def _extract_hours(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return UTC hour array for intraday data, None for daily+ data."""
        if not self.enable_session_filter:
            return None
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return None
        if (df.index[1] - df.index[0]).total_seconds() >= 86_400:
            return None
        return df.index.hour.values
