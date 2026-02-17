"""
Market Structure Strategy (MSS)

Trades price structure itself: swing points, trend classification (HH/HL/LH/LL),
dynamic support/resistance zones, and chart pattern detection.

Architecture:
    Layer 1 - SwingPointDetector:        Finds swing highs/lows, classifies HH/HL/LH/LL
    Layer 2 - MarketStructureClassifier: Determines trend regime from swing sequence
    Layer 3 - SupportResistanceMapper:   Clusters swings into S/R zones with strength scores
    Layer 4 - PatternDetector:           Detects H&S, IHS, Double Top/Bottom, Triangles
    Layer 5 - Signal generation:         Combines layers into entry/exit signals
    Layer 6 - Filters & risk management: ATR filter, R:R ratio, cooldown
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from forex_strategies.base_strategy import BaseForexStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class SwingType(Enum):
    """Classification of a swing point relative to the previous swing of the same kind."""
    HIGHER_HIGH = "HH"
    LOWER_HIGH = "LH"
    HIGHER_LOW = "HL"
    LOWER_LOW = "LL"


class TrendRegime(Enum):
    """Market trend regime derived from swing sequence."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"


class PatternType(Enum):
    """Recognized chart patterns."""
    HEAD_AND_SHOULDERS = "HS"
    INVERSE_HEAD_AND_SHOULDERS = "IHS"
    DOUBLE_TOP = "DT"
    DOUBLE_BOTTOM = "DB"
    TRIANGLE_TOP = "TTOP"
    TRIANGLE_BOTTOM = "TBOT"
    RECTANGLE_TOP = "RTOP"
    RECTANGLE_BOTTOM = "RBOT"


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    index: int          # Position in the DataFrame
    price: float        # Price at the swing point
    is_high: bool       # True for swing high, False for swing low
    swing_type: Optional[SwingType] = None  # HH, LH, HL, LL


@dataclass
class SRZone:
    """A support or resistance zone."""
    level: float        # Central price level
    strength: int       # Number of touches
    is_resistance: bool # True for resistance, False for support
    last_touch_idx: int # Index of the most recent touch
    touches: List[int] = field(default_factory=list)  # Indices of all touches


@dataclass
class DetectedPattern:
    """A detected chart pattern."""
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    neckline: float        # Neckline price level (for H&S and Double patterns)
    pattern_height: float  # Height of the pattern (for measured move targets)
    is_bullish: bool


# ---------------------------------------------------------------------------
# Layer 1: Swing Point Detection
# ---------------------------------------------------------------------------

class SwingPointDetector:
    """
    Detects swing highs and swing lows using left/right bar confirmation.

    A swing high requires the high to be >= all highs in the lookback window
    on both sides. Similarly for swing lows.

    Each swing is then classified relative to the previous swing of the same
    kind: HH, LH, HL, LL.
    """

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """
        Args:
            left_bars:  Number of bars to the left for confirmation.
            right_bars: Number of bars to the right for confirmation.
        """
        self.left_bars = left_bars
        self.right_bars = right_bars

    def detect(self, highs: np.ndarray, lows: np.ndarray) -> List[SwingPoint]:
        """
        Detect and classify swing points.

        Args:
            highs: Array of high prices.
            lows:  Array of low prices.

        Returns:
            List of SwingPoint sorted by index.
        """
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)
        swings = self._classify_swings(swing_highs, swing_lows)
        return swings

    def _find_swing_highs(self, highs: np.ndarray) -> List[SwingPoint]:
        """Find indices where high[i] is a local maximum (strictly higher than neighbours)."""
        results = []
        n = len(highs)
        for i in range(self.left_bars, n - self.right_bars):
            left_window = highs[i - self.left_bars: i]
            right_window = highs[i + 1: i + 1 + self.right_bars]
            if highs[i] > np.max(left_window) and highs[i] > np.max(right_window):
                results.append(SwingPoint(index=i, price=float(highs[i]), is_high=True))
        return results

    def _find_swing_lows(self, lows: np.ndarray) -> List[SwingPoint]:
        """Find indices where low[i] is a local minimum (strictly lower than neighbours)."""
        results = []
        n = len(lows)
        for i in range(self.left_bars, n - self.right_bars):
            left_window = lows[i - self.left_bars: i]
            right_window = lows[i + 1: i + 1 + self.right_bars]
            if lows[i] < np.min(left_window) and lows[i] < np.min(right_window):
                results.append(SwingPoint(index=i, price=float(lows[i]), is_high=False))
        return results

    def _classify_swings(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> List[SwingPoint]:
        """
        Classify each swing as HH/LH or HL/LL relative to the previous
        swing of the same kind, then merge and sort by index.
        """
        # Classify highs
        prev_high: Optional[SwingPoint] = None
        for sh in swing_highs:
            if prev_high is not None:
                sh.swing_type = (
                    SwingType.HIGHER_HIGH if sh.price > prev_high.price
                    else SwingType.LOWER_HIGH
                )
            prev_high = sh

        # Classify lows
        prev_low: Optional[SwingPoint] = None
        for sl in swing_lows:
            if prev_low is not None:
                sl.swing_type = (
                    SwingType.HIGHER_LOW if sl.price > prev_low.price
                    else SwingType.LOWER_LOW
                )
            prev_low = sl

        # Merge and sort
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda s: s.index)
        return all_swings


# ---------------------------------------------------------------------------
# Layer 2: Market Structure Classifier
# ---------------------------------------------------------------------------

class MarketStructureClassifier:
    """
    Determines the current trend regime from a sequence of classified swing points.

    Uptrend:   Recent swings show HH + HL
    Downtrend: Recent swings show LH + LL
    Ranging:   Mixed signals
    """

    def __init__(self, lookback_swings: int = 4):
        """
        Args:
            lookback_swings: How many recent classified swings to consider.
        """
        self.lookback_swings = lookback_swings

    def classify(self, swings: List[SwingPoint]) -> TrendRegime:
        """
        Classify the current market structure.

        Args:
            swings: All detected swing points (sorted by index).

        Returns:
            Current TrendRegime.
        """
        # Only look at swings that have been classified (skip the very first of each kind)
        classified = [s for s in swings if s.swing_type is not None]
        if len(classified) < 2:
            return TrendRegime.RANGING

        recent = classified[-self.lookback_swings:]
        return self._regime_from_swings(recent)

    def classify_at_each_bar(
        self, swings: List[SwingPoint], num_bars: int
    ) -> np.ndarray:
        """
        Produce a per-bar regime array.

        For each bar index ``i``, the regime is determined from all swings
        with ``swing.index <= i``.

        Args:
            swings:   All swing points (sorted by index).
            num_bars: Length of the price series.

        Returns:
            Array of TrendRegime values (as strings), length ``num_bars``.
        """
        regimes = np.full(num_bars, TrendRegime.RANGING.value, dtype=object)
        swing_idx = 0
        current_regime = TrendRegime.RANGING

        for bar in range(num_bars):
            # Advance the swing pointer to include all swings up to this bar
            while swing_idx < len(swings) and swings[swing_idx].index <= bar:
                swing_idx += 1
            # Classify from the swings seen so far
            visible = swings[:swing_idx]
            classified = [s for s in visible if s.swing_type is not None]
            if len(classified) >= 2:
                recent = classified[-self.lookback_swings:]
                current_regime = self._regime_from_swings(recent)
            regimes[bar] = current_regime.value

        return regimes

    def _regime_from_swings(self, recent: List[SwingPoint]) -> TrendRegime:
        """Determine regime from a short list of recent classified swings."""
        types = [s.swing_type for s in recent]
        bullish_count = sum(
            1 for t in types if t in (SwingType.HIGHER_HIGH, SwingType.HIGHER_LOW)
        )
        bearish_count = sum(
            1 for t in types if t in (SwingType.LOWER_HIGH, SwingType.LOWER_LOW)
        )
        total = len(types)
        if total == 0:
            return TrendRegime.RANGING
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total

        if bullish_ratio >= 0.6:
            return TrendRegime.UPTREND
        if bearish_ratio >= 0.6:
            return TrendRegime.DOWNTREND
        return TrendRegime.RANGING


# ---------------------------------------------------------------------------
# Layer 3: Support & Resistance Mapper
# ---------------------------------------------------------------------------

class SupportResistanceMapper:
    """
    Clusters swing points into support and resistance *zones*.

    Proximity-based grouping: if a swing price is within ``zone_merge_pct``
    of an existing zone's level, they belong to the same zone. Zones track
    a strength score (# touches) and recency.
    """

    def __init__(self, zone_merge_pct: float = 0.003, min_touches: int = 1):
        """
        Args:
            zone_merge_pct: Maximum relative distance to merge into the same
                            zone (0.003 = 0.3%).
            min_touches:    Minimum touches for a zone to be considered valid.
        """
        self.zone_merge_pct = zone_merge_pct
        self.min_touches = min_touches

    def build_zones(self, swings: List[SwingPoint]) -> List[SRZone]:
        """
        Build S/R zones from swing points.

        Swing highs create resistance zones; swing lows create support zones.
        """
        resistance_zones = self._cluster(
            [s for s in swings if s.is_high], is_resistance=True
        )
        support_zones = self._cluster(
            [s for s in swings if not s.is_high], is_resistance=False
        )
        all_zones = resistance_zones + support_zones
        # Filter by minimum touches
        all_zones = [z for z in all_zones if z.strength >= self.min_touches]
        return all_zones

    def nearest_support(
        self, zones: List[SRZone], price: float
    ) -> Optional[SRZone]:
        """Return the nearest support zone below ``price``, or None."""
        supports = [z for z in zones if not z.is_resistance and z.level < price]
        if not supports:
            return None
        return min(supports, key=lambda z: abs(z.level - price))

    def nearest_resistance(
        self, zones: List[SRZone], price: float
    ) -> Optional[SRZone]:
        """Return the nearest resistance zone above ``price``, or None."""
        resistances = [z for z in zones if z.is_resistance and z.level > price]
        if not resistances:
            return None
        return min(resistances, key=lambda z: abs(z.level - price))

    def _cluster(
        self, swing_points: List[SwingPoint], is_resistance: bool
    ) -> List[SRZone]:
        """Group swing points by proximity into zones."""
        if not swing_points:
            return []

        # Sort by price
        sorted_pts = sorted(swing_points, key=lambda s: s.price)
        zones: List[SRZone] = []

        for sp in sorted_pts:
            merged = False
            for zone in zones:
                if zone.level == 0:
                    continue
                if abs(sp.price - zone.level) / zone.level <= self.zone_merge_pct:
                    # Merge into existing zone (weighted average)
                    total_touches = zone.strength + 1
                    zone.level = (
                        (zone.level * zone.strength + sp.price) / total_touches
                    )
                    zone.strength = total_touches
                    zone.last_touch_idx = max(zone.last_touch_idx, sp.index)
                    zone.touches.append(sp.index)
                    merged = True
                    break
            if not merged:
                zones.append(
                    SRZone(
                        level=sp.price,
                        strength=1,
                        is_resistance=is_resistance,
                        last_touch_idx=sp.index,
                        touches=[sp.index],
                    )
                )
        return zones


# ---------------------------------------------------------------------------
# Layer 4: Pattern Detector
# ---------------------------------------------------------------------------

class PatternDetector:
    """
    Detects chart patterns from swing points.

    Supports: Head & Shoulders, Inverse H&S, Double Top, Double Bottom,
    Triangle Top/Bottom, Rectangle Top/Bottom.

    Uses the swing points from Layer 1 (no kernel regression needed).
    """

    def __init__(
        self,
        max_bars: int = 60,
        symmetry_tolerance: float = 0.03,
        double_tolerance: float = 0.015,
        rectangle_tolerance: float = 0.0075,
    ):
        """
        Args:
            max_bars:             Maximum number of bars a pattern can span.
            symmetry_tolerance:   Relative tolerance for shoulder/neckline
                                  symmetry in H&S patterns.
            double_tolerance:     Relative tolerance for double top/bottom peaks.
            rectangle_tolerance:  Relative tolerance for rectangle pattern levels.
        """
        self.max_bars = max_bars
        self.symmetry_tolerance = symmetry_tolerance
        self.double_tolerance = double_tolerance
        self.rectangle_tolerance = rectangle_tolerance

    def detect(self, swings: List[SwingPoint]) -> List[DetectedPattern]:
        """
        Detect patterns from swing points.

        Returns:
            List of DetectedPattern.
        """
        patterns: List[DetectedPattern] = []
        patterns.extend(self._detect_5point_patterns(swings))
        patterns.extend(self._detect_double_patterns(swings))
        return patterns

    # --- 5-point patterns (H&S, IHS, Triangles, Rectangles) ---------------

    def _detect_5point_patterns(
        self, swings: List[SwingPoint]
    ) -> List[DetectedPattern]:
        """Detect patterns that require 5 consecutive extrema."""
        patterns = []
        for i in range(5, len(swings) + 1):
            window = swings[i - 5: i]
            bar_span = window[-1].index - window[0].index
            if bar_span > self.max_bars:
                continue

            e1, e2, e3, e4, e5 = [s.price for s in window]
            start_idx = window[0].index
            end_idx = window[-1].index

            # --- Head and Shoulders (bearish) ---
            if self._is_head_and_shoulders(e1, e2, e3, e4, e5):
                neckline = (e2 + e4) / 2
                height = e3 - neckline
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.HEAD_AND_SHOULDERS,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=False,
                ))

            # --- Inverse Head and Shoulders (bullish) ---
            elif self._is_inverse_head_and_shoulders(e1, e2, e3, e4, e5):
                neckline = (e2 + e4) / 2
                height = neckline - e3
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=True,
                ))

            # --- Triangle Top (bearish) ---
            elif (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
                neckline = e4
                height = e3 - e4
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.TRIANGLE_TOP,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=False,
                ))

            # --- Triangle Bottom (bullish) ---
            elif (e1 < e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
                neckline = e4
                height = e4 - e3
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.TRIANGLE_BOTTOM,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=True,
                ))

            # --- Rectangle Top (bearish) ---
            elif self._is_rectangle_top(e1, e2, e3, e4, e5):
                g1 = np.mean([e1, e3, e5])
                g2 = np.mean([e2, e4])
                neckline = g2
                height = g1 - g2
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.RECTANGLE_TOP,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=False,
                ))

            # --- Rectangle Bottom (bullish) ---
            elif self._is_rectangle_bottom(e1, e2, e3, e4, e5):
                g1 = np.mean([e1, e3, e5])
                g2 = np.mean([e2, e4])
                neckline = g2
                height = g2 - g1
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.RECTANGLE_BOTTOM,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=True,
                ))

        return patterns

    # --- Double Top / Double Bottom (3-point patterns) ---------------------

    def _detect_double_patterns(
        self, swings: List[SwingPoint]
    ) -> List[DetectedPattern]:
        """Detect Double Top and Double Bottom from 3 consecutive extrema."""
        patterns = []
        for i in range(3, len(swings) + 1):
            window = swings[i - 3: i]
            bar_span = window[-1].index - window[0].index
            if bar_span > self.max_bars:
                continue

            e1, e2, e3 = [s.price for s in window]
            start_idx = window[0].index
            end_idx = window[-1].index

            # Double Top: high - low - high (with similar highs)
            if (
                window[0].is_high
                and not window[1].is_high
                and window[2].is_high
                and self._prices_equal(e1, e3)
                and e1 > e2
            ):
                neckline = e2
                height = ((e1 + e3) / 2) - e2
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOUBLE_TOP,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=False,
                ))

            # Double Bottom: low - high - low (with similar lows)
            elif (
                not window[0].is_high
                and window[1].is_high
                and not window[2].is_high
                and self._prices_equal(e1, e3)
                and e1 < e2
            ):
                neckline = e2
                height = e2 - ((e1 + e3) / 2)
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    start_idx=start_idx, end_idx=end_idx,
                    neckline=neckline, pattern_height=height,
                    is_bullish=True,
                ))

        return patterns

    # --- Helper methods ---

    def _is_head_and_shoulders(self, e1, e2, e3, e4, e5) -> bool:
        """H&S: e1(shoulder) > e2(neck) and e3(head) > e1,e5, shoulders similar, neckline similar."""
        mean_shoulder = np.mean([e1, e5])
        return (
            (e1 > e2)
            and (e3 > e1)
            and (e3 > e5)
            and (abs(e1 - e5) <= self.symmetry_tolerance * mean_shoulder)
            and (abs(e2 - e4) <= self.symmetry_tolerance * mean_shoulder)
        )

    def _is_inverse_head_and_shoulders(self, e1, e2, e3, e4, e5) -> bool:
        """IHS: e1(shoulder) < e2(neck) and e3(head) < e1,e5."""
        mean_shoulder = np.mean([e1, e5])
        return (
            (e1 < e2)
            and (e3 < e1)
            and (e3 < e5)
            and (abs(e1 - e5) <= self.symmetry_tolerance * mean_shoulder)
            and (abs(e2 - e4) <= self.symmetry_tolerance * mean_shoulder)
        )

    def _is_rectangle_top(self, e1, e2, e3, e4, e5) -> bool:
        g1 = np.mean([e1, e3, e5])
        g2 = np.mean([e2, e4])
        tol = self.rectangle_tolerance
        return (
            (e1 > e2)
            and (abs(e1 - g1) / g1 < tol)
            and (abs(e3 - g1) / g1 < tol)
            and (abs(e5 - g1) / g1 < tol)
            and (abs(e2 - g2) / g2 < tol)
            and (abs(e4 - g2) / g2 < tol)
            and (min(e1, e3, e5) > max(e2, e4))
        )

    def _is_rectangle_bottom(self, e1, e2, e3, e4, e5) -> bool:
        g1 = np.mean([e1, e3, e5])
        g2 = np.mean([e2, e4])
        tol = self.rectangle_tolerance
        return (
            (e1 < e2)
            and (abs(e1 - g1) / g1 < tol)
            and (abs(e3 - g1) / g1 < tol)
            and (abs(e5 - g1) / g1 < tol)
            and (abs(e2 - g2) / g2 < tol)
            and (abs(e4 - g2) / g2 < tol)
            and (max(e1, e3, e5) < min(e2, e4))
        )

    def _prices_equal(self, p1: float, p2: float) -> bool:
        """Check if two prices are approximately equal."""
        avg = (p1 + p2) / 2
        if avg == 0:
            return p1 == p2
        return abs(p1 - p2) / avg <= self.double_tolerance


# ---------------------------------------------------------------------------
# Layer 5 + 6: MarketStructureStrategy
# ---------------------------------------------------------------------------

class MarketStructureStrategy(BaseForexStrategy):
    """
    A price-structure-based trading strategy that combines:

    - Swing point detection (HH/HL/LH/LL)
    - Market structure / trend classification
    - Dynamic support & resistance zones
    - Chart pattern detection (H&S, IHS, Double Top/Bottom, Triangles)
    - ATR-based risk management with minimum R:R filtering

    Signal types:
      1. Trend continuation:  Pullback to S/R zone in a confirmed trend
      2. Structure break:     Pattern completion + neckline break against current trend
      3. Breakout:            Price breaks a key S/R zone with trend confirmation
    """

    def __init__(
        self,
        initial_cash: int = 10000,
        commission: float = 0.0002,
        spread: float = 0.0001,
        # Swing detection
        swing_left_bars: int = 5,
        swing_right_bars: int = 5,
        # Structure classifier
        lookback_swings: int = 4,
        # S/R zones
        zone_merge_pct: float = 0.003,
        sr_min_touches: int = 2,
        near_zone_pct: float = 0.003,
        # Pattern detection
        pattern_max_bars: int = 60,
        # Risk management
        atr_stop_multiplier: float = 1.5,
        min_rr_ratio: float = 1.5,
        atr_extreme_multiplier: float = 3.0,
        cooldown_bars: int = 5,
        # Optional ADX filter (set to 0 to disable)
        adx_trend_min: float = 20.0,
        adx_range_max: float = 25.0,
    ):
        super().__init__(initial_cash, commission, spread)

        # Layer 1
        self.swing_detector = SwingPointDetector(swing_left_bars, swing_right_bars)
        # Layer 2
        self.structure_classifier = MarketStructureClassifier(lookback_swings)
        # Layer 3
        self.sr_mapper = SupportResistanceMapper(zone_merge_pct, sr_min_touches)
        # Layer 4
        self.pattern_detector = PatternDetector(max_bars=pattern_max_bars)

        # Parameters
        self.near_zone_pct = near_zone_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_rr_ratio = min_rr_ratio
        self.atr_extreme_multiplier = atr_extreme_multiplier
        self.cooldown_bars = cooldown_bars
        self.adx_trend_min = adx_trend_min
        self.adx_range_max = adx_range_max

    # ------------------------------------------------------------------
    # BaseForexStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market structure analysis."""
        df = df.copy()
        df["execute_buy"] = np.nan
        df["execute_sell"] = np.nan

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)

        if len(df) < 20:
            return df

        # ----- Layer 1: Swing detection -----
        swings = self.swing_detector.detect(highs, lows)
        if len(swings) < 4:
            return df

        # ----- Layer 2: Trend regime per bar -----
        regimes = self.structure_classifier.classify_at_each_bar(swings, len(df))

        # ----- Layer 3: Build S/R zones (incrementally per bar) -----
        # Pre-compute per-bar nearest S/R for efficiency
        support_levels = np.full(len(df), np.nan)
        resistance_levels = np.full(len(df), np.nan)
        self._compute_sr_levels(swings, closes, support_levels, resistance_levels)

        # ----- Layer 4: Pattern detection -----
        patterns = self.pattern_detector.detect(swings)

        # Build per-bar pattern lookup: for each bar, what pattern just completed?
        pattern_at_bar: Dict[int, DetectedPattern] = {}
        for p in patterns:
            pattern_at_bar.setdefault(p.end_idx, [])
            pattern_at_bar[p.end_idx].append(p)

        # ----- ATR for stops / filters -----
        atr = self._get_atr(df)
        atr_avg = pd.Series(atr).rolling(window=20, min_periods=1).mean().values

        # Optional ADX filter
        has_adx = "adx" in df.columns
        adx_values = df["adx"].values.astype(float) if has_adx else None

        # ----- Layer 5+6: Signal generation -----
        last_trade_bar = -self.cooldown_bars - 1  # Allow first trade

        for i in range(1, len(df)):
            # Cooldown filter
            if (i - last_trade_bar) < self.cooldown_bars:
                continue
            # Extreme volatility filter
            if atr_avg[i] > 0 and atr[i] > atr_avg[i] * self.atr_extreme_multiplier:
                continue

            regime = regimes[i]
            close = closes[i]
            current_atr = atr[i]
            sup = support_levels[i]
            res = resistance_levels[i]

            # --- Signal A: Trend continuation (pullback to zone) ---
            signal = self._check_trend_continuation(
                regime, close, sup, res, current_atr,
                adx_values, i, has_adx,
            )

            # --- Signal B: Pattern-based structure break ---
            if signal is None and i in pattern_at_bar:
                signal = self._check_pattern_signal(
                    pattern_at_bar[i], close, current_atr, regime,
                )

            # --- Signal C: S/R breakout ---
            if signal is None:
                signal = self._check_breakout(
                    regime, close, closes, i, sup, res, current_atr,
                    adx_values, has_adx,
                )

            # Apply signal
            if signal is not None:
                direction, price = signal
                if direction == "buy":
                    df.iloc[i, df.columns.get_loc("execute_buy")] = price
                else:
                    df.iloc[i, df.columns.get_loc("execute_sell")] = price
                last_trade_bar = i

        return df

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _check_trend_continuation(
        self,
        regime: str,
        close: float,
        sup: float,
        res: float,
        current_atr: float,
        adx_values: Optional[np.ndarray],
        bar_idx: int,
        has_adx: bool,
    ) -> Optional[Tuple[str, float]]:
        """
        Trend continuation: price pulls back to a support zone in an uptrend
        (or resistance zone in a downtrend).
        """
        if current_atr <= 0:
            return None

        # Optional ADX filter: require minimum ADX for trend trades
        if has_adx and self.adx_trend_min > 0:
            if np.isnan(adx_values[bar_idx]) or adx_values[bar_idx] < self.adx_trend_min:
                return None

        if regime == TrendRegime.UPTREND.value and not np.isnan(sup):
            # Price near support in uptrend -> buy
            if self._price_near_level(close, sup):
                stop_loss = sup - current_atr * self.atr_stop_multiplier
                if not np.isnan(res):
                    take_profit = res
                else:
                    take_profit = close + current_atr * self.atr_stop_multiplier * self.min_rr_ratio
                risk = close - stop_loss
                reward = take_profit - close
                if risk > 0 and (reward / risk) >= self.min_rr_ratio:
                    return ("buy", close)

        elif regime == TrendRegime.DOWNTREND.value and not np.isnan(res):
            # Price near resistance in downtrend -> sell
            if self._price_near_level(close, res):
                stop_loss = res + current_atr * self.atr_stop_multiplier
                if not np.isnan(sup):
                    take_profit = sup
                else:
                    take_profit = close - current_atr * self.atr_stop_multiplier * self.min_rr_ratio
                risk = stop_loss - close
                reward = close - take_profit
                if risk > 0 and (reward / risk) >= self.min_rr_ratio:
                    return ("sell", close)

        return None

    def _check_pattern_signal(
        self,
        bar_patterns: List[DetectedPattern],
        close: float,
        current_atr: float,
        regime: str,
    ) -> Optional[Tuple[str, float]]:
        """
        Pattern-based signal: look for a completed pattern with neckline break.
        Bullish patterns -> buy; bearish patterns -> sell.
        """
        if current_atr <= 0:
            return None

        for pat in bar_patterns:
            if pat.is_bullish:
                # Bullish pattern: buy if close is above the neckline
                if close > pat.neckline:
                    # Measured move target
                    target = close + pat.pattern_height
                    stop = close - current_atr * self.atr_stop_multiplier
                    risk = close - stop
                    reward = target - close
                    if risk > 0 and (reward / risk) >= self.min_rr_ratio:
                        return ("buy", close)
            else:
                # Bearish pattern: sell if close is below the neckline
                if close < pat.neckline:
                    target = close - pat.pattern_height
                    stop = close + current_atr * self.atr_stop_multiplier
                    risk = stop - close
                    reward = close - target
                    if risk > 0 and (reward / risk) >= self.min_rr_ratio:
                        return ("sell", close)

        return None

    def _check_breakout(
        self,
        regime: str,
        close: float,
        closes: np.ndarray,
        bar_idx: int,
        sup: float,
        res: float,
        current_atr: float,
        adx_values: Optional[np.ndarray],
        has_adx: bool,
    ) -> Optional[Tuple[str, float]]:
        """
        Breakout signal: price closes through a key S/R level with trend
        confirmation. Requires the previous close to be on the other side.
        """
        if current_atr <= 0 or bar_idx < 1:
            return None

        prev_close = closes[bar_idx - 1]

        # Breakout above resistance
        if not np.isnan(res) and close > res and prev_close <= res:
            if regime == TrendRegime.UPTREND.value:
                stop = res - current_atr * self.atr_stop_multiplier
                target = close + current_atr * self.atr_stop_multiplier * self.min_rr_ratio
                risk = close - stop
                reward = target - close
                if risk > 0 and (reward / risk) >= self.min_rr_ratio:
                    return ("buy", close)

        # Breakout below support
        if not np.isnan(sup) and close < sup and prev_close >= sup:
            if regime == TrendRegime.DOWNTREND.value:
                stop = sup + current_atr * self.atr_stop_multiplier
                target = close - current_atr * self.atr_stop_multiplier * self.min_rr_ratio
                risk = stop - close
                reward = close - target
                if risk > 0 and (reward / risk) >= self.min_rr_ratio:
                    return ("sell", close)

        return None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _price_near_level(self, price: float, level: float) -> bool:
        """Check if price is within near_zone_pct of a level."""
        if level == 0:
            return False
        return abs(price - level) / abs(level) <= self.near_zone_pct

    def _get_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Extract or compute ATR from the DataFrame."""
        if "atr" in df.columns:
            atr = df["atr"].values.astype(float)
            # Fill leading NaNs with the first valid value
            first_valid = np.nanmean(atr[:20]) if np.any(~np.isnan(atr[:20])) else 0
            return np.where(np.isnan(atr), first_valid, atr)

        # Compute ATR manually if not provided (14-period)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        return self._compute_atr(highs, lows, closes, period=14)

    @staticmethod
    def _compute_atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Compute Average True Range."""
        n = len(highs)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        atr = np.zeros(n)
        atr[:period] = np.nan
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        # Fill NaN with first valid
        first_valid = atr[period - 1] if n >= period else tr[0]
        atr = np.where(np.isnan(atr), first_valid, atr)
        return atr

    def _compute_sr_levels(
        self,
        swings: List[SwingPoint],
        closes: np.ndarray,
        support_out: np.ndarray,
        resistance_out: np.ndarray,
    ):
        """
        Compute nearest support and resistance level for each bar.

        This is done incrementally: for bar ``i`` we only consider swings
        with ``swing.index <= i`` to avoid look-ahead bias.
        """
        swing_idx = 0
        visible_swings: List[SwingPoint] = []

        for bar in range(len(closes)):
            # Add swings that have been confirmed by this bar
            while swing_idx < len(swings) and swings[swing_idx].index <= bar:
                visible_swings.append(swings[swing_idx])
                swing_idx += 1

            if not visible_swings:
                continue

            zones = self.sr_mapper.build_zones(visible_swings)
            close = closes[bar]

            sup_zone = self.sr_mapper.nearest_support(zones, close)
            res_zone = self.sr_mapper.nearest_resistance(zones, close)

            if sup_zone is not None:
                support_out[bar] = sup_zone.level
            if res_zone is not None:
                resistance_out[bar] = res_zone.level
