"""
Comprehensive tests for the MarketStructureStrategy and all its layers.

Tests cover:
    - Layer 1: SwingPointDetector
    - Layer 2: MarketStructureClassifier
    - Layer 3: SupportResistanceMapper
    - Layer 4: PatternDetector
    - Layer 5+6: MarketStructureStrategy (integration)
    - Backtesting fixes: ForexBacktestingStrategy limit price handling
"""

import numpy as np
import pandas as pd
import pytest

from forex_strategies.market_structure_strategy import (
    SwingPointDetector,
    SwingPoint,
    SwingType,
    MarketStructureClassifier,
    TrendRegime,
    SupportResistanceMapper,
    SRZone,
    PatternDetector,
    PatternType,
    DetectedPattern,
    MarketStructureStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(closes, spread=0.005):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    closes = np.array(closes, dtype=float)
    highs = closes + spread
    lows = closes - spread
    opens = (closes + np.roll(closes, 1)) / 2
    opens[0] = closes[0]
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.ones(n) * 1000,
    })
    return df


def _make_trend_series(
    n: int = 100,
    start: float = 1.0,
    end: float = 1.5,
    noise_std: float = 0.005,
) -> np.ndarray:
    """Create a trending price series with noise."""
    rng = np.random.RandomState(42)
    trend = np.linspace(start, end, n)
    noise = rng.normal(0, noise_std, n)
    return trend + noise


def _make_zigzag_series(
    peaks: list, troughs: list, points_per_leg: int = 12
) -> np.ndarray:
    """
    Create a zigzag price series from alternating peaks and troughs.

    peaks and troughs should alternate and can be of different lengths
    (one list may be 1 element longer).
    """
    series = []
    # Interleave peaks and troughs
    all_pts = []
    for i in range(max(len(peaks), len(troughs))):
        if i < len(troughs):
            all_pts.append(troughs[i])
        if i < len(peaks):
            all_pts.append(peaks[i])

    for i in range(len(all_pts) - 1):
        leg = np.linspace(all_pts[i], all_pts[i + 1], points_per_leg, endpoint=False)
        series.extend(leg)
    series.append(all_pts[-1])
    return np.array(series)


# ===========================================================================
# Layer 1: SwingPointDetector
# ===========================================================================

class TestSwingPointDetector:
    """Tests for swing high / swing low detection and classification."""

    def test_simple_peak_detection(self):
        """A single peak surrounded by lower values should be detected."""
        # Shape: / \ 
        prices_high = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        prices_low = [p - 0.01 for p in prices_high]

        detector = SwingPointDetector(left_bars=3, right_bars=3)
        swings = detector.detect(np.array(prices_high), np.array(prices_low))

        highs = [s for s in swings if s.is_high]
        assert len(highs) >= 1
        # The peak should be at index 5
        assert any(s.index == 5 for s in highs)

    def test_simple_trough_detection(self):
        """A single trough surrounded by higher values should be detected."""
        prices_low = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        prices_high = [p + 0.01 for p in prices_low]

        detector = SwingPointDetector(left_bars=3, right_bars=3)
        swings = detector.detect(np.array(prices_high), np.array(prices_low))

        lows = [s for s in swings if not s.is_high]
        assert len(lows) >= 1
        assert any(s.index == 5 for s in lows)

    def test_no_swings_in_flat_market(self):
        """No swings should be detected in a perfectly flat series."""
        flat = np.ones(20)
        detector = SwingPointDetector(left_bars=3, right_bars=3)
        swings = detector.detect(flat, flat)
        assert len(swings) == 0

    def test_classification_higher_high(self):
        """Two consecutive swing highs where the second is higher -> HH."""
        # Two peaks: first at 1.3, second at 1.5
        highs = np.array([1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2])
        lows = highs - 0.01
        detector = SwingPointDetector(left_bars=3, right_bars=3)
        swings = detector.detect(highs, lows)

        swing_highs = [s for s in swings if s.is_high and s.swing_type is not None]
        hh_swings = [s for s in swing_highs if s.swing_type == SwingType.HIGHER_HIGH]
        assert len(hh_swings) >= 1

    def test_classification_lower_low(self):
        """Two consecutive swing lows where the second is lower -> LL."""
        # Two troughs: first at 1.0, second at 0.8
        lows = np.array([1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1])
        highs = lows + 0.01
        detector = SwingPointDetector(left_bars=3, right_bars=3)
        swings = detector.detect(highs, lows)

        swing_lows = [s for s in swings if not s.is_high and s.swing_type is not None]
        ll_swings = [s for s in swing_lows if s.swing_type == SwingType.LOWER_LOW]
        assert len(ll_swings) >= 1

    def test_swings_sorted_by_index(self):
        """Returned swings should be sorted by index."""
        series = _make_zigzag_series(
            peaks=[1.5, 1.6, 1.7],
            troughs=[1.0, 1.1, 1.2],
            points_per_leg=8,
        )
        detector = SwingPointDetector(left_bars=3, right_bars=3)
        swings = detector.detect(series + 0.01, series - 0.01)
        indices = [s.index for s in swings]
        assert indices == sorted(indices)

    def test_left_right_bar_config(self):
        """Wider lookback should find fewer swings."""
        series = _make_zigzag_series(
            peaks=[1.5, 1.6, 1.7, 1.8],
            troughs=[1.0, 1.1, 1.0, 1.1],
            points_per_leg=6,
        )
        highs = series + 0.01
        lows = series - 0.01

        narrow = SwingPointDetector(left_bars=2, right_bars=2)
        wide = SwingPointDetector(left_bars=5, right_bars=5)

        swings_narrow = narrow.detect(highs, lows)
        swings_wide = wide.detect(highs, lows)

        assert len(swings_narrow) >= len(swings_wide)


# ===========================================================================
# Layer 2: MarketStructureClassifier
# ===========================================================================

class TestMarketStructureClassifier:
    """Tests for trend regime classification from swing sequences."""

    def test_uptrend_from_hh_hl(self):
        """A sequence of HH + HL should yield UPTREND."""
        swings = [
            SwingPoint(0, 1.0, False),  # first low (no type)
            SwingPoint(5, 1.3, True),   # first high (no type)
            SwingPoint(10, 1.1, False, SwingType.HIGHER_LOW),
            SwingPoint(15, 1.5, True, SwingType.HIGHER_HIGH),
            SwingPoint(20, 1.2, False, SwingType.HIGHER_LOW),
            SwingPoint(25, 1.7, True, SwingType.HIGHER_HIGH),
        ]
        classifier = MarketStructureClassifier(lookback_swings=4)
        regime = classifier.classify(swings)
        assert regime == TrendRegime.UPTREND

    def test_downtrend_from_lh_ll(self):
        """A sequence of LH + LL should yield DOWNTREND."""
        swings = [
            SwingPoint(0, 1.5, True),
            SwingPoint(5, 1.2, False),
            SwingPoint(10, 1.4, True, SwingType.LOWER_HIGH),
            SwingPoint(15, 1.1, False, SwingType.LOWER_LOW),
            SwingPoint(20, 1.3, True, SwingType.LOWER_HIGH),
            SwingPoint(25, 0.9, False, SwingType.LOWER_LOW),
        ]
        classifier = MarketStructureClassifier(lookback_swings=4)
        regime = classifier.classify(swings)
        assert regime == TrendRegime.DOWNTREND

    def test_ranging_from_mixed_signals(self):
        """Alternating bullish/bearish signals should yield RANGING."""
        swings = [
            SwingPoint(0, 1.0, False),
            SwingPoint(5, 1.5, True),
            SwingPoint(10, 1.1, False, SwingType.HIGHER_LOW),   # bullish
            SwingPoint(15, 1.4, True, SwingType.LOWER_HIGH),    # bearish
            SwingPoint(20, 0.9, False, SwingType.LOWER_LOW),    # bearish
            SwingPoint(25, 1.6, True, SwingType.HIGHER_HIGH),   # bullish
        ]
        classifier = MarketStructureClassifier(lookback_swings=4)
        regime = classifier.classify(swings)
        assert regime == TrendRegime.RANGING

    def test_too_few_swings_returns_ranging(self):
        """Fewer than 2 classified swings should return RANGING."""
        swings = [SwingPoint(0, 1.0, False)]
        classifier = MarketStructureClassifier()
        assert classifier.classify(swings) == TrendRegime.RANGING

    def test_classify_at_each_bar_length(self):
        """Per-bar regime array should have the same length as the input."""
        swings = [
            SwingPoint(5, 1.0, False),
            SwingPoint(10, 1.5, True),
            SwingPoint(15, 1.1, False, SwingType.HIGHER_LOW),
            SwingPoint(20, 1.6, True, SwingType.HIGHER_HIGH),
        ]
        classifier = MarketStructureClassifier()
        regimes = classifier.classify_at_each_bar(swings, 30)
        assert len(regimes) == 30

    def test_classify_at_each_bar_no_lookahead(self):
        """Regime at bar 12 should not see swings after bar 12."""
        swings = [
            SwingPoint(5, 1.0, False),
            SwingPoint(10, 1.5, True),
            SwingPoint(15, 1.1, False, SwingType.HIGHER_LOW),
            SwingPoint(20, 1.6, True, SwingType.HIGHER_HIGH),
            SwingPoint(25, 0.5, False, SwingType.LOWER_LOW),   # late reversal
            SwingPoint(30, 1.0, True, SwingType.LOWER_HIGH),
        ]
        classifier = MarketStructureClassifier(lookback_swings=4)
        regimes = classifier.classify_at_each_bar(swings, 35)
        # At bar 12, only swings at 5 and 10 are visible -- not enough classified
        assert regimes[12] == TrendRegime.RANGING.value


# ===========================================================================
# Layer 3: SupportResistanceMapper
# ===========================================================================

class TestSupportResistanceMapper:
    """Tests for S/R zone clustering and lookup."""

    def test_single_swing_creates_zone(self):
        """A single swing low should create a support zone."""
        mapper = SupportResistanceMapper(min_touches=1)
        swings = [SwingPoint(0, 1.1000, False)]
        zones = mapper.build_zones(swings)
        assert len(zones) == 1
        assert zones[0].is_resistance is False
        assert abs(zones[0].level - 1.1000) < 1e-6

    def test_nearby_swings_merge(self):
        """Two swing lows within zone_merge_pct should merge into one zone."""
        mapper = SupportResistanceMapper(zone_merge_pct=0.005, min_touches=1)
        swings = [
            SwingPoint(0, 1.1000, False),
            SwingPoint(10, 1.1004, False),  # 0.036% difference
        ]
        zones = mapper.build_zones(swings)
        support_zones = [z for z in zones if not z.is_resistance]
        assert len(support_zones) == 1
        assert support_zones[0].strength == 2

    def test_distant_swings_separate(self):
        """Two swing lows far apart should create separate zones."""
        mapper = SupportResistanceMapper(zone_merge_pct=0.003, min_touches=1)
        swings = [
            SwingPoint(0, 1.1000, False),
            SwingPoint(10, 1.2000, False),  # ~9% difference
        ]
        zones = mapper.build_zones(swings)
        support_zones = [z for z in zones if not z.is_resistance]
        assert len(support_zones) == 2

    def test_min_touches_filter(self):
        """Zones with fewer than min_touches should be excluded."""
        mapper = SupportResistanceMapper(zone_merge_pct=0.003, min_touches=2)
        swings = [
            SwingPoint(0, 1.1000, False),
            SwingPoint(10, 1.5000, True),
        ]
        zones = mapper.build_zones(swings)
        # Both zones have only 1 touch, should be filtered out
        assert len(zones) == 0

    def test_nearest_support(self):
        """nearest_support should return the closest support below price."""
        mapper = SupportResistanceMapper(min_touches=1)
        zones = [
            SRZone(level=1.0, strength=2, is_resistance=False, last_touch_idx=0),
            SRZone(level=1.2, strength=3, is_resistance=False, last_touch_idx=5),
            SRZone(level=1.5, strength=1, is_resistance=True, last_touch_idx=10),
        ]
        result = mapper.nearest_support(zones, price=1.3)
        assert result is not None
        assert result.level == 1.2

    def test_nearest_resistance(self):
        """nearest_resistance should return the closest resistance above price."""
        mapper = SupportResistanceMapper(min_touches=1)
        zones = [
            SRZone(level=1.0, strength=2, is_resistance=False, last_touch_idx=0),
            SRZone(level=1.5, strength=3, is_resistance=True, last_touch_idx=5),
            SRZone(level=1.8, strength=1, is_resistance=True, last_touch_idx=10),
        ]
        result = mapper.nearest_resistance(zones, price=1.3)
        assert result is not None
        assert result.level == 1.5

    def test_nearest_support_returns_none_when_no_support_below(self):
        """Should return None if there is no support below the price."""
        mapper = SupportResistanceMapper(min_touches=1)
        zones = [
            SRZone(level=1.5, strength=2, is_resistance=False, last_touch_idx=0),
        ]
        result = mapper.nearest_support(zones, price=1.3)
        assert result is None

    def test_strength_reflects_touch_count(self):
        """Zone strength should equal the number of merged swing points."""
        mapper = SupportResistanceMapper(zone_merge_pct=0.01, min_touches=1)
        swings = [
            SwingPoint(0, 1.100, True),
            SwingPoint(10, 1.102, True),
            SwingPoint(20, 1.098, True),
        ]
        zones = mapper.build_zones(swings)
        res_zones = [z for z in zones if z.is_resistance]
        assert len(res_zones) == 1
        assert res_zones[0].strength == 3


# ===========================================================================
# Layer 4: PatternDetector
# ===========================================================================

class TestPatternDetector:
    """Tests for chart pattern detection."""

    def _make_swings(self, points):
        """
        Build a list of SwingPoints from (price, is_high) tuples.
        Indices are auto-assigned 0, 10, 20, ...
        """
        return [
            SwingPoint(index=i * 10, price=p, is_high=h)
            for i, (p, h) in enumerate(points)
        ]

    def test_head_and_shoulders(self):
        """Classic H&S: shoulder-neck-head-neck-shoulder (bearish)."""
        detector = PatternDetector(max_bars=100)
        swings = self._make_swings([
            (1.30, True),   # left shoulder (high)
            (1.20, False),  # neckline (low)
            (1.40, True),   # head (high, above shoulders)
            (1.20, False),  # neckline (low, similar to e2)
            (1.30, True),   # right shoulder (high, similar to e1)
        ])
        patterns = detector.detect(swings)
        hs = [p for p in patterns if p.pattern_type == PatternType.HEAD_AND_SHOULDERS]
        assert len(hs) >= 1
        assert hs[0].is_bullish is False
        assert abs(hs[0].neckline - 1.20) < 0.01

    def test_inverse_head_and_shoulders(self):
        """Classic IHS (bullish)."""
        detector = PatternDetector(max_bars=100)
        swings = self._make_swings([
            (1.10, False),  # left shoulder (low)
            (1.20, True),   # neckline (high)
            (1.00, False),  # head (low, below shoulders)
            (1.20, True),   # neckline (high)
            (1.10, False),  # right shoulder (low, similar to e1)
        ])
        patterns = detector.detect(swings)
        ihs = [p for p in patterns if p.pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS]
        assert len(ihs) >= 1
        assert ihs[0].is_bullish is True

    def test_double_top(self):
        """Two similar peaks with a trough between them."""
        detector = PatternDetector(max_bars=100, double_tolerance=0.02)
        swings = self._make_swings([
            (1.50, True),   # first peak
            (1.30, False),  # trough
            (1.50, True),   # second peak (same level)
        ])
        patterns = detector.detect(swings)
        dt = [p for p in patterns if p.pattern_type == PatternType.DOUBLE_TOP]
        assert len(dt) >= 1
        assert dt[0].is_bullish is False
        assert abs(dt[0].neckline - 1.30) < 0.01

    def test_double_bottom(self):
        """Two similar troughs with a peak between them."""
        detector = PatternDetector(max_bars=100, double_tolerance=0.02)
        swings = self._make_swings([
            (1.10, False),  # first trough
            (1.30, True),   # peak
            (1.10, False),  # second trough (same level)
        ])
        patterns = detector.detect(swings)
        db = [p for p in patterns if p.pattern_type == PatternType.DOUBLE_BOTTOM]
        assert len(db) >= 1
        assert db[0].is_bullish is True

    def test_max_bars_filter(self):
        """Patterns spanning more than max_bars should be excluded."""
        detector = PatternDetector(max_bars=25)
        # Swings spaced 10 bars apart -> 5 points span 40 bars
        swings = self._make_swings([
            (1.30, True),
            (1.20, False),
            (1.40, True),
            (1.20, False),
            (1.30, True),
        ])
        patterns = detector.detect(swings)
        hs = [p for p in patterns if p.pattern_type == PatternType.HEAD_AND_SHOULDERS]
        # 40-bar span > 25 max_bars -> should be filtered
        assert len(hs) == 0

    def test_no_false_patterns_on_flat_data(self):
        """Flat prices should not produce any patterns."""
        detector = PatternDetector(max_bars=100)
        swings = self._make_swings([
            (1.0, True),
            (1.0, False),
            (1.0, True),
            (1.0, False),
            (1.0, True),
        ])
        patterns = detector.detect(swings)
        assert len(patterns) == 0

    def test_pattern_height_positive(self):
        """Detected pattern heights should be positive."""
        detector = PatternDetector(max_bars=100)
        swings = self._make_swings([
            (1.30, True),
            (1.20, False),
            (1.40, True),
            (1.20, False),
            (1.30, True),
        ])
        patterns = detector.detect(swings)
        for p in patterns:
            assert p.pattern_height > 0, f"Pattern {p.pattern_type} has non-positive height"


# ===========================================================================
# Layer 5+6: MarketStructureStrategy (integration)
# ===========================================================================

class TestMarketStructureStrategy:
    """Integration tests for the full strategy."""

    def test_generate_signals_returns_correct_columns(self):
        """Output should always contain execute_buy and execute_sell."""
        closes = _make_trend_series(n=100, start=1.0, end=1.3)
        df = _make_ohlcv_df(closes)
        strategy = MarketStructureStrategy(
            swing_left_bars=3, swing_right_bars=3,
            sr_min_touches=1,
        )
        result = strategy.generate_signals(df)
        assert "execute_buy" in result.columns
        assert "execute_sell" in result.columns
        assert len(result) == len(df)

    def test_no_signals_on_short_data(self):
        """Very short data should produce no signals."""
        df = _make_ohlcv_df([1.0, 1.1, 1.2, 1.1, 1.0])
        strategy = MarketStructureStrategy()
        result = strategy.generate_signals(df)
        assert result["execute_buy"].isna().all()
        assert result["execute_sell"].isna().all()

    def test_signals_are_not_nan_on_sufficient_data(self):
        """With enough trending + reversal data, at least one signal should appear."""
        # Build a pronounced zigzag with many turning points to generate
        # multiple swing points and S/R zones.
        series = _make_zigzag_series(
            peaks=[1.50, 1.55, 1.52, 1.58, 1.48, 1.54, 1.51, 1.56],
            troughs=[1.00, 1.05, 1.02, 1.08, 0.98, 1.04, 1.01, 1.06],
            points_per_leg=12,
        )
        # Add small noise to avoid perfectly flat segments
        rng = np.random.RandomState(123)
        series += rng.normal(0, 0.003, len(series))
        df = _make_ohlcv_df(series, spread=0.02)
        strategy = MarketStructureStrategy(
            swing_left_bars=3, swing_right_bars=3,
            sr_min_touches=1, min_rr_ratio=1.0,
            cooldown_bars=3,
            near_zone_pct=0.01,
            atr_extreme_multiplier=10.0,  # Relax extreme filter for synthetic data
        )
        result = strategy.generate_signals(df)
        has_buy = result["execute_buy"].notna().any()
        has_sell = result["execute_sell"].notna().any()
        assert has_buy or has_sell, "Expected at least one signal on zigzag data"

    def test_cooldown_prevents_rapid_signals(self):
        """Signals should be at least cooldown_bars apart."""
        series = _make_zigzag_series(
            peaks=[1.5, 1.6, 1.7, 1.5, 1.6, 1.7],
            troughs=[1.0, 1.1, 1.0, 1.1, 1.0, 1.1],
            points_per_leg=10,
        )
        rng = np.random.RandomState(99)
        series += rng.normal(0, 0.003, len(series))
        df = _make_ohlcv_df(series, spread=0.01)
        cooldown = 10
        strategy = MarketStructureStrategy(
            swing_left_bars=3, swing_right_bars=3,
            sr_min_touches=1, min_rr_ratio=1.0,
            cooldown_bars=cooldown,
        )
        result = strategy.generate_signals(df)

        signal_indices = result.index[
            result["execute_buy"].notna() | result["execute_sell"].notna()
        ].tolist()
        for i in range(1, len(signal_indices)):
            gap = signal_indices[i] - signal_indices[i - 1]
            assert gap >= cooldown, (
                f"Signals at {signal_indices[i-1]} and {signal_indices[i]} "
                f"are only {gap} bars apart (cooldown={cooldown})"
            )

    def test_no_lookahead_in_sr_computation(self):
        """
        S/R levels at bar i should not be influenced by data after bar i.
        We verify this by running the strategy on a subset vs the full data
        and checking that signals on the overlap are identical.
        """
        series = _make_zigzag_series(
            peaks=[1.5, 1.6, 1.5, 1.6],
            troughs=[1.0, 1.1, 1.0, 1.1],
            points_per_leg=15,
        )
        df_full = _make_ohlcv_df(series, spread=0.01)
        half = len(df_full) // 2
        df_half = df_full.iloc[:half].reset_index(drop=True)

        strategy = MarketStructureStrategy(
            swing_left_bars=3, swing_right_bars=3,
            sr_min_touches=1, min_rr_ratio=1.0,
            cooldown_bars=3,
        )
        result_full = strategy.generate_signals(df_full)
        result_half = strategy.generate_signals(df_half)

        # Signals on the first half should be identical
        for col in ["execute_buy", "execute_sell"]:
            full_vals = result_full[col].iloc[:half].values
            half_vals = result_half[col].values
            # NaN == NaN should be True
            mask = ~(np.isnan(full_vals) & np.isnan(half_vals))
            np.testing.assert_array_equal(
                full_vals[mask], half_vals[mask],
                err_msg=f"Look-ahead detected in {col}",
            )

    def test_strategy_works_with_atr_column(self):
        """Strategy should use pre-computed ATR if available."""
        closes = _make_trend_series(n=80, start=1.0, end=1.3)
        df = _make_ohlcv_df(closes)
        df["atr"] = 0.005  # Constant ATR
        strategy = MarketStructureStrategy(
            swing_left_bars=3, swing_right_bars=3, sr_min_touches=1,
        )
        result = strategy.generate_signals(df)
        assert "execute_buy" in result.columns

    def test_strategy_works_without_atr_column(self):
        """Strategy should compute ATR internally if column is missing."""
        closes = _make_trend_series(n=80, start=1.0, end=1.3)
        df = _make_ohlcv_df(closes)
        assert "atr" not in df.columns
        strategy = MarketStructureStrategy(
            swing_left_bars=3, swing_right_bars=3, sr_min_touches=1,
        )
        result = strategy.generate_signals(df)
        assert "execute_buy" in result.columns

    def test_atr_computation_static_method(self):
        """_compute_atr should return an array of the correct length."""
        highs = np.array([1.1, 1.2, 1.15, 1.25, 1.2, 1.3, 1.25, 1.35,
                          1.3, 1.4, 1.35, 1.45, 1.4, 1.5, 1.45])
        lows = highs - 0.05
        closes = (highs + lows) / 2

        atr = MarketStructureStrategy._compute_atr(highs, lows, closes, period=5)
        assert len(atr) == len(highs)
        assert np.all(atr >= 0)
        assert not np.any(np.isnan(atr))


# ===========================================================================
# Backtesting fixes: ForexBacktestingStrategy
# ===========================================================================

class TestForexBacktestingStrategyFixes:
    """Tests for the limit-price fix in ForexBacktestingStrategy."""

    def test_is_market_order_when_price_equals_close(self):
        """Signal price == close should be treated as market order."""
        from forex_strategies.backtesting_strategy import ForexBacktestingStrategy
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        assert strategy._is_market_order(1.2345, 1.2345) is True

    def test_is_not_market_order_when_price_differs(self):
        """Signal price != close should NOT be treated as market order."""
        from forex_strategies.backtesting_strategy import ForexBacktestingStrategy
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        assert strategy._is_market_order(1.2500, 1.2345) is False

    def test_is_market_order_zero_price(self):
        """Edge case: both prices are zero."""
        from forex_strategies.backtesting_strategy import ForexBacktestingStrategy
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        assert strategy._is_market_order(0.0, 0.0) is True

    def test_tolerance_boundary(self):
        """Prices very close but not equal should still be market orders."""
        from forex_strategies.backtesting_strategy import ForexBacktestingStrategy
        strategy = ForexBacktestingStrategy.__new__(ForexBacktestingStrategy)
        # 1e-12 relative difference -> within 1e-9 tolerance
        assert strategy._is_market_order(1.0000000000001, 1.0) is True
        # 1% difference -> not a market order
        assert strategy._is_market_order(1.01, 1.0) is False


# ===========================================================================
# BaseForexStrategy spread parameter
# ===========================================================================

class TestBaseStrategySpread:
    """Tests for the spread parameter added to BaseForexStrategy."""

    def test_default_spread_is_set(self):
        """New strategies should have a default spread value."""
        strategy = MarketStructureStrategy()
        assert strategy.spread == 0.0001

    def test_custom_spread_is_passed(self):
        """Custom spread should be stored correctly."""
        strategy = MarketStructureStrategy(spread=0.0005)
        assert strategy.spread == 0.0005

    def test_zero_spread(self):
        """Spread of 0 should be allowed (disables spread modeling)."""
        strategy = MarketStructureStrategy(spread=0)
        assert strategy.spread == 0
