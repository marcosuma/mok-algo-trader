"""
Tests for the shared risk management module.

Ensures SL/TP calculation logic is correct and identical for both
backtesting and live trading paths.
"""
import pytest

from forex_strategies.risk_management import (
    calculate_stop_loss,
    calculate_take_profit,
    compute_sl_tp,
)


# ---------------------------------------------------------------------------
# Stop-loss tests
# ---------------------------------------------------------------------------

class TestCalculateStopLoss:

    def test_atr_long(self):
        sl = calculate_stop_loss(1.2000, "LONG", "ATR", 1.5, atr_value=0.0010)
        assert sl == pytest.approx(1.2000 - 1.5 * 0.0010)

    def test_atr_short(self):
        sl = calculate_stop_loss(1.2000, "SHORT", "ATR", 1.5, atr_value=0.0010)
        assert sl == pytest.approx(1.2000 + 1.5 * 0.0010)

    def test_atr_returns_none_when_no_atr(self):
        assert calculate_stop_loss(1.2000, "LONG", "ATR", 1.5, atr_value=None) is None

    def test_atr_returns_none_when_atr_zero(self):
        assert calculate_stop_loss(1.2000, "LONG", "ATR", 1.5, atr_value=0.0) is None

    def test_percentage_long(self):
        sl = calculate_stop_loss(1.2000, "LONG", "PERCENTAGE", 0.02)
        assert sl == pytest.approx(1.2000 - 0.02 * 1.2000)

    def test_percentage_short(self):
        sl = calculate_stop_loss(1.2000, "SHORT", "PERCENTAGE", 0.02)
        assert sl == pytest.approx(1.2000 + 0.02 * 1.2000)

    def test_fixed_long(self):
        sl = calculate_stop_loss(1.2000, "LONG", "FIXED", 0.0050)
        assert sl == pytest.approx(1.2000 - 0.0050)

    def test_fixed_short(self):
        sl = calculate_stop_loss(1.2000, "SHORT", "FIXED", 0.0050)
        assert sl == pytest.approx(1.2000 + 0.0050)

    def test_none_type(self):
        assert calculate_stop_loss(1.2000, "LONG", "NONE", 0) is None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown stop_loss_type"):
            calculate_stop_loss(1.2000, "LONG", "UNKNOWN", 1.0)


# ---------------------------------------------------------------------------
# Take-profit tests
# ---------------------------------------------------------------------------

class TestCalculateTakeProfit:

    def test_risk_reward_long(self):
        sl = 1.1985  # 15 pip risk
        tp = calculate_take_profit(1.2000, sl, "LONG", "RISK_REWARD", 2.0)
        expected = 1.2000 + (1.2000 - 1.1985) * 2.0
        assert tp == pytest.approx(expected)

    def test_risk_reward_short(self):
        sl = 1.2015
        tp = calculate_take_profit(1.2000, sl, "SHORT", "RISK_REWARD", 2.0)
        expected = 1.2000 - (1.2015 - 1.2000) * 2.0
        assert tp == pytest.approx(expected)

    def test_risk_reward_returns_none_without_sl(self):
        assert calculate_take_profit(1.2000, None, "LONG", "RISK_REWARD", 2.0) is None

    def test_atr_long(self):
        tp = calculate_take_profit(1.2000, 1.1985, "LONG", "ATR", 2.5, atr_value=0.0010)
        assert tp == pytest.approx(1.2000 + 2.5 * 0.0010)

    def test_atr_short(self):
        tp = calculate_take_profit(1.2000, 1.2015, "SHORT", "ATR", 2.5, atr_value=0.0010)
        assert tp == pytest.approx(1.2000 - 2.5 * 0.0010)

    def test_atr_returns_none_when_no_atr(self):
        assert calculate_take_profit(1.2000, 1.1985, "LONG", "ATR", 2.0) is None

    def test_percentage_long(self):
        tp = calculate_take_profit(1.2000, 1.1985, "LONG", "PERCENTAGE", 0.03)
        assert tp == pytest.approx(1.2000 + 0.03 * 1.2000)

    def test_percentage_short(self):
        tp = calculate_take_profit(1.2000, 1.2015, "SHORT", "PERCENTAGE", 0.03)
        assert tp == pytest.approx(1.2000 - 0.03 * 1.2000)

    def test_fixed_returns_absolute_price(self):
        tp = calculate_take_profit(1.2000, 1.1985, "LONG", "FIXED", 1.2500)
        assert tp == pytest.approx(1.2500)

    def test_none_type(self):
        assert calculate_take_profit(1.2000, 1.1985, "LONG", "NONE", 0) is None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown take_profit_type"):
            calculate_take_profit(1.2000, 1.1985, "LONG", "UNKNOWN", 1.0)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class TestComputeSlTp:

    def test_returns_tuple(self):
        sl, tp = compute_sl_tp(
            entry_price=1.2000,
            position_type="LONG",
            sl_type="ATR",
            sl_value=1.5,
            tp_type="RISK_REWARD",
            tp_value=2.0,
            atr_value=0.0010,
        )
        assert sl is not None
        assert tp is not None
        assert sl < 1.2000 < tp

    def test_none_types_return_none_tuple(self):
        sl, tp = compute_sl_tp(
            entry_price=1.2000,
            position_type="LONG",
            sl_type="NONE",
            sl_value=0,
            tp_type="NONE",
            tp_value=0,
        )
        assert sl is None
        assert tp is None

    def test_risk_reward_depends_on_sl(self):
        """RISK_REWARD TP needs SL; if SL is NONE, TP should also be None."""
        sl, tp = compute_sl_tp(
            entry_price=1.2000,
            position_type="LONG",
            sl_type="NONE",
            sl_value=0,
            tp_type="RISK_REWARD",
            tp_value=2.0,
        )
        assert sl is None
        assert tp is None
