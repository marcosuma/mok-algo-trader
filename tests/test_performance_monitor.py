# tests/test_performance_monitor.py
import pytest
from scripts.performance_monitor import compute_sharpe, detect_drift


class TestComputeSharpe:

    def test_positive_returns_positive_sharpe(self):
        daily_pnl = [10.0, 20.0, 15.0, 10.0, 25.0]
        result = compute_sharpe(daily_pnl, capital=10000)
        assert result > 0

    def test_negative_returns_negative_sharpe(self):
        daily_pnl = [-10.0, -20.0, -15.0, -10.0, -25.0]
        result = compute_sharpe(daily_pnl, capital=10000)
        assert result < 0

    def test_zero_std_returns_zero(self):
        # All same return → std dev = 0 → division by zero risk → return 0.0 as safe fallback
        daily_pnl = [10.0, 10.0, 10.0, 10.0, 10.0]
        result = compute_sharpe(daily_pnl, capital=10000)
        assert result == 0.0

    def test_insufficient_data_returns_none(self):
        # Fewer than 5 data points → can't compute reliably
        daily_pnl = [10.0, 20.0]
        result = compute_sharpe(daily_pnl, capital=10000)
        assert result is None

    def test_known_value(self):
        # 10% daily returns on $10k capital → $1000/day, std=0
        # But zero std → Sharpe = 0, so use varying returns
        # Mean daily return = 1%, std = 0.5% → Sharpe = (1/0.5)*sqrt(252) ≈ 31.7
        import math
        daily_pnl = [100.0, 150.0, 50.0, 100.0, 100.0]  # mean=100, std≈35.36
        result = compute_sharpe(daily_pnl, capital=10000)
        # mean_daily_ret = 1.0%, std = 0.3536% → annualized Sharpe ≈ (1.0/0.3536)*sqrt(252) ≈ 44.8
        assert result == pytest.approx(44.8, rel=0.05)


class TestDetectDrift:

    def test_no_drift_when_params_match(self):
        live = {"lookback_period": 10, "atr_multiplier": 1.5}
        optimal = {"lookback_period": 10, "atr_multiplier": 1.5}
        assert detect_drift(live, optimal) == []

    def test_drift_detected_when_value_differs(self):
        live = {"lookback_period": 10, "atr_multiplier": 1.5}
        optimal = {"lookback_period": 20, "atr_multiplier": 1.5}
        drifted = detect_drift(live, optimal)
        assert len(drifted) == 1
        assert drifted[0]["param"] == "lookback_period"
        assert drifted[0]["live"] == 10
        assert drifted[0]["optimal"] == 20

    def test_drift_detected_when_param_missing_from_live(self):
        live = {"lookback_period": 10}
        optimal = {"lookback_period": 10, "atr_multiplier": 1.5}
        drifted = detect_drift(live, optimal)
        assert len(drifted) == 1
        assert drifted[0]["param"] == "atr_multiplier"

    def test_extra_live_params_not_flagged(self):
        # Params in live but not in optimal are NOT drift
        live = {"lookback_period": 10, "trend_filter": False}
        optimal = {"lookback_period": 10}
        assert detect_drift(live, optimal) == []
