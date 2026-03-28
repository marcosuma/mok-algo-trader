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


class TestParseBestConfigs:

    def test_parses_single_row(self):
        markdown = """
| Rank | Strategy | Params | Mean Sharpe | Beat B&H % | Windows |
|------|----------|--------|-------------|------------|---------|
| 1 | ATRBreakout | lookback=10, mult=1.5 | 2.12 | 75% | 8 |
"""
        from scripts.scheduled_walkforward import parse_best_configs
        results = parse_best_configs(markdown)
        assert len(results) == 1
        assert results[0]["strategy_name"] == "ATRBreakout"
        assert results[0]["mean_sharpe"] == pytest.approx(2.12)
        assert results[0]["beat_bh_pct"] == pytest.approx(0.75)

    def test_parses_multiple_rows(self):
        markdown = """
| 1 | ATRBreakout | lookback=10, mult=1.5 | 2.12 | 75% | 8 |
| 2 | ATRBreakout | lookback=20, mult=1.0 | 1.84 | 60% | 7 |
"""
        from scripts.scheduled_walkforward import parse_best_configs
        results = parse_best_configs(markdown)
        assert len(results) == 2

    def test_returns_empty_for_no_table(self):
        from scripts.scheduled_walkforward import parse_best_configs
        assert parse_best_configs("No table here") == []


class TestParseParamsString:

    def test_parses_int_and_float(self):
        from scripts.scheduled_walkforward import parse_params_string
        result = parse_params_string("lookback=10, mult=1.5")
        assert result["lookback"] == 10
        assert isinstance(result["lookback"], int)
        assert result["mult"] == pytest.approx(1.5)
        assert isinstance(result["mult"], float)

    def test_handles_empty_string(self):
        from scripts.scheduled_walkforward import parse_params_string
        assert parse_params_string("") == {}

    def test_handles_single_param(self):
        from scripts.scheduled_walkforward import parse_params_string
        result = parse_params_string("lookback=20")
        assert result == {"lookback": 20}


class TestEstimateSharpeStd:

    def test_extracts_std_value_from_markdown(self):
        from scripts.scheduled_walkforward import _estimate_sharpe_std
        markdown = "### ATRBreakout Results\nMean Sharpe: 2.1, std: 0.45\nOther data"
        result = _estimate_sharpe_std(markdown, "ATRBreakout")
        assert result == pytest.approx(0.45)

    def test_returns_fallback_when_not_found(self):
        from scripts.scheduled_walkforward import _estimate_sharpe_std
        result = _estimate_sharpe_std("No std here", "ATRBreakout")
        assert result == pytest.approx(0.5)
