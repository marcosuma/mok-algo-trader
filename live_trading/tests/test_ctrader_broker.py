"""
Unit tests for CTraderBroker price handling and order validation.
"""
import pytest


class TestRoundPriceForSymbol:
    """Tests for _round_price_for_symbol (isolated, no network)."""

    def _make_broker(self, symbol_digits=None):
        """Create a minimal CTraderBroker with a fake _symbol_digits cache."""
        from live_trading.brokers.ctrader_broker import CTraderBroker

        broker = object.__new__(CTraderBroker)
        broker._symbol_digits = symbol_digits or {}
        return broker

    def test_rounds_to_2_digits_for_gold(self):
        broker = self._make_broker({1: 2})
        assert broker._round_price_for_symbol(1994.2692857142856, 1) == 1994.27

    def test_rounds_to_5_digits_for_forex(self):
        broker = self._make_broker({2: 5})
        assert broker._round_price_for_symbol(1.123456789, 2) == 1.12346

    def test_rounds_to_3_digits(self):
        broker = self._make_broker({3: 3})
        assert broker._round_price_for_symbol(123.45678, 3) == 123.457

    def test_defaults_to_5_digits_for_unknown_symbol(self):
        broker = self._make_broker({})
        assert broker._round_price_for_symbol(1.123456789, 999) == 1.12346

    def test_no_op_when_price_already_within_digits(self):
        broker = self._make_broker({1: 2})
        assert broker._round_price_for_symbol(1994.27, 1) == 1994.27

    def test_rounds_to_0_digits(self):
        broker = self._make_broker({4: 0})
        assert broker._round_price_for_symbol(1994.56, 4) == 1995.0

    def test_rounds_negative_price(self):
        """Sanity check — negative prices shouldn't appear but rounding must still work."""
        broker = self._make_broker({1: 2})
        assert broker._round_price_for_symbol(-0.005, 1) == -0.01


class TestConvertQuantityToVolume:
    """Tests for _convert_quantity_to_volume."""

    def _make_broker(self):
        from live_trading.brokers.ctrader_broker import CTraderBroker

        broker = object.__new__(CTraderBroker)
        broker._symbol_digits = {}
        return broker

    def test_one_lot(self):
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(1.0) == 100_000

    def test_point_zero_one_lot(self):
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(0.01) == 1_000

    def test_minimum_volume_enforced(self):
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(0.001) == 1_000

    def test_fractional_lot(self):
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(0.1) == 10_000

    def test_max_volume_clamped(self):
        """Regression: absurd lot sizes must be clamped to prevent broker rejection."""
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(313_199.0) == 10_000_000

    def test_100_lots_at_boundary(self):
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(100.0) == 10_000_000

    def test_99_lots_below_boundary(self):
        broker = self._make_broker()
        assert broker._convert_quantity_to_volume(99.0) == 9_900_000
