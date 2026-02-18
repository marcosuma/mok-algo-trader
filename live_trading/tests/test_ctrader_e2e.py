"""
End-to-end tests for CTraderBroker against the DEMO (paper) account.

These tests connect to the real cTrader DEMO API and place/cancel orders.
They require valid credentials in the .env file and network access.

Run with:
    python -m pytest live_trading/tests/test_ctrader_e2e.py -v -s

Skip in CI or unit-test runs by excluding the 'e2e' marker:
    python -m pytest -m "not e2e"
"""
import asyncio
import logging
import os
import pytest

logger = logging.getLogger(__name__)

E2E_REASON = "cTrader e2e tests require CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, and CTRADER_ACCESS_TOKEN"

# Load environment early so the check below works
try:
    from dotenv import load_dotenv
    _test_dir = os.path.dirname(os.path.abspath(__file__))
    _live_trading_dir = os.path.dirname(_test_dir)
    _project_root = os.path.dirname(_live_trading_dir)
    load_dotenv(os.path.join(_live_trading_dir, ".env"))
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

_has_credentials = all([
    os.getenv("CTRADER_CLIENT_ID"),
    os.getenv("CTRADER_CLIENT_SECRET"),
    os.getenv("CTRADER_ACCESS_TOKEN"),
])

e2e = pytest.mark.skipif(not _has_credentials, reason=E2E_REASON)

# Reusable test config
TEST_FOREX_PAIR = "EUR-USD"
TEST_GOLD = "XAU-USD"
# Minimum lot size varies by broker/symbol — Pepperstone DEMO requires 1.0 lot
# for forex. Adjust if your broker allows micro lots (0.01).
MIN_LOT_SIZE_FOREX = 1.0
MIN_LOT_SIZE_GOLD = 0.01


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def broker(event_loop):
    """Create and connect a CTraderBroker for the whole test module."""
    from live_trading.brokers.ctrader_broker import CTraderBroker

    b = CTraderBroker()
    connected = event_loop.run_until_complete(b.connect())
    assert connected, "Failed to connect to cTrader DEMO — check credentials"
    yield b
    event_loop.run_until_complete(b.disconnect())


# ---------------------------------------------------------------------------
# Connection & account tests
# ---------------------------------------------------------------------------

@e2e
class TestConnection:

    def test_connect_and_authenticate(self, broker):
        assert broker.connected
        assert broker.authenticated
        assert broker.account_id is not None

    def test_get_account_info(self, broker, event_loop):
        info = event_loop.run_until_complete(broker.get_account_info())
        assert info, "get_account_info returned empty"
        assert "account_id" in info

    def test_get_positions(self, broker, event_loop):
        positions = event_loop.run_until_complete(broker.get_positions())
        assert isinstance(positions, list)


# ---------------------------------------------------------------------------
# Market order tests
# ---------------------------------------------------------------------------

@e2e
class TestMarketOrder:

    def test_market_buy_forex(self, broker, event_loop):
        """Place a minimum-size market BUY on EUR-USD and verify acceptance."""
        order_id = event_loop.run_until_complete(
            broker.place_order(
                asset=TEST_FOREX_PAIR,
                action="BUY",
                quantity=MIN_LOT_SIZE_FOREX,
                order_type="MARKET",
            )
        )
        assert order_id, f"Market BUY on {TEST_FOREX_PAIR} failed (empty order_id)"
        logger.info(f"Market BUY order accepted: {order_id}")

    def test_market_sell_forex(self, broker, event_loop):
        """Place a minimum-size market SELL on EUR-USD to flatten the position."""
        order_id = event_loop.run_until_complete(
            broker.place_order(
                asset=TEST_FOREX_PAIR,
                action="SELL",
                quantity=MIN_LOT_SIZE_FOREX,
                order_type="MARKET",
            )
        )
        assert order_id, f"Market SELL on {TEST_FOREX_PAIR} failed (empty order_id)"
        logger.info(f"Market SELL order accepted: {order_id}")


# ---------------------------------------------------------------------------
# Limit order with SL/TP (price precision test)
# ---------------------------------------------------------------------------

@e2e
class TestLimitOrderPrecision:
    """
    Regression test for the INVALID_REQUEST price-digits bug.

    Places a limit order with stop_loss and take_profit values that have
    excessive decimals, verifying they are properly rounded before submission.
    """

    def test_limit_order_forex_with_sl_tp(self, broker, event_loop):
        """LIMIT BUY on EUR-USD far below market so it won't fill, then cancel."""
        limit_price = 0.90000
        sl = 0.8912345678
        tp = 0.9512345678

        order_id = event_loop.run_until_complete(
            broker.place_order(
                asset=TEST_FOREX_PAIR,
                action="BUY",
                quantity=MIN_LOT_SIZE_FOREX,
                order_type="LIMIT",
                price=limit_price,
                stop_loss=sl,
                take_profit=tp,
            )
        )
        assert order_id, (
            f"Limit order on {TEST_FOREX_PAIR} with SL={sl} TP={tp} was rejected "
            "(price digits not rounded?)"
        )
        logger.info(f"Limit order accepted: {order_id}")

        # Best-effort cleanup — order might already be gone if the broker
        # matched or rejected it between acceptance and our cancel call.
        event_loop.run_until_complete(broker.cancel_order(order_id))

    def test_limit_order_gold_with_sl_tp(self, broker, event_loop):
        """
        Reproduces the original bug: XAU-USD with 2-digit precision.

        Places a LIMIT BUY far below market with high-precision SL/TP
        that would previously cause INVALID_REQUEST.
        """
        limit_price = 1800.00
        sl = 1780.2692857142856
        tp = 1850.9387142857143

        order_id = event_loop.run_until_complete(
            broker.place_order(
                asset=TEST_GOLD,
                action="BUY",
                quantity=MIN_LOT_SIZE_GOLD,
                order_type="LIMIT",
                price=limit_price,
                stop_loss=sl,
                take_profit=tp,
            )
        )
        assert order_id, (
            f"Limit order on {TEST_GOLD} with SL={sl} TP={tp} was rejected — "
            "price digits regression"
        )
        logger.info(f"Gold limit order accepted: {order_id}")

        event_loop.run_until_complete(broker.cancel_order(order_id))


# ---------------------------------------------------------------------------
# Stop order precision test
# ---------------------------------------------------------------------------

@e2e
class TestStopOrderPrecision:

    def test_stop_order_with_excessive_decimals(self, broker, event_loop):
        """STOP BUY on EUR-USD far above market so it won't fill, then cancel."""
        stop_price = 1.50000
        sl = 1.4812345678
        tp = 1.5512345678

        order_id = event_loop.run_until_complete(
            broker.place_order(
                asset=TEST_FOREX_PAIR,
                action="BUY",
                quantity=MIN_LOT_SIZE_FOREX,
                order_type="STOP",
                price=stop_price,
                stop_loss=sl,
                take_profit=tp,
            )
        )
        assert order_id, "Stop order was rejected (price digits not rounded?)"
        logger.info(f"Stop order accepted: {order_id}")

        event_loop.run_until_complete(broker.cancel_order(order_id))
