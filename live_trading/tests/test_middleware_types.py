"""Tests for broker middleware types and enums."""
import pytest
from datetime import datetime

from live_trading.brokers.middleware.types import (
    AccountMode,
    BrokerAccountInfo,
    BrokerDeal,
    BrokerOrderResult,
    BrokerOrderUpdate,
    BrokerPosition,
    CloseReason,
    OrderIntent,
    OrderParams,
    OrderSide,
    OrderSource,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionStatus,
)


class TestEnumValues:
    def test_order_side_values(self):
        assert OrderSide.BUY == "BUY"
        assert OrderSide.SELL == "SELL"

    def test_order_status_lifecycle(self):
        assert OrderStatus.PENDING_SUBMIT == "PENDING_SUBMIT"
        assert OrderStatus.SUBMITTED == "SUBMITTED"
        assert OrderStatus.FILLED == "FILLED"
        assert OrderStatus.REJECTED == "REJECTED"
        assert OrderStatus.CANCELLED == "CANCELLED"
        assert OrderStatus.EXPIRED == "EXPIRED"

    def test_position_status(self):
        assert PositionStatus.OPEN == "OPEN"
        assert PositionStatus.CLOSED == "CLOSED"

    def test_close_reason(self):
        assert CloseReason.STOP_LOSS == "STOP_LOSS"
        assert CloseReason.TAKE_PROFIT == "TAKE_PROFIT"
        assert CloseReason.STRATEGY_SIGNAL == "STRATEGY_SIGNAL"
        assert CloseReason.MANUAL == "MANUAL"
        assert CloseReason.UNKNOWN == "UNKNOWN"

    def test_account_mode(self):
        assert AccountMode.HEDGING == "HEDGING"
        assert AccountMode.NETTING == "NETTING"


class TestOrderParams:
    def test_minimal_market_order(self):
        params = OrderParams(
            symbol="EUR-USD",
            side=OrderSide.BUY,
            quantity=0.1,
        )
        assert params.order_type == OrderType.MARKET
        assert params.price is None
        assert params.stop_loss is None

    def test_limit_order_with_sl_tp(self):
        params = OrderParams(
            symbol="GBP-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=1.2500,
            stop_loss=1.2550,
            take_profit=1.2400,
        )
        assert params.price == 1.2500
        assert params.stop_loss == 1.2550

    def test_close_order_with_position_id(self):
        params = OrderParams(
            symbol="USD-CAD",
            side=OrderSide.SELL,
            quantity=0.5,
            broker_position_id="12345",
        )
        assert params.broker_position_id == "12345"

    def test_quantity_must_be_positive(self):
        with pytest.raises(Exception):
            OrderParams(symbol="X", side=OrderSide.BUY, quantity=-1.0)


class TestBrokerOrderResult:
    def test_filled_result(self):
        r = BrokerOrderResult(
            broker_order_id="999",
            status=OrderStatus.FILLED,
            filled_quantity=0.5,
            avg_fill_price=1.1234,
            broker_position_id="5678",
            commission=0.07,
        )
        assert r.status == OrderStatus.FILLED
        assert r.avg_fill_price == 1.1234

    def test_rejected_result(self):
        r = BrokerOrderResult(broker_order_id="", status=OrderStatus.REJECTED)
        assert r.filled_quantity == 0.0


class TestBrokerPosition:
    def test_long_position(self):
        pos = BrokerPosition(
            broker_position_id="100",
            symbol="EUR-USD",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=1.10,
            current_price=1.11,
            unrealized_pnl=1000.0,
        )
        assert pos.side == PositionSide.LONG
        assert pos.quantity == 1.0

    def test_quantity_must_be_positive(self):
        with pytest.raises(Exception):
            BrokerPosition(
                broker_position_id="x",
                symbol="x",
                side=PositionSide.LONG,
                quantity=-1.0,
                entry_price=1.0,
                current_price=1.0,
            )


class TestBrokerDeal:
    def test_close_deal(self):
        deal = BrokerDeal(
            deal_id="d1",
            broker_position_id="p1",
            broker_order_id="o1",
            symbol="EUR-USD",
            side=OrderSide.SELL,
            execution_price=1.1050,
            quantity=1.0,
            is_close=True,
            timestamp=datetime(2026, 3, 7, 12, 0, 0),
        )
        assert deal.is_close is True
        assert deal.execution_price == 1.1050


class TestBrokerAccountInfo:
    def test_defaults(self):
        info = BrokerAccountInfo(balance=10000.0)
        assert info.currency == "USD"
        assert info.equity is None
