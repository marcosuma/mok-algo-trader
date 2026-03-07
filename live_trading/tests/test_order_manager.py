"""
Tests for the redesigned OrderManager.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from bson import ObjectId

from live_trading.brokers.middleware.types import (
    AccountMode,
    BrokerOrderResult,
    BrokerOrderUpdate,
    BrokerPosition,
    CloseReason,
    OrderIntent,
    OrderSide,
    OrderSource,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionStatus,
)
from live_trading.orders.order_manager import OrderManager, _detect_close_reason


@pytest.fixture
def adapter():
    adapter = AsyncMock()
    adapter.get_account_mode = MagicMock(return_value=AccountMode.HEDGING)
    adapter.register_execution_callback = MagicMock()
    return adapter


@pytest.fixture
def order_manager(adapter):
    journal = AsyncMock()
    return OrderManager(adapter=adapter, journal_manager=journal)


class TestNettingModeBlocked:
    def test_raises_on_netting(self):
        adapter = MagicMock()
        adapter.get_account_mode.return_value = AccountMode.NETTING
        adapter.register_execution_callback = MagicMock()
        journal = AsyncMock()

        with pytest.raises(NotImplementedError, match="Netting"):
            OrderManager(adapter=adapter, journal_manager=journal)


class TestDetectCloseReason:
    def test_stop_loss(self):
        pos = MagicMock(stop_loss=1.10, take_profit=1.12)
        assert _detect_close_reason(pos, 1.10005) == CloseReason.STOP_LOSS

    def test_take_profit(self):
        pos = MagicMock(stop_loss=1.10, take_profit=1.12)
        assert _detect_close_reason(pos, 1.12010) == CloseReason.TAKE_PROFIT

    def test_unknown(self):
        pos = MagicMock(stop_loss=1.10, take_profit=1.12)
        assert _detect_close_reason(pos, 1.11) == CloseReason.UNKNOWN

    def test_no_sl_tp(self):
        pos = MagicMock(stop_loss=None, take_profit=None)
        assert _detect_close_reason(pos, 1.11) == CloseReason.UNKNOWN


class TestCalculateDefaultQuantity:
    @pytest.mark.asyncio
    async def test_returns_lots_not_units(self, order_manager):
        operation = MagicMock()
        operation.current_capital = 100_000.0

        lots = await order_manager._calculate_default_quantity(
            operation=operation,
            entry_price=1.17712,
            stop_loss=1.17680,
        )
        assert 25.0 < lots < 40.0, f"Expected ~31 lots, got {lots}"

    @pytest.mark.asyncio
    async def test_small_account(self, order_manager):
        operation = MagicMock()
        operation.current_capital = 1_000.0

        lots = await order_manager._calculate_default_quantity(
            operation=operation,
            entry_price=1.17712,
            stop_loss=1.17680,
        )
        assert 0.1 < lots < 1.0, f"Expected ~0.3 lots, got {lots}"

    @pytest.mark.asyncio
    async def test_fallback_without_stop_loss(self, order_manager):
        operation = MagicMock()
        operation.current_capital = 100_000.0

        lots = await order_manager._calculate_default_quantity(
            operation=operation,
            entry_price=1.17712,
            stop_loss=None,
        )
        assert lots > 0.0
        assert lots < 100, f"Capital-based fallback should be reasonable, got {lots}"

    @pytest.mark.asyncio
    async def test_minimum_fallback(self, order_manager):
        operation = MagicMock()
        operation.current_capital = 100.0

        lots = await order_manager._calculate_default_quantity(
            operation=operation,
            entry_price=0.0,
            stop_loss=None,
        )
        assert lots == 0.01


class TestPlaceOrder:
    @pytest.mark.asyncio
    async def test_successful_market_order(self, order_manager, adapter):
        """A filled market order should create an Order and a Position."""
        operation = MagicMock()
        operation.id = ObjectId()
        operation.asset = "EUR-USD"
        operation.stop_loss_type = "FIXED"
        operation.stop_loss_value = 0.003
        operation.take_profit_type = "RISK_REWARD"
        operation.take_profit_value = 2.0
        operation.primary_bar_size = "1 hour"
        operation.current_capital = 10000.0

        adapter.submit_order.return_value = BrokerOrderResult(
            broker_order_id="BO-100",
            status=OrderStatus.FILLED,
            filled_quantity=0.1,
            avg_fill_price=1.1050,
            broker_position_id="POS-200",
            commission=0.07,
        )
        adapter.get_positions.return_value = []

        with patch("live_trading.orders.order_manager.TradingOperation") as MockOp, \
             patch("live_trading.orders.order_manager.Order") as MockOrder, \
             patch("live_trading.orders.order_manager.Position") as MockPos, \
             patch("live_trading.orders.order_manager.MarketData"):
            MockOp.get = AsyncMock(return_value=operation)
            mock_order_instance = MagicMock()
            mock_order_instance.id = ObjectId()
            mock_order_instance.insert = AsyncMock()
            mock_order_instance.save = AsyncMock()
            mock_order_instance.status = OrderStatus.PENDING_SUBMIT
            mock_order_instance.side = OrderSide.BUY
            mock_order_instance.intent = OrderIntent.OPEN
            mock_order_instance.broker_position_id = "POS-200"
            mock_order_instance.avg_fill_price = 1.1050
            mock_order_instance.filled_quantity = 0.1
            mock_order_instance.commission = 0.07
            mock_order_instance.stop_loss = 1.1020
            mock_order_instance.take_profit = 1.1110
            mock_order_instance.symbol = "EUR-USD"
            mock_order_instance.operation_id = operation.id
            MockOrder.return_value = mock_order_instance

            mock_pos_instance = MagicMock()
            mock_pos_instance.insert = AsyncMock()
            MockPos.return_value = mock_pos_instance

            result = await order_manager.place_order(
                operation_id=operation.id,
                asset="EUR-USD",
                signal_type="BUY",
                quantity=0.1,
                stop_loss=1.1020,
                take_profit=1.1110,
            )

        adapter.submit_order.assert_awaited_once()
        mock_order_instance.insert.assert_awaited_once()
        mock_pos_instance.insert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejected_order(self, order_manager, adapter):
        """A rejected order should set status to REJECTED and not create a position."""
        operation = MagicMock()
        operation.id = ObjectId()
        operation.current_capital = 10000.0
        operation.primary_bar_size = "1 hour"

        adapter.submit_order.return_value = BrokerOrderResult(
            broker_order_id="",
            status=OrderStatus.REJECTED,
        )
        adapter.get_positions.return_value = []

        with patch("live_trading.orders.order_manager.TradingOperation") as MockOp, \
             patch("live_trading.orders.order_manager.Order") as MockOrder, \
             patch("live_trading.orders.order_manager.Position") as MockPos, \
             patch("live_trading.orders.order_manager.MarketData"):
            MockOp.get = AsyncMock(return_value=operation)
            mock_order_instance = MagicMock()
            mock_order_instance.id = ObjectId()
            mock_order_instance.insert = AsyncMock()
            mock_order_instance.save = AsyncMock()
            mock_order_instance.status = OrderStatus.PENDING_SUBMIT
            mock_order_instance.intent = OrderIntent.OPEN
            MockOrder.return_value = mock_order_instance

            result = await order_manager.place_order(
                operation_id=operation.id,
                asset="EUR-USD",
                signal_type="BUY",
                quantity=0.1,
                stop_loss=1.10,
                take_profit=1.12,
            )

        assert mock_order_instance.status == OrderStatus.REJECTED


class TestHandleBrokerPositionClose:
    @pytest.mark.asyncio
    async def test_closes_position_on_sl(self, order_manager):
        """When broker fires a close event near SL, position should be closed with STOP_LOSS."""
        position = MagicMock()
        position.id = ObjectId()
        position.operation_id = ObjectId()
        position.broker_position_id = "POS-300"
        position.symbol = "EUR-USD"
        position.side = PositionSide.LONG
        position.quantity = 0.5
        position.entry_price = 1.1100
        position.current_price = 1.1050
        position.stop_loss = 1.1050
        position.take_profit = 1.1200
        position.commission = 0.07
        position.swap = 0.0
        position.status = PositionStatus.OPEN
        position.opened_at = datetime(2026, 3, 7, 10, 0, 0)
        position.save = AsyncMock()

        update = BrokerOrderUpdate(
            broker_order_id="CLOSE-1",
            status=OrderStatus.FILLED,
            filled_quantity=0.5,
            avg_fill_price=1.1050,
            broker_position_id="POS-300",
            is_position_close=True,
        )

        with patch("live_trading.orders.order_manager.Position") as MockPos, \
             patch("live_trading.orders.order_manager.Order") as MockOrder:
            MockPos.find_one = AsyncMock(return_value=position)
            mock_close_order = MagicMock()
            mock_close_order.id = ObjectId()
            mock_close_order.insert = AsyncMock()
            MockOrder.return_value = mock_close_order

            await order_manager._handle_broker_position_close(update)

        assert position.status == PositionStatus.CLOSED
        assert position.close_price == 1.1050
        assert position.close_reason == CloseReason.STOP_LOSS
        assert position.realized_pnl is not None
        position.save.assert_awaited()
