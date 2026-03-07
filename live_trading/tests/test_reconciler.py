"""Tests for the Reconciler and close-reason detection."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from live_trading.brokers.middleware.reconciler import Reconciler, detect_close_reason
from live_trading.brokers.middleware.types import (
    BrokerDeal,
    BrokerOrderResult,
    BrokerPosition,
    BrokerAccountInfo,
    CloseReason,
    OrderSide,
    OrderStatus,
    PositionSide,
    PositionStatus,
)


class TestDetectCloseReason:
    def test_stop_loss_exact(self):
        pos = MagicMock()
        pos.stop_loss = 1.10000
        pos.take_profit = 1.12000
        assert detect_close_reason(pos, 1.10000) == CloseReason.STOP_LOSS

    def test_stop_loss_within_tolerance(self):
        pos = MagicMock()
        pos.stop_loss = 1.10000
        pos.take_profit = 1.12000
        assert detect_close_reason(pos, 1.10010) == CloseReason.STOP_LOSS

    def test_take_profit_exact(self):
        pos = MagicMock()
        pos.stop_loss = 1.10000
        pos.take_profit = 1.12000
        assert detect_close_reason(pos, 1.12000) == CloseReason.TAKE_PROFIT

    def test_take_profit_within_tolerance(self):
        pos = MagicMock()
        pos.stop_loss = 1.10000
        pos.take_profit = 1.12000
        assert detect_close_reason(pos, 1.11990) == CloseReason.TAKE_PROFIT

    def test_unknown_when_neither(self):
        pos = MagicMock()
        pos.stop_loss = 1.10000
        pos.take_profit = 1.12000
        assert detect_close_reason(pos, 1.11000) == CloseReason.UNKNOWN

    def test_unknown_when_no_sl_tp(self):
        pos = MagicMock()
        pos.stop_loss = None
        pos.take_profit = None
        assert detect_close_reason(pos, 1.11000) == CloseReason.UNKNOWN

    def test_custom_tolerance(self):
        pos = MagicMock()
        pos.stop_loss = 1.10000
        pos.take_profit = None
        assert detect_close_reason(pos, 1.10030, tolerance=0.0005) == CloseReason.STOP_LOSS
        assert detect_close_reason(pos, 1.10030, tolerance=0.0001) == CloseReason.UNKNOWN


class TestReconcilerSyncPositions:
    @pytest.fixture
    def adapter(self):
        adapter = AsyncMock()
        adapter.get_account_mode.return_value = "HEDGING"
        return adapter

    @pytest.fixture
    def reconciler(self, adapter):
        return Reconciler(adapter)

    @pytest.mark.asyncio
    async def test_updates_existing_position(self, reconciler, adapter):
        """When broker reports a position we already have, update its fields."""
        adapter.get_positions.return_value = [
            BrokerPosition(
                broker_position_id="111",
                symbol="EUR-USD",
                side=PositionSide.LONG,
                quantity=1.0,
                entry_price=1.10,
                current_price=1.12,
                unrealized_pnl=200.0,
                unrealized_pnl_pct=1.82,
                stop_loss=1.09,
                take_profit=1.13,
            )
        ]
        adapter.get_open_orders.return_value = []
        adapter.get_account_info.return_value = BrokerAccountInfo(balance=10000.0)

        db_pos = MagicMock()
        db_pos.broker_position_id = "111"
        db_pos.operation_id = ObjectId()
        db_pos.status = PositionStatus.OPEN
        db_pos.save = AsyncMock()

        operation = MagicMock()
        operation.id = db_pos.operation_id
        operation.asset = "EUR-USD"
        operation.save = AsyncMock()

        with patch("live_trading.brokers.middleware.reconciler.Position") as MockPosition, \
             patch("live_trading.brokers.middleware.reconciler.Order") as MockOrder:
            MockPosition.find.return_value.to_list = AsyncMock(return_value=[db_pos])
            MockOrder.find.return_value.to_list = AsyncMock(return_value=[])

            await reconciler.reconcile(operation)

        assert db_pos.current_price == 1.12
        assert db_pos.unrealized_pnl == 200.0
        assert db_pos.stop_loss == 1.09
        db_pos.save.assert_awaited()

    @pytest.mark.asyncio
    async def test_creates_missing_position(self, reconciler, adapter):
        """When broker has a position we don't, create it."""
        adapter.get_positions.return_value = [
            BrokerPosition(
                broker_position_id="222",
                symbol="GBP-USD",
                side=PositionSide.SHORT,
                quantity=0.5,
                entry_price=1.25,
                current_price=1.24,
            )
        ]
        adapter.get_open_orders.return_value = []
        adapter.get_account_info.return_value = BrokerAccountInfo(balance=10000.0)

        operation = MagicMock()
        operation.id = ObjectId()
        operation.asset = "GBP-USD"
        operation.save = AsyncMock()

        with patch("live_trading.brokers.middleware.reconciler.Position") as MockPosition, \
             patch("live_trading.brokers.middleware.reconciler.Order") as MockOrder:
            MockPosition.find.return_value.to_list = AsyncMock(return_value=[])
            MockPosition.return_value.insert = AsyncMock()
            MockOrder.find.return_value.to_list = AsyncMock(return_value=[])

            await reconciler.reconcile(operation)

        MockPosition.return_value.insert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_removed_position(self, reconciler, adapter):
        """When DB has a position the broker doesn't, close it."""
        adapter.get_positions.return_value = []
        adapter.get_open_orders.return_value = []
        adapter.get_deals.return_value = [
            BrokerDeal(
                deal_id="d1",
                broker_position_id="333",
                side=OrderSide.SELL,
                execution_price=1.1100,
                quantity=1.0,
                is_close=True,
                timestamp=datetime.utcnow(),
            )
        ]
        adapter.get_account_info.return_value = BrokerAccountInfo(balance=10000.0)

        db_pos = MagicMock()
        db_pos.broker_position_id = "333"
        db_pos.operation_id = ObjectId()
        db_pos.symbol = "EUR-USD"
        db_pos.side = PositionSide.LONG
        db_pos.quantity = 1.0
        db_pos.entry_price = 1.10
        db_pos.current_price = 1.11
        db_pos.stop_loss = 1.0900
        db_pos.take_profit = 1.1100
        db_pos.commission = 0.07
        db_pos.swap = 0.0
        db_pos.status = PositionStatus.OPEN
        db_pos.opened_at = datetime.utcnow() - timedelta(hours=2)
        db_pos.save = AsyncMock()

        operation = MagicMock()
        operation.id = db_pos.operation_id
        operation.asset = "EUR-USD"
        operation.save = AsyncMock()

        with patch("live_trading.brokers.middleware.reconciler.Position") as MockPosition, \
             patch("live_trading.brokers.middleware.reconciler.Order") as MockOrder:
            MockPosition.find.return_value.to_list = AsyncMock(return_value=[db_pos])
            MockOrder.find.return_value.to_list = AsyncMock(return_value=[])
            MockOrder.return_value.insert = AsyncMock()

            await reconciler.reconcile(operation)

        assert db_pos.status == PositionStatus.CLOSED
        assert db_pos.close_price == 1.1100
        assert db_pos.close_reason == CloseReason.TAKE_PROFIT
        assert db_pos.realized_pnl is not None
        db_pos.save.assert_awaited()


class TestReconcilerFixStuckOrders:
    @pytest.fixture
    def adapter(self):
        adapter = AsyncMock()
        return adapter

    @pytest.fixture
    def reconciler(self, adapter):
        return Reconciler(adapter)

    @pytest.mark.asyncio
    async def test_marks_stuck_order_as_cancelled(self, reconciler, adapter):
        """Order not on broker and no matching position → CANCELLED."""
        adapter.get_positions.return_value = []
        adapter.get_open_orders.return_value = []
        adapter.get_account_info.return_value = BrokerAccountInfo(balance=10000.0)

        stuck_order = MagicMock()
        stuck_order.broker_order_id = "ord_999"
        stuck_order.broker_position_id = None
        stuck_order.status = OrderStatus.SUBMITTED
        stuck_order.cancelled_at = None
        stuck_order.save = AsyncMock()

        operation = MagicMock()
        operation.id = ObjectId()
        operation.asset = "EUR-USD"
        operation.save = AsyncMock()

        with patch("live_trading.brokers.middleware.reconciler.Position") as MockPosition, \
             patch("live_trading.brokers.middleware.reconciler.Order") as MockOrder:
            MockPosition.find.return_value.to_list = AsyncMock(return_value=[])
            MockOrder.find.return_value.to_list = AsyncMock(return_value=[stuck_order])

            await reconciler.reconcile(operation)

        assert stuck_order.status == OrderStatus.CANCELLED
        stuck_order.save.assert_awaited()
