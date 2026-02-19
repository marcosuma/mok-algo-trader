"""
Tests for OrderManager order status callback logic.

Verifies that the callback uses the stable order _id (not broker_order_id)
to find and update orders, avoiding the race condition where the broker fires
the callback before broker_order_id is persisted to the database.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from bson import ObjectId


@pytest.fixture
def order_manager():
    """Create an OrderManager with mocked broker and journal."""
    from live_trading.orders.order_manager import OrderManager

    broker = AsyncMock()
    journal = AsyncMock()
    return OrderManager(broker=broker, journal_manager=journal)


class TestOrderStatusCallbackUsesStableId:
    """
    Regression test for the race condition where update_order() tried to
    find the order by broker_order_id before it was saved to the DB.
    The fix captures order._id at callback-creation time instead.
    """

    @pytest.mark.asyncio
    async def test_callback_finds_order_by_id_not_broker_order_id(self, order_manager):
        """
        Simulate the race: broker fires the callback before broker_order_id
        is written to the DB. The callback must still find and update the order.
        """
        fake_order_id = ObjectId()
        fake_broker_order_id = "99999"

        # Mock Order model
        mock_order = MagicMock()
        mock_order.id = fake_order_id
        mock_order.broker_order_id = None  # Not saved yet!
        mock_order.status = "PENDING"
        mock_order.filled_quantity = 0.0
        mock_order.avg_fill_price = None
        mock_order.filled_at = None
        mock_order.save = AsyncMock()

        mock_operation = MagicMock()
        mock_operation.asset = "EUR-USD"
        mock_operation.stop_loss_type = "ATR"
        mock_operation.stop_loss_value = 1.5
        mock_operation.take_profit_type = "RISK_REWARD"
        mock_operation.take_profit_value = 2.0
        mock_operation.current_capital = 10000.0

        with patch("live_trading.orders.order_manager.Order") as MockOrder, \
             patch("live_trading.orders.order_manager.TradingOperation") as MockOp, \
             patch("live_trading.orders.order_manager.MarketData") as MockMD:

            MockOp.get = AsyncMock(return_value=mock_operation)

            # Order.insert() returns immediately, Order.get() returns our mock
            mock_order_instance = MagicMock()
            mock_order_instance.id = fake_order_id
            mock_order_instance.insert = AsyncMock()
            mock_order_instance.save = AsyncMock()
            MockOrder.return_value = mock_order_instance
            MockOrder.get = AsyncMock(return_value=mock_order)

            # Broker returns the order_id but fires callback BEFORE returning
            captured_callback = None

            async def fake_place_order(**kwargs):
                nonlocal captured_callback
                captured_callback = kwargs.get("order_status_callback")
                # Simulate broker calling back immediately (before we return)
                if captured_callback:
                    captured_callback({
                        "order_id": fake_broker_order_id,
                        "status": "FILLED",
                        "filled": 1.0,
                        "avg_fill_price": 1.1050,
                    })
                return fake_broker_order_id

            order_manager.broker.place_order = fake_place_order
            order_manager.on_order_filled = AsyncMock()

            # Mock _get_current_price and _get_atr_value
            order_manager._get_current_price = AsyncMock(return_value=1.1050)
            order_manager._get_atr_value = AsyncMock(return_value=0.001)

            # Set up mock for MarketData find chain
            mock_md_chain = MagicMock()
            mock_md_chain.sort = MagicMock(return_value=mock_md_chain)
            mock_md_chain.limit = MagicMock(return_value=mock_md_chain)
            mock_md_chain.to_list = AsyncMock(return_value=[])
            MockMD.find = MagicMock(return_value=mock_md_chain)

            # Place the order
            result = await order_manager.place_order(
                operation_id=ObjectId(),
                asset="EUR-USD",
                signal_type="BUY",
            )

            # Let the scheduled update_order() coroutine run
            await asyncio.sleep(0.1)

            # The callback should have used Order.get(order_db_id), NOT
            # Order.find_one(broker_order_id=...). Verify Order.get was called.
            MockOrder.get.assert_called()
            call_args = MockOrder.get.call_args
            assert call_args[0][0] == fake_order_id, (
                f"Expected Order.get({fake_order_id}), got Order.get({call_args}). "
                "Callback is not using the stable order _id."
            )

    @pytest.mark.asyncio
    async def test_callback_backfills_broker_order_id(self, order_manager):
        """
        Verify that the callback sets broker_order_id on the order when it
        wasn't saved yet (backfill from callback payload).
        """
        fake_order_id = ObjectId()
        fake_broker_order_id = "12345"

        mock_order = MagicMock()
        mock_order.id = fake_order_id
        mock_order.broker_order_id = None  # Not saved yet
        mock_order.status = "PENDING"
        mock_order.filled_quantity = 0.0
        mock_order.avg_fill_price = None
        mock_order.filled_at = None
        mock_order.save = AsyncMock()

        mock_operation = MagicMock()
        mock_operation.asset = "EUR-USD"
        mock_operation.stop_loss_type = "ATR"
        mock_operation.stop_loss_value = 1.5
        mock_operation.take_profit_type = "RISK_REWARD"
        mock_operation.take_profit_value = 2.0
        mock_operation.current_capital = 10000.0

        with patch("live_trading.orders.order_manager.Order") as MockOrder, \
             patch("live_trading.orders.order_manager.TradingOperation") as MockOp, \
             patch("live_trading.orders.order_manager.MarketData") as MockMD:

            MockOp.get = AsyncMock(return_value=mock_operation)

            mock_order_instance = MagicMock()
            mock_order_instance.id = fake_order_id
            mock_order_instance.insert = AsyncMock()
            mock_order_instance.save = AsyncMock()
            MockOrder.return_value = mock_order_instance
            MockOrder.get = AsyncMock(return_value=mock_order)

            captured_callback = None

            async def fake_place_order(**kwargs):
                nonlocal captured_callback
                captured_callback = kwargs.get("order_status_callback")
                if captured_callback:
                    captured_callback({
                        "order_id": fake_broker_order_id,
                        "status": "ACCEPTED",
                        "filled": 0.0,
                        "avg_fill_price": 0.0,
                    })
                return fake_broker_order_id

            order_manager.broker.place_order = fake_place_order
            order_manager._get_current_price = AsyncMock(return_value=1.1050)
            order_manager._get_atr_value = AsyncMock(return_value=0.001)

            mock_md_chain = MagicMock()
            mock_md_chain.sort = MagicMock(return_value=mock_md_chain)
            mock_md_chain.limit = MagicMock(return_value=mock_md_chain)
            mock_md_chain.to_list = AsyncMock(return_value=[])
            MockMD.find = MagicMock(return_value=mock_md_chain)

            await order_manager.place_order(
                operation_id=ObjectId(),
                asset="EUR-USD",
                signal_type="BUY",
            )

            await asyncio.sleep(0.1)

            # The callback should have set broker_order_id on the order
            assert mock_order.broker_order_id == fake_broker_order_id, (
                "Callback should backfill broker_order_id from the callback payload"
            )
