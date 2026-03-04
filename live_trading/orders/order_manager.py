"""
Order Manager - Handles order placement, position management, and P/L calculation.
"""
import asyncio
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

from live_trading.models.order import Order
from live_trading.models.transaction import Transaction
from live_trading.models.position import Position
from live_trading.models.trade import Trade
from live_trading.models.trading_operation import TradingOperation
from live_trading.models.market_data import MarketData
from live_trading.brokers.base_broker import BaseBroker
from live_trading.journal.journal_manager import JournalManager
from forex_strategies.risk_management import (
    calculate_stop_loss as _calc_sl,
    calculate_take_profit as _calc_tp,
)

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages orders, positions, and trades"""

    def __init__(
        self,
        broker: BaseBroker,
        journal_manager: JournalManager
    ):
        self.broker = broker
        self.journal = journal_manager

    async def place_order(
        self,
        operation_id: ObjectId,
        asset: str,
        signal_type: str,  # 'BUY' or 'SELL'
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Order:
        """
        Place an order based on strategy signal.

        Args:
            operation_id: Trading operation ID
            asset: Asset symbol
            signal_type: 'BUY' or 'SELL'
            price: Entry price (for limit orders) or None (for market orders)
            quantity: Order quantity (if None, uses default from operation)
            stop_loss: Stop loss price (if None, calculated from operation config)
            take_profit: Take profit price (if None, calculated from operation config)

        Returns:
            Created Order document
        """
        # Build operation tag for log messages
        short_id = str(operation_id)[-6:]
        op_tag = f"[OP:{short_id} {asset}]"

        logger.info(f"[ORDER] {op_tag} Preparing order: {signal_type} @ {price or 'MARKET'}")

        # Get operation
        operation = await TradingOperation.get(operation_id)
        if not operation:
            logger.error(f"[ORDER] {op_tag} FAILED - Operation not found")
            raise ValueError(f"Operation {operation_id} not found")

        # Determine order type
        order_type = "LIMIT" if price else "MARKET"
        logger.info(f"[ORDER] {op_tag} Order type: {order_type}")

        # Calculate stop loss and take profit if not provided
        if stop_loss is None or take_profit is None:
            # Get current market price if needed
            entry_price = price
            if entry_price is None:
                # Get current market price from broker positions or latest market data
                entry_price = await self._get_current_price(operation_id, asset)
                if entry_price is None or entry_price == 0.0:
                    logger.warning(f"Could not get current price for {asset}, using 0.0 as fallback")
                    entry_price = 0.0

            # Determine position type
            position_type = "LONG" if signal_type == "BUY" else "SHORT"

            if stop_loss is None:
                stop_loss = await self.calculate_stop_loss(
                    operation_id,
                    entry_price,
                    position_type
                )

            if take_profit is None:
                take_profit = await self.calculate_take_profit(
                    operation_id,
                    entry_price,
                    stop_loss,
                    position_type
                )

        # Create order record first (before placing with broker)
        order = Order(
            operation_id=operation_id,
            broker_order_id=None,  # Will be set after broker placement
            order_type=order_type,
            action=signal_type,
            quantity=quantity or 1.0,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="PENDING"
        )
        await order.insert()

        # Create order status callback to update database
        # Get the current event loop to schedule async updates from IBKR thread
        import asyncio
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        # Capture the order's DB _id so the callback can always find it,
        # even before broker_order_id is saved (avoids a race condition where
        # the broker fires the callback before we write broker_order_id to DB).
        order_db_id = order.id

        def order_status_callback(status_data: Dict[str, Any]):
            """Callback to update order status in database when broker sends updates"""
            from datetime import datetime

            # Map broker status to our status
            broker_status = status_data.get("status", "").upper()
            status_map = {
                "SUBMITTED": "PENDING",
                "PRESUBMITTED": "PENDING",
                "PENDINGSUBMIT": "PENDING",
                "PENDINGCANCEL": "PENDING",
                "PRECANCELED": "CANCELLED",
                "CANCELLED": "CANCELLED",
                "FILLED": "FILLED",
                "PARTIALLYFILLED": "PARTIALLY_FILLED",
                "INACTIVE": "REJECTED",
                "REJECTED": "REJECTED",
                # cTrader specific statuses
                "ACCEPTED": "PENDING",
                "EXECUTED": "FILLED",
                "EXPIRED": "CANCELLED",
            }

            new_status = status_map.get(broker_status, "PENDING")
            filled = status_data.get("filled", 0.0)
            avg_fill_price = status_data.get("avg_fill_price", 0.0)

            # Log status change with clear emoji indicators
            status_emoji = {
                "PENDING": "⏳",
                "PARTIALLY_FILLED": "🔄",
                "FILLED": "✅",
                "CANCELLED": "🚫",
                "REJECTED": "❌",
            }
            emoji = status_emoji.get(new_status, "❓")
            logger.info(f"[ORDER] {op_tag} {emoji} Status: {broker_status} -> {new_status} (filled: {filled}, price: {avg_fill_price})")

            # Schedule async database update
            async def update_order():
                try:
                    # Look up by stable _id — broker_order_id may not be saved
                    # to the DB yet when fast execution events arrive.
                    order_to_update = await Order.get(order_db_id)
                    if not order_to_update:
                        logger.error(
                            f"[ORDER] {op_tag} Order {order_db_id} not found in DB "
                            f"during status callback ({broker_status} -> {new_status})"
                        )
                        return

                    # Also backfill broker_order_id from the callback payload
                    # in case the main flow hasn't saved it yet.
                    callback_order_id = status_data.get("order_id")
                    if callback_order_id and not order_to_update.broker_order_id:
                        order_to_update.broker_order_id = str(callback_order_id)

                    old_status = order_to_update.status
                    order_to_update.status = new_status
                    order_to_update.filled_quantity = filled
                    if avg_fill_price > 0:
                        order_to_update.avg_fill_price = avg_fill_price
                    if new_status == "FILLED" and not order_to_update.filled_at:
                        order_to_update.filled_at = datetime.utcnow()
                    await order_to_update.save()

                    logger.info(f"[ORDER] {op_tag} DB updated: order {order_to_update.id} ({old_status} -> {new_status})")

                    # If order is filled, trigger on_order_filled
                    if new_status == "FILLED":
                        logger.info(f"[ORDER] {op_tag} FILLED! {signal_type} @ {avg_fill_price} (qty: {filled})")
                        await self.on_order_filled(
                            order_to_update.id,
                            {
                                "filled": filled,
                                "avg_fill_price": avg_fill_price,
                                # Pass broker_position_id so on_order_filled can store it on Position.
                                "broker_position_id": status_data.get("broker_position_id"),
                            }
                        )
                        # Register a callback so broker-initiated TP/SL closes are tracked.
                        bp_id = status_data.get("broker_position_id")
                        if bp_id and hasattr(self.broker, "register_position_close_callback"):
                            self.broker.register_position_close_callback(
                                bp_id,
                                self._make_position_close_callback(
                                    order_to_update.operation_id, bp_id, event_loop
                                )
                            )
                            logger.info(f"[ORDER] {op_tag} Registered position-close callback for broker_position_id={bp_id}")
                    elif new_status == "REJECTED":
                        logger.error(f"[ORDER] {op_tag} REJECTED! {signal_type} - Reason: {status_data.get('reason', 'unknown')}")
                    elif new_status == "CANCELLED":
                        logger.warning(f"[ORDER] {op_tag} CANCELLED: {signal_type}")
                except Exception as e:
                    logger.error(f"[ORDER] {op_tag} Error updating order status in callback: {e}", exc_info=True)

            # Schedule async update in the event loop (callback runs in broker thread)
            if event_loop:
                asyncio.run_coroutine_threadsafe(update_order(), event_loop)
            else:
                logger.warning(f"[ORDER] {op_tag} No event loop available to update order status")

        # Calculate default quantity if not provided
        if quantity is None:
            quantity = await self._calculate_default_quantity(operation, entry_price, stop_loss)

        # Place order with broker and register callback
        logger.info(f"[ORDER] {op_tag} Sending to broker: {signal_type} {quantity:.2f} @ {price or 'MARKET'} (SL: {stop_loss}, TP: {take_profit})")

        broker_order_id = await self.broker.place_order(
            asset=asset,
            action=signal_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_status_callback=order_status_callback
        )

        # Check if order was accepted by broker
        if not broker_order_id:
            order.status = "REJECTED"
            await order.save()
            logger.error(f"[ORDER] {op_tag} BROKER REJECTED - No order ID returned for {signal_type}")
            return order

        # Update order with broker_order_id
        order.broker_order_id = broker_order_id
        await order.save()

        logger.info(f"[ORDER] {op_tag} SUBMITTED (order_id: {broker_order_id}) - Awaiting execution...")

        # Log to journal
        await self.journal.log_action(
            action_type="ORDER_PLACED",
            action_data={
                "order_id": str(order.id),
                "broker_order_id": broker_order_id,
                "asset": asset,
                "action": signal_type,
                "quantity": quantity,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            },
            operation_id=operation_id
        )

        logger.info(f"[ORDER] {op_tag} Summary: {signal_type} {quantity:.2f} @ {price or 'MARKET'} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        return order

    async def calculate_stop_loss(
        self,
        operation_id: ObjectId,
        entry_price: float,
        position_type: str
    ) -> float:
        """
        Calculate stop loss price based on operation configuration.

        Delegates the pure calculation to ``forex_strategies.risk_management``
        so backtesting and live trading share identical logic.
        """
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        atr_value = None
        if operation.stop_loss_type == "ATR":
            atr_value = await self._get_atr_value(operation_id, operation.primary_bar_size)
            if atr_value is None or atr_value == 0.0:
                logger.warning(f"Could not get ATR value for operation {operation_id}, using default 0.001")
                atr_value = 0.001

        result = _calc_sl(
            entry_price=entry_price,
            position_type=position_type,
            sl_type=operation.stop_loss_type,
            sl_value=operation.stop_loss_value,
            atr_value=atr_value,
        )
        return result if result is not None else entry_price

    async def calculate_take_profit(
        self,
        operation_id: ObjectId,
        entry_price: float,
        stop_loss_price: float,
        position_type: str
    ) -> float:
        """
        Calculate take profit price based on operation configuration.

        Delegates the pure calculation to ``forex_strategies.risk_management``
        so backtesting and live trading share identical logic.
        """
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        atr_value = None
        if operation.take_profit_type == "ATR":
            atr_value = await self._get_atr_value(operation_id, operation.primary_bar_size)
            if atr_value is None or atr_value == 0.0:
                logger.warning(f"Could not get ATR value for operation {operation_id}, using default 0.001")
                atr_value = 0.001

        result = _calc_tp(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            position_type=position_type,
            tp_type=operation.take_profit_type,
            tp_value=operation.take_profit_value,
            atr_value=atr_value,
        )
        return result if result is not None else entry_price

    async def on_order_filled(
        self,
        order_id: ObjectId,
        fill_data: Dict[str, Any]
    ):
        """
        Handle order fill callback.

        - Create transaction record (BUY or SELL, ENTRY or EXIT)
        - Calculate profit for EXIT transactions
        - Link EXIT to corresponding ENTRY transaction
        - Create trade record if position is closed
        """
        order = await Order.get(order_id)
        if not order:
            logger.error(f"[ORDER] Order {order_id} not found")
            return

        # Build operation tag for log messages
        operation_for_tag = await TradingOperation.get(order.operation_id)
        asset_label = operation_for_tag.asset if operation_for_tag else "?"
        short_id = str(order.operation_id)[-6:]
        op_tag = f"[OP:{short_id} {asset_label}]"

        # Update order
        order.status = "FILLED"
        order.filled_quantity = fill_data.get("filled", order.quantity)
        order.avg_fill_price = fill_data.get("avg_fill_price", order.price)
        order.filled_at = datetime.utcnow()
        await order.save()

        logger.info(f"[ORDER] {op_tag} Processing fill: {order.action} qty={order.filled_quantity} @ {order.avg_fill_price}")

        # Determine if this is ENTRY, EXIT, or SCALE IN
        # Check if there's an open position for this operation
        open_position = await Position.find_one(
            Position.operation_id == order.operation_id,
            Position.closed_at == None
        )

        if open_position:
            position_type = "LONG" if open_position.quantity > 0 else "SHORT"
            position_quantity = open_position.quantity

            # Determine if order direction matches position direction
            # LONG position: BUY = scale in, SELL = scale out/close
            # SHORT position: SELL = scale in, BUY = scale out/close
            is_scale_in = (
                (position_type == "LONG" and order.action == "BUY") or
                (position_type == "SHORT" and order.action == "SELL")
            )

            if is_scale_in:
                # SCALE IN: Add to existing position
                transaction_role = "ENTRY"

                # Create ENTRY transaction (for scaling in)
                entry_transaction = await self.create_transaction(
                    operation_id=order.operation_id,
                    order_id=order_id,
                    transaction_type=order.action,
                    transaction_role=transaction_role,
                    position_type=position_type,
                    price=order.avg_fill_price,
                    quantity=order.filled_quantity,
                    commission=order.commission
                )

                # Calculate new average entry price (weighted average)
                # Total cost = (old_quantity * old_price) + (new_quantity * new_price)
                # New average = Total cost / Total quantity
                old_total_cost = abs(position_quantity) * open_position.entry_price
                new_total_cost = order.filled_quantity * order.avg_fill_price
                new_total_quantity = abs(position_quantity) + order.filled_quantity
                new_avg_entry_price = (old_total_cost + new_total_cost) / new_total_quantity

                # Update position: add to quantity and recalculate average entry price
                if position_type == "LONG":
                    open_position.quantity += order.filled_quantity
                else:  # SHORT
                    open_position.quantity -= order.filled_quantity  # quantity is negative, so subtract

                open_position.entry_price = new_avg_entry_price
                open_position.current_price = order.avg_fill_price  # Update current price
                await open_position.save()

                logger.info(f"[POSITION] {op_tag} Scaled in: {order.action} {order.filled_quantity} units. Position now: {open_position.quantity} @ {new_avg_entry_price:.5f}")

            else:
                # SCALE OUT / CLOSE: Reduce or close position
                transaction_role = "EXIT"
                order_quantity = order.filled_quantity
                position_abs_quantity = abs(position_quantity)

                # Determine how much to close (can't close more than position size)
                close_quantity = min(order_quantity, position_abs_quantity)

                # Get entry transactions for this position (FIFO - first in, first out)
                # Find all ENTRY transactions for this position that haven't been fully closed
                entry_transactions = await Transaction.find(
                    Transaction.operation_id == order.operation_id,
                    Transaction.transaction_role == "ENTRY",
                    Transaction.position_type == position_type
                ).sort(Transaction.executed_at).to_list()

                # Find transactions that haven't been fully matched with exits
                # For simplicity, we'll use FIFO: match oldest entries first
                remaining_to_close = close_quantity
                matched_entries = []

                for entry_tx in entry_transactions:
                    if remaining_to_close <= 0:
                        break

                    # Check how much of this entry has already been closed
                    # Find all EXIT transactions linked to this entry
                    exit_transactions = await Transaction.find(
                        Transaction.related_entry_transaction_id == entry_tx.id
                    ).to_list()
                    already_closed = sum(tx.quantity for tx in exit_transactions)
                    available = entry_tx.quantity - already_closed

                    if available > 0:
                        close_amount = min(remaining_to_close, available)
                        matched_entries.append({
                            "transaction": entry_tx,
                            "quantity": close_amount
                        })
                        remaining_to_close -= close_amount

                # If we couldn't match enough entries, use the position's entry price as fallback
                if remaining_to_close > 0:
                    logger.warning(f"Could not match all entry transactions, using position entry price for {remaining_to_close} units")
                    matched_entries.append({
                        "transaction": None,  # Will use position entry price
                        "quantity": remaining_to_close
                    })

                # Create trades for each matched entry (FIFO)
                # Each matched entry gets its own exit transaction for proper tracking
                total_pnl = 0.0
                for match in matched_entries:
                    entry_tx = match["transaction"]
                    match_quantity = match["quantity"]

                    if entry_tx:
                        # Create a separate exit transaction for this matched portion
                        partial_exit_transaction = await self.create_transaction(
                            operation_id=order.operation_id,
                            order_id=order_id,
                            transaction_type=order.action,
                            transaction_role="EXIT",
                            position_type=position_type,
                            price=order.avg_fill_price,
                            quantity=match_quantity,
                            commission=order.commission * (match_quantity / close_quantity),  # Proportional commission
                            related_entry_transaction_id=entry_tx.id
                        )

                        # Calculate P/L for this portion
                        if position_type == "LONG":
                            partial_pnl = (partial_exit_transaction.price - entry_tx.price) * match_quantity
                        else:  # SHORT
                            partial_pnl = (entry_tx.price - partial_exit_transaction.price) * match_quantity

                        partial_pnl_pct = (partial_pnl / (entry_tx.price * match_quantity)) * 100
                        partial_exit_transaction.profit = partial_pnl
                        partial_exit_transaction.profit_pct = partial_pnl_pct
                        await partial_exit_transaction.save()

                        total_pnl += partial_pnl

                        # Create trade for this matched portion
                        await self.create_trade(
                            operation_id=order.operation_id,
                            entry_transaction_id=entry_tx.id,
                            exit_transaction_id=partial_exit_transaction.id,
                            position_type=position_type,
                            quantity=match_quantity
                        )
                    else:
                        # Use position entry price as fallback
                        # Create a synthetic exit transaction for tracking
                        synthetic_exit_transaction = await self.create_transaction(
                            operation_id=order.operation_id,
                            order_id=order_id,
                            transaction_type=order.action,
                            transaction_role="EXIT",
                            position_type=position_type,
                            price=order.avg_fill_price,
                            quantity=match_quantity,
                            commission=order.commission * (match_quantity / close_quantity),
                            related_entry_transaction_id=None  # No matching entry transaction
                        )

                        if position_type == "LONG":
                            pnl = (synthetic_exit_transaction.price - open_position.entry_price) * match_quantity
                        else:  # SHORT
                            pnl = (open_position.entry_price - synthetic_exit_transaction.price) * match_quantity

                        pnl_pct = (pnl / (open_position.entry_price * match_quantity)) * 100
                        synthetic_exit_transaction.profit = pnl
                        synthetic_exit_transaction.profit_pct = pnl_pct
                        await synthetic_exit_transaction.save()

                        total_pnl += pnl

                        # Can't create trade without entry transaction - log warning
                        logger.warning(f"Could not match entry transaction for {match_quantity} units. Created exit transaction but no trade.")

                # Update or close position
                if close_quantity >= position_abs_quantity:
                    # Full close
                    open_position.closed_at = datetime.utcnow()
                    open_position.quantity = 0.0
                    logger.info(f"[POSITION] {op_tag} CLOSED: {position_type} {position_abs_quantity} units | P/L: {total_pnl:.2f}")
                else:
                    # Partial close - reduce quantity
                    if position_type == "LONG":
                        open_position.quantity -= close_quantity
                    else:  # SHORT
                        open_position.quantity += close_quantity  # quantity is negative, so add
                    open_position.current_price = order.avg_fill_price
                    logger.info(f"[POSITION] {op_tag} Partial close: {close_quantity} of {position_abs_quantity} units | Remaining: {open_position.quantity} | Partial P/L: {total_pnl:.2f}")

                await open_position.save()

        else:
            # This is an ENTRY transaction
            transaction_role = "ENTRY"
            position_type = "LONG" if order.action == "BUY" else "SHORT"

            # Create ENTRY transaction
            entry_transaction = await self.create_transaction(
                operation_id=order.operation_id,
                order_id=order_id,
                transaction_type=order.action,
                transaction_role=transaction_role,
                position_type=position_type,
                price=order.avg_fill_price,
                quantity=order.filled_quantity,
                commission=order.commission
            )

            # Fetch operation to get contract symbol
            operation = await TradingOperation.get(order.operation_id)
            if not operation:
                logger.error(f"Operation {order.operation_id} not found when creating position")
                return

            # Create position
            position = Position(
                operation_id=order.operation_id,
                contract_symbol=operation.asset,
                quantity=order.filled_quantity if position_type == "LONG" else -order.filled_quantity,
                entry_price=order.avg_fill_price,
                current_price=order.avg_fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                broker_position_id=fill_data.get("broker_position_id"),
            )
            await position.insert()

            logger.info(f"[POSITION] {op_tag} OPENED: {position_type} | Qty: {order.filled_quantity} @ {order.avg_fill_price:.5f} | SL: {order.stop_loss:.5f if order.stop_loss else 'N/A'} | TP: {order.take_profit:.5f if order.take_profit else 'N/A'}")

        # Log to journal
        await self.journal.log_action(
            action_type="ORDER_FILLED",
            action_data={
                "order_id": str(order_id),
                "filled_quantity": order.filled_quantity,
                "avg_fill_price": order.avg_fill_price
            },
            operation_id=order.operation_id
        )

    async def create_transaction(
        self,
        operation_id: ObjectId,
        order_id: ObjectId,
        transaction_type: str,
        transaction_role: str,
        position_type: str,
        price: float,
        quantity: float,
        commission: float,
        related_entry_transaction_id: Optional[ObjectId] = None
    ) -> Transaction:
        """Create a transaction record"""
        transaction = Transaction(
            operation_id=operation_id,
            order_id=order_id,
            transaction_type=transaction_type,
            transaction_role=transaction_role,
            position_type=position_type,
            price=price,
            quantity=quantity,
            commission=commission,
            related_entry_transaction_id=related_entry_transaction_id
        )
        await transaction.insert()
        return transaction

    async def create_trade(
        self,
        operation_id: ObjectId,
        entry_transaction_id: ObjectId,
        exit_transaction_id: ObjectId,
        position_type: str,
        quantity: Optional[float] = None
    ) -> Trade:
        """
        Create a trade record (completed round-trip).

        Args:
            operation_id: Trading operation ID
            entry_transaction_id: Entry transaction ID
            exit_transaction_id: Exit transaction ID
            position_type: Position type ('LONG' or 'SHORT')
            quantity: Trade quantity (if None, uses exit_transaction.quantity)
        """
        entry_transaction = await Transaction.get(entry_transaction_id)
        exit_transaction = await Transaction.get(exit_transaction_id)

        # Use provided quantity or exit transaction quantity (for partial closes)
        trade_quantity = quantity if quantity is not None else exit_transaction.quantity

        if position_type == "LONG":
            pnl = (exit_transaction.price - entry_transaction.price) * trade_quantity
        else:  # SHORT
            pnl = (entry_transaction.price - exit_transaction.price) * trade_quantity

        pnl_pct = (pnl / (entry_transaction.price * trade_quantity)) * 100

        # Calculate proportional commission for partial closes
        entry_commission = entry_transaction.commission * (trade_quantity / entry_transaction.quantity) if entry_transaction.quantity > 0 else entry_transaction.commission
        exit_commission = exit_transaction.commission
        total_commission = entry_commission + exit_commission

        duration_seconds = (exit_transaction.executed_at - entry_transaction.executed_at).total_seconds()

        trade = Trade(
            operation_id=operation_id,
            position_type=position_type,
            entry_transaction_id=entry_transaction_id,
            exit_transaction_id=exit_transaction_id,
            entry_price=entry_transaction.price,
            exit_price=exit_transaction.price,
            quantity=abs(trade_quantity),
            pnl=pnl,
            pnl_pct=pnl_pct,
            total_commission=total_commission,
            entry_time=entry_transaction.executed_at,
            exit_time=exit_transaction.executed_at,
            duration_seconds=duration_seconds
        )
        await trade.insert()
        return trade

    async def update_positions(self, operation_id: ObjectId):
        """Update position P/L"""
        positions = await Position.find(
            Position.operation_id == operation_id,
            Position.closed_at == None
        ).to_list()

        for position in positions:
            # Get current market price from broker positions or latest market data
            current_price = await self._get_current_price(operation_id, position.contract_symbol)
            if current_price is None or current_price == 0.0:
                # Fallback to stored price
                current_price = position.current_price
                logger.warning(f"Could not get current price for {position.contract_symbol}, using stored price {current_price}")

            # Calculate unrealized P/L
            if position.quantity > 0:  # LONG
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # SHORT
                unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)

            unrealized_pnl_pct = (unrealized_pnl / (position.entry_price * abs(position.quantity))) * 100

            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
            position.unrealized_pnl_pct = unrealized_pnl_pct
            await position.save()

    def _make_position_close_callback(
        self,
        operation_id: ObjectId,
        broker_position_id: str,
        event_loop,
    ) -> Callable:
        """Return a thread-safe callback that processes a broker-initiated position close."""
        def callback(close_data: Dict[str, Any]):
            close_price = close_data.get("avg_fill_price") or close_data.get("last_fill_price")
            filled_volume = close_data.get("filled", 0.0)
            logger.info(
                f"[ORDER] Position-close callback fired: op={operation_id}, "
                f"broker_position_id={broker_position_id}, price={close_price}, volume={filled_volume}"
            )
            if event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._process_external_close(operation_id, close_price, filled_volume),
                    event_loop,
                )
        return callback

    async def _process_external_close(
        self,
        operation_id: ObjectId,
        close_price: Optional[float],
        filled_volume: float = 0.0,
        close_time: Optional[datetime] = None,
    ):
        """Create close records for a position closed externally (TP/SL/manual from broker UI).

        Idempotent: skips if no open position is found.
        Creates a synthetic FILLED Order then delegates to on_order_filled() so that
        FIFO transaction matching, Trade record creation, and position close all happen
        via the same code path as a normal exit order.
        """
        try:
            open_position = await Position.find_one(
                Position.operation_id == operation_id,
                Position.closed_at == None
            )
            if not open_position:
                logger.info(
                    f"[ORDER] _process_external_close: no open position for op={operation_id} — skipping"
                )
                return

            position_type = "LONG" if open_position.quantity > 0 else "SHORT"
            close_action = "SELL" if position_type == "LONG" else "BUY"
            close_qty = abs(open_position.quantity)
            effective_price = close_price if close_price else open_position.current_price

            # Create a synthetic close order so on_order_filled can find it by ID
            close_order = Order(
                operation_id=operation_id,
                broker_order_id=None,
                order_type="MARKET",
                action=close_action,
                quantity=close_qty,
                price=effective_price,
                stop_loss=None,
                take_profit=None,
                status="FILLED",
                filled_quantity=close_qty,
                avg_fill_price=effective_price,
                filled_at=close_time or datetime.utcnow(),
            )
            await close_order.insert()

            logger.info(
                f"[ORDER] External close: {position_type} op={operation_id} "
                f"@ {effective_price:.5f} (qty={close_qty}) — processing via on_order_filled"
            )

            await self.on_order_filled(
                close_order.id,
                {"filled": close_qty, "avg_fill_price": effective_price},
            )

            await self.journal.log_action(
                action_type="EXTERNAL_POSITION_CLOSE",
                action_data={
                    "order_id": str(close_order.id),
                    "position_type": position_type,
                    "close_price": effective_price,
                    "quantity": close_qty,
                },
                operation_id=operation_id,
            )
        except Exception as e:
            logger.error(
                f"[ORDER] _process_external_close error for op={operation_id}: {e}",
                exc_info=True,
            )

    async def close_position(self, operation_id: ObjectId, position_id: ObjectId):
        """Manually close a position"""
        position = await Position.get(position_id)
        if not position or position.closed_at:
            raise ValueError(f"Position {position_id} not found or already closed")

        # Determine exit transaction type
        if position.quantity > 0:  # LONG
            exit_action = "SELL"
        else:  # SHORT
            exit_action = "BUY"

        # Get operation to get asset
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        # Place market order to close
        order = await self.place_order(
            operation_id=operation_id,
            asset=operation.asset,
            signal_type=exit_action,
            quantity=abs(position.quantity)
        )

        logger.info(f"Closing position {position_id} with order {order.id}")

    async def handle_crash_recovery(self, operation_id: ObjectId):
        """
        Handle crash recovery for an operation with open positions.

        Checks crash_recovery_mode and applies appropriate strategy:
        - CLOSE_ALL: Close all open positions immediately via market orders
        - RESUME: Resume normal operation, apply emergency stop loss if needed
        - EMERGENCY_EXIT: Close positions if unrealized loss > emergency_stop_loss_pct
        """
        operation = await TradingOperation.get(operation_id)
        if not operation:
            logger.error(f"Operation {operation_id} not found")
            return

        # Get open positions
        open_positions = await Position.find(
            Position.operation_id == operation_id,
            Position.closed_at == None
        ).to_list()

        if not open_positions:
            logger.info(f"No open positions for operation {operation_id}")
            return

        logger.info(f"Found {len(open_positions)} open positions for operation {operation_id}")

        # Update position P/L first
        await self.update_positions(operation_id)

        if operation.crash_recovery_mode == "CLOSE_ALL":
            # Close all positions immediately
            for position in open_positions:
                try:
                    await self.close_position(operation_id, position.id)
                    await self.journal.log_action(
                        action_type="CRASH_RECOVERY_CLOSE",
                        action_data={
                            "position_id": str(position.id),
                            "mode": "CLOSE_ALL"
                        },
                        operation_id=operation_id,
                        notes="Position closed during crash recovery"
                    )
                except Exception as e:
                    logger.error(f"Error closing position {position.id}: {e}")

        elif operation.crash_recovery_mode == "RESUME":
            # Resume normal operation, but check for emergency stop loss
            for position in open_positions:
                if abs(position.unrealized_pnl_pct) > operation.emergency_stop_loss_pct * 100:
                    # Emergency exit
                    try:
                        await self.close_position(operation_id, position.id)
                        await self.journal.log_action(
                            action_type="CRASH_RECOVERY_EMERGENCY_EXIT",
                            action_data={
                                "position_id": str(position.id),
                                "unrealized_pnl_pct": position.unrealized_pnl_pct
                            },
                            operation_id=operation_id,
                            notes=f"Emergency exit due to {position.unrealized_pnl_pct:.2f}% loss"
                        )
                    except Exception as e:
                        logger.error(f"Error in emergency exit for position {position.id}: {e}")

        elif operation.crash_recovery_mode == "EMERGENCY_EXIT":
            # Only close positions exceeding threshold
            for position in open_positions:
                if position.unrealized_pnl_pct < -operation.emergency_stop_loss_pct * 100:
                    try:
                        await self.close_position(operation_id, position.id)
                        await self.journal.log_action(
                            action_type="CRASH_RECOVERY_EMERGENCY_EXIT",
                            action_data={
                                "position_id": str(position.id),
                                "unrealized_pnl_pct": position.unrealized_pnl_pct
                            },
                            operation_id=operation_id,
                            notes=f"Emergency exit due to {position.unrealized_pnl_pct:.2f}% loss"
                        )
                    except Exception as e:
                        logger.error(f"Error in emergency exit for position {position.id}: {e}")

    async def _get_current_price(self, operation_id: ObjectId, asset: str) -> Optional[float]:
        """
        Get current market price from broker positions or latest market data.

        Args:
            operation_id: Trading operation ID
            asset: Asset symbol

        Returns:
            Current price or None if not available
        """
        try:
            # First, try to get price from broker positions
            broker_positions = await self.broker.get_positions()
            for pos in broker_positions:
                if pos.get("asset") == asset:
                    current_price = pos.get("current_price")
                    if current_price and current_price > 0:
                        return float(current_price)

            # Fallback: Get from latest market data
            operation = await TradingOperation.get(operation_id)
            if operation:
                # Get latest market data for primary bar size
                latest_bar = await MarketData.find(
                    MarketData.operation_id == operation_id,
                    MarketData.bar_size == operation.primary_bar_size
                ).sort(-MarketData.timestamp).limit(1).to_list()

                if latest_bar and len(latest_bar) > 0:
                    return float(latest_bar[0].close)

            return None
        except Exception as e:
            logger.error(f"Error getting current price for {asset}: {e}", exc_info=True)
            return None

    async def _get_atr_value(self, operation_id: ObjectId, bar_size: str) -> Optional[float]:
        """
        Get ATR value from latest market data bar.

        Args:
            operation_id: Trading operation ID
            bar_size: Bar size to get ATR from

        Returns:
            ATR value or None if not available
        """
        try:
            # Get latest market data bar
            latest_bar = await MarketData.find(
                MarketData.operation_id == operation_id,
                MarketData.bar_size == bar_size
            ).sort(-MarketData.timestamp).limit(1).to_list()

            if latest_bar and len(latest_bar) > 0:
                indicators = latest_bar[0].indicators
                if indicators and "atr" in indicators:
                    atr_value = indicators.get("atr")
                    if atr_value is not None:
                        return float(atr_value)

            return None
        except Exception as e:
            logger.error(f"Error getting ATR value for operation {operation_id}: {e}", exc_info=True)
            return None

    async def _calculate_default_quantity(
        self,
        operation: TradingOperation,
        entry_price: float,
        stop_loss: Optional[float]
    ) -> float:
        """
        Calculate default order quantity in **lots** based on risk management.

        The broker's place_order expects lots (1 lot = UNITS_PER_LOT currency
        units).  risk_amount / risk_per_unit gives currency units, so we must
        convert to lots before returning.

        Returns:
            Quantity in lots (e.g. 0.5 = half a standard lot)
        """
        UNITS_PER_LOT = 100_000

        try:
            risk_per_trade_pct = 0.01  # 1% of capital

            if entry_price > 0:
                if stop_loss and stop_loss > 0:
                    risk_amount = operation.current_capital * risk_per_trade_pct
                    risk_per_unit = abs(entry_price - stop_loss)

                    if risk_per_unit > 0:
                        units = risk_amount / risk_per_unit
                        lots = units / UNITS_PER_LOT
                        logger.info(
                            f"[SIZING] Risk-based: capital={operation.current_capital}, "
                            f"risk={risk_amount}, risk/unit={risk_per_unit:.6f}, "
                            f"units={units:.0f}, lots={lots:.4f}"
                        )
                        return lots

                # Fallback: fixed fraction of capital expressed in lots
                position_value = operation.current_capital * 0.01
                units = position_value / entry_price if entry_price > 0 else UNITS_PER_LOT
                lots = units / UNITS_PER_LOT
                logger.info(
                    f"[SIZING] Capital-based fallback: lots={lots:.4f} "
                    f"(capital={operation.current_capital}, entry={entry_price})"
                )
                return lots

            # Final fallback: minimum lot size
            return 0.01
        except Exception as e:
            logger.error(f"Error calculating default quantity: {e}", exc_info=True)
            return 0.01

