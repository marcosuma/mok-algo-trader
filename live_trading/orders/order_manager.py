"""
Order Manager — places orders, creates positions, handles fills and closes.

Uses the BrokerAdapter middleware so all broker interaction goes through
typed contracts.  No FIFO matching — each filled order creates its own
Position (hedging mode).
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId

from live_trading.brokers.middleware.adapter import BrokerAdapter
from live_trading.brokers.middleware.types import (
    AccountMode,
    BrokerOrderUpdate,
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
from live_trading.journal.journal_manager import JournalManager
from live_trading.models.market_data import MarketData
from live_trading.models.order import Order
from live_trading.models.position import Position
from live_trading.models.trading_operation import TradingOperation
from forex_strategies.risk_management import (
    calculate_stop_loss as _calc_sl,
    calculate_take_profit as _calc_tp,
)

logger = logging.getLogger(__name__)


def _detect_close_reason(
    position: Position,
    close_price: float,
    tolerance: float = 0.00015,
) -> CloseReason:
    """Infer *why* a position was closed based on the execution price."""
    if position.stop_loss and abs(close_price - position.stop_loss) <= tolerance:
        return CloseReason.STOP_LOSS
    if position.take_profit and abs(close_price - position.take_profit) <= tolerance:
        return CloseReason.TAKE_PROFIT
    return CloseReason.UNKNOWN


class OrderManager:
    """Manages orders and positions via the BrokerAdapter middleware."""

    def __init__(
        self,
        adapter: BrokerAdapter,
        journal_manager: JournalManager,
    ):
        self.adapter = adapter
        self.journal = journal_manager
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        if adapter.get_account_mode() == AccountMode.NETTING:
            raise NotImplementedError("Netting mode is not yet supported")

        adapter.register_execution_callback(self._on_execution_update)

    def bind_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio event loop so broker-thread callbacks can
        schedule coroutines on it."""
        self._event_loop = loop

    async def place_order(
        self,
        operation_id: ObjectId,
        asset: str,
        signal_type: str,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        source: OrderSource = OrderSource.STRATEGY,
        intent: OrderIntent = OrderIntent.OPEN,
        close_reason: Optional[CloseReason] = None,
        broker_position_id: Optional[str] = None,
    ) -> Order:
        """Place an order and return the persisted Order document."""
        short_id = str(operation_id)[-6:]
        tag = f"[OP:{short_id} {asset}]"

        logger.info(f"[ORDER] {tag} Preparing order: {signal_type} @ {price or 'MARKET'}")

        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        order_type = OrderType.LIMIT if price else OrderType.MARKET
        side = OrderSide.BUY if signal_type == "BUY" else OrderSide.SELL

        if stop_loss is None or take_profit is None:
            entry_price = price or await self._get_current_price(operation_id, asset)
            if not entry_price or entry_price == 0.0:
                entry_price = 0.0

            position_type = "LONG" if signal_type == "BUY" else "SHORT"

            if stop_loss is None:
                stop_loss = await self.calculate_stop_loss(
                    operation_id, entry_price, position_type,
                )
            if take_profit is None:
                take_profit = await self.calculate_take_profit(
                    operation_id, entry_price, stop_loss, position_type,
                )

        if quantity is None:
            entry_price_for_sizing = price or await self._get_current_price(operation_id, asset) or 0.0
            quantity = await self._calculate_default_quantity(operation, entry_price_for_sizing, stop_loss)

        order = Order(
            operation_id=operation_id,
            symbol=asset,
            side=side,
            order_type=order_type,
            requested_quantity=quantity,
            requested_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.PENDING_SUBMIT,
            intent=intent,
            close_reason=close_reason,
            source=source,
        )
        await order.insert()

        logger.info(
            f"[ORDER] {tag} Sending to broker: {signal_type} {quantity:.4f} "
            f"@ {price or 'MARKET'} (SL: {stop_loss}, TP: {take_profit})"
        )

        params = OrderParams(
            symbol=asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            broker_position_id=broker_position_id,
        )

        result = await self.adapter.submit_order(params)

        if not result.broker_order_id or result.status == OrderStatus.REJECTED:
            order.status = OrderStatus.REJECTED
            await order.save()
            logger.error(f"[ORDER] {tag} BROKER REJECTED — no order ID returned")
            return order

        order.broker_order_id = result.broker_order_id
        order.submitted_at = datetime.utcnow()

        if result.status == OrderStatus.FILLED:
            order.status = OrderStatus.FILLED
            order.filled_quantity = result.filled_quantity
            order.avg_fill_price = result.avg_fill_price
            order.commission = result.commission
            order.filled_at = datetime.utcnow()
            order.broker_position_id = result.broker_position_id
        else:
            order.status = result.status

        await order.save()

        logger.info(
            f"[ORDER] {tag} SUBMITTED (broker_id: {result.broker_order_id}, "
            f"status: {order.status.value})"
        )

        if order.status == OrderStatus.FILLED and order.intent == OrderIntent.OPEN:
            await self._create_position_from_fill(order, operation)

        await self.journal.log_action(
            action_type="ORDER_PLACED",
            action_data={
                "order_id": str(order.id),
                "broker_order_id": order.broker_order_id,
                "asset": asset,
                "side": side.value,
                "quantity": quantity,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "status": order.status.value,
            },
            operation_id=operation_id,
        )

        return order

    async def close_position(
        self,
        operation_id: ObjectId,
        position_id: ObjectId,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> Order:
        """Close an open position via the broker adapter."""
        position = await Position.get(position_id)
        if not position or position.status != PositionStatus.OPEN:
            raise ValueError(f"Position {position_id} not found or not open")

        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        order = Order(
            operation_id=operation_id,
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            requested_quantity=position.quantity,
            status=OrderStatus.PENDING_SUBMIT,
            intent=OrderIntent.CLOSE,
            close_reason=reason,
            source=OrderSource.MANUAL,
            broker_position_id=position.broker_position_id,
        )
        await order.insert()

        result = await self.adapter.close_position(position.broker_position_id)

        if not result.broker_order_id or result.status == OrderStatus.REJECTED:
            order.status = OrderStatus.REJECTED
            await order.save()
            raise RuntimeError(f"Broker rejected close for position {position.broker_position_id}")

        order.broker_order_id = result.broker_order_id
        order.status = OrderStatus.FILLED
        order.filled_quantity = result.filled_quantity or position.quantity
        order.avg_fill_price = result.avg_fill_price
        order.commission = result.commission
        order.submitted_at = datetime.utcnow()
        order.filled_at = datetime.utcnow()
        await order.save()

        await self._close_position_record(
            position,
            close_price=result.avg_fill_price or position.current_price,
            close_reason=reason,
            exit_order_id=order.id,
            exit_commission=result.commission,
        )

        logger.info(
            f"[ORDER] Closed position {position.broker_position_id} "
            f"({position.side.value}) via broker order {result.broker_order_id}"
        )
        return order

    async def close_all_positions(self, operation_id: ObjectId) -> None:
        """Close every open position belonging to an operation."""
        positions = await Position.find(
            Position.operation_id == operation_id,
            Position.status == PositionStatus.OPEN,
        ).to_list()

        for pos in positions:
            try:
                await self.close_position(operation_id, pos.id, CloseReason.MANUAL)
            except Exception as e:
                logger.error(f"Error closing position {pos.id}: {e}")

    async def handle_crash_recovery(self, operation_id: ObjectId) -> None:
        """Apply crash-recovery strategy for open positions."""
        operation = await TradingOperation.get(operation_id)
        if not operation:
            logger.error(f"Operation {operation_id} not found")
            return

        open_positions = await Position.find(
            Position.operation_id == operation_id,
            Position.status == PositionStatus.OPEN,
        ).to_list()

        if not open_positions:
            logger.info(f"No open positions for operation {operation_id}")
            return

        logger.info(f"Found {len(open_positions)} open positions for operation {operation_id}")

        if operation.crash_recovery_mode == "CLOSE_ALL":
            for pos in open_positions:
                try:
                    await self.close_position(operation_id, pos.id, CloseReason.MANUAL)
                    await self.journal.log_action(
                        action_type="CRASH_RECOVERY_CLOSE",
                        action_data={"position_id": str(pos.id), "mode": "CLOSE_ALL"},
                        operation_id=operation_id,
                    )
                except Exception as e:
                    logger.error(f"Error closing position {pos.id}: {e}")

        elif operation.crash_recovery_mode in ("RESUME", "EMERGENCY_EXIT"):
            threshold = operation.emergency_stop_loss_pct * 100
            for pos in open_positions:
                should_close = (
                    (operation.crash_recovery_mode == "RESUME" and abs(pos.unrealized_pnl_pct) > threshold)
                    or (operation.crash_recovery_mode == "EMERGENCY_EXIT" and pos.unrealized_pnl_pct < -threshold)
                )
                if should_close:
                    try:
                        await self.close_position(operation_id, pos.id, CloseReason.MANUAL)
                        await self.journal.log_action(
                            action_type="CRASH_RECOVERY_EMERGENCY_EXIT",
                            action_data={
                                "position_id": str(pos.id),
                                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                            },
                            operation_id=operation_id,
                        )
                    except Exception as e:
                        logger.error(f"Error in emergency exit for position {pos.id}: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _create_position_from_fill(
        self, order: Order, operation: TradingOperation,
    ) -> Position:
        """Create a new Position document from a filled entry order."""
        side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        position = Position(
            operation_id=order.operation_id,
            broker_position_id=order.broker_position_id or "",
            symbol=order.symbol,
            side=side,
            entry_price=order.avg_fill_price or 0.0,
            quantity=order.filled_quantity,
            current_price=order.avg_fill_price or 0.0,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            commission=order.commission,
            status=PositionStatus.OPEN,
            entry_order_id=order.id,
        )
        await position.insert()

        tag = f"[OP:{str(order.operation_id)[-6:]} {order.symbol}]"
        logger.info(
            f"[POSITION] {tag} OPENED: {side.value} | "
            f"Qty: {order.filled_quantity} @ {order.avg_fill_price:.5f} | "
            f"SL: {order.stop_loss or 'N/A'} | TP: {order.take_profit or 'N/A'} | "
            f"broker_pos_id: {order.broker_position_id}"
        )
        return position

    async def _close_position_record(
        self,
        position: Position,
        close_price: float,
        close_reason: CloseReason,
        exit_order_id: Optional[ObjectId] = None,
        exit_commission: float = 0.0,
        close_time: Optional[datetime] = None,
    ) -> None:
        """Freeze a Position as CLOSED with its final P/L."""
        direction = 1.0 if position.side == PositionSide.LONG else -1.0
        realized_pnl = (close_price - position.entry_price) * direction * position.quantity
        notional = position.entry_price * position.quantity
        realized_pnl_pct = (realized_pnl / notional * 100) if notional > 0 else 0.0

        now = close_time or datetime.utcnow()

        position.status = PositionStatus.CLOSED
        position.close_price = close_price
        position.close_reason = close_reason
        position.realized_pnl = realized_pnl
        position.realized_pnl_pct = realized_pnl_pct
        position.total_commission = position.commission + exit_commission
        position.total_swap = position.swap
        position.duration_seconds = (now - position.opened_at).total_seconds()
        position.exit_order_id = exit_order_id
        position.closed_at = now
        position.updated_at = now
        position.current_price = close_price
        position.unrealized_pnl = 0.0
        position.unrealized_pnl_pct = 0.0
        await position.save()

        tag = f"[OP:{str(position.operation_id)[-6:]} {position.symbol}]"
        logger.info(
            f"[POSITION] {tag} CLOSED: {position.side.value} | "
            f"P/L: {realized_pnl:.2f} ({realized_pnl_pct:.2f}%) | "
            f"Reason: {close_reason.value} | Duration: {position.duration_seconds:.0f}s"
        )

    def _on_execution_update(self, update: BrokerOrderUpdate) -> None:
        """Called from the broker thread on every execution event.

        Schedules async handling on the application event loop.
        """
        if self._event_loop is None:
            logger.warning("[ORDER] No event loop bound — cannot handle execution update")
            return
        asyncio.run_coroutine_threadsafe(
            self._handle_execution_update(update), self._event_loop,
        )

    async def _handle_execution_update(self, update: BrokerOrderUpdate) -> None:
        """Process an asynchronous execution update from the broker."""
        if update.is_position_close and update.broker_position_id:
            await self._handle_broker_position_close(update)

    async def _handle_broker_position_close(self, update: BrokerOrderUpdate) -> None:
        """Handle a position closed by the broker (TP / SL / manual)."""
        position = await Position.find_one(
            Position.broker_position_id == update.broker_position_id,
            Position.status == PositionStatus.OPEN,
        )
        if not position:
            logger.info(
                f"[ORDER] Position close event for {update.broker_position_id} "
                f"but no matching OPEN position — skipping"
            )
            return

        close_price = update.avg_fill_price or position.current_price
        reason = _detect_close_reason(position, close_price)

        close_order = Order(
            operation_id=position.operation_id,
            symbol=position.symbol,
            side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_quantity=position.quantity,
            status=OrderStatus.FILLED,
            filled_quantity=update.filled_quantity or position.quantity,
            avg_fill_price=close_price,
            commission=update.commission,
            intent=OrderIntent.CLOSE,
            close_reason=reason,
            source=OrderSource.BROKER_DETECTED,
            broker_order_id=update.broker_order_id,
            broker_position_id=update.broker_position_id,
            submitted_at=datetime.utcnow(),
            filled_at=datetime.utcnow(),
        )
        await close_order.insert()

        await self._close_position_record(
            position,
            close_price=close_price,
            close_reason=reason,
            exit_order_id=close_order.id,
            exit_commission=update.commission,
        )

        await self.journal.log_action(
            action_type="BROKER_POSITION_CLOSE",
            action_data={
                "order_id": str(close_order.id),
                "position_id": str(position.id),
                "broker_position_id": update.broker_position_id,
                "close_price": close_price,
                "close_reason": reason.value,
            },
            operation_id=position.operation_id,
        )

    # ------------------------------------------------------------------
    # SL / TP / Sizing helpers (unchanged logic)
    # ------------------------------------------------------------------

    async def calculate_stop_loss(
        self, operation_id: ObjectId, entry_price: float, position_type: str,
    ) -> float:
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        atr_value = None
        if operation.stop_loss_type == "ATR":
            atr_value = await self._get_atr_value(operation_id, operation.primary_bar_size)
            if atr_value is None or atr_value == 0.0:
                logger.warning(f"Could not get ATR for op {operation_id}, using 0.001")
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
        position_type: str,
    ) -> float:
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        atr_value = None
        if operation.take_profit_type == "ATR":
            atr_value = await self._get_atr_value(operation_id, operation.primary_bar_size)
            if atr_value is None or atr_value == 0.0:
                logger.warning(f"Could not get ATR for op {operation_id}, using 0.001")
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

    async def _get_current_price(
        self, operation_id: ObjectId, asset: str,
    ) -> Optional[float]:
        try:
            broker_positions = await self.adapter.get_positions()
            for pos in broker_positions:
                if pos.symbol == asset and pos.current_price > 0:
                    return pos.current_price

            operation = await TradingOperation.get(operation_id)
            if operation:
                latest_bar = await MarketData.find(
                    MarketData.operation_id == operation_id,
                    MarketData.bar_size == operation.primary_bar_size,
                ).sort(-MarketData.timestamp).limit(1).to_list()
                if latest_bar:
                    return float(latest_bar[0].close)

            return None
        except Exception as e:
            logger.error(f"Error getting current price for {asset}: {e}", exc_info=True)
            return None

    async def _get_atr_value(
        self, operation_id: ObjectId, bar_size: str,
    ) -> Optional[float]:
        try:
            latest_bar = await MarketData.find(
                MarketData.operation_id == operation_id,
                MarketData.bar_size == bar_size,
            ).sort(-MarketData.timestamp).limit(1).to_list()

            if latest_bar:
                indicators = latest_bar[0].indicators
                if indicators and "atr" in indicators:
                    atr_value = indicators.get("atr")
                    if atr_value is not None:
                        return float(atr_value)
            return None
        except Exception as e:
            logger.error(f"Error getting ATR for op {operation_id}: {e}", exc_info=True)
            return None

    async def _calculate_default_quantity(
        self,
        operation: TradingOperation,
        entry_price: float,
        stop_loss: Optional[float],
    ) -> float:
        """Calculate order quantity in lots based on risk management."""
        UNITS_PER_LOT = 100_000

        try:
            risk_per_trade_pct = 0.01

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

                position_value = operation.current_capital * 0.01
                units = position_value / entry_price if entry_price > 0 else UNITS_PER_LOT
                lots = units / UNITS_PER_LOT
                logger.info(
                    f"[SIZING] Capital-based fallback: lots={lots:.4f} "
                    f"(capital={operation.current_capital}, entry={entry_price})"
                )
                return lots

            return 0.01
        except Exception as e:
            logger.error(f"Error calculating default quantity: {e}", exc_info=True)
            return 0.01
