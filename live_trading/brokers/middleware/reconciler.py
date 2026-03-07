"""
Reconciler — keeps the local database in sync with the broker.

Runs periodically (e.g. every 30 s) and on startup.  The key improvement
over the previous implementation is matching positions by
``broker_position_id`` rather than by asset name, which correctly handles
pyramiding (multiple positions on the same symbol).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bson import ObjectId

from live_trading.brokers.middleware.adapter import BrokerAdapter
from live_trading.brokers.middleware.types import (
    BrokerDeal,
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
from live_trading.models.order import Order
from live_trading.models.position import Position
from live_trading.models.trading_operation import TradingOperation

logger = logging.getLogger(__name__)


def detect_close_reason(
    position: Position,
    close_price: float,
    tolerance: float = 0.00015,
) -> CloseReason:
    """Infer the close reason from the execution price vs SL/TP."""
    if position.stop_loss and abs(close_price - position.stop_loss) <= tolerance:
        return CloseReason.STOP_LOSS
    if position.take_profit and abs(close_price - position.take_profit) <= tolerance:
        return CloseReason.TAKE_PROFIT
    return CloseReason.UNKNOWN


class Reconciler:
    """Broker ↔ database reconciliation engine."""

    def __init__(self, adapter: BrokerAdapter):
        self.adapter = adapter

    async def reconcile(self, operation: TradingOperation) -> None:
        """Run a full reconciliation cycle for a single operation.

        Steps:
          1. Sync positions (update existing, create missing, close removed).
          2. Fix stuck PENDING orders.
          3. Update current_capital from broker balance.
        """
        tag = f"[SYNC op={str(operation.id)[-6:]} {operation.asset}]"
        logger.info(f"{tag} Starting reconciliation...")

        try:
            await self._sync_positions(operation, tag)
            await self._fix_stuck_orders(operation, tag)
            await self._update_capital(operation, tag)
            logger.info(f"{tag} Reconciliation complete")
        except Exception as e:
            logger.error(f"{tag} Reconciliation error: {e}", exc_info=True)

    async def _sync_positions(
        self, operation: TradingOperation, tag: str,
    ) -> None:
        broker_positions = await self.adapter.get_positions()

        our_positions = [
            p for p in broker_positions if p.symbol == operation.asset
        ]

        db_positions = await Position.find(
            Position.operation_id == operation.id,
            Position.status == PositionStatus.OPEN,
        ).to_list()
        db_map: Dict[str, Position] = {
            p.broker_position_id: p for p in db_positions
        }

        broker_ids: set[str] = set()

        for bp in our_positions:
            broker_ids.add(bp.broker_position_id)

            if bp.broker_position_id in db_map:
                await self._update_existing_position(db_map[bp.broker_position_id], bp, tag)
            else:
                await self._create_missing_position(operation.id, bp, tag)

        closed_positions = [
            p for p in db_positions if p.broker_position_id not in broker_ids
        ]
        if closed_positions:
            await self._handle_closed_positions(operation, closed_positions, tag)

    async def _update_existing_position(
        self, db_pos: Position, bp: BrokerPosition, tag: str,
    ) -> None:
        db_pos.entry_price = bp.entry_price
        db_pos.current_price = bp.current_price
        db_pos.unrealized_pnl = bp.unrealized_pnl
        db_pos.unrealized_pnl_pct = bp.unrealized_pnl_pct
        db_pos.stop_loss = bp.stop_loss
        db_pos.take_profit = bp.take_profit
        db_pos.swap = bp.swap
        db_pos.commission = bp.commission
        db_pos.quantity = bp.quantity
        db_pos.updated_at = datetime.utcnow()
        await db_pos.save()
        logger.debug(
            f"{tag} Updated position {bp.broker_position_id}: "
            f"price={bp.current_price}, upnl={bp.unrealized_pnl:.2f}"
        )

    async def _create_missing_position(
        self, operation_id: ObjectId, bp: BrokerPosition, tag: str,
    ) -> None:
        new_pos = Position(
            operation_id=operation_id,
            broker_position_id=bp.broker_position_id,
            symbol=bp.symbol,
            side=bp.side,
            entry_price=bp.entry_price,
            quantity=bp.quantity,
            current_price=bp.current_price,
            unrealized_pnl=bp.unrealized_pnl,
            unrealized_pnl_pct=bp.unrealized_pnl_pct,
            stop_loss=bp.stop_loss,
            take_profit=bp.take_profit,
            swap=bp.swap,
            commission=bp.commission,
            status=PositionStatus.OPEN,
        )
        await new_pos.insert()
        logger.info(
            f"{tag} Created missing position {bp.broker_position_id} "
            f"({bp.side.value} {bp.quantity} @ {bp.entry_price})"
        )

    async def _handle_closed_positions(
        self,
        operation: TradingOperation,
        closed: List[Position],
        tag: str,
    ) -> None:
        deals = await self._fetch_close_deals(closed)

        for pos in closed:
            close_price = pos.current_price
            close_time: Optional[datetime] = None

            matching_deals = deals.get(pos.broker_position_id, [])
            if matching_deals:
                best = max(matching_deals, key=lambda d: d.timestamp or datetime.min)
                close_price = best.execution_price or close_price
                close_time = best.timestamp

                logger.info(
                    f"{tag} Matched close deal for {pos.broker_position_id}: "
                    f"price={close_price}, time={close_time}"
                )

            reason = detect_close_reason(pos, close_price)

            close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            close_order = Order(
                operation_id=pos.operation_id,
                symbol=pos.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                requested_quantity=pos.quantity,
                status=OrderStatus.FILLED,
                filled_quantity=pos.quantity,
                avg_fill_price=close_price,
                intent=OrderIntent.CLOSE,
                close_reason=reason,
                source=OrderSource.BROKER_DETECTED,
                broker_position_id=pos.broker_position_id,
                submitted_at=close_time or datetime.utcnow(),
                filled_at=close_time or datetime.utcnow(),
            )
            await close_order.insert()

            direction = 1.0 if pos.side == PositionSide.LONG else -1.0
            realized_pnl = (close_price - pos.entry_price) * direction * pos.quantity
            notional = pos.entry_price * pos.quantity
            realized_pnl_pct = (realized_pnl / notional * 100) if notional > 0 else 0.0

            now = close_time or datetime.utcnow()
            pos.status = PositionStatus.CLOSED
            pos.close_price = close_price
            pos.close_reason = reason
            pos.realized_pnl = realized_pnl
            pos.realized_pnl_pct = realized_pnl_pct
            pos.total_commission = pos.commission
            pos.total_swap = pos.swap
            pos.duration_seconds = (now - pos.opened_at).total_seconds()
            pos.exit_order_id = close_order.id
            pos.closed_at = now
            pos.updated_at = now
            pos.current_price = close_price
            pos.unrealized_pnl = 0.0
            pos.unrealized_pnl_pct = 0.0
            await pos.save()

            logger.info(
                f"{tag} Position {pos.broker_position_id} closed externally: "
                f"{pos.side.value} P/L={realized_pnl:.2f} reason={reason.value}"
            )

    async def _fetch_close_deals(
        self, positions: List[Position],
    ) -> Dict[str, List[BrokerDeal]]:
        """Fetch deal history and group closing deals by broker_position_id."""
        if not positions:
            return {}

        try:
            earliest = min(p.opened_at for p in positions)
            from_dt = min(earliest, datetime.utcnow() - timedelta(hours=24))
            to_dt = datetime.utcnow()

            deals = await self.adapter.get_deals(from_dt, to_dt)

            result: Dict[str, List[BrokerDeal]] = {}
            for deal in deals:
                if deal.is_close and deal.broker_position_id:
                    result.setdefault(deal.broker_position_id, []).append(deal)
            return result

        except Exception as e:
            logger.warning(f"[SYNC] Could not fetch deal history: {e}")
            return {}

    async def _fix_stuck_orders(
        self, operation: TradingOperation, tag: str,
    ) -> None:
        pending_orders = await Order.find(
            Order.operation_id == operation.id,
            Order.status.in_([OrderStatus.PENDING_SUBMIT, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]),
        ).to_list()

        if not pending_orders:
            return

        broker_open_orders = await self.adapter.get_open_orders()
        broker_open_ids = {o.broker_order_id for o in broker_open_orders}

        open_positions = await self.adapter.get_positions()
        broker_position_ids = {p.broker_position_id for p in open_positions}

        for db_order in pending_orders:
            if not db_order.broker_order_id:
                db_order.status = OrderStatus.REJECTED
                await db_order.save()
                logger.warning(f"{tag} Order {db_order.id} had no broker_order_id — marked REJECTED")
                continue

            if db_order.broker_order_id in broker_open_ids:
                continue

            if db_order.broker_position_id and db_order.broker_position_id in broker_position_ids:
                db_order.status = OrderStatus.FILLED
                db_order.filled_at = db_order.filled_at or datetime.utcnow()
                await db_order.save()
                logger.info(f"{tag} Order {db_order.broker_order_id} → FILLED (position exists)")
            else:
                db_order.status = OrderStatus.CANCELLED
                db_order.cancelled_at = datetime.utcnow()
                await db_order.save()
                logger.info(f"{tag} Order {db_order.broker_order_id} → CANCELLED (not on broker)")

    async def _update_capital(
        self, operation: TradingOperation, tag: str,
    ) -> None:
        try:
            info = await self.adapter.get_account_info()
            if info.balance > 0:
                operation.current_capital = info.balance
                await operation.save()
                logger.info(f"{tag} Capital updated to {info.balance:.2f}")
        except Exception as e:
            logger.warning(f"{tag} Could not fetch broker balance: {e}")
