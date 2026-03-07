"""
CTrader adapter — translates the existing CTraderBroker into the typed
BrokerAdapter interface.

The underlying CTraderBroker handles all Twisted/reactor concerns, connection
management and data-feed subscriptions.  This adapter only wraps the
*trading* surface and normalises every return value into the middleware types.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

from live_trading.brokers.middleware.types import (
    AccountMode,
    BrokerAccountInfo,
    BrokerDeal,
    BrokerOrderResult,
    BrokerOrderUpdate,
    BrokerPosition,
    OrderParams,
    OrderSide,
    OrderStatus,
    PositionSide,
)
from live_trading.brokers.middleware.adapter import BrokerAdapter

logger = logging.getLogger(__name__)

_STATUS_MAP: Dict[str, OrderStatus] = {
    "ACCEPTED": OrderStatus.ACCEPTED,
    "SUBMITTED": OrderStatus.SUBMITTED,
    "FILLED": OrderStatus.FILLED,
    "REJECTED": OrderStatus.REJECTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "EXPIRED": OrderStatus.EXPIRED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
}


def _map_status(raw: str) -> OrderStatus:
    return _STATUS_MAP.get(raw.upper(), OrderStatus.SUBMITTED)


class CTraderAdapter(BrokerAdapter):
    """Wraps ``CTraderBroker`` and exposes the ``BrokerAdapter`` interface."""

    def __init__(self, ctrader_broker):
        from live_trading.brokers.ctrader_broker import CTraderBroker
        if not isinstance(ctrader_broker, CTraderBroker):
            raise TypeError("CTraderAdapter requires a CTraderBroker instance")

        self._broker = ctrader_broker
        self._execution_callback: Optional[Callable[[BrokerOrderUpdate], None]] = None

        self._broker.add_execution_listener(self._on_execution_event)

    def get_account_mode(self) -> AccountMode:
        return AccountMode.HEDGING

    async def submit_order(self, params: OrderParams) -> BrokerOrderResult:
        result_data: Dict = {}
        result_ready = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _order_callback(status_data: Dict):
            result_data.update(status_data)
            loop.call_soon_threadsafe(result_ready.set)

        broker_order_id = await self._broker.place_order(
            asset=params.symbol,
            action=params.side.value,
            quantity=params.quantity,
            order_type=params.order_type.value,
            price=params.price,
            stop_loss=params.stop_loss,
            take_profit=params.take_profit,
            order_status_callback=_order_callback,
        )

        if not broker_order_id:
            return BrokerOrderResult(
                broker_order_id="",
                status=OrderStatus.REJECTED,
            )

        try:
            await asyncio.wait_for(result_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        status = _map_status(result_data.get("status", "SUBMITTED"))
        return BrokerOrderResult(
            broker_order_id=broker_order_id,
            status=status,
            filled_quantity=result_data.get("filled", 0.0),
            avg_fill_price=result_data.get("avg_fill_price"),
            broker_position_id=result_data.get("broker_position_id"),
            commission=result_data.get("commission", 0.0),
        )

    async def close_position(
        self,
        broker_position_id: str,
        quantity: Optional[float] = None,
    ) -> BrokerOrderResult:
        volume: Optional[int] = None
        if quantity is not None:
            volume = self._broker._convert_quantity_to_volume(quantity)

        broker_order_id = await self._broker.close_position_by_id(
            broker_position_id, volume=volume,
        )

        if not broker_order_id:
            return BrokerOrderResult(
                broker_order_id="",
                status=OrderStatus.REJECTED,
            )

        return BrokerOrderResult(
            broker_order_id=broker_order_id,
            status=OrderStatus.FILLED,
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        return await self._broker.cancel_order(broker_order_id)

    async def get_positions(self) -> List[BrokerPosition]:
        raw_positions = await self._broker.get_positions()
        result: List[BrokerPosition] = []
        for pos in raw_positions:
            qty = pos.get("quantity", 0.0)
            side = PositionSide.LONG if qty >= 0 else PositionSide.SHORT
            result.append(BrokerPosition(
                broker_position_id=str(pos.get("position_id", "")),
                symbol=pos.get("asset", ""),
                side=side,
                quantity=abs(qty),
                entry_price=pos.get("avg_price", 0.0),
                current_price=pos.get("current_price", 0.0),
                unrealized_pnl=pos.get("unrealized_pnl", 0.0),
                unrealized_pnl_pct=pos.get("unrealized_pnl_pct", 0.0),
                stop_loss=pos.get("stop_loss"),
                take_profit=pos.get("take_profit"),
            ))
        return result

    async def get_open_orders(self) -> List[BrokerOrderResult]:
        raw = await self._broker.get_open_broker_orders()
        return [
            BrokerOrderResult(
                broker_order_id=o.get("broker_order_id", ""),
                status=OrderStatus.SUBMITTED,
            )
            for o in raw
        ]

    async def get_deals(
        self,
        from_dt: datetime,
        to_dt: datetime,
    ) -> List[BrokerDeal]:
        from_ms = int(from_dt.timestamp() * 1000)
        to_ms = int(to_dt.timestamp() * 1000)
        raw_deals = await self._broker.get_deal_history(from_ms, to_ms)

        result: List[BrokerDeal] = []
        for d in raw_deals:
            ts_ms = d.get("execution_timestamp_ms", 0)
            timestamp = datetime.utcfromtimestamp(ts_ms / 1000) if ts_ms else None

            result.append(BrokerDeal(
                deal_id=str(d.get("deal_id", "")),
                broker_position_id=d.get("position_id"),
                broker_order_id=d.get("order_id"),
                symbol=d.get("asset"),
                side=OrderSide.BUY if d.get("trade_side") == "BUY" else OrderSide.SELL,
                execution_price=d.get("execution_price", 0.0),
                quantity=d.get("volume_units", 0.0),
                is_close=d.get("is_close", False),
                timestamp=timestamp,
            ))
        return result

    async def get_account_info(self) -> BrokerAccountInfo:
        raw = await self._broker.get_account_info()
        return BrokerAccountInfo(
            balance=raw.get("balance", 0.0),
            equity=raw.get("equity"),
            margin_used=raw.get("margin_used"),
            margin_free=raw.get("margin_available"),
            currency=raw.get("currency", "USD"),
        )

    def register_execution_callback(
        self,
        callback: Callable[[BrokerOrderUpdate], None],
    ) -> None:
        self._execution_callback = callback

    def _on_execution_event(self, event_data: Dict) -> None:
        """Translate a raw execution event into ``BrokerOrderUpdate``."""
        if self._execution_callback is None:
            return

        status_raw = event_data.get("status", "")
        is_close = event_data.get("is_position_close", False)

        update = BrokerOrderUpdate(
            broker_order_id=str(event_data.get("order_id", "")),
            status=_map_status(status_raw),
            filled_quantity=event_data.get("filled", 0.0),
            avg_fill_price=event_data.get("avg_fill_price"),
            broker_position_id=event_data.get("broker_position_id"),
            is_position_close=is_close,
        )
        try:
            self._execution_callback(update)
        except Exception as e:
            logger.error(f"Execution callback error: {e}", exc_info=True)
