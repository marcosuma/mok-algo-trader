"""
Abstract broker adapter interface.

Every broker integration must implement this interface so that the application
layer (OrderManager, Reconciler) works exclusively with the typed contracts
defined in ``types.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional

from .types import (
    AccountMode,
    BrokerAccountInfo,
    BrokerDeal,
    BrokerOrderResult,
    BrokerOrderUpdate,
    BrokerPosition,
    OrderParams,
)


class BrokerAdapter(ABC):
    """Broker-agnostic trading interface.

    Data-feed concerns (market-data subscriptions, tick streaming) are handled
    separately by the underlying broker connection and are NOT part of this
    interface.
    """

    @abstractmethod
    def get_account_mode(self) -> AccountMode:
        """Return whether the account operates in HEDGING or NETTING mode."""
        ...

    @abstractmethod
    async def submit_order(self, params: OrderParams) -> BrokerOrderResult:
        """Submit an order to the broker.

        For MARKET orders the returned ``BrokerOrderResult`` typically already
        contains the fill details.  For LIMIT / STOP orders the result will
        have status ``SUBMITTED`` or ``ACCEPTED`` and actual fills arrive later
        via the execution callback registered with
        ``register_execution_callback``.
        """
        ...

    @abstractmethod
    async def close_position(
        self,
        broker_position_id: str,
        quantity: Optional[float] = None,
    ) -> BrokerOrderResult:
        """Close (or partially close) an existing position by its broker ID.

        If *quantity* is ``None`` the full position is closed.
        """
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a pending order.  Returns ``True`` on success."""
        ...

    @abstractmethod
    async def get_positions(self) -> List[BrokerPosition]:
        """Return all open positions on the account."""
        ...

    @abstractmethod
    async def get_open_orders(self) -> List[BrokerOrderResult]:
        """Return all orders that are still open / pending on the broker."""
        ...

    @abstractmethod
    async def get_deals(
        self,
        from_dt: datetime,
        to_dt: datetime,
    ) -> List[BrokerDeal]:
        """Return deal (execution) history for the given time range."""
        ...

    @abstractmethod
    async def get_account_info(self) -> BrokerAccountInfo:
        """Return account-level information (balance, equity, etc.)."""
        ...

    @abstractmethod
    def register_execution_callback(
        self,
        callback: Callable[[BrokerOrderUpdate], None],
    ) -> None:
        """Register a callback that fires on every order status change.

        The adapter must translate broker-native events into
        ``BrokerOrderUpdate`` before invoking the callback.
        """
        ...
