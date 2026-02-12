"""
Base Broker interface.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
from datetime import datetime


class BaseBroker(ABC):
    """Base class for broker adapters"""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker. Returns True if successful."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def subscribe_market_data(
        self,
        asset: str,
        callback: Callable[[Dict[str, Any]], None],
        callback_id: str = None,
        **kwargs
    ) -> bool:
        """
        Subscribe to real-time market data for an asset.

        Supports multiple callbacks per asset (e.g., multiple operations on same asset).
        Each callback should have a unique callback_id for proper tracking.

        Args:
            asset: Asset symbol (e.g., "USD-CAD")
            callback: Callback function to receive market data updates
            callback_id: Optional unique identifier for this callback (for tracking/removal)

        Returns:
            True if subscription successful
        """
        pass

    @abstractmethod
    async def unsubscribe_market_data(self, asset: str, callback_id: str = None):
        """
        Unsubscribe from market data for an asset.

        If callback_id is provided, removes only that specific callback.
        If callback_id is None, removes ALL callbacks for the asset.
        """
        pass

    @abstractmethod
    async def place_order(
        self,
        asset: str,
        action: str,  # 'BUY' or 'SELL'
        quantity: float,
        order_type: str = "MARKET",  # 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> str:
        """
        Place an order.

        Args:
            asset: Asset symbol
            action: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Broker order ID
        """
        pass

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order. Returns True if successful."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dictionaries with keys: symbol, quantity, avg_price, etc.
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status including subscription details.

        Returns:
            Dictionary with connection status, subscriptions, and callback counts
        """
        return {
            "connected": getattr(self, 'connected', False),
            "authenticated": getattr(self, 'authenticated', False),
            "subscriptions": list(getattr(self, '_data_subscriptions', {}).keys()),
            "subscription_details": {}
        }
