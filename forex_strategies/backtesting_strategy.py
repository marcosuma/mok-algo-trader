"""
Backtesting strategy class for forex strategies.
"""
from backtesting import Strategy
import numpy as np
import pandas as pd


class ForexBacktestingStrategy(Strategy):
    """Generic backtesting strategy that uses execute_buy and execute_sell signals.

    Reads ``execute_buy`` / ``execute_sell`` columns from the data.
    When a value is present (not NaN) a trade is placed:
      - If the value equals the bar's close price the order is a market order
        (filled at next bar open).
      - Otherwise the value is forwarded as a *limit* price so the backtesting
        engine only fills the order when the market reaches that level.
    """

    # Tolerance for comparing the signal price to the close price.
    # If the difference is within this fraction the order is treated as
    # a market order (avoids floating-point comparison issues).
    _MARKET_ORDER_TOL = 1e-9

    def init(self):
        super().init()

    def _is_market_order(self, signal_price: float, close_price: float) -> bool:
        """Return True when the signal price is essentially equal to close."""
        if close_price == 0:
            return signal_price == 0
        return abs(signal_price - close_price) / abs(close_price) < self._MARKET_ORDER_TOL

    def next(self):
        super().next()

        close_price = self.data.Close[-1]

        # Check for buy signal
        if hasattr(self.data, "execute_buy"):
            buy_signal = self.data.execute_buy[-1]
            if not pd.isna(buy_signal):
                if not self.position or not self.position.is_long:
                    if self._is_market_order(buy_signal, close_price):
                        self.buy()
                    else:
                        self.buy(limit=buy_signal)

        # Check for sell signal
        if hasattr(self.data, "execute_sell"):
            sell_signal = self.data.execute_sell[-1]
            if not pd.isna(sell_signal):
                if not self.position or not self.position.is_short:
                    if self._is_market_order(sell_signal, close_price):
                        self.sell()
                    else:
                        self.sell(limit=sell_signal)

