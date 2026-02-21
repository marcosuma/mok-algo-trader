"""
Backtesting strategy class for forex strategies.
"""
from backtesting import Strategy
import numpy as np
import pandas as pd

from forex_strategies.risk_management import compute_sl_tp


class ForexBacktestingStrategy(Strategy):
    """Generic backtesting strategy that uses execute_buy and execute_sell signals.

    Reads ``execute_buy`` / ``execute_sell`` columns from the data.
    When a value is present (not NaN) a trade is placed:
      - If the value equals the bar's close price the order is a market order
        (filled at next bar open).
      - Otherwise the value is forwarded as a *limit* price so the backtesting
        engine only fills the order when the market reaches that level.

    Risk management and pyramiding parameters are injected as class-level
    attributes by ``BaseForexStrategy.execute()`` via ``bt.run(**params)``.
    """

    # Tolerance for comparing the signal price to the close price.
    _MARKET_ORDER_TOL = 1e-9

    # --- Parameters injected by BaseForexStrategy.execute() -----------------
    stop_loss_type = "NONE"
    stop_loss_value = 1.5
    take_profit_type = "NONE"
    take_profit_value = 2.0
    allow_pyramiding = False
    max_pyramid_entries = 1

    def init(self):
        super().init()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_market_order(self, signal_price: float, close_price: float) -> bool:
        """Return True when the signal price is essentially equal to close."""
        if close_price == 0:
            return signal_price == 0
        return abs(signal_price - close_price) / abs(close_price) < self._MARKET_ORDER_TOL

    def _get_atr(self) -> float | None:
        """Read the current ATR from data if available."""
        if hasattr(self.data, "atr"):
            val = self.data.atr[-1]
            if not pd.isna(val) and val > 0:
                return float(val)
        return None

    def _build_order_kwargs(
        self,
        signal_price: float,
        close_price: float,
        position_type: str,
    ) -> dict:
        """Build the kwargs dict for ``self.buy()`` / ``self.sell()``."""
        kwargs: dict = {}

        if not self._is_market_order(signal_price, close_price):
            kwargs["limit"] = signal_price

        entry_estimate = (
            close_price
            if self._is_market_order(signal_price, close_price)
            else signal_price
        )

        sl, tp = compute_sl_tp(
            entry_price=entry_estimate,
            position_type=position_type,
            sl_type=self.stop_loss_type,
            sl_value=self.stop_loss_value,
            tp_type=self.take_profit_type,
            tp_value=self.take_profit_value,
            atr_value=self._get_atr(),
        )

        if sl is not None:
            kwargs["sl"] = sl
        if tp is not None:
            kwargs["tp"] = tp

        if self.allow_pyramiding:
            kwargs["size"] = 1.0 / self.max_pyramid_entries

        return kwargs

    def _close_all_trades(self):
        """Explicitly close every open trade (used in pyramiding mode)."""
        for trade in list(self.trades):
            trade.close()

    def _open_long_count(self) -> int:
        return sum(1 for t in self.trades if t.is_long)

    def _open_short_count(self) -> int:
        return sum(1 for t in self.trades if t.is_short)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def next(self):
        super().next()

        close_price = self.data.Close[-1]

        if hasattr(self.data, "execute_buy"):
            buy_signal = self.data.execute_buy[-1]
            if not pd.isna(buy_signal):
                self._handle_buy(buy_signal, close_price)

        if hasattr(self.data, "execute_sell"):
            sell_signal = self.data.execute_sell[-1]
            if not pd.isna(sell_signal):
                self._handle_sell(sell_signal, close_price)

    def _handle_buy(self, signal_price: float, close_price: float):
        should_trade = False

        if not self.position:
            should_trade = True
        elif self.position.is_short:
            if self.allow_pyramiding:
                self._close_all_trades()
            should_trade = True
        elif self.allow_pyramiding:
            should_trade = self._open_long_count() < self.max_pyramid_entries

        if should_trade:
            kwargs = self._build_order_kwargs(signal_price, close_price, "LONG")
            self.buy(**kwargs)

    def _handle_sell(self, signal_price: float, close_price: float):
        should_trade = False

        if not self.position:
            should_trade = True
        elif self.position.is_long:
            if self.allow_pyramiding:
                self._close_all_trades()
            should_trade = True
        elif self.allow_pyramiding:
            should_trade = self._open_short_count() < self.max_pyramid_entries

        if should_trade:
            kwargs = self._build_order_kwargs(signal_price, close_price, "SHORT")
            self.sell(**kwargs)
