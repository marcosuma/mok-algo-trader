"""
Strategy Adapter - Adapts backtesting strategies for live trading.
"""
import logging
from typing import Dict, Optional, List
import pandas as pd

from live_trading.data.data_manager import DataManager
from forex_strategies.base_strategy import BaseForexStrategy

logger = logging.getLogger(__name__)


class StrategyAdapter:
    """Adapts backtesting strategies for incremental live trading"""

    def __init__(
        self,
        strategy_class: type[BaseForexStrategy],
        strategy_config: Dict,
        data_manager: DataManager,
        operation_id: str,
        bar_sizes: List[str],
        primary_bar_size: str,
        asset: Optional[str] = None
    ):
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config.copy()  # Make a copy to avoid modifying original
        self.data_manager = data_manager
        self.operation_id = operation_id
        self.bar_sizes = bar_sizes
        self.primary_bar_size = primary_bar_size
        self.asset = asset

        # For multi-timeframe strategies, ensure contract_name is provided
        # Check if strategy inherits from MultiTimeframeStrategy
        try:
            from forex_strategies.multi_timeframe_strategy import MultiTimeframeStrategy
            is_multi_timeframe = issubclass(strategy_class, MultiTimeframeStrategy)
        except ImportError:
            # Fallback: check by class name if import fails
            strategy_name = strategy_class.__name__
            is_multi_timeframe = strategy_name in ['MultiTimeframeStrategy', 'AdaptiveMultiTimeframeStrategy']

        if is_multi_timeframe:
            # If contract_name not in config and asset is available, use asset as contract_name
            if 'contract_name' not in self.strategy_config or self.strategy_config.get('contract_name') is None:
                if asset:
                    self.strategy_config['contract_name'] = asset
                    logger.info(f"[SIGNAL] Added contract_name={asset} to strategy_config for {strategy_class.__name__} (op: {operation_id})")
                else:
                    logger.warning(f"contract_name not provided for {strategy_class.__name__} and asset is not available")

        # Initialize strategy instance
        self.strategy = strategy_class(**self.strategy_config)

        # Short tag for log messages: [OP:abc123 EUR-USD StrategyName]
        short_id = str(operation_id)[-6:]
        strategy_short = strategy_class.__name__
        self._op_tag = f"[OP:{short_id} {asset or '?'} {strategy_short}]"

        # Signal callback
        self.signal_callback: Optional[callable] = None

    def set_signal_callback(self, callback: callable):
        """Set callback for when signals are generated"""
        self.signal_callback = callback

    async def on_new_bar(self, bar_size: str, bar_data: Dict, indicators: Dict):
        """Called when a new bar is completed"""
        tag = self._op_tag
        logger.info(f"[SIGNAL] {tag} on_new_bar: bar_size={bar_size}, primary={self.primary_bar_size}")

        # Only generate signals when primary bar_size completes
        if bar_size != self.primary_bar_size:
            logger.debug(f"[SIGNAL] {tag} Bar size {bar_size} != primary {self.primary_bar_size}, skipping")
            return

        logger.info(f"[SIGNAL] {tag} Primary bar matched, generating signals...")

        # Get aligned dataframes for all timeframes
        aligned_data = await self._align_timeframes()

        if aligned_data is None or aligned_data.empty:
            logger.warning(f"[SIGNAL] {tag} No aligned data available for signal generation")
            return

        logger.info(f"[SIGNAL] {tag} Aligned data: {len(aligned_data)} rows, columns: {list(aligned_data.columns)}")

        # Diagnostic: Check for required indicators and their validity
        # Note: local_extrema is expected to be NaN in recent bars (needs future data to confirm)
        required_indicators = [
            "adx", "plus_di", "minus_di", "RSI_14", "macd", "macd_s", "macd_h",
            "SMA_50", "bollinger_up", "bollinger_down", "atr", "local_extrema"
        ]
        # Indicators that are expected to be NaN in the last row due to their nature
        expected_nan_indicators = {"local_extrema"}  # Needs future data to confirm extrema

        missing_indicators = [col for col in required_indicators if col not in aligned_data.columns]
        if missing_indicators:
            logger.warning(f"[SIGNAL] {tag} Missing indicators: {missing_indicators}")

        # Check for NaN values in the last row of key indicators
        if len(aligned_data) > 0:
            last_row = aligned_data.iloc[-1]
            nan_indicators = []
            for col in required_indicators:
                if col in aligned_data.columns:
                    val = last_row.get(col)
                    if pd.isna(val):
                        nan_indicators.append(col)
                    else:
                        logger.debug(f"[SIGNAL] {tag} {col} = {val} (type: {type(val).__name__})")

            # Separate expected vs unexpected NaN indicators
            unexpected_nan = [ind for ind in nan_indicators if ind not in expected_nan_indicators]
            expected_nan = [ind for ind in nan_indicators if ind in expected_nan_indicators]

            if unexpected_nan:
                logger.warning(f"[SIGNAL] {tag} Unexpected NaN indicators: {unexpected_nan}")
            if expected_nan:
                logger.debug(f"[SIGNAL] {tag} Expected NaN indicators (normal): {expected_nan}")

            # Log OHLC values for debugging
            logger.info(f"[SIGNAL] {tag} Last bar OHLC: O={last_row.get('open')}, H={last_row.get('high')}, L={last_row.get('low')}, C={last_row.get('close')}")

        # Generate signals using strategy
        try:
            logger.info(f"[SIGNAL] {tag} Calling strategy.generate_signals() with {len(aligned_data)} rows...")
            signals_df = self.strategy.generate_signals(aligned_data)
            logger.info(f"[SIGNAL] {tag} Signal generation completed, checking last row")

            # Check for signals in the last row
            last_row = signals_df.iloc[-1]

            # Log signal values for debugging
            execute_buy = last_row.get("execute_buy")
            execute_sell = last_row.get("execute_sell")
            logger.info(f"[SIGNAL] {tag} Signal values - execute_buy: {execute_buy}, execute_sell: {execute_sell}")

            # Also log raw buy_signal/sell_signal if they exist
            if "buy_signal" in last_row:
                logger.debug(f"[SIGNAL] {tag} buy_signal (raw): {last_row['buy_signal']}")
            if "sell_signal" in last_row:
                logger.debug(f"[SIGNAL] {tag} sell_signal (raw): {last_row['sell_signal']}")

            # Check for buy signal
            if pd.notna(execute_buy):
                logger.info(f"[SIGNAL] {tag} 🟢 BUY SIGNAL @ {execute_buy:.5f}")
                await self._handle_signal("BUY", execute_buy)

            # Check for sell signal
            if pd.notna(execute_sell):
                logger.info(f"[SIGNAL] {tag} 🔴 SELL SIGNAL @ {execute_sell:.5f}")
                await self._handle_signal("SELL", execute_sell)

            # Log when no signal is generated (this helps debugging)
            if pd.isna(execute_buy) and pd.isna(execute_sell):
                logger.info(f"[SIGNAL] {tag} ⚪ No signal generated (HOLD)")

        except ValueError as e:
            # ValueError is typically raised for missing required indicators.
            # Log at WARNING (not just ERROR) so it is visible at the same
            # level as the surrounding INFO messages.
            logger.warning(f"[SIGNAL] {tag} ❌ STRATEGY ERROR - Missing required data: {e}")
            logger.warning(f"[SIGNAL] {tag} ❌ Available columns: {list(aligned_data.columns)}")
            logger.warning(f"[SIGNAL] {tag} ❌ Signal generation FAILED - no signal will be produced for this bar")
        except Exception as e:
            logger.warning(f"[SIGNAL] {tag} ❌ Unexpected error generating signals: {type(e).__name__}: {e}")
            logger.error(f"[SIGNAL] {tag} ❌ Exception details:", exc_info=True)

    async def _align_timeframes(self) -> Optional[pd.DataFrame]:
        """Align data from multiple timeframes"""
        # Get latest bars for primary timeframe
        primary_df = await self.data_manager.get_dataframe(
            self.operation_id,
            self.primary_bar_size
        )

        if primary_df.empty:
            return None

        # For other timeframes, get most recent available bar
        for bar_size in self.bar_sizes:
            if bar_size == self.primary_bar_size:
                continue

            other_df = await self.data_manager.get_dataframe(
                self.operation_id,
                bar_size
            )

            if not other_df.empty:
                # Get most recent bar
                latest_bar = other_df.iloc[-1]

                # Add columns with bar_size prefix
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in latest_bar:
                        primary_df[f"{bar_size}_{col}"] = latest_bar[col]

                # Add indicators with bar_size prefix
                if "indicators" in latest_bar and isinstance(latest_bar["indicators"], dict):
                    for indicator_name, indicator_value in latest_bar["indicators"].items():
                        primary_df[f"{bar_size}_{indicator_name}"] = indicator_value

        return primary_df

    async def _handle_signal(self, signal_type: str, price: float):
        """Handle a trading signal"""
        tag = self._op_tag
        if self.signal_callback:
            logger.info(f"[SIGNAL] {tag} Forwarding {signal_type} signal @ {price:.5f} to order manager")
            await self.signal_callback(
                operation_id=self.operation_id,
                signal_type=signal_type,
                price=price
            )
        else:
            logger.warning(f"[SIGNAL] {tag} Signal generated but no callback set: {signal_type} @ {price}")

