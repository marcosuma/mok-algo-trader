"""
Data Manager - Handles real-time data collection, bar aggregation, and indicator calculation.
"""
import logging
import asyncio
import inspect
from typing import Dict, Callable, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

from live_trading.models.market_data import MarketData
from live_trading.models.trading_operation import TradingOperation
from live_trading.config import config

logger = logging.getLogger(__name__)


class BarAggregator:
    """Aggregates ticks into bars"""

    def __init__(self, bar_size: str):
        self.bar_size = bar_size
        self.current_bar: Optional[Dict] = None
        self.bar_start_time: Optional[datetime] = None
        self._parse_bar_size()

    def _parse_bar_size(self):
        """Parse bar size string to timedelta"""
        # Parse strings like "1 hour", "15 mins", "1 day"
        parts = self.bar_size.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid bar size format: {self.bar_size}")

        value = int(parts[0])
        unit = parts[1].lower()

        if unit in ["min", "mins", "minute", "minutes"]:
            self.bar_duration = timedelta(minutes=value)
        elif unit in ["hour", "hours"]:
            self.bar_duration = timedelta(hours=value)
        elif unit in ["day", "days"]:
            self.bar_duration = timedelta(days=value)
        elif unit in ["week", "weeks"]:
            self.bar_duration = timedelta(weeks=value)
        else:
            raise ValueError(f"Unknown bar size unit: {unit}")

    def add_tick(self, price: float, size: float, timestamp: datetime) -> Optional[Dict]:
        """
        Add a tick and return completed bar if bar is complete.

        Returns:
            Completed bar dict or None if bar not yet complete
        """
        # Determine bar start time (round down to bar boundary)
        bar_start = self._get_bar_start_time(timestamp)

        # If new bar started, finalize previous bar
        if self.current_bar is not None and bar_start != self.bar_start_time:
            completed_bar = self.current_bar.copy()
            completed_bar["timestamp"] = self.bar_start_time
            completed_bar["close"] = self.current_bar["close"]  # Last price
            completed_bar["volume"] = self.current_bar["volume"]

            # Validate OHLC consistency before returning
            if not self._validate_ohlc(completed_bar):
                logger.error(f"[BarAggregator] Invalid OHLC data detected! Bar: {completed_bar}")
                # Don't return invalid bar - discard it
                self._start_new_bar(bar_start, price, size)
                return None

            # Start new bar
            self._start_new_bar(bar_start, price, size)
            return completed_bar

        # Start first bar if needed
        if self.current_bar is None:
            self._start_new_bar(bar_start, price, size)
            return None

        # Update current bar
        self.current_bar["high"] = max(self.current_bar["high"], price)
        self.current_bar["low"] = min(self.current_bar["low"], price)
        self.current_bar["close"] = price
        self.current_bar["volume"] += size

        return None

    def _validate_ohlc(self, bar: Dict) -> bool:
        """
        Validate OHLC data consistency.
        Returns True if valid, False if invalid.
        """
        o, h, l, c = bar.get("open", 0), bar.get("high", 0), bar.get("low", 0), bar.get("close", 0)

        # Basic OHLC rules:
        # 1. High must be >= Open, Close, Low
        # 2. Low must be <= Open, Close, High
        # 3. All values must be positive
        # 4. Range shouldn't be unreasonably large (>10% is suspicious for forex)

        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            logger.warning(f"[BarAggregator] Invalid: non-positive price detected")
            return False

        if h < o or h < c or h < l:
            logger.warning(f"[BarAggregator] Invalid: high ({h}) is not the highest value")
            return False

        if l > o or l > c or l > h:
            logger.warning(f"[BarAggregator] Invalid: low ({l}) is not the lowest value")
            return False

        # Check for unreasonably large range (>10% of the price)
        if l > 0:
            range_pct = (h - l) / l * 100
            if range_pct > 10:
                logger.warning(f"[BarAggregator] Suspicious: bar range is {range_pct:.2f}% (o={o}, h={h}, l={l}, c={c})")
                # Still allow it but log warning - might be legitimate during high volatility

        return True

    def _get_bar_start_time(self, timestamp: datetime) -> datetime:
        """Get the start time of the bar containing this timestamp"""
        if self.bar_duration >= timedelta(days=1):
            # For daily/weekly bars, use date boundaries
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # For intraday bars, round down to bar boundary
            total_seconds = int(self.bar_duration.total_seconds())
            timestamp_seconds = int(timestamp.timestamp())
            bar_start_seconds = (timestamp_seconds // total_seconds) * total_seconds
            return datetime.fromtimestamp(bar_start_seconds)

    def _start_new_bar(self, bar_start: datetime, first_price: float, first_size: float):
        """Start a new bar"""
        self.bar_start_time = bar_start
        self.current_bar = {
            "open": first_price,
            "high": first_price,
            "low": first_price,
            "close": first_price,
            "volume": first_size
        }


class DataManager:
    """Manages real-time data collection and bar aggregation"""

    # Staleness threshold - if no ticks for this long, consider data stale
    STALENESS_THRESHOLD_SECONDS = 120  # 2 minutes

    def __init__(self, broker_adapter, indicator_calculator=None):
        self.broker = broker_adapter
        self.indicator_calculator = indicator_calculator

        # Per-operation data buffers: operation_id -> bar_size -> list of bars
        self.data_buffers: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

        # Bar aggregators: operation_id -> bar_size -> BarAggregator
        self.aggregators: Dict[str, Dict[str, BarAggregator]] = defaultdict(dict)

        # Callbacks: operation_id -> bar_size -> callback
        self.bar_callbacks: Dict[str, Dict[str, Callable]] = defaultdict(dict)

        # Track last tick time per operation for staleness detection
        self.last_tick_time: Dict[str, datetime] = {}

        # Track tick counts for monitoring
        self.tick_counts: Dict[str, int] = defaultdict(int)

    def register_operation(
        self,
        operation_id: str,
        bar_sizes: list[str],
        callback: Optional[Callable] = None
    ):
        """Register a trading operation for data collection"""
        for bar_size in bar_sizes:
            self.aggregators[operation_id][bar_size] = BarAggregator(bar_size)
            if callback:
                self.bar_callbacks[operation_id][bar_size] = callback

        short_id = str(operation_id)[-6:]
        logger.info(f"[DataManager] [OP:{short_id}] Registered for bar sizes: {bar_sizes}")

    def unregister_operation(self, operation_id: str):
        """Unregister a trading operation"""
        if operation_id in self.aggregators:
            del self.aggregators[operation_id]
        if operation_id in self.data_buffers:
            del self.data_buffers[operation_id]
        if operation_id in self.bar_callbacks:
            del self.bar_callbacks[operation_id]
        short_id = str(operation_id)[-6:]
        logger.info(f"[DataManager] [OP:{short_id}] Unregistered")

    async def handle_tick(
        self,
        operation_id: str,
        asset: str,
        price: float,
        size: float,
        timestamp: datetime
    ):
        """Handle incoming tick data"""
        # Track last tick time and count for staleness monitoring
        now = datetime.utcnow()
        self.last_tick_time[operation_id] = now
        self.tick_counts[operation_id] += 1

        # Log periodically (every 100 ticks)
        if self.tick_counts[operation_id] % 100 == 0:
            short_id = str(operation_id)[-6:]
            logger.debug(f"[DataManager] [OP:{short_id} {asset}] {self.tick_counts[operation_id]} ticks received")

        # Process tick through all bar aggregators for this operation
        for bar_size, aggregator in self.aggregators[operation_id].items():
            completed_bar = aggregator.add_tick(price, size, timestamp)

            if completed_bar:
                # Bar is complete, process it
                await self._process_completed_bar(operation_id, asset, bar_size, completed_bar)

    def _is_ohlc_valid(self, bar_data: Dict) -> bool:
        """Validate OHLC data consistency"""
        o = bar_data.get("open", 0)
        h = bar_data.get("high", 0)
        l = bar_data.get("low", 0)
        c = bar_data.get("close", 0)

        # All values must be positive
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return False

        # High must be >= all others, Low must be <= all others
        if h < o or h < c or h < l:
            return False
        if l > o or l > c or l > h:
            return False

        # Check for unreasonable range (>5% is very suspicious for forex in a single bar)
        if l > 0:
            range_pct = (h - l) / l * 100
            if range_pct > 5:
                logger.warning(f"[DataManager] Large bar range: {range_pct:.2f}% (o={o:.5f}, h={h:.5f}, l={l:.5f}, c={c:.5f})")
                if range_pct > 20:  # >20% is definitely wrong
                    return False

        return True

    async def _process_completed_bar(
        self,
        operation_id: str,
        asset: str,
        bar_size: str,
        bar_data: Dict
    ):
        """Process a completed bar"""
        bar_timestamp = bar_data.get("timestamp")
        short_id = str(operation_id)[-6:]
        op_tag = f"[OP:{short_id} {asset}]"

        # Validate incoming bar data
        if not self._is_ohlc_valid(bar_data):
            logger.error(
                f"[DataManager] {op_tag} REJECTING invalid bar {bar_size} at {bar_timestamp}: "
                f"o={bar_data.get('open')}, h={bar_data.get('high')}, "
                f"l={bar_data.get('low')}, c={bar_data.get('close')}, v={bar_data.get('volume')}"
            )
            return  # Don't process invalid data

        # Add to buffer with deduplication
        # Check if a bar with the same timestamp already exists
        existing_bars = self.data_buffers[operation_id][bar_size]

        # Find and update existing bar if present, otherwise append
        found_existing = False
        for i, existing_bar in enumerate(existing_bars):
            if existing_bar.get("timestamp") == bar_timestamp:
                # Check if existing bar has volume (historical data)
                existing_has_volume = existing_bar.get("volume", 0) > 0
                new_has_volume = bar_data.get("volume", 0) > 0

                if existing_has_volume and not new_has_volume:
                    # Existing bar is from historical data (has volume), new is real-time (no volume)
                    # Only update the close price, don't touch OHLC
                    logger.debug(
                        f"[DataManager] Updating only close for historical bar at {bar_timestamp}: "
                        f"close {existing_bar.get('close'):.5f} -> {bar_data['close']:.5f}"
                    )
                    existing_bar["close"] = bar_data["close"]
                else:
                    # Both are real-time or both are historical - merge normally
                    existing_bar["high"] = max(existing_bar.get("high", bar_data["high"]), bar_data["high"])
                    existing_bar["low"] = min(existing_bar.get("low", bar_data["low"]), bar_data["low"])
                    existing_bar["close"] = bar_data["close"]
                    if new_has_volume:
                        existing_bar["volume"] = bar_data.get("volume", 0)

                found_existing = True
                logger.debug(f"[DataManager] {op_tag} Updated bar in buffer: {bar_size} @ {bar_timestamp}")
                break

        if not found_existing:
            existing_bars.append(bar_data)
            logger.debug(f"[DataManager] {op_tag} New bar in buffer: {bar_size} @ {bar_timestamp}")

        # Apply data retention limit
        operation = await TradingOperation.get(operation_id)
        retention_bars = operation.data_retention_bars if operation else config.DEFAULT_DATA_RETENTION_BARS

        if len(self.data_buffers[operation_id][bar_size]) > retention_bars:
            # Remove oldest bars
            excess = len(self.data_buffers[operation_id][bar_size]) - retention_bars
            self.data_buffers[operation_id][bar_size] = self.data_buffers[operation_id][bar_size][excess:]

        # Calculate indicators if calculator available
        indicators = {}
        if self.indicator_calculator:
            df = self._buffer_to_dataframe(operation_id, bar_size)
            if len(df) > 0:
                try:
                    # TechnicalIndicators expects a DataFrame and returns it with indicators added
                    df_with_indicators = self.indicator_calculator.execute(df)
                    # Extract indicators as dict (or use the DataFrame columns)
                    # For now, store the indicator values from the last row
                    if len(df_with_indicators) > 0:
                        last_row = df_with_indicators.iloc[-1]
                        # Extract indicator columns (exclude OHLCV and timestamp)
                        indicator_cols = [col for col in df_with_indicators.columns
                                         if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
                        indicators = {col: last_row[col] for col in indicator_cols if pd.notna(last_row.get(col))}

                        # Update the last bar in buffer with indicators so they're available in get_dataframe
                        if self.data_buffers[operation_id][bar_size]:
                            last_bar = self.data_buffers[operation_id][bar_size][-1]
                            # Add indicators to the bar data in buffer
                            if 'indicators' not in last_bar:
                                last_bar['indicators'] = {}
                            last_bar['indicators'].update(indicators)
                            # Also add indicators as direct columns for easier DataFrame conversion
                            for indicator_name, indicator_value in indicators.items():
                                last_bar[indicator_name] = indicator_value
                except Exception as e:
                    logger.error(f"[DataManager] {op_tag} Error calculating indicators for {bar_size}: {e}")

        # Store in database using upsert to prevent duplicates
        # If a bar already exists for this timestamp (e.g., from historical data), update it
        try:
            from bson import ObjectId

            # Use find_one + update with upsert to handle duplicates gracefully
            existing = await MarketData.find_one(
                MarketData.operation_id == ObjectId(operation_id),
                MarketData.bar_size == bar_size,
                MarketData.timestamp == bar_data["timestamp"]
            )

            if existing:
                # Check if existing bar is from historical data (has volume)
                existing_has_volume = existing.volume > 0
                new_has_volume = bar_data.get("volume", 0) > 0

                if existing_has_volume and not new_has_volume:
                    # Existing is historical (has volume), new is real-time (no volume)
                    # Only update close price and indicators, preserve OHLC
                    logger.debug(
                        f"[DB] Updating only close for historical bar at {bar_data['timestamp']}: "
                        f"close {existing.close:.5f} -> {bar_data['close']:.5f}"
                    )
                    existing.close = bar_data["close"]
                    # Merge indicators (keep existing, add new)
                    if indicators:
                        merged_indicators = {**existing.indicators, **indicators}
                        existing.indicators = merged_indicators
                    await existing.save()
                else:
                    # Both are real-time or new has volume - update OHLC carefully
                    existing.high = max(existing.high, bar_data["high"])
                    existing.low = min(existing.low, bar_data["low"])
                    existing.close = bar_data["close"]
                    if new_has_volume:
                        existing.volume = bar_data.get("volume", 0.0)
                    # Merge indicators
                    if indicators:
                        merged_indicators = {**existing.indicators, **indicators}
                        existing.indicators = merged_indicators
                    await existing.save()
                    logger.debug(f"[DataManager] {op_tag} Updated DB bar: {bar_size} @ {bar_data['timestamp']}")
            else:
                # Create new bar
                market_data = MarketData(
                    operation_id=ObjectId(operation_id),
                    bar_size=bar_size,
                    timestamp=bar_data["timestamp"],
                    open=bar_data["open"],
                    high=bar_data["high"],
                    low=bar_data["low"],
                    close=bar_data["close"],
                    volume=bar_data.get("volume", 0.0),
                    indicators=indicators
                )
                await market_data.insert()
                logger.debug(f"[DataManager] {op_tag} Stored new bar: {bar_size} @ {bar_data['timestamp']}")
        except Exception as e:
            logger.error(f"[DataManager] {op_tag} Error storing bar {bar_size}: {e}", exc_info=True)

        # Notify callback
        if operation_id in self.bar_callbacks and bar_size in self.bar_callbacks[operation_id]:
            callback = self.bar_callbacks[operation_id][bar_size]
            logger.debug(f"[DATA] {op_tag} Bar completed: {bar_size} @ {bar_data.get('timestamp')} close={bar_data.get('close')}")
            logger.debug(f"[DATA] {op_tag} Indicators: {list(indicators.keys()) if indicators else 'none'}")

            # Handle both sync and async callbacks
            # For bound methods, check the underlying function
            func = getattr(callback, '__func__', callback)
            if inspect.iscoroutinefunction(func):
                await callback(bar_data, indicators)
            else:
                # Try calling it - if it returns a coroutine, await it
                result = callback(bar_data, indicators)
                if asyncio.iscoroutine(result):
                    await result
        else:
            logger.debug(f"[DATA] {op_tag} No callback registered for {bar_size}")

    def _buffer_to_dataframe(self, operation_id: str, bar_size: str) -> pd.DataFrame:
        """Convert buffer to DataFrame for indicator calculation"""
        bars = self.data_buffers[operation_id][bar_size]
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)

        # Expand the 'indicators' dict column into separate columns if present
        # This handles cases where indicators are stored in a nested dict
        if 'indicators' in df.columns:
            # Get all unique indicator keys from the indicators dicts
            all_indicator_keys = set()
            for bar in bars:
                if isinstance(bar.get('indicators'), dict):
                    all_indicator_keys.update(bar['indicators'].keys())

            # Add indicator columns from the nested dict if not already present
            for indicator_key in all_indicator_keys:
                if indicator_key not in df.columns:
                    df[indicator_key] = df['indicators'].apply(
                        lambda x: x.get(indicator_key) if isinstance(x, dict) else np.nan
                    )

            # Drop the indicators dict column to keep DataFrame clean
            df = df.drop(columns=['indicators'], errors='ignore')

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    async def _store_bar(
        self,
        operation_id: str,
        bar_size: str,
        bar_data: Dict,
        indicators: Dict
    ):
        """Store bar in database"""
        market_data = MarketData(
            operation_id=operation_id,
            bar_size=bar_size,
            timestamp=bar_data["timestamp"],
            open=bar_data["open"],
            high=bar_data["high"],
            low=bar_data["low"],
            close=bar_data["close"],
            volume=bar_data.get("volume", 0.0),
            indicators=indicators
        )
        await market_data.insert()

    async def get_latest_bars(
        self,
        operation_id: str,
        bar_size: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, list]:
        """Get latest bars from buffer"""
        if bar_size:
            bars = self.data_buffers[operation_id].get(bar_size, [])
            return {bar_size: bars[-limit:]}
        else:
            return {
                size: bars[-limit:]
                for size, bars in self.data_buffers[operation_id].items()
            }

    async def get_dataframe(
        self,
        operation_id: str,
        bar_size: str
    ) -> pd.DataFrame:
        """Get DataFrame for a specific bar size"""
        return self._buffer_to_dataframe(operation_id, bar_size)

    def is_data_stale(self, operation_id: str) -> bool:
        """Check if data for an operation is stale (no ticks received recently)"""
        if operation_id not in self.last_tick_time:
            return True  # No ticks ever received

        last_tick = self.last_tick_time[operation_id]
        seconds_since_tick = (datetime.utcnow() - last_tick).total_seconds()
        return seconds_since_tick > self.STALENESS_THRESHOLD_SECONDS

    def get_staleness_info(self, operation_id: str) -> Dict:
        """Get staleness information for an operation"""
        if operation_id not in self.last_tick_time:
            return {
                "operation_id": operation_id,
                "is_stale": True,
                "last_tick_time": None,
                "seconds_since_tick": None,
                "tick_count": 0
            }

        last_tick = self.last_tick_time[operation_id]
        seconds_since_tick = (datetime.utcnow() - last_tick).total_seconds()

        return {
            "operation_id": operation_id,
            "is_stale": seconds_since_tick > self.STALENESS_THRESHOLD_SECONDS,
            "last_tick_time": last_tick.isoformat(),
            "seconds_since_tick": seconds_since_tick,
            "tick_count": self.tick_counts.get(operation_id, 0)
        }

    def get_all_staleness_info(self) -> list:
        """Get staleness info for all registered operations"""
        all_operations = set(self.aggregators.keys())
        return [self.get_staleness_info(op_id) for op_id in all_operations]
