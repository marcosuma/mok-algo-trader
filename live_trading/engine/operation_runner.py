"""
Operation Runner - Manages a single trading operation.
"""
import logging
import asyncio
from collections import deque
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from bson import ObjectId

from live_trading.data.data_manager import DataManager
from live_trading.strategies.strategy_adapter import StrategyAdapter
from live_trading.orders.order_manager import OrderManager
from live_trading.models.trading_operation import TradingOperation
from forex_strategies.strategy_registry import get_strategy

logger = logging.getLogger(__name__)


class OperationRunner:
    """Manages a single trading operation"""

    def __init__(
        self,
        operation_id: ObjectId,
        data_manager: DataManager,
        order_manager: OrderManager
    ):
        self.operation_id = operation_id
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.strategy_adapter: Optional[StrategyAdapter] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _build_op_tag(self, asset: str = "?", strategy: str = "?") -> str:
        """Build a short operation tag for log messages"""
        short_id = str(self.operation_id)[-6:]
        return f"[OP:{short_id} {asset} {strategy}]"

    async def start(self):
        """Start the operation"""
        operation = await TradingOperation.get(self.operation_id)
        if not operation:
            raise ValueError(f"Operation {self.operation_id} not found")

        if operation.status != "active":
            raise ValueError(f"Operation {self.operation_id} is not active")

        # Build operation tag for log messages
        self._op_tag = self._build_op_tag(operation.asset, operation.strategy_name)

        # Get strategy class
        strategy_class = get_strategy(operation.strategy_name)

        if strategy_class is None:
            raise ValueError(f"Strategy {operation.strategy_name} not found")

        # Create strategy adapter
        self.strategy_adapter = StrategyAdapter(
            strategy_class=strategy_class,
            strategy_config=operation.strategy_config,
            data_manager=self.data_manager,
            operation_id=str(self.operation_id),
            bar_sizes=operation.bar_sizes,
            primary_bar_size=operation.primary_bar_size,
            asset=operation.asset
        )

        # Set signal callback
        self.strategy_adapter.set_signal_callback(self._handle_signal)

        # Register with data manager
        self.data_manager.register_operation(
            operation_id=str(self.operation_id),
            bar_sizes=operation.bar_sizes,
            callback=self._on_new_bar
        )

        # Store the event loop for use in the callback
        # This is the event loop from the async context where start() is called
        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # If we're not in an async context, try to get the current loop
            try:
                self.event_loop = asyncio.get_event_loop()
            except RuntimeError:
                logger.warning("No event loop available - tick processing may not work")
                self.event_loop = None

        # Fetch historical data before subscribing to real-time data
        # This ensures we have context for strategy decisions
        if self.data_manager.broker and hasattr(self.data_manager.broker, 'fetch_historical_data'):
            await self._fetch_historical_data(operation)

            # Check for gaps in market data and fill them
            await self._fill_data_gaps(operation)

        # Subscribe to market data from broker
        # Ticks arrive on the broker thread and are batched into a deque.
        # A single drain coroutine is scheduled on the event loop to process
        # each batch, reducing scheduling overhead from ~50/sec to ~10/sec.
        TICK_THROTTLE_SECONDS = 0.1  # 100 ms batching window

        op_tag = self._op_tag
        op_id_str = str(self.operation_id)
        op_asset = operation.asset
        tick_buffer: deque = deque()
        drain_pending = [False]  # mutable flag for the closure

        async def _drain_tick_buffer():
            """Wait briefly to accumulate ticks, then process the batch."""
            await asyncio.sleep(TICK_THROTTLE_SECONDS)

            batch = []
            while tick_buffer:
                batch.append(tick_buffer.popleft())
            drain_pending[0] = False

            for price, size, ts in batch:
                await self.data_manager.handle_tick(
                    operation_id=op_id_str,
                    asset=op_asset,
                    price=price,
                    size=size,
                    timestamp=ts,
                )

            if tick_buffer and not drain_pending[0]:
                drain_pending[0] = True
                asyncio.ensure_future(_drain_tick_buffer())

        def broker_tick_callback(tick_data: Dict):
            """Callback to route broker ticks to data manager (throttled)."""
            if tick_data.get("type") == "tick":
                price = tick_data.get("price")
                size = tick_data.get("size", 0.0)
                timestamp = tick_data.get("timestamp", datetime.utcnow())

                tick_count = getattr(broker_tick_callback, '_tick_count', 0) + 1
                broker_tick_callback._tick_count = tick_count
                if tick_count == 1:
                    logger.info(f"[TICK] {op_tag} First tick: price={price}, size={size}")
                elif tick_count % 500 == 0:
                    logger.info(f"[TICK] {op_tag} Tick #{tick_count}: price={price}")

                if price is not None and self.event_loop is not None:
                    tick_buffer.append((price, size, timestamp))
                    if not drain_pending[0]:
                        drain_pending[0] = True
                        try:
                            asyncio.run_coroutine_threadsafe(
                                _drain_tick_buffer(), self.event_loop
                            )
                        except Exception as e:
                            drain_pending[0] = False
                            logger.error(f"[TICK] {op_tag} Error scheduling tick drain: {e}", exc_info=True)
                elif price is not None:
                    logger.warning(f"[TICK] {op_tag} No event loop available (price: {price})")

        # Get broker from data manager and subscribe
        # Use operation_id as callback_id to allow multiple operations on same asset
        # Pass bar_sizes to enable live trendbar subscriptions (for volume data)
        if self.data_manager.broker:
            callback_id = f"op_{self.operation_id}"
            subscribe_kwargs = {
                "asset": operation.asset,
                "callback": broker_tick_callback,
                "callback_id": callback_id,
            }
            # Pass bar_sizes if the broker supports it (cTrader uses them for live trendbar volume)
            if hasattr(self.data_manager.broker, 'subscribe_live_trendbars'):
                subscribe_kwargs["bar_sizes"] = operation.bar_sizes
            subscribed = await self.data_manager.broker.subscribe_market_data(**subscribe_kwargs)
            if subscribed:
                logger.info(f"[DATA] {self._op_tag} Subscribed to market data (bar_sizes: {operation.bar_sizes})")
            else:
                logger.warning(f"[DATA] {self._op_tag} Failed to subscribe to market data")
        else:
            logger.warning(f"[DATA] {self._op_tag} Broker not available - cannot subscribe to market data")

        self.running = True
        logger.info(f"{self._op_tag} Operation started")

    async def stop(self):
        """Stop the operation"""
        self.running = False

        # Unsubscribe from market data with our specific callback_id
        operation = await TradingOperation.get(self.operation_id)
        if operation and self.data_manager.broker:
            callback_id = f"op_{self.operation_id}"
            try:
                await self.data_manager.broker.unsubscribe_market_data(
                    asset=operation.asset,
                    callback_id=callback_id
                )
                tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
                logger.info(f"{tag} Unsubscribed from market data")
            except Exception as e:
                tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
                logger.error(f"{tag} Error unsubscribing from market data: {e}")

        # Unregister from data manager
        self.data_manager.unregister_operation(str(self.operation_id))

        # Cancel task if running
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
        logger.info(f"{tag} Operation stopped")

    async def pause(self):
        """Pause the operation"""
        operation = await TradingOperation.get(self.operation_id)
        if operation:
            operation.status = "paused"
            await operation.save()
            tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
            logger.info(f"{tag} Operation paused")

    async def resume(self):
        """Resume the operation"""
        operation = await TradingOperation.get(self.operation_id)
        if operation:
            operation.status = "active"
            await operation.save()
            tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
            logger.info(f"{tag} Operation resumed")

    async def _on_new_bar(self, bar_data: Dict, indicators: Dict):
        """Callback when a new bar is completed"""
        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
        if not self.running:
            logger.debug(f"[DATA] {tag} Not running, ignoring bar")
            return

        operation = await TradingOperation.get(self.operation_id)
        if operation and self.strategy_adapter:
            logger.info(f"[DATA] {tag} New bar @ {bar_data.get('timestamp')} close={bar_data.get('close')}, passing to strategy")
            try:
                await self.strategy_adapter.on_new_bar(
                    operation.primary_bar_size,
                    bar_data,
                    indicators
                )
            except Exception as e:
                logger.warning(f"[DATA] {tag} ❌ Error in strategy adapter on_new_bar: {type(e).__name__}: {e}")
                logger.error(f"[DATA] {tag} ❌ Strategy adapter exception details:", exc_info=True)
        else:
            logger.warning(f"[DATA] {tag} No operation or strategy adapter")

    async def _check_existing_historical_data(self, bar_size: str, min_bars: int = 100) -> bool:
        """Check if sufficient historical data already exists in database"""
        from live_trading.models.market_data import MarketData
        from bson import ObjectId

        count = await MarketData.find(
            MarketData.operation_id == ObjectId(str(self.operation_id)),
            MarketData.bar_size == bar_size
        ).count()

        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
        has_sufficient_data = count >= min_bars
        if has_sufficient_data:
            logger.info(f"{tag} Found {count} existing bars for {bar_size} - loading from database")
        else:
            logger.info(f"{tag} Found only {count} bars for {bar_size} - will fetch from broker")

        return has_sufficient_data

    async def _load_historical_data_from_db(self, bar_size: str) -> Tuple[list, dict]:
        """Load historical data from database

        Returns:
            tuple: (bars_list, indicators_dict) where bars_list is in IBKR format
                   and indicators_dict maps timestamp -> indicators dict
        """
        from live_trading.models.market_data import MarketData
        from bson import ObjectId
        import pandas as pd

        # Fetch all bars for this operation and bar_size, sorted by timestamp
        market_data_list = await MarketData.find(
            MarketData.operation_id == ObjectId(str(self.operation_id)),
            MarketData.bar_size == bar_size
        ).sort(+MarketData.timestamp).to_list()

        # Convert to list of dicts (same format as IBKR callback)
        bars = []
        indicators_dict = {}  # timestamp -> indicators dict

        for md in market_data_list:
            bars.append({
                "date": md.timestamp,
                "open": md.open,
                "high": md.high,
                "low": md.low,
                "close": md.close,
                "volume": md.volume
            })
            # Store indicators separately
            indicators_dict[md.timestamp] = md.indicators or {}

        # Log indicator statistics
        bars_with_indicators = sum(1 for v in indicators_dict.values() if v and len(v) > 0)
        if indicators_dict:
            sample_indicators = next((v for v in indicators_dict.values() if v), {})
            indicator_keys = list(sample_indicators.keys())[:10]  # First 10 indicator names
        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
        if indicators_dict:
            logger.info(f"{tag} Loaded {len(bars)} bars from DB for {bar_size}, "
                       f"{bars_with_indicators} have indicators. Sample: {indicator_keys}")
        else:
            logger.warning(f"{tag} Loaded {len(bars)} bars from DB for {bar_size} - NO indicators found!")
        return bars, indicators_dict

    def _calculate_timeout(self, interval: str) -> int:
        """Calculate timeout based on interval size"""
        # Base timeout: 5 minutes
        base_timeout = 300

        # Parse interval (e.g., "1 Y", "6 M", "1 W")
        interval_upper = interval.upper()

        if "Y" in interval_upper:
            # Years - large data, need more time
            years = int(interval_upper.split()[0]) if interval_upper.split()[0].isdigit() else 1
            timeout = base_timeout * (2 + years)  # 2-4 minutes per year
        elif "M" in interval_upper:
            # Months
            months = int(interval_upper.split()[0]) if interval_upper.split()[0].isdigit() else 6
            timeout = base_timeout + (months * 10)  # Add 10 seconds per month
        elif "W" in interval_upper:
            # Weeks
            timeout = base_timeout + 60
        elif "D" in interval_upper:
            # Days
            timeout = base_timeout + 30
        else:
            timeout = base_timeout

        # Cap at 10 minutes maximum
        return min(timeout, 600)

    # Bar size interval limits (originally from data_manager.data_downloader.DataDownloader)
    # Duplicated here to avoid importing ibapi-dependent module
    BAR_SIZE_INTERVAL_LIMITS = {
        "1 min": "1 M",
        "5 mins": "1 Y",
        "15 mins": "1 Y",
        "1 hour": "1 Y",
        "4 hours": "1 Y",
        "1 day": "10 Y",
        "1 week": "10 Y",
    }

    async def _fetch_historical_data(self, operation: TradingOperation):
        """Fetch historical data for all bar sizes before starting real-time collection"""
        from datetime import datetime
        import pandas as pd
        from bson import ObjectId
        from live_trading.models.market_data import MarketData

        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")

        if not self.data_manager.broker:
            logger.warning(f"{tag} Broker not available - cannot fetch historical data")
            return

        # Verify broker has fetch_historical_data method
        if not hasattr(self.data_manager.broker, 'fetch_historical_data'):
            logger.warning(f"{tag} Broker {type(self.data_manager.broker).__name__} does not support fetch_historical_data")
            return

        logger.info(f"{tag} Broker: {type(self.data_manager.broker).__name__}, connected={getattr(self.data_manager.broker, 'connected', 'N/A')}, authenticated={getattr(self.data_manager.broker, 'authenticated', 'N/A')}")

        # Default interval: 6 months (can be made configurable)
        default_interval = "6 M"

        # Get interval limits
        interval_limits = self.BAR_SIZE_INTERVAL_LIMITS

        logger.info(f"{tag} Fetching historical data...")

        # Track pending historical data requests
        # Use asyncio.Event for each bar size to signal completion
        pending_requests = {}
        completed_data = {}  # bar_size -> bars
        completed_indicators = {}  # bar_size -> {timestamp: indicators_dict}
        completion_events = {}  # bar_size -> asyncio.Event

        def historical_data_callback(bars: list, context: dict):
            """Callback when historical data is received (runs in IBKR thread)"""
            bar_size = context.get("bar_size")
            operation_id = context.get("operation_id")

            if not bar_size or not operation_id:
                logger.error("Missing context in historical data callback")
                return

            logger.info(f"{tag} Received {len(bars)} historical bars for {bar_size}")

            # Store data
            completed_data[bar_size] = bars

            # Signal completion in main event loop
            if self.event_loop and bar_size in completion_events:
                event = completion_events[bar_size]
                # Schedule event.set() in main event loop
                asyncio.run_coroutine_threadsafe(
                    self._signal_historical_data_complete(bar_size, event),
                    self.event_loop
                )

        # Request historical data for each bar size
        for bar_size in operation.bar_sizes:
            # Determine interval based on bar size
            interval = interval_limits.get(bar_size, default_interval)

            # Check if data already exists in database
            # Use a reasonable minimum (e.g., 100 bars) to determine if we have sufficient data
            has_existing_data = await self._check_existing_historical_data(bar_size, min_bars=100)

            if has_existing_data:
                # Load from database instead of fetching from broker
                try:
                    bars, indicators = await self._load_historical_data_from_db(bar_size)
                    if bars:
                        completed_data[bar_size] = bars
                        completed_indicators[bar_size] = indicators
                        logger.info(f"{tag} Loaded historical data from database for {bar_size}")
                        continue  # Skip broker request for this bar_size
                except Exception as e:
                    logger.error(f"{tag} Error loading historical data from database for {bar_size}: {e}")
                    logger.info(f"{tag} Will fetch from broker instead")
                    # Fall through to fetch from broker

            # Create completion event for this bar size
            completion_events[bar_size] = asyncio.Event()
            pending_requests[bar_size] = True

            # Request historical data from broker
            context = {
                "bar_size": bar_size,
                "operation_id": str(self.operation_id),
                "asset": operation.asset
            }

            try:
                logger.info(f"{tag} Fetching from broker: {bar_size} ({interval})")
                requested = await self.data_manager.broker.fetch_historical_data(
                    asset=operation.asset,
                    bar_size=bar_size,
                    interval=interval,
                    callback=historical_data_callback,
                    context=context
                )
                logger.info(f"{tag} fetch_historical_data returned: {requested} (type: {type(requested).__name__})")
            except Exception as e:
                logger.error(f"{tag} Exception fetching historical data for {bar_size}: {e}", exc_info=True)
                requested = False

            if requested is True:
                logger.info(f"{tag} Requested historical data: {bar_size} ({interval})")
            else:
                logger.warning(f"{tag} Failed to request historical data for {bar_size} (requested={requested})")
                # Remove from pending if request failed
                if bar_size in pending_requests:
                    del pending_requests[bar_size]

        # Wait for all historical data requests to complete
        # IBKR API is asynchronous, so we need to wait for callbacks
        # Calculate timeout based on the largest interval requested
        if pending_requests:
            max_interval = max(
                [interval_limits.get(bs, default_interval) for bs in operation.bar_sizes if bs in pending_requests],
                default=default_interval
            )
            max_wait = self._calculate_timeout(max_interval)
            logger.info(f"{tag} Timeout: {max_wait}s for historical data (interval: {max_interval})")
        else:
            logger.info(f"{tag} No pending broker requests - all data loaded from database")

        wait_tasks = []

        for bar_size, event in completion_events.items():
            if bar_size in pending_requests:
                # Calculate timeout for this specific bar_size
                interval = interval_limits.get(bar_size, default_interval)
                timeout = self._calculate_timeout(interval)

                # Create a task that waits for this event with timeout
                # Use default argument to capture loop variables correctly
                async def wait_for_data(bs=bar_size, evt=event, to=timeout):
                    try:
                        await asyncio.wait_for(evt.wait(), timeout=to)
                        logger.info(f"{tag} Historical data received for {bs}")
                    except asyncio.TimeoutError:
                        logger.warning(f"{tag} Timeout waiting for historical data for {bs} (timeout: {to}s)")

                wait_tasks.append(wait_for_data())

        # Wait for all requests (or timeout)
        if wait_tasks:
            await asyncio.gather(*wait_tasks, return_exceptions=True)

        # Process historical data (from database or broker)
        for bar_size, bars in completed_data.items():
            if not bars:
                logger.warning(f"{tag} No historical data received for {bar_size}")
                continue

            # Check if data came from database (already has indicators) or from broker (needs processing)
            data_from_db = bar_size in completed_indicators
            indicators_map = completed_indicators.get(bar_size, {})

            try:
                # Convert bars to DataFrame for indicator calculation
                df = pd.DataFrame(bars)

                # Convert date from IBKR format (unix timestamp string) to datetime
                # IBKR returns date as string like "20240101 12:00:00" or unix timestamp
                # Database data already has datetime objects
                if df["date"].dtype == 'object':
                    # Try to parse as datetime string first
                    try:
                        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M:%S")
                    except:
                        # If that fails, try unix timestamp
                        try:
                            df["date"] = pd.to_datetime(df["date"].astype(int), unit='s')
                        except:
                            # If that fails, try parsing as datetime directly (for DB data)
                            try:
                                df["date"] = pd.to_datetime(df["date"])
                            except:
                                logger.warning(f"Could not parse date format for {bar_size}")
                                continue
                else:
                    # Already numeric, assume unix timestamp
                    df["date"] = pd.to_datetime(df["date"], unit='s')

                df.set_index("date", inplace=True)
                df = df.sort_index()

                if data_from_db:
                    # Data already in database - just populate buffers
                    logger.info(f"{tag} {bar_size} data from DB - populating buffers")
                    df_with_indicators = df.copy()
                    # Add indicators from database to DataFrame
                    # Convert indicators_map to a DataFrame and join (much faster than row-by-row)
                    if indicators_map:
                        indicators_map_pd = {pd.Timestamp(k): v for k, v in indicators_map.items()}
                        indicators_df = pd.DataFrame.from_dict(indicators_map_pd, orient='index')
                        # Join on index (timestamp) - only matching rows get indicator values
                        indicators_matched = len(indicators_df.index.intersection(df_with_indicators.index))
                        df_with_indicators = df_with_indicators.join(indicators_df, how='left')
                        logger.info(f"{tag} Matched indicators for {indicators_matched}/{len(df)} bars ({bar_size})")
                    else:
                        logger.info(f"{tag} No indicators to match for {bar_size}")
                else:
                    # Data from broker - calculate indicators and store in database
                    df_with_indicators = df.copy()
                    if self.data_manager.indicator_calculator:
                        try:
                            df_with_indicators = self.data_manager.indicator_calculator.execute(df.copy())
                            logger.info(f"{tag} Calculated indicators for {len(df_with_indicators)} bars ({bar_size})")
                        except Exception as e:
                            logger.error(f"{tag} Error calculating indicators ({bar_size}): {e}")

                    # Store bars in database with indicators using batch insert
                    # Process in batches to avoid memory issues and show progress
                    batch_size = 1000  # Insert 1000 bars at a time
                    total_bars = len(df_with_indicators)
                    stored_count = 0

                    logger.info(f"{tag} Storing {total_bars} bars for {bar_size} (batch_size={batch_size})")

                    # Get indicator columns once
                    indicator_cols = [col for col in df_with_indicators.columns
                                    if col not in ['open', 'high', 'low', 'close', 'volume']]

                    # Process in batches
                    for batch_start in range(0, total_bars, batch_size):
                        batch_end = min(batch_start + batch_size, total_bars)
                        batch_data = []

                        # Prepare batch of MarketData objects
                        for idx in range(batch_start, batch_end):
                            try:
                                timestamp = df_with_indicators.index[idx]
                                row = df_with_indicators.iloc[idx]

                                # Extract indicators for this bar
                                indicators_dict = {}
                                for col in indicator_cols:
                                    if pd.notna(row.get(col)):
                                        value = row[col]
                                        # Handle both numeric and string indicator values
                                        # Some indicators like local_extrema return strings ('LOCAL_MAX', 'LOCAL_MIN')
                                        if isinstance(value, (int, float, bool)):
                                            # Numeric or boolean values - convert to float (bool becomes 0.0 or 1.0)
                                            indicators_dict[col] = float(value)
                                        elif isinstance(value, str):
                                            # String values - try to convert to float if numeric, otherwise keep as string
                                            try:
                                                # Try to convert string to float (handles numeric strings)
                                                indicators_dict[col] = float(value)
                                            except (ValueError, TypeError):
                                                # If conversion fails, store as string (e.g., 'LOCAL_MAX', 'LOCAL_MIN')
                                                indicators_dict[col] = value
                                        else:
                                            # Other types (e.g., None, NaN) - convert to string or skip
                                            if value is not None:
                                                indicators_dict[col] = str(value)

                                # Convert timestamp to datetime if needed
                                if hasattr(timestamp, 'to_pydatetime'):
                                    bar_timestamp = timestamp.to_pydatetime()
                                else:
                                    bar_timestamp = pd.to_datetime(timestamp)

                                market_data = MarketData(
                                    operation_id=ObjectId(str(self.operation_id)),
                                    bar_size=bar_size,
                                    timestamp=bar_timestamp,
                                    open=float(row["open"]),
                                    high=float(row["high"]),
                                    low=float(row["low"]),
                                    close=float(row["close"]),
                                    volume=float(row.get("volume", 0.0)),
                                    indicators=indicators_dict
                                )
                                batch_data.append(market_data)
                            except Exception as e:
                                logger.error(f"{tag} Error preparing historical bar {idx} for {bar_size}: {e}")

                        # Batch insert
                        if batch_data:
                            try:
                                await MarketData.insert_many(batch_data)
                                stored_count += len(batch_data)
                                logger.info(f"{tag} Stored batch {batch_start//batch_size + 1} ({batch_end}/{total_bars} bars) for {bar_size}")
                            except Exception as e:
                                logger.error(f"{tag} Error batch inserting bars {batch_start}-{batch_end} for {bar_size}: {e}")
                                # Fallback to individual inserts for this batch
                                for md in batch_data:
                                    try:
                                        await md.insert()
                                        stored_count += 1
                                    except Exception as e2:
                                        logger.error(f"{tag} Error storing individual bar for {bar_size}: {e2}")

                    logger.info(f"{tag} Completed storing {stored_count}/{total_bars} bars for {bar_size}")

                # Add to data manager buffer so strategy can use it immediately
                # Use the DataFrame with indicators
                existing_bars = self.data_manager.data_buffers[str(self.operation_id)][bar_size]
                existing_timestamps = {
                    existing.get("timestamp") for existing in existing_bars
                }

                # Get indicator columns once (outside the loop)
                indicator_cols = [col for col in df_with_indicators.columns
                                 if col not in ['open', 'high', 'low', 'close', 'volume']]

                duplicates_skipped = 0
                bars_added = 0

                for timestamp, row in df_with_indicators.iterrows():
                    # Convert timestamp to datetime if needed
                    if hasattr(timestamp, 'to_pydatetime'):
                        bar_timestamp = timestamp.to_pydatetime()
                    else:
                        bar_timestamp = pd.to_datetime(timestamp)

                    # Fast duplicate check using set lookup
                    if bar_timestamp in existing_timestamps:
                        duplicates_skipped += 1
                        continue

                    bar_data = {
                        "timestamp": bar_timestamp,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0.0))
                    }

                    # Add indicators to bar_data so they're available in get_dataframe
                    indicators_dict = {}
                    for col in indicator_cols:
                        if col in row and pd.notna(row[col]):
                            value = row[col]
                            # Store as float if numeric, otherwise as string
                            if isinstance(value, (int, float, bool)):
                                indicators_dict[col] = float(value)
                            elif isinstance(value, str):
                                try:
                                    indicators_dict[col] = float(value)
                                except (ValueError, TypeError):
                                    indicators_dict[col] = value
                            else:
                                indicators_dict[col] = value
                            # Also add as direct column for easier DataFrame conversion
                            bar_data[col] = indicators_dict[col]

                    bar_data["indicators"] = indicators_dict
                    existing_bars.append(bar_data)
                    existing_timestamps.add(bar_timestamp)
                    bars_added += 1

                logger.info(f"{tag} Buffer populated for {bar_size}: {bars_added} added, {duplicates_skipped} duplicates skipped")

            except Exception as e:
                logger.error(f"{tag} Error processing historical data for {bar_size}: {e}", exc_info=True)

        logger.info(f"{tag} Historical data fetch complete")

    async def _signal_historical_data_complete(self, bar_size: str, event: asyncio.Event):
        """Helper to signal historical data completion (runs in main event loop)"""
        event.set()

    def _parse_bar_size_to_duration(self, bar_size: str) -> timedelta:
        """Parse bar size string to timedelta duration"""
        parts = bar_size.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid bar size format: {bar_size}")

        value = int(parts[0])
        unit = parts[1].lower()

        if unit in ["min", "mins", "minute", "minutes"]:
            return timedelta(minutes=value)
        elif unit in ["hour", "hours"]:
            return timedelta(hours=value)
        elif unit in ["day", "days"]:
            return timedelta(days=value)
        elif unit in ["week", "weeks"]:
            return timedelta(weeks=value)
        else:
            raise ValueError(f"Unknown bar size unit: {unit}")

    async def _get_last_timestamp(self, bar_size: str) -> Optional[datetime]:
        """Get the last available timestamp for a bar_size"""
        from live_trading.models.market_data import MarketData
        from bson import ObjectId

        last_bar = await MarketData.find(
            MarketData.operation_id == ObjectId(str(self.operation_id)),
            MarketData.bar_size == bar_size
        ).sort(-MarketData.timestamp).limit(1).to_list()

        if last_bar and len(last_bar) > 0:
            return last_bar[0].timestamp
        return None

    async def _fill_data_gaps(self, operation: TradingOperation):
        """Check for gaps in market data and fill them with historical data"""
        from live_trading.models.market_data import MarketData
        from bson import ObjectId
        import pandas as pd

        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")

        if not self.data_manager.broker:
            logger.warning(f"{tag} Broker not available - cannot fill data gaps")
            return

        logger.info(f"{tag} Checking for data gaps...")
        current_time = datetime.utcnow()

        for bar_size in operation.bar_sizes:
            try:
                # Get last available timestamp
                last_timestamp = await self._get_last_timestamp(bar_size)

                if last_timestamp is None:
                    logger.info(f"{tag} No existing data for {bar_size} - skipping gap check")
                    continue

                # Calculate time gap
                time_gap = current_time - last_timestamp

                # Parse bar size to get duration
                bar_duration = self._parse_bar_size_to_duration(bar_size)

                # Calculate expected number of bars in the gap
                # Add one bar duration to account for the current incomplete bar
                expected_bars = int(time_gap.total_seconds() / bar_duration.total_seconds())

                # Only fill gaps if there's at least one missing bar
                # (accounting for some tolerance - e.g., if gap is less than 2 bar durations, might just be normal delay)
                if expected_bars < 2:
                    logger.debug(f"No significant gap for {bar_size} (gap: {time_gap}, expected bars: {expected_bars})")
                    continue

                logger.info(f"{tag} Gap found for {bar_size}: last={last_timestamp}, gap={time_gap}, expected_bars={expected_bars}")

                # Calculate interval string for the gap
                # Request data from last_timestamp to current_time
                # Convert gap to IBKR interval format
                gap_days = time_gap.total_seconds() / 86400
                if gap_days < 1:
                    interval = f"{int(gap_days * 24)} H"  # Hours
                elif gap_days < 30:
                    interval = f"{int(gap_days)} D"  # Days
                elif gap_days < 365:
                    interval = f"{int(gap_days / 30)} M"  # Months
                else:
                    interval = f"{int(gap_days / 365)} Y"  # Years

                # Cap interval at bar size limits
                interval_limits = self.BAR_SIZE_INTERVAL_LIMITS
                max_interval = interval_limits.get(bar_size, "6 M")

                # Use the smaller of calculated interval or max allowed
                # For now, just use a reasonable interval (e.g., "1 M" for gaps)
                # IBKR will return data up to the limit
                request_interval = "1 M"  # Request 1 month, IBKR will return what's available

                logger.info(f"{tag} Requesting gap-fill for {bar_size}: interval={request_interval}")

                # Request historical data to fill the gap
                # Use a callback to process the received data
                completed_data = []
                completion_event = asyncio.Event()

                def gap_fill_callback(bars: list, context: dict):
                    """Callback when gap-fill data is received"""
                    bar_size_cb = context.get("bar_size")
                    if bar_size_cb == bar_size:
                        # Filter bars to only include those after last_timestamp
                        filtered_bars = [
                            bar for bar in bars
                            if isinstance(bar.get("date"), (int, float, str)) and
                            self._parse_bar_timestamp(bar["date"]) > last_timestamp
                        ]
                        completed_data.extend(filtered_bars)
                        logger.info(f"{tag} Received {len(filtered_bars)} gap-fill bars for {bar_size}")

                        # Signal completion
                        if self.event_loop:
                            asyncio.run_coroutine_threadsafe(
                                self._signal_historical_data_complete(bar_size, completion_event),
                                self.event_loop
                            )

                context = {
                    "bar_size": bar_size,
                    "operation_id": str(self.operation_id),
                    "asset": operation.asset
                }

                requested = await self.data_manager.broker.fetch_historical_data(
                    asset=operation.asset,
                    bar_size=bar_size,
                    interval=request_interval,
                    callback=gap_fill_callback,
                    context=context
                )

                if requested:
                    # Wait for data with timeout
                    timeout = self._calculate_timeout(request_interval)
                    try:
                        await asyncio.wait_for(completion_event.wait(), timeout=timeout)

                        if completed_data:
                            # Process and store the gap-fill data
                            await self._process_and_store_historical_bars(
                                bar_size=bar_size,
                                bars=completed_data,
                                operation=operation
                            )
                            logger.info(f"{tag} Gap filled for {bar_size}: stored {len(completed_data)} bars")
                        else:
                            logger.warning(f"{tag} No gap-fill data received for {bar_size}")
                    except asyncio.TimeoutError:
                        logger.warning(f"{tag} Timeout waiting for gap-fill data for {bar_size}")
                else:
                    logger.warning(f"{tag} Failed to request gap-fill data for {bar_size}")

            except Exception as e:
                logger.error(f"{tag} Error filling gap for {bar_size}: {e}", exc_info=True)

    def _parse_bar_timestamp(self, timestamp_value) -> datetime:
        """Parse timestamp from IBKR bar data (can be unix timestamp or string)"""
        from datetime import datetime
        import pandas as pd

        if isinstance(timestamp_value, (int, float)):
            return datetime.fromtimestamp(timestamp_value)
        elif isinstance(timestamp_value, str):
            try:
                # Try parsing as datetime string
                return pd.to_datetime(timestamp_value, format="%Y%m%d %H:%M:%S")
            except:
                try:
                    # Try as unix timestamp string
                    return datetime.fromtimestamp(int(timestamp_value))
                except:
                    return pd.to_datetime(timestamp_value)
        else:
            return pd.to_datetime(timestamp_value)

    async def _process_and_store_historical_bars(
        self,
        bar_size: str,
        bars: list,
        operation: TradingOperation
    ):
        """Process and store historical bars (used for gap filling)"""
        from datetime import datetime
        import pandas as pd
        from bson import ObjectId
        from live_trading.models.market_data import MarketData

        if not bars:
            return

        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
        try:
            # Convert bars to DataFrame
            df = pd.DataFrame(bars)

            # Parse timestamps
            if df["date"].dtype == 'object':
                try:
                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M:%S")
                except:
                    try:
                        df["date"] = pd.to_datetime(df["date"].astype(int), unit='s')
                    except:
                        df["date"] = pd.to_datetime(df["date"])
            else:
                df["date"] = pd.to_datetime(df["date"], unit='s')

            df.set_index("date", inplace=True)
            df = df.sort_index()

            # Calculate indicators if calculator available
            df_with_indicators = df.copy()
            if self.data_manager.indicator_calculator:
                try:
                    df_with_indicators = self.data_manager.indicator_calculator.execute(df.copy())
                    logger.info(f"{tag} Calculated indicators for {len(df_with_indicators)} gap-fill bars ({bar_size})")
                except Exception as e:
                    logger.error(f"{tag} Error calculating gap-fill indicators ({bar_size}): {e}")

            # Store in batches (reuse the batch insert logic)
            batch_size = 1000
            total_bars = len(df_with_indicators)
            stored_count = 0

            indicator_cols = [col for col in df_with_indicators.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume']]

            for batch_start in range(0, total_bars, batch_size):
                batch_end = min(batch_start + batch_size, total_bars)
                batch_data = []

                for idx in range(batch_start, batch_end):
                    try:
                        timestamp = df_with_indicators.index[idx]
                        row = df_with_indicators.iloc[idx]

                        # Extract indicators
                        indicators_dict = {}
                        for col in indicator_cols:
                            if pd.notna(row.get(col)):
                                value = row[col]
                                if isinstance(value, (int, float, bool)):
                                    indicators_dict[col] = float(value)
                                elif isinstance(value, str):
                                    try:
                                        indicators_dict[col] = float(value)
                                    except (ValueError, TypeError):
                                        indicators_dict[col] = value
                                else:
                                    if value is not None:
                                        indicators_dict[col] = str(value)

                        if hasattr(timestamp, 'to_pydatetime'):
                            bar_timestamp = timestamp.to_pydatetime()
                        else:
                            bar_timestamp = pd.to_datetime(timestamp)

                        market_data = MarketData(
                            operation_id=ObjectId(str(self.operation_id)),
                            bar_size=bar_size,
                            timestamp=bar_timestamp,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row.get("volume", 0.0)),
                            indicators=indicators_dict
                        )
                        batch_data.append(market_data)
                    except Exception as e:
                        logger.error(f"{tag} Error preparing gap-fill bar {idx} for {bar_size}: {e}")

                # Batch insert
                if batch_data:
                    try:
                        await MarketData.insert_many(batch_data)
                        stored_count += len(batch_data)
                    except Exception as e:
                        logger.error(f"{tag} Error batch inserting gap-fill bars {batch_start}-{batch_end} for {bar_size}: {e}")

            logger.info(f"{tag} Stored {stored_count} gap-fill bars for {bar_size}")

        except Exception as e:
            logger.error(f"{tag} Error processing gap-fill data for {bar_size}: {e}", exc_info=True)

    async def _handle_signal(
        self,
        operation_id: str,
        signal_type: str,
        price: float
    ):
        """Handle trading signal from strategy"""
        from live_trading.models.position import Position

        tag = getattr(self, '_op_tag', f"[OP:{str(self.operation_id)[-6:]}]")
        operation = await TradingOperation.get(self.operation_id)
        if not operation:
            logger.warning(f"[ORDER] {tag} Operation not found, cannot place order")
            return

        if operation.status != "active":
            logger.warning(f"[ORDER] {tag} Status '{operation.status}' - order blocked: {signal_type}")
            return

        # Check existing position to decide if this signal should be acted on
        open_position = await Position.find_one(
            Position.operation_id == self.operation_id,
            Position.closed_at == None,  # noqa: E711
        )

        if open_position:
            is_long = open_position.quantity > 0
            same_direction = (
                (is_long and signal_type == "BUY")
                or (not is_long and signal_type == "SELL")
            )
            if same_direction and not operation.allow_pyramiding:
                logger.info(
                    f"[ORDER] {tag} Duplicate {signal_type} blocked "
                    f"(already {'LONG' if is_long else 'SHORT'}, pyramiding disabled)"
                )
                return

        # Place order
        try:
            order = await self.order_manager.place_order(
                operation_id=self.operation_id,
                asset=operation.asset,
                signal_type=signal_type,
                price=price
            )
            if order.status == "REJECTED":
                logger.error(f"[ORDER] {tag} Order rejected by broker: {signal_type}")
        except Exception as e:
            logger.error(f"[ORDER] {tag} Error placing order for {signal_type}: {e}", exc_info=True)

