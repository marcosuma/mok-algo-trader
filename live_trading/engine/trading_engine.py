"""
Trading Engine - Orchestrates all trading operations.
"""
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
from bson import ObjectId

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from live_trading.models import (
    TradingOperation,
    Transaction,
    Position,
    Order,
    Trade,
    MarketData,
    JournalEntry
)
from live_trading.brokers.base_broker import BaseBroker
from live_trading.data.data_manager import DataManager
from live_trading.orders.order_manager import OrderManager
from live_trading.journal.journal_manager import JournalManager
from live_trading.engine.operation_runner import OperationRunner
from live_trading.config import config

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that orchestrates all operations"""

    def __init__(
        self,
        broker: BaseBroker,
        journal_manager: JournalManager
    ):
        self.broker = broker
        self.journal = journal_manager

        # Initialize indicator calculator
        from technical_indicators.technical_indicators import TechnicalIndicators
        indicator_calculator = TechnicalIndicators(candlestickData=None, fileToSave=None)

        # Initialize data manager and order manager
        self.data_manager = DataManager(broker, indicator_calculator)
        self.order_manager = OrderManager(broker, journal_manager)

        # Active operations: operation_id -> OperationRunner
        self.active_operations: Dict[ObjectId, OperationRunner] = {}

        # Database client
        self.db_client: Optional[AsyncIOMotorClient] = None

    async def initialize(self):
        """Initialize the trading engine (connect to database, etc.)"""
        # Connect to MongoDB (supports both local and Atlas)
        try:
            # Determine if this is an Atlas connection (mongodb+srv://)
            is_atlas = config.MONGODB_URL.startswith("mongodb+srv://")

            # Connection options
            # Note: mongodb+srv:// automatically handles TLS/SSL, don't set tls=True explicitly
            connection_options = {
                "serverSelectionTimeoutMS": config.MONGODB_CONNECT_TIMEOUT_MS,
                "connectTimeoutMS": config.MONGODB_CONNECT_TIMEOUT_MS,
            }

            # For Atlas connections, ensure connection string has required parameters
            if is_atlas:
                # Ensure connection string includes retryWrites and w=majority if not present
                connection_url = config.MONGODB_URL
                if "retryWrites" not in connection_url:
                    separator = "&" if "?" in connection_url else "?"
                    connection_url = f"{connection_url}{separator}retryWrites=true"
                if "w=majority" not in connection_url:
                    separator = "&" if "?" in connection_url or "&" in connection_url else "?"
                    connection_url = f"{connection_url}{separator}w=majority"

                # Use the updated connection URL
                logger.info("Connecting to MongoDB Atlas with SSL/TLS...")
                self.db_client = AsyncIOMotorClient(
                    connection_url,
                    **connection_options
                )
            else:
                logger.info("Connecting to local MongoDB...")
                self.db_client = AsyncIOMotorClient(
                    config.MONGODB_URL,
                    **connection_options
                )

            # Test connection with a longer timeout for Atlas
            try:
                await self.db_client.admin.command('ping')
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                logger.error(
                    "MongoDB connection failed. Please ensure MongoDB is running.\n"
                    "  - For local MongoDB: Start with 'brew services start mongodb-community' (macOS) or 'sudo systemctl start mongod' (Linux)\n"
                    "  - For MongoDB Atlas: Check your connection string and network access settings\n"
                    "  - The application will continue but database operations will fail until MongoDB is available."
                )
                # Don't raise - allow the app to start but database operations will fail
                # This allows the API to start and show helpful error messages
                raise

            # Log connection success (mask password in URL)
            safe_url = config.MONGODB_URL
            if "@" in safe_url:
                # Mask password in connection string
                parts = safe_url.split("@")
                if len(parts) == 2:
                    user_pass = parts[0].split("://")[1] if "://" in parts[0] else parts[0]
                    if ":" in user_pass:
                        user, _ = user_pass.split(":", 1)
                        safe_url = safe_url.replace(user_pass, f"{user}:***")

            logger.info(f"Successfully connected to MongoDB: {safe_url.split('@')[-1] if '@' in safe_url else safe_url}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            if is_atlas:
                logger.error("For MongoDB Atlas connections, please check:")
                logger.error("  1. Your IP address is whitelisted in Atlas Network Access")
                logger.error("     Go to: Network Access > Add IP Address (or use 0.0.0.0/0 for testing)")
                logger.error("  2. Your connection string is correct (mongodb+srv://...)")
                logger.error("  3. Your username and password are correct")
                logger.error("  4. Your cluster is running and accessible")
                logger.error("  5. Your connection string includes: ?retryWrites=true&w=majority")
            raise

        # Initialize Beanie
        await init_beanie(
            database=self.db_client[config.MONGODB_DB_NAME],
            document_models=[
                TradingOperation,
                Transaction,
                Position,
                Order,
                Trade,
                MarketData,
                JournalEntry
            ]
        )

        # Initialize journal sequence counter
        await self.journal.initialize_sequence_counter()

        logger.info("Trading engine initialized")

    async def start_operation(
        self,
        asset: str,
        bar_sizes: List[str],
        primary_bar_size: str,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        initial_capital: float = 10000.0,
        **kwargs
    ) -> TradingOperation:
        """
        Start a new trading operation.

        Args:
            asset: Asset symbol (e.g., "USD-CAD")
            bar_sizes: List of bar sizes (e.g., ["1 hour", "15 mins", "1 day"])
            primary_bar_size: Primary timeframe for entry/exit
            strategy_name: Strategy class name
            strategy_config: Strategy parameters
            initial_capital: Initial capital
            **kwargs: Additional operation config (stop_loss_type, etc.)

        Returns:
            Created TradingOperation
        """
        # Create operation record
        operation = TradingOperation(
            asset=asset,
            bar_sizes=bar_sizes,
            primary_bar_size=primary_bar_size,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            broker_type=kwargs.get("broker_type", config.BROKER_TYPE),
            stop_loss_type=kwargs.get("stop_loss_type", config.DEFAULT_STOP_LOSS_TYPE),
            stop_loss_value=kwargs.get("stop_loss_value", config.DEFAULT_STOP_LOSS_VALUE),
            take_profit_type=kwargs.get("take_profit_type", config.DEFAULT_TAKE_PROFIT_TYPE),
            take_profit_value=kwargs.get("take_profit_value", config.DEFAULT_TAKE_PROFIT_VALUE),
            crash_recovery_mode=kwargs.get("crash_recovery_mode", config.DEFAULT_CRASH_RECOVERY_MODE),
            emergency_stop_loss_pct=kwargs.get("emergency_stop_loss_pct", config.DEFAULT_EMERGENCY_STOP_LOSS_PCT),
            data_retention_bars=kwargs.get("data_retention_bars", config.DEFAULT_DATA_RETENTION_BARS)
        )
        await operation.insert()

        # Create operation runner
        runner = OperationRunner(
            operation_id=operation.id,
            data_manager=self.data_manager,
            order_manager=self.order_manager
        )

        # Start the operation
        await runner.start()

        # Store runner
        self.active_operations[operation.id] = runner

        # Log to journal
        await self.journal.log_action(
            action_type="OPERATION_STARTED",
            action_data={
                "operation_id": str(operation.id),
                "asset": asset,
                "strategy": strategy_name
            },
            operation_id=operation.id
        )

        logger.info(f"Started operation {operation.id} for {asset} with {strategy_name}")
        return operation

    async def stop_operation(self, operation_id: ObjectId):
        """Stop an active trading operation"""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} is not active")

        runner = self.active_operations[operation_id]
        await runner.stop()
        del self.active_operations[operation_id]

        # Update operation status
        operation = await TradingOperation.get(operation_id)
        if operation:
            operation.status = "closed"
            operation.closed_at = datetime.utcnow()
            await operation.save()

        # Log to journal
        await self.journal.log_action(
            action_type="OPERATION_STOPPED",
            action_data={"operation_id": str(operation_id)},
            operation_id=operation_id
        )

        logger.info(f"Stopped operation {operation_id}")

    async def pause_operation(self, operation_id: ObjectId):
        """Pause an operation"""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} is not active")

        runner = self.active_operations[operation_id]
        await runner.pause()

    async def resume_operation(self, operation_id: ObjectId):
        """Resume a paused operation"""
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")

        if operation.status != "paused":
            raise ValueError(f"Operation {operation_id} is not paused")

        # Create runner if not exists
        if operation_id not in self.active_operations:
            runner = OperationRunner(
                operation_id=operation_id,
                data_manager=self.data_manager,
                order_manager=self.order_manager
            )
            await runner.start()
            self.active_operations[operation_id] = runner
        else:
            runner = self.active_operations[operation_id]
            await runner.resume()

    async def recover_from_journal(self):
        """
        Recover state from journal on startup.

        Steps:
        1. Load all active operations from database
        2. Check for open positions (crashed positions)
        3. Apply crash recovery strategy
        4. Reconstruct strategy state
        5. Resume data collection and trading
        """
        logger.info("Starting crash recovery...")

        # 1. Load all active operations
        active_operations = await TradingOperation.find(
            TradingOperation.status == "active"
        ).to_list()

        logger.info(f"Found {len(active_operations)} active operations")

        # 2. Check for open positions and apply crash recovery
        for operation in active_operations:
            open_positions = await Position.find(
                Position.operation_id == operation.id,
                Position.closed_at == None
            ).to_list()

            if open_positions:
                logger.info(f"Operation {operation.id} has {len(open_positions)} open positions")
                await self.order_manager.handle_crash_recovery(operation.id)

            # 3. Sync positions from broker to database
            await self._sync_positions_from_broker(operation)

            # 4. Reconstruct strategy state from last N bars
            # Load last N bars for each bar_size
            for bar_size in operation.bar_sizes:
                bars = await MarketData.find(
                    MarketData.operation_id == operation.id,
                    MarketData.bar_size == bar_size
                ).sort(-MarketData.timestamp).limit(operation.data_retention_bars).to_list()

                # Add to data manager buffer
                for bar in reversed(bars):  # Oldest first
                    await self.data_manager.handle_tick(
                        operation_id=str(operation.id),
                        asset=operation.asset,
                        price=bar.close,  # Use close as tick price
                        size=bar.volume,
                        timestamp=bar.timestamp
                    )

            # 5. Resume operation
            runner = OperationRunner(
                operation_id=operation.id,
                data_manager=self.data_manager,
                order_manager=self.order_manager
            )
            await runner.start()
            self.active_operations[operation.id] = runner

            # 6. Check for gaps in market data and fill them
            if self.broker and hasattr(self.broker, 'fetch_historical_data'):
                logger.info(f"Checking for data gaps in operation {operation.id}...")
                await runner._fill_data_gaps(operation)

        logger.info("Crash recovery completed")

    async def _sync_positions_from_broker(self, operation: TradingOperation):
        """Reconcile positions and pending orders from the broker into the database.

        This is the single source of truth sync that runs on startup and every 30 s.
        It fixes three categories of divergence:
          1. Positions that exist on the broker but not in the DB (e.g. order filled
             while the bot was down, or placed directly on cTrader).
          2. DB positions that no longer exist on the broker (closed externally).
          3. DB orders stuck in PENDING status whose broker order has already been
             resolved (filled, cancelled, or rejected).
        """
        try:
            logger.info(f"[SYNC] Reconciling broker state for operation {operation.id} ({operation.asset})...")

            # --- 1. Fetch broker positions (also populates _open_orders_cache) ---
            broker_positions = await self.broker.get_positions()

            # get_positions() also refreshes _open_orders_cache via the reconcile response
            broker_open_orders: List[Dict[str, Any]] = []
            if hasattr(self.broker, 'get_open_broker_orders'):
                broker_open_orders = await self.broker.get_open_broker_orders()

            # Build a set of broker_order_ids that are still open on the broker
            broker_open_order_ids = {o["broker_order_id"] for o in broker_open_orders}

            # --- 2. Sync positions ---
            db_positions = await Position.find(
                Position.operation_id == operation.id,
                Position.closed_at == None
            ).to_list()
            db_positions_map = {pos.contract_symbol: pos for pos in db_positions}

            # Filter to positions for this operation's asset only
            our_broker_positions = [p for p in broker_positions if p.get("asset") == operation.asset]

            for broker_pos in our_broker_positions:
                asset = broker_pos["asset"]
                quantity = broker_pos.get("quantity", 0.0)
                avg_price = broker_pos.get("avg_price", 0.0)
                current_price = broker_pos.get("current_price", avg_price)
                unrealized_pnl = broker_pos.get("unrealized_pnl", 0.0)
                unrealized_pnl_pct = broker_pos.get("unrealized_pnl_pct", 0.0)
                stop_loss = broker_pos.get("stop_loss")
                take_profit = broker_pos.get("take_profit")

                if asset in db_positions_map:
                    db_pos = db_positions_map[asset]
                    db_pos.entry_price = avg_price  # keep in sync with broker (fixes stale/wrong values)
                    db_pos.current_price = current_price
                    db_pos.unrealized_pnl = unrealized_pnl
                    db_pos.unrealized_pnl_pct = unrealized_pnl_pct
                    db_pos.stop_loss = stop_loss
                    db_pos.take_profit = take_profit
                    await db_pos.save()
                    logger.debug(f"[SYNC] Updated position {asset}: entry={avg_price}, price={current_price}, upnl={unrealized_pnl:.2f}")
                else:
                    if abs(quantity) > 0.0001:
                        new_pos = Position(
                            operation_id=operation.id,
                            contract_symbol=asset,
                            quantity=quantity,
                            entry_price=avg_price,
                            current_price=current_price,
                            unrealized_pnl=unrealized_pnl,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                        )
                        await new_pos.insert()
                        logger.info(f"[SYNC] Created missing DB position for {asset} @ {avg_price} (qty={quantity})")

            # Close DB positions no longer present on the broker
            broker_asset_set = {p["asset"] for p in our_broker_positions}
            for db_pos in db_positions:
                if db_pos.contract_symbol not in broker_asset_set:
                    logger.info(
                        f"[SYNC] Position {db_pos.contract_symbol} gone from broker "
                        f"— creating close records via _process_external_close"
                    )
                    await self.order_manager._process_external_close(
                        operation_id=operation.id,
                        close_price=db_pos.current_price,
                        filled_volume=abs(db_pos.quantity),
                    )

            # --- 3. Reconcile stuck PENDING orders ---
            # Find all orders in DB that the bot believes are still pending
            pending_db_orders = await Order.find(
                Order.operation_id == operation.id,
                Order.status == "PENDING"
            ).to_list()

            for db_order in pending_db_orders:
                if not db_order.broker_order_id:
                    # No broker ID means place_order returned "" → already REJECTED in DB
                    # (this shouldn't be PENDING, but fix it defensively)
                    db_order.status = "REJECTED"
                    await db_order.save()
                    logger.warning(f"[SYNC] Order {db_order.id} had no broker_order_id but was PENDING — marked REJECTED")
                    continue

                if db_order.broker_order_id in broker_open_order_ids:
                    # Still genuinely pending on broker — nothing to do
                    continue

                # Order is not in broker's open orders. Determine what happened.
                # If the broker now has a position for this asset it was likely filled.
                if operation.asset in broker_asset_set:
                    # A position exists — the order was probably FILLED (missed callback)
                    db_order.status = "FILLED"
                    db_order.filled_at = db_order.filled_at or datetime.utcnow()
                    await db_order.save()
                    logger.info(
                        f"[SYNC] Order {db_order.broker_order_id} not in broker open orders but position exists "
                        f"→ marking FILLED (callback was missed)"
                    )
                else:
                    # No position → order was cancelled or rejected on the broker side
                    db_order.status = "CANCELLED"
                    await db_order.save()
                    logger.info(
                        f"[SYNC] Order {db_order.broker_order_id} not in broker open orders and no position "
                        f"→ marking CANCELLED"
                    )

            # --- 4. Update current_capital from real broker balance ---
            if hasattr(self.broker, 'get_account_info'):
                try:
                    account_info = await self.broker.get_account_info()
                    broker_balance = account_info.get("balance")
                    if broker_balance is not None and broker_balance > 0:
                        operation.current_capital = float(broker_balance)
                        await operation.save()
                        logger.info(f"[SYNC] Updated current_capital to broker balance: {broker_balance:.2f}")
                except Exception as bal_err:
                    logger.warning(f"[SYNC] Could not fetch broker balance: {bal_err}")

            logger.info(f"[SYNC] Reconciliation complete for operation {operation.id}")

        except Exception as e:
            logger.error(f"[SYNC] Error during broker reconciliation: {e}", exc_info=True)

    async def shutdown(self):
        """Shutdown the trading engine"""
        # Stop all operations
        for operation_id, runner in list(self.active_operations.items()):
            await runner.stop()

        self.active_operations.clear()

        # Disconnect from broker
        await self.broker.disconnect()

        # Close database connection
        if self.db_client:
            self.db_client.close()

        logger.info("Trading engine shut down")

