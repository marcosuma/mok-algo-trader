"""
Trading Engine — orchestrates all trading operations.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from bson import ObjectId

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from live_trading.models import (
    TradingOperation,
    Position,
    Order,
    MarketData,
    JournalEntry,
)
from live_trading.brokers.base_broker import BaseBroker
from live_trading.brokers.middleware.adapter import BrokerAdapter
from live_trading.brokers.middleware.reconciler import Reconciler
from live_trading.brokers.middleware.types import PositionStatus
from live_trading.data.data_manager import DataManager
from live_trading.orders.order_manager import OrderManager
from live_trading.journal.journal_manager import JournalManager
from live_trading.engine.operation_runner import OperationRunner
from live_trading.config import config

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that orchestrates all operations."""

    def __init__(
        self,
        broker: BaseBroker,
        adapter: BrokerAdapter,
        journal_manager: JournalManager,
    ):
        self.broker = broker
        self.adapter = adapter
        self.journal = journal_manager

        from technical_indicators.technical_indicators import TechnicalIndicators
        indicator_calculator = TechnicalIndicators(candlestickData=None, fileToSave=None)

        self.data_manager = DataManager(broker, indicator_calculator)
        self.order_manager = OrderManager(adapter, journal_manager)
        self.reconciler = Reconciler(adapter)

        self.active_operations: Dict[ObjectId, OperationRunner] = {}
        self.db_client: Optional[AsyncIOMotorClient] = None

    async def initialize(self):
        """Initialize the trading engine (connect to database, etc.)."""
        try:
            is_atlas = config.MONGODB_URL.startswith("mongodb+srv://")

            connection_options = {
                "serverSelectionTimeoutMS": config.MONGODB_CONNECT_TIMEOUT_MS,
                "connectTimeoutMS": config.MONGODB_CONNECT_TIMEOUT_MS,
            }

            if is_atlas:
                connection_url = config.MONGODB_URL
                if "retryWrites" not in connection_url:
                    separator = "&" if "?" in connection_url else "?"
                    connection_url = f"{connection_url}{separator}retryWrites=true"
                if "w=majority" not in connection_url:
                    separator = "&" if "?" in connection_url or "&" in connection_url else "?"
                    connection_url = f"{connection_url}{separator}w=majority"

                logger.info("Connecting to MongoDB Atlas with SSL/TLS...")
                self.db_client = AsyncIOMotorClient(connection_url, **connection_options)
            else:
                logger.info("Connecting to local MongoDB...")
                self.db_client = AsyncIOMotorClient(config.MONGODB_URL, **connection_options)

            try:
                await self.db_client.admin.command("ping")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                logger.error(
                    "MongoDB connection failed. Please ensure MongoDB is running.\n"
                    "  - For local MongoDB: Start with 'brew services start mongodb-community' (macOS) or 'sudo systemctl start mongod' (Linux)\n"
                    "  - For MongoDB Atlas: Check your connection string and network access settings\n"
                    "  - The application will continue but database operations will fail until MongoDB is available."
                )
                raise

            safe_url = config.MONGODB_URL
            if "@" in safe_url:
                parts = safe_url.split("@")
                if len(parts) == 2:
                    user_pass = parts[0].split("://")[1] if "://" in parts[0] else parts[0]
                    if ":" in user_pass:
                        user, _ = user_pass.split(":", 1)
                        safe_url = safe_url.replace(user_pass, f"{user}:***")

            logger.info(f"Successfully connected to MongoDB: {safe_url.split('@')[-1] if '@' in safe_url else safe_url}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            if config.MONGODB_URL.startswith("mongodb+srv://"):
                logger.error("For MongoDB Atlas connections, please check:")
                logger.error("  1. Your IP address is whitelisted in Atlas Network Access")
                logger.error("  2. Your connection string is correct (mongodb+srv://...)")
                logger.error("  3. Your username and password are correct")
                logger.error("  4. Your cluster is running and accessible")
                logger.error("  5. Your connection string includes: ?retryWrites=true&w=majority")
            raise

        await init_beanie(
            database=self.db_client[config.MONGODB_DB_NAME],
            document_models=[
                TradingOperation,
                Position,
                Order,
                MarketData,
                JournalEntry,
            ],
        )

        await self.journal.initialize_sequence_counter()

        self.order_manager.bind_event_loop(asyncio.get_running_loop())

        logger.info("Trading engine initialized")

    async def start_operation(
        self,
        asset: str,
        bar_sizes: List[str],
        primary_bar_size: str,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        initial_capital: float = 10000.0,
        **kwargs,
    ) -> TradingOperation:
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
            data_retention_bars=kwargs.get("data_retention_bars", config.DEFAULT_DATA_RETENTION_BARS),
        )
        await operation.insert()

        runner = OperationRunner(
            operation_id=operation.id,
            data_manager=self.data_manager,
            order_manager=self.order_manager,
        )
        await runner.start()
        self.active_operations[operation.id] = runner

        await self.journal.log_action(
            action_type="OPERATION_STARTED",
            action_data={
                "operation_id": str(operation.id),
                "asset": asset,
                "strategy": strategy_name,
            },
            operation_id=operation.id,
        )

        logger.info(f"Started operation {operation.id} for {asset} with {strategy_name}")
        return operation

    async def stop_operation(self, operation_id: ObjectId):
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} is not active")

        runner = self.active_operations[operation_id]
        await runner.stop()
        del self.active_operations[operation_id]

        operation = await TradingOperation.get(operation_id)
        if operation:
            operation.status = "closed"
            operation.closed_at = datetime.utcnow()
            await operation.save()

        await self.journal.log_action(
            action_type="OPERATION_STOPPED",
            action_data={"operation_id": str(operation_id)},
            operation_id=operation_id,
        )
        logger.info(f"Stopped operation {operation_id}")

    async def pause_operation(self, operation_id: ObjectId):
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} is not active")
        runner = self.active_operations[operation_id]
        await runner.pause()

    async def resume_operation(self, operation_id: ObjectId):
        operation = await TradingOperation.get(operation_id)
        if not operation:
            raise ValueError(f"Operation {operation_id} not found")
        if operation.status != "paused":
            raise ValueError(f"Operation {operation_id} is not paused")

        if operation_id not in self.active_operations:
            runner = OperationRunner(
                operation_id=operation_id,
                data_manager=self.data_manager,
                order_manager=self.order_manager,
            )
            await runner.start()
            self.active_operations[operation_id] = runner
        else:
            runner = self.active_operations[operation_id]
            await runner.resume()

    async def recover_from_journal(self):
        """Recover state from journal on startup."""
        logger.info("Starting crash recovery...")

        active_operations = await TradingOperation.find(
            TradingOperation.status == "active",
        ).to_list()

        logger.info(f"Found {len(active_operations)} active operations")

        for operation in active_operations:
            open_positions = await Position.find(
                Position.operation_id == operation.id,
                Position.status == PositionStatus.OPEN,
            ).to_list()

            if open_positions:
                logger.info(f"Operation {operation.id} has {len(open_positions)} open positions")
                await self.order_manager.handle_crash_recovery(operation.id)

            await self.reconciler.reconcile(operation)

            for bar_size in operation.bar_sizes:
                bars = await MarketData.find(
                    MarketData.operation_id == operation.id,
                    MarketData.bar_size == bar_size,
                ).sort(-MarketData.timestamp).limit(operation.data_retention_bars).to_list()

                for bar in reversed(bars):
                    await self.data_manager.handle_tick(
                        operation_id=str(operation.id),
                        asset=operation.asset,
                        price=bar.close,
                        size=bar.volume,
                        timestamp=bar.timestamp,
                    )

            runner = OperationRunner(
                operation_id=operation.id,
                data_manager=self.data_manager,
                order_manager=self.order_manager,
            )
            await runner.start()
            self.active_operations[operation.id] = runner

            if self.broker and hasattr(self.broker, "fetch_historical_data"):
                logger.info(f"Checking for data gaps in operation {operation.id}...")
                await runner._fill_data_gaps(operation)

        logger.info("Crash recovery completed")

    async def sync_positions_from_broker(self, operation: TradingOperation):
        """Public entry-point for periodic reconciliation."""
        await self.reconciler.reconcile(operation)

    async def shutdown(self):
        for operation_id, runner in list(self.active_operations.items()):
            await runner.stop()
        self.active_operations.clear()
        await self.broker.disconnect()
        if self.db_client:
            self.db_client.close()
        logger.info("Trading engine shut down")
