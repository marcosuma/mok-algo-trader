"""
FastAPI main application.
"""
import logging
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

import inspect
from live_trading.engine.trading_engine import TradingEngine

logger = logging.getLogger(__name__)
from live_trading.models.trading_operation import TradingOperation
from live_trading.models.position import Position
from live_trading.models.order import Order
from live_trading.brokers.middleware.types import PositionStatus, OrderStatus
from live_trading.config import config

app = FastAPI(title="Live Trading System API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trading engine (will be initialized in startup)
trading_engine: Optional[TradingEngine] = None


def get_trading_engine() -> TradingEngine:
    """Dependency to get trading engine"""
    if trading_engine is None:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    return trading_engine


# Pydantic models for request/response
class CreateOperationRequest(BaseModel):
    asset: str
    bar_sizes: List[str]
    primary_bar_size: str
    strategy_name: str
    strategy_config: Dict[str, Any] = {}
    initial_capital: Optional[float] = None
    stop_loss_type: Optional[str] = None
    stop_loss_value: Optional[float] = None
    take_profit_type: Optional[str] = None
    take_profit_value: Optional[float] = None
    crash_recovery_mode: Optional[str] = None
    emergency_stop_loss_pct: Optional[float] = None
    data_retention_bars: Optional[int] = None


class OperationResponse(BaseModel):
    id: str
    asset: str
    bar_sizes: List[str]
    primary_bar_size: str
    strategy_name: str
    status: str
    broker_type: str = "IBKR"
    initial_capital: float
    current_capital: float
    total_pnl: float
    total_pnl_pct: float
    created_at: datetime
    updated_at: datetime


# Operations endpoints
@app.post("/api/operations", response_model=OperationResponse)
async def create_operation(
    request: CreateOperationRequest,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Create a new trading operation"""
    try:
        operation = await engine.start_operation(
            asset=request.asset,
            bar_sizes=request.bar_sizes,
            primary_bar_size=request.primary_bar_size,
            strategy_name=request.strategy_name,
            strategy_config=request.strategy_config,
            initial_capital=request.initial_capital or 0.0,
            stop_loss_type=request.stop_loss_type,
            stop_loss_value=request.stop_loss_value,
            take_profit_type=request.take_profit_type,
            take_profit_value=request.take_profit_value,
            crash_recovery_mode=request.crash_recovery_mode,
            emergency_stop_loss_pct=request.emergency_stop_loss_pct,
            data_retention_bars=request.data_retention_bars
        )

        return OperationResponse(
            id=str(operation.id),
            asset=operation.asset,
            bar_sizes=operation.bar_sizes,
            primary_bar_size=operation.primary_bar_size,
            strategy_name=operation.strategy_name,
            status=operation.status,
            broker_type=operation.broker_type,
            initial_capital=operation.initial_capital,
            current_capital=operation.current_capital,
            total_pnl=operation.total_pnl,
            total_pnl_pct=operation.total_pnl_pct,
            created_at=operation.created_at,
            updated_at=operation.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/operations", response_model=List[OperationResponse])
async def list_operations(
    status: Optional[str] = None,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """List all trading operations"""
    query = {}
    if status:
        query["status"] = status

    operations = await TradingOperation.find(query).sort(-TradingOperation.created_at).to_list()

    return [
        OperationResponse(
            id=str(op.id),
            asset=op.asset,
            bar_sizes=op.bar_sizes,
            primary_bar_size=op.primary_bar_size,
            strategy_name=op.strategy_name,
            status=op.status,
            broker_type=op.broker_type,
            initial_capital=op.initial_capital,
            current_capital=op.current_capital,
            total_pnl=op.total_pnl,
            total_pnl_pct=op.total_pnl_pct,
            created_at=op.created_at,
            updated_at=op.updated_at
        )
        for op in operations
    ]


@app.get("/api/operations/{operation_id}", response_model=OperationResponse)
async def get_operation(
    operation_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Get operation details"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    operation = await TradingOperation.get(op_id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")

    return OperationResponse(
        id=str(operation.id),
        asset=operation.asset,
        bar_sizes=operation.bar_sizes,
        primary_bar_size=operation.primary_bar_size,
        strategy_name=operation.strategy_name,
        status=operation.status,
        broker_type=operation.broker_type,
        initial_capital=operation.initial_capital,
        current_capital=operation.current_capital,
        total_pnl=operation.total_pnl,
        total_pnl_pct=operation.total_pnl_pct,
        created_at=operation.created_at,
        updated_at=operation.updated_at
    )


@app.delete("/api/operations/{operation_id}")
async def delete_operation(
    operation_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Stop and close an operation"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    try:
        await engine.stop_operation(op_id)
        return {"message": "Operation stopped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/operations/{operation_id}/pause")
async def pause_operation(
    operation_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Pause an operation"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    try:
        await engine.pause_operation(op_id)
        return {"message": "Operation paused"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/operations/{operation_id}/resume")
async def resume_operation(
    operation_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Resume a paused operation"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    try:
        await engine.resume_operation(op_id)
        return {"message": "Operation resumed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Strategies endpoint
_BASE_STRATEGY_PARAMS = {
    "self", "initial_cash", "commission", "spread",
    "stop_loss_type", "stop_loss_value", "take_profit_type", "take_profit_value",
    "allow_pyramiding", "max_pyramid_entries", "slippage",
}

@app.get("/api/strategies")
async def list_strategies():
    """
    List all available strategy names with their strategy-specific default parameters.
    Base-class parameters (stop_loss_type, initial_cash, etc.) are excluded because
    they are configured separately in the operation creation form.
    """
    from forex_strategies.strategy_registry import get_all_strategies

    result = []
    for name, cls in sorted(get_all_strategies().items()):
        try:
            sig = inspect.signature(cls.__init__)
            default_config: Dict[str, Any] = {}
            for param_name, param in sig.parameters.items():
                if param_name in _BASE_STRATEGY_PARAMS:
                    continue
                if param.default is not inspect.Parameter.empty:
                    default_config[param_name] = param.default
            result.append({"name": name, "default_config": default_config})
        except Exception:
            result.append({"name": name, "default_config": {}})

    return result


# Positions endpoints
@app.get("/api/operations/{operation_id}/positions")
async def get_positions(
    operation_id: str,
    status_filter: Optional[str] = None,
    engine: TradingEngine = Depends(get_trading_engine),
):
    """Get positions for an operation.

    Query params:
        status_filter: "OPEN", "CLOSED", or omit for all.
    """
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    query = {"operation_id": op_id}
    if status_filter:
        query["status"] = status_filter.upper()

    positions = await Position.find(query).sort(-Position.opened_at).to_list()

    live_spot = getattr(engine.broker, '_last_spot_prices', {})
    symbol_cache = getattr(engine.broker, '_symbol_cache', {})

    result = []
    for pos in positions:
        current_price = pos.current_price
        unrealized_pnl = pos.unrealized_pnl
        unrealized_pnl_pct = pos.unrealized_pnl_pct

        if pos.status == PositionStatus.OPEN:
            symbol_id = symbol_cache.get(pos.symbol)
            if symbol_id and symbol_id in live_spot:
                spot = live_spot[symbol_id]
                current_price = spot["bid"] if pos.side.value == "LONG" else spot["ask"]
                direction = 1.0 if pos.side.value == "LONG" else -1.0
                gross_pnl = (current_price - pos.entry_price) * direction * pos.quantity
                notional = pos.entry_price * pos.quantity
                unrealized_pnl = gross_pnl
                unrealized_pnl_pct = (gross_pnl / notional * 100) if notional > 0 else 0.0

        result.append({
            "id": str(pos.id),
            "broker_position_id": pos.broker_position_id,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "status": pos.status.value,
            "close_price": pos.close_price,
            "close_reason": pos.close_reason.value if pos.close_reason else None,
            "realized_pnl": pos.realized_pnl,
            "realized_pnl_pct": pos.realized_pnl_pct,
            "total_commission": pos.total_commission,
            "duration_seconds": pos.duration_seconds,
            "opened_at": pos.opened_at,
            "closed_at": pos.closed_at,
        })

    return result


@app.get("/api/operations/{operation_id}/trades")
async def get_trades(operation_id: str):
    """Get completed trades (closed positions) for an operation."""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    closed = await Position.find(
        Position.operation_id == op_id,
        Position.status == PositionStatus.CLOSED,
    ).sort(-Position.closed_at).to_list()

    return [
        {
            "id": str(pos.id),
            "broker_position_id": pos.broker_position_id,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "close_price": pos.close_price,
            "quantity": pos.quantity,
            "realized_pnl": pos.realized_pnl,
            "realized_pnl_pct": pos.realized_pnl_pct,
            "close_reason": pos.close_reason.value if pos.close_reason else None,
            "total_commission": pos.total_commission,
            "total_swap": pos.total_swap,
            "duration_seconds": pos.duration_seconds,
            "opened_at": pos.opened_at,
            "closed_at": pos.closed_at,
        }
        for pos in closed
    ]


# Orders endpoints
@app.get("/api/operations/{operation_id}/orders")
async def get_orders(operation_id: str):
    """Get orders for an operation"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    orders = await Order.find(
        Order.operation_id == op_id
    ).sort(-Order.created_at).to_list()

    return [
        {
            "id": str(order.id),
            "broker_order_id": order.broker_order_id,
            "broker_position_id": order.broker_position_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "requested_quantity": order.requested_quantity,
            "requested_price": order.requested_price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
            "intent": order.intent.value,
            "close_reason": order.close_reason.value if order.close_reason else None,
            "source": order.source.value,
            "created_at": order.created_at,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
        }
        for order in orders
    ]


# Market Data endpoints
def sanitize_value(value):
    """Sanitize a value to be JSON-compliant (replace inf, -inf, nan with None)"""
    import math
    if isinstance(value, (int, float)):
        if math.isinf(value) or math.isnan(value):
            return None
    return value

def sanitize_indicators(indicators: dict) -> dict:
    """Sanitize indicators dictionary to remove inf, -inf, and nan values"""
    if not indicators:
        return {}
    return {
        key: sanitize_value(value)
        for key, value in indicators.items()
    }

@app.get("/api/operations/{operation_id}/market-data")
async def get_market_data(
    operation_id: str,
    bar_size: Optional[str] = None,
    limit: int = 1000
):
    """Get market data for an operation"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    from live_trading.models.market_data import MarketData

    # Build query with filters
    if bar_size:
        market_data = await MarketData.find(
            MarketData.operation_id == op_id,
            MarketData.bar_size == bar_size
        ).sort(-MarketData.timestamp).limit(limit).to_list()
    else:
        market_data = await MarketData.find(
            MarketData.operation_id == op_id
        ).sort(-MarketData.timestamp).limit(limit).to_list()

    return [
        {
            "id": str(md.id),
            "bar_size": md.bar_size,
            "timestamp": md.timestamp,
            "open": sanitize_value(md.open),
            "high": sanitize_value(md.high),
            "low": sanitize_value(md.low),
            "close": sanitize_value(md.close),
            "volume": sanitize_value(md.volume),
            "indicators": sanitize_indicators(md.indicators)
        }
        for md in market_data
    ]


@app.get("/api/operations/{operation_id}/market-data/count")
async def get_market_data_count(operation_id: str):
    """Get market data count for an operation (for tab display)"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    from live_trading.models.market_data import MarketData

    count = await MarketData.find(MarketData.operation_id == op_id).count()
    return {"count": count}


@app.post("/api/operations/{operation_id}/market-data/cleanup-duplicates")
async def cleanup_duplicate_market_data(operation_id: str, bar_size: Optional[str] = None):
    """
    Clean up duplicate market data bars for an operation.

    This endpoint removes duplicate bars (same operation_id, bar_size, timestamp),
    keeping only the bar with the highest volume (historical data typically has actual volume).
    """
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    from live_trading.models.market_data import MarketData
    from collections import defaultdict

    # Build query
    query = {"operation_id": op_id}
    if bar_size:
        query["bar_size"] = bar_size

    # Get all market data for this operation
    all_bars = await MarketData.find(MarketData.operation_id == op_id).to_list()

    # Group by (bar_size, timestamp)
    grouped = defaultdict(list)
    for bar in all_bars:
        key = (bar.bar_size, bar.timestamp)
        grouped[key].append(bar)

    # Find and clean up duplicates
    duplicates_removed = 0
    for key, bars in grouped.items():
        if len(bars) > 1:
            # Sort by volume (descending) - keep the one with highest volume
            # Historical data typically has actual volume, real-time often has 0
            bars_sorted = sorted(bars, key=lambda b: b.volume or 0, reverse=True)

            # Keep the first one (highest volume), delete the rest
            for bar_to_delete in bars_sorted[1:]:
                await bar_to_delete.delete()
                duplicates_removed += 1

    return {
        "operation_id": operation_id,
        "bar_size": bar_size,
        "duplicates_removed": duplicates_removed,
        "total_bars_checked": len(all_bars)
    }


@app.post("/api/operations/{operation_id}/market-data/cleanup-invalid")
async def cleanup_invalid_market_data(
    operation_id: str,
    bar_size: Optional[str] = None,
    dry_run: bool = True
):
    """
    Find and optionally remove invalid OHLC bars (where high < low, or values are clearly wrong).

    Args:
        dry_run: If True, only report invalid bars without deleting them
    """
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    from live_trading.models.market_data import MarketData

    # Get all market data for this operation
    query = {"operation_id": op_id}
    if bar_size:
        query["bar_size"] = bar_size

    all_bars = await MarketData.find(MarketData.operation_id == op_id).to_list()

    invalid_bars = []
    for bar in all_bars:
        is_invalid = False
        reasons = []

        o, h, l, c = bar.open, bar.high, bar.low, bar.close

        # Check OHLC consistency
        if h < l:
            is_invalid = True
            reasons.append(f"high ({h}) < low ({l})")
        if h < o or h < c:
            is_invalid = True
            reasons.append(f"high ({h}) is not highest")
        if l > o or l > c:
            is_invalid = True
            reasons.append(f"low ({l}) is not lowest")

        # Check for unreasonable range (>5% for a single bar is suspicious)
        if l > 0:
            range_pct = (h - l) / l * 100
            if range_pct > 5:
                is_invalid = True
                reasons.append(f"range {range_pct:.2f}% too large")

        # Check for zero volume on bars that should have it
        # (this is informational, not necessarily invalid)

        if is_invalid:
            invalid_bars.append({
                "id": str(bar.id),
                "bar_size": bar.bar_size,
                "timestamp": bar.timestamp.isoformat(),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": bar.volume,
                "reasons": reasons
            })

    # Delete if not dry run
    deleted_count = 0
    if not dry_run and invalid_bars:
        for invalid_info in invalid_bars:
            bar = await MarketData.get(invalid_info["id"])
            if bar:
                await bar.delete()
                deleted_count += 1

    return {
        "operation_id": operation_id,
        "bar_size": bar_size,
        "dry_run": dry_run,
        "invalid_bars_found": len(invalid_bars),
        "deleted_count": deleted_count,
        "invalid_bars": invalid_bars[:100]  # Limit to first 100 for response size
    }


# Statistics endpoints
@app.get("/api/operations/{operation_id}/stats")
async def get_operation_stats(operation_id: str):
    """Get operation statistics"""
    try:
        op_id = ObjectId(operation_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    operation = await TradingOperation.get(op_id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")

    closed_positions = await Position.find(
        Position.operation_id == op_id,
        Position.status == PositionStatus.CLOSED,
    ).to_list()
    open_positions = await Position.find(
        Position.operation_id == op_id,
        Position.status == PositionStatus.OPEN,
    ).to_list()

    total_unrealized_pnl = sum(p.unrealized_pnl or 0.0 for p in open_positions)

    return {
        "operation_id": str(operation.id),
        "broker_type": operation.broker_type,
        "total_trades": len(closed_positions),
        "winning_trades": len([p for p in closed_positions if (p.realized_pnl or 0) > 0]),
        "losing_trades": len([p for p in closed_positions if (p.realized_pnl or 0) < 0]),
        "total_pnl": operation.total_pnl,
        "total_pnl_pct": operation.total_pnl_pct,
        "open_positions": len(open_positions),
        "unrealized_pnl": total_unrealized_pnl,
        "current_capital": operation.current_capital,
        "last_synced_at": operation.updated_at.isoformat() if operation.updated_at else None,
    }


@app.post("/api/operations/{operation_id}/reconcile")
async def reconcile_operation(
    operation_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Force an immediate broker reconciliation for this operation.

    Syncs positions, marks stuck PENDING orders correctly, and refreshes
    current_capital from the real broker balance.
    """
    try:
        op_id = ObjectId(operation_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid operation ID")

    operation = await TradingOperation.get(op_id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")

    try:
        await engine.sync_positions_from_broker(operation)
        return {"message": "Reconciliation complete", "operation_id": operation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {e}")


@app.get("/api/stats/overall")
async def get_overall_stats(engine: TradingEngine = Depends(get_trading_engine)):
    """Get overall statistics across all operations, with live account info from broker"""
    operations = await TradingOperation.find_all().to_list()
    all_closed = await Position.find(Position.status == PositionStatus.CLOSED).to_list()

    total_pnl = sum(op.total_pnl for op in operations)

    account_info = await engine.adapter.get_account_info()

    return {
        "total_operations": len(operations),
        "active_operations": len([op for op in operations if op.status == "active"]),
        "total_trades": len(all_closed),
        "total_pnl": total_pnl,
        "balance": account_info.balance,
        "equity": account_info.equity or 0.0,
        "floating_pnl": 0.0,
        "margin_used": account_info.margin_used or 0.0,
        "margin_available": account_info.margin_free or 0.0,
        "currency": account_info.currency,
    }


@app.on_event("startup")
async def startup_event():
    """Initialize trading engine on startup"""
    global trading_engine

    # Import here to avoid circular imports
    from live_trading.brokers import IBKRBroker, OANDABroker, PepperstoneBroker, CTraderBroker
    from live_trading.journal.journal_manager import JournalManager

    # Log broker type for debugging
    logger.info(f"Initializing broker: {config.BROKER_TYPE} (from env: {os.getenv('BROKER_TYPE', 'not set')})")

    # Initialize broker based on config
    if config.BROKER_TYPE == "IBKR":
        if IBKRBroker is None:
            raise ImportError(
                "IBKR broker requested but 'ibapi' module not found. "
                "Please install IBKR API: pip install -e ./IBJts/source/pythonclient"
            )
        broker = IBKRBroker()
        connected = await broker.connect()
        if not connected:
            logger.warning(
                "Failed to connect to IBKR. Make sure TWS/Gateway is running and API is enabled. "
                "The system will continue but trading operations will not work until IBKR is connected."
            )
    elif config.BROKER_TYPE == "OANDA":
        if OANDABroker is None:
            raise ImportError(
                "OANDA broker requested but 'oandapyV20' module not found. "
                "Please install OANDA API: pip install oandapyV20"
            )
        broker = OANDABroker()
        connected = await broker.connect()
        if not connected:
            logger.warning(
                "Failed to connect to OANDA. Make sure OANDA_API_KEY and OANDA_ACCOUNT_ID are set correctly. "
                "The system will continue but trading operations will not work until OANDA is connected."
            )
    elif config.BROKER_TYPE == "PEPPERSTONE":
        if PepperstoneBroker is None:
            raise ImportError(
                "Pepperstone broker requested but 'MetaTrader5' module not found. "
                "Please install MetaTrader5: pip install MetaTrader5"
            )
        broker = PepperstoneBroker()
        connected = await broker.connect()
        if not connected:
            logger.warning(
                "Failed to connect to Pepperstone. Make sure MetaTrader5 terminal is installed and running, "
                "and PEPPERSTONE_LOGIN, PEPPERSTONE_PASSWORD, and PEPPERSTONE_SERVER are set correctly. "
                "The system will continue but trading operations will not work until Pepperstone is connected."
            )
    elif config.BROKER_TYPE == "CTRADER":
        if CTraderBroker is None:
            raise ImportError(
                "cTrader broker requested but 'ctrader-open-api' module not found. "
                "Please install cTrader Open API: pip install ctrader-open-api"
            )
        from live_trading.notifications.connection_manager import ConnectionManager
        from live_trading.notifications.telegram_notifier import TelegramNotifier

        def _make_ctrader_broker(bus):
            return CTraderBroker(event_bus=bus)

        _manager = ConnectionManager(broker_factory=_make_ctrader_broker)
        _notifier = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            environment=config.CTRADER_ENVIRONMENT,
        )
        _manager._bus.subscribe(_notifier.on_event)

        connected = await _manager.connect()
        broker = _manager._broker
        if not connected:
            logger.warning(
                "Failed to connect to cTrader. Make sure CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, "
                "and CTRADER_ACCESS_TOKEN are set correctly. "
                "The system will continue but trading operations will not work until cTrader is connected."
            )
    else:
        raise ValueError(
            f"Unsupported broker type: {config.BROKER_TYPE}. "
            f"Supported types: IBKR, OANDA, PEPPERSTONE, CTRADER"
        )

    # Build the broker adapter
    from live_trading.brokers.adapters.ctrader_adapter import CTraderAdapter as _CTA

    if config.BROKER_TYPE == "CTRADER":
        adapter = _CTA(broker)
    else:
        raise ValueError(
            f"No BrokerAdapter implemented for {config.BROKER_TYPE}. "
            "Only CTRADER is currently supported."
        )

    journal_manager = JournalManager()

    trading_engine = TradingEngine(broker, adapter, journal_manager)
    await trading_engine.initialize()

    await trading_engine.recover_from_journal()

    import asyncio
    async def periodic_position_sync():
        while True:
            try:
                await asyncio.sleep(30)
                if trading_engine:
                    active_operations = await TradingOperation.find(
                        TradingOperation.status == "active"
                    ).to_list()
                    for operation in active_operations:
                        await trading_engine.sync_positions_from_broker(operation)
            except Exception as e:
                logger.error(f"Error in periodic position sync: {e}", exc_info=True)

    asyncio.create_task(periodic_position_sync())

    print(f"Trading engine started on {config.API_HOST}:{config.API_PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown trading engine"""
    global trading_engine
    if trading_engine:
        await trading_engine.shutdown()


# ============================================================================
# Connection Health Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/connection/status")
async def get_connection_status(engine: TradingEngine = Depends(get_trading_engine)):
    """Get detailed broker connection status"""
    status = {
        "broker_type": config.BROKER_TYPE,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Get broker-specific status
    if hasattr(engine.broker, 'get_connection_status'):
        status["broker"] = engine.broker.get_connection_status()
    else:
        status["broker"] = {
            "connected": getattr(engine.broker, 'connected', None),
            "authenticated": getattr(engine.broker, 'authenticated', None)
        }

    # Get data manager staleness info
    if hasattr(engine, 'data_manager') and engine.data_manager:
        status["data_staleness"] = engine.data_manager.get_all_staleness_info()

    return status


@app.get("/api/subscriptions")
async def get_subscriptions(engine: TradingEngine = Depends(get_trading_engine)):
    """Get active market data subscriptions with callback details"""
    if not engine.broker:
        raise HTTPException(status_code=503, detail="Broker not available")

    status = engine.broker.get_connection_status()

    return {
        "broker_type": config.BROKER_TYPE,
        "timestamp": datetime.utcnow().isoformat(),
        "subscriptions": status.get("subscriptions", []),
        "subscription_details": status.get("subscription_details", {}),
        "connected": status.get("connected", False),
        "authenticated": status.get("authenticated", False)
    }


@app.post("/api/connection/reconnect")
async def force_reconnect(engine: TradingEngine = Depends(get_trading_engine)):
    """Force a broker reconnection"""
    if not hasattr(engine.broker, '_attempt_reconnect'):
        raise HTTPException(status_code=400, detail="Broker does not support reconnection")

    try:
        # Reset reconnect attempts to allow manual reconnect
        engine.broker._reconnect_attempts = 0
        await engine.broker._attempt_reconnect()
        return {"status": "reconnecting", "message": "Reconnection initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconnection failed: {str(e)}")


# ============================================================================
# Logging & Daemon Management Endpoints
# ============================================================================

@app.get("/api/logs")
async def get_logs(
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """
    Retrieve application logs with optional filters.

    Query Parameters:
    - level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger_name: Filter by logger name (partial match)
    - search: Search in log messages
    - limit: Maximum number of logs to return (default: 100)
    - offset: Offset for pagination
    - start_time: ISO format start time filter
    - end_time: ISO format end time filter
    """
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()

        # Parse time filters
        start_dt = None
        end_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_time format")
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_time format")

        logs = log_manager.get_logs(
            start_time=start_dt,
            end_time=end_dt,
            level=level,
            logger=logger_name,
            search=search,
            limit=limit,
            offset=offset
        )

        return {
            "logs": logs,
            "count": len(logs),
            "limit": limit,
            "offset": offset,
            "filters": {
                "level": level,
                "logger": logger_name,
                "search": search,
                "start_time": start_time,
                "end_time": end_time
            }
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/stats")
async def get_log_stats():
    """Get logging statistics including file sizes and log level counts"""
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()
        return log_manager.get_stats()
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/files")
async def get_log_files():
    """Get list of available log files with metadata"""
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()
        storage = log_manager.storage

        files = []

        # Current log file
        if storage.current_file.exists():
            stat = storage.current_file.stat()
            files.append({
                "name": storage.current_file.name,
                "path": str(storage.current_file),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "current"
            })

        # Rotated files
        for i in range(1, storage.max_files + 1):
            path = storage.log_dir / f"live_trading.log.{i}"
            path_gz = storage.log_dir / f"live_trading.log.{i}.gz"

            if path.exists():
                stat = path.stat()
                files.append({
                    "name": path.name,
                    "path": str(path),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": "rotated",
                    "compressed": False
                })
            elif path_gz.exists():
                stat = path_gz.stat()
                files.append({
                    "name": path_gz.name,
                    "path": str(path_gz),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": "rotated",
                    "compressed": True
                })

        # Archive files
        for archive_file in sorted(storage.archive_dir.glob("*.log.gz"), reverse=True):
            stat = archive_file.stat()
            files.append({
                "name": archive_file.name,
                "path": str(archive_file),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "archive",
                "compressed": True
            })

        return {
            "files": files,
            "total_count": len(files),
            "log_directory": str(storage.log_dir)
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/count")
async def get_logs_count(
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    search: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get total count of logs matching filters (for pagination)"""
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()

        # Parse time filters
        start_dt = None
        end_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                pass
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            except ValueError:
                pass

        # Get all matching logs (with high limit) to count
        logs = log_manager.get_logs(
            level=level,
            logger=logger_name,
            search=search,
            start_time=start_dt,
            end_time=end_dt,
            limit=100000,  # High limit for counting
            offset=0
        )

        return {
            "count": len(logs),
            "filters": {
                "level": level,
                "logger_name": logger_name,
                "search": search,
                "start_time": start_time,
                "end_time": end_time
            }
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/errors")
async def get_recent_errors(limit: int = 50):
    """Get recent error logs"""
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()
        return {
            "errors": log_manager.get_recent_errors(limit=limit),
            "limit": limit
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")


@app.get("/api/logs/warnings")
async def get_recent_warnings(limit: int = 50):
    """Get recent warning logs"""
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()
        return {
            "warnings": log_manager.get_recent_warnings(limit=limit),
            "limit": limit
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")


@app.post("/api/logs/cleanup")
async def cleanup_old_logs(older_than_days: int = 30):
    """Clean up old log files"""
    try:
        from live_trading.logging import get_log_manager
        log_manager = get_log_manager()
        result = log_manager.cleanup_old_logs(older_than_days=older_than_days)
        return {
            "success": True,
            "older_than_days": older_than_days,
            **result
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Logging system not initialized")


@app.get("/api/daemon/status")
async def get_daemon_status():
    """Get daemon process status"""
    try:
        from live_trading.daemon import get_daemon_manager
        daemon_manager = get_daemon_manager()
        return daemon_manager.get_status()
    except ImportError:
        return {
            "running": True,
            "mode": "interactive",
            "message": "Running in interactive mode (not as daemon)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

