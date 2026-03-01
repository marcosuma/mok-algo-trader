"""
cTrader Broker Adapter using Open API.
"""
import asyncio
import atexit
import logging
import threading
import weakref
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
from collections import defaultdict

# Keep track of all CTraderBroker instances for cleanup
_broker_instances: List[weakref.ref] = []


def _cleanup_all_brokers():
    """Cleanup function called at exit to stop all reactor threads"""
    for ref in _broker_instances:
        broker = ref()
        if broker is not None:
            try:
                broker._force_stop()
            except Exception:
                pass


# Register cleanup at module level
atexit.register(_cleanup_all_brokers)

try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, Auth, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
    from twisted.internet import reactor, defer
    from twisted.internet.defer import Deferred
    CTRADER_AVAILABLE = True
except ImportError:
    CTRADER_AVAILABLE = False
    Client = None
    Protobuf = None
    TcpProtocol = None
    Auth = None
    EndPoints = None
    reactor = None  # type: ignore[assignment]
    defer = None  # type: ignore[assignment]
    Deferred = None  # type: ignore[assignment,misc]

from live_trading.brokers.base_broker import BaseBroker
from live_trading.config import config

logger = logging.getLogger(__name__)


def twisted_to_asyncio(deferred: Deferred, loop: asyncio.AbstractEventLoop):
    """Convert a Twisted Deferred to an asyncio Future"""
    future = loop.create_future()

    def callback(result):
        if not future.done():
            future.set_result(result)

    def errback(failure):
        if not future.done():
            future.set_exception(failure.value)

    deferred.addCallbacks(callback, errback)
    return future


class CTraderBroker(BaseBroker):
    """cTrader Broker Adapter using Open API"""

    def __init__(self):
        if not CTRADER_AVAILABLE:
            logger.warning(
                "ctrader-open-api module not found. Please install it: pip install ctrader-open-api"
            )
        self.client: Optional[Client] = None
        self.connected = False
        self.authenticated = False
        self.account_id: Optional[int] = None
        self._data_subscriptions: Dict[str, int] = {}  # asset -> symbol_id
        # Multiple callbacks per asset (supports multiple operations on same asset)
        self._data_callbacks: Dict[str, List[Callable]] = {}  # asset -> [callbacks]
        self._data_callback_ids: Dict[str, List[str]] = {}  # asset -> [callback_ids] for tracking
        self._trendbar_subscriptions: Dict[str, set] = {}  # asset -> set of subscribed period IDs
        self._live_trendbar_volumes: Dict[str, Dict[int, int]] = {}  # symbol_id -> {period -> volume}
        self._symbol_cache: Dict[str, int] = {}  # asset -> symbol_id
        self._symbol_id_to_name: Dict[int, str] = {}  # symbol_id -> asset (reverse lookup)
        self._symbol_digits: Dict[int, int] = {}  # symbol_id -> digits (decimal places)
        self._symbol_digits_verified: set = set()  # symbol_ids with digits fetched from full symbol info
        self._order_callbacks: Dict[str, Callable] = {}  # order_id -> callback
        self._pending_order_futures: Dict[str, Any] = {}  # clientMsgId -> (asyncio.Future, loop, order_status_callback)
        self._reactor_thread: Optional[threading.Thread] = None
        self._pending_requests: Dict[int, Deferred] = {}  # request_id -> deferred
        self._request_id_counter = 0
        self._positions_cache: List[Dict[str, Any]] = []
        self._open_orders_cache: List[Dict[str, Any]] = []  # pending broker orders from last reconcile
        self._account_info_cache: Dict[str, Any] = {}
        self._last_spot_prices: Dict[int, Dict[str, float]] = {}  # symbol_id -> {bid, ask, mid}
        self._historical_data_callbacks: Dict[int, Callable] = {}  # request_id -> callback
        self._historical_data_context: Dict[int, Dict] = {}  # request_id -> context
        self._connection_error: Optional[str] = None  # Store connection errors
        self._auth_error: Optional[str] = None  # Store authentication errors
        self._shutdown_requested = False

        # Reconnection state
        self._reconnecting = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds
        self._last_message_time: Optional[datetime] = None
        self._connection_monitor_task: Optional[asyncio.Task] = None

        # Register this instance for cleanup at exit
        _broker_instances.append(weakref.ref(self))

    def _get_next_request_id(self) -> int:
        """Get next request ID"""
        self._request_id_counter += 1
        return self._request_id_counter

    def _convert_asset_to_symbol(self, asset: str) -> str:
        """
        Convert asset format (USD-CAD) to cTrader symbol format.
        cTrader brokers use different formats: 'USD/CAD', 'USDCAD', etc.
        """
        # Return the base format without hyphen - we'll match flexibly
        return asset.replace("-", "")

    def _convert_symbol_to_asset(self, symbol: str) -> str:
        """Convert cTrader symbol to asset format (EUR-USD)"""
        # Remove common suffixes
        clean_symbol = symbol
        for suffix in [".fx", ".FX", ".pro", ".PRO", ".ecn", ".ECN", ".m", ".M"]:
            if clean_symbol.endswith(suffix):
                clean_symbol = clean_symbol[:-len(suffix)]
                break

        # Handle different formats
        clean_symbol = clean_symbol.replace("/", "").replace(".", "").replace("_", "")

        # For 6-character forex pairs (e.g., EURUSD), insert hyphen
        if len(clean_symbol) == 6 and clean_symbol.isalpha():
            return f"{clean_symbol[:3]}-{clean_symbol[3:]}"

        # For other formats, just replace separators with hyphen
        return symbol.replace("/", "-").replace("_", "-")

    def _symbol_matches_asset(self, symbol_name: str, asset: str) -> bool:
        """Check if a cTrader symbol matches our asset format"""
        # Normalize both to uppercase without separators
        asset_normalized = asset.replace("-", "").upper()
        symbol_normalized = symbol_name.upper()

        # Remove common suffixes from symbol
        for suffix in [".FX", ".PRO", ".ECN", ".M", ".C"]:
            if symbol_normalized.endswith(suffix):
                symbol_normalized = symbol_normalized[:-len(suffix)]
                break

        # Remove separators
        symbol_normalized = symbol_normalized.replace("/", "").replace(".", "").replace("_", "")

        return symbol_normalized == asset_normalized

    def _convert_price(self, raw_price: float, symbol_id: int) -> float:
        """
        Convert a cTrader raw tick integer to an actual price.

        All integer price fields in the cTrader Open API (ProtoOASpotEvent bid/ask,
        ProtoOATrendbar low/delta*, etc.) use a FIXED scale of 1/100,000 regardless
        of the symbol's 'digits' value.  The 'digits' field is purely for
        display/rounding purposes, not for the primary unit conversion.

        Reference:
          https://help.ctrader.com/open-api/protocol/trendbars/
          "All price values are represented in 1/100000 of the base currency pair.
           E.g. EURUSD 1.08500 → 108500; ETHUSD 1992.68 → 199268000"

        NOTE: proto `double` fields (ProtoOAPosition.price, order.executionPrice)
        are already in actual price units and must NOT be passed through here.
        """
        digits = self._symbol_digits.get(symbol_id, 5)
        return round(raw_price / 100000.0, max(digits, 2))

    def _round_price_for_symbol(self, price: float, symbol_id: int) -> float:
        """
        Round a price to the number of decimal places allowed by the symbol.

        cTrader rejects orders whose price has more decimal digits than
        the symbol permits (e.g. XAUUSD allows 2 digits).
        """
        digits = self._symbol_digits.get(symbol_id, 5)
        return round(price, digits)

    def _start_reactor(self):
        """Start Twisted reactor in a separate thread"""
        def run_reactor():
            try:
                logger.info("[Reactor Thread] Starting Twisted reactor...")
                reactor.run(installSignalHandlers=0)
                logger.info("[Reactor Thread] Twisted reactor stopped")
            except Exception as e:
                logger.error(f"[Reactor Thread] Twisted reactor error: {e}", exc_info=True)
                self._connection_error = f"Twisted reactor error: {e}"

        if not reactor.running:
            self._reactor_thread = threading.Thread(target=run_reactor, daemon=True, name="TwistedReactor")
            self._reactor_thread.start()
            # Wait for reactor to actually be running
            import time
            max_wait = 5.0
            waited = 0
            while not reactor.running and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            if reactor.running:
                logger.info(f"✓ Twisted reactor started successfully (waited {waited:.1f}s)")
            else:
                logger.error("✗ Twisted reactor failed to start within timeout")
        else:
            logger.info("Twisted reactor already running")

    def _stop_reactor(self):
        """Stop Twisted reactor safely from any thread"""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True

        logger.info("Stopping Twisted reactor...")
        try:
            if CTRADER_AVAILABLE and reactor.running:
                stop_complete = threading.Event()

                def do_stop():
                    try:
                        # First, try to close the client connection
                        if self.client:
                            try:
                                self.client.stopService()
                            except Exception:
                                pass
                        # Then stop the reactor
                        reactor.stop()
                    except Exception as e:
                        logger.debug(f"Error in reactor stop: {e}")
                    finally:
                        stop_complete.set()

                # Schedule the stop on the reactor thread
                reactor.callFromThread(do_stop)

                # Wait for stop to complete with short timeout
                if not stop_complete.wait(timeout=2.0):
                    logger.warning("Reactor stop timed out, forcing...")
                    # Try crash as last resort (more aggressive)
                    try:
                        reactor.callFromThread(reactor.crash)
                    except Exception:
                        pass

                # Wait for reactor thread to finish
                if self._reactor_thread and self._reactor_thread.is_alive():
                    logger.info("Waiting for reactor thread to finish...")
                    self._reactor_thread.join(timeout=2.0)
                    if self._reactor_thread.is_alive():
                        logger.warning("Reactor thread did not stop within timeout (will be cleaned up on exit)")
                    else:
                        logger.info("✓ Reactor thread stopped")
            else:
                logger.info("Reactor was not running")
        except Exception as e:
            logger.error(f"Error stopping reactor: {e}", exc_info=True)

    def _force_stop(self):
        """Force stop the reactor - called at exit"""
        self._shutdown_requested = True
        try:
            if CTRADER_AVAILABLE and reactor.running:
                try:
                    reactor.callFromThread(reactor.crash)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_connected(self, client: Client):
        """Callback when client connects - runs on reactor thread"""
        logger.info("[CONNECTION] ✅ Connected to cTrader server")
        self.connected = True
        self._connection_error = None
        self._last_message_time = datetime.utcnow()  # Reset message time on connect

        # Authenticate application
        if not config.CTRADER_CLIENT_ID:
            error_msg = "CTRADER_CLIENT_ID environment variable is not set"
            logger.error(f"[Reactor Thread] ✗ {error_msg}")
            self._auth_error = error_msg
            return

        if not config.CTRADER_CLIENT_SECRET:
            error_msg = "CTRADER_CLIENT_SECRET environment variable is not set"
            logger.error(f"[Reactor Thread] ✗ {error_msg}")
            self._auth_error = error_msg
            return

        logger.info(f"[Reactor Thread] Authenticating application (Client ID: {config.CTRADER_CLIENT_ID[:8]}...)")

        try:
            request = ProtoOAApplicationAuthReq()
            request.clientId = config.CTRADER_CLIENT_ID
            request.clientSecret = config.CTRADER_CLIENT_SECRET

            deferred = client.send(request)
            deferred.addCallbacks(self._on_application_auth_success, self._on_auth_error)
            logger.info("[Reactor Thread] Application authentication request sent")
        except Exception as e:
            error_msg = f"Failed to send application auth request: {e}"
            logger.error(f"[Reactor Thread] ✗ {error_msg}", exc_info=True)
            self._auth_error = error_msg

    def _on_application_auth_success(self, message):
        """Callback when application authentication succeeds - runs on reactor thread"""
        # Extract the actual message from the wrapper
        try:
            if CTRADER_AVAILABLE and Protobuf:
                extracted = Protobuf.extract(message)
                logger.info(f"[Reactor Thread] App auth response type: {type(extracted).__name__}")
            else:
                extracted = message
        except Exception as e:
            logger.error(f"[Reactor Thread] Failed to extract message: {e}")
            extracted = message

        logger.info("[Reactor Thread] ✓ Application authenticated successfully")

        # Check for errors in the response
        if hasattr(extracted, 'errorCode') and extracted.errorCode:
            error_msg = f"Application auth error code: {extracted.errorCode}"
            logger.error(f"[Reactor Thread] ✗ {error_msg}")
            self._auth_error = error_msg
            return

        # Get account list - use ProtoOAGetAccountListByAccessTokenReq
        if self.client:
            access_token = config.CTRADER_ACCESS_TOKEN
            if not access_token:
                error_msg = (
                    "CTRADER_ACCESS_TOKEN is not set - cannot retrieve accounts. "
                    "Generate one with 'trading' scope: python -m live_trading.scripts.get_ctrader_token"
                )
                logger.error(f"[Reactor Thread] ✗ {error_msg}")
                self._auth_error = error_msg
                return

            logger.info("[Reactor Thread] Requesting account list...")
            try:
                request = ProtoOAGetAccountListByAccessTokenReq()
                request.accessToken = access_token
                deferred = self.client.send(request)
                deferred.addCallbacks(self._on_account_list, self._on_auth_error)
            except Exception as e:
                error_msg = f"Failed to request account list: {e}"
                logger.error(f"[Reactor Thread] ✗ {error_msg}", exc_info=True)
                self._auth_error = error_msg

    def _on_account_list(self, message):  # type: ignore
        """Callback when account list is received - runs on reactor thread"""
        logger.info(f"[Reactor Thread] Account list raw response: {message}")

        # Extract the actual message from the wrapper using Protobuf helper
        try:
            if CTRADER_AVAILABLE and Protobuf:
                extracted = Protobuf.extract(message)
                logger.info(f"[Reactor Thread] Extracted message type: {type(extracted).__name__}")
                logger.info(f"[Reactor Thread] Extracted message: {extracted}")
            else:
                extracted = message
        except Exception as e:
            logger.error(f"[Reactor Thread] Failed to extract message: {e}")
            extracted = message

        # Check permission scope - warn if view-only
        permission_scope = getattr(extracted, 'permissionScope', None)
        if permission_scope is not None:
            # ProtoOAAccessRights: 0 = SCOPE_VIEW, 1 = SCOPE_TRADE
            scope_name = "SCOPE_TRADE" if permission_scope == 1 else "SCOPE_VIEW"
            logger.info(f"[Reactor Thread] Access token permission scope: {scope_name} ({permission_scope})")
            if permission_scope == 0:
                logger.warning(
                    "[Reactor Thread] ⚠️ Access token has SCOPE_VIEW only - TRADING WILL NOT WORK! "
                    "Regenerate your access token with 'trading' scope using: "
                    "python -m live_trading.scripts.get_ctrader_token"
                )

        # Check for ctidTraderAccount (correct field name for the response)
        accounts = getattr(extracted, 'ctidTraderAccount', None) or getattr(extracted, 'account', None)

        # Log available fields for debugging
        if hasattr(extracted, 'DESCRIPTOR'):
            fields = [f.name for f in extracted.DESCRIPTOR.fields]
            logger.info(f"[Reactor Thread] Available fields: {fields}")

        if not accounts or len(accounts) == 0:
            error_msg = "No trading accounts found. Check your access token and account permissions."
            logger.error(f"[Reactor Thread] ✗ {error_msg}")
            self._auth_error = error_msg
            return

        # Use first account (or find by account_id if specified)
        account = accounts[0]
        logger.info(f"[Reactor Thread] First account: {account}")
        self.account_id = account.ctidTraderAccountId if hasattr(account, 'ctidTraderAccountId') else account.accountId
        logger.info(f"[Reactor Thread] ✓ Found {len(accounts)} account(s). Using account ID: {self.account_id}")

        # Authenticate account
        if self.client:
            access_token = config.CTRADER_ACCESS_TOKEN
            if not access_token:
                error_msg = (
                    "CTRADER_ACCESS_TOKEN is not set. Generate one with 'trading' scope: "
                    "python -m live_trading.scripts.get_ctrader_token"
                )
                logger.error(f"[Reactor Thread] ✗ {error_msg}")
                self._auth_error = error_msg
                return

            logger.info("[Reactor Thread] Authenticating trading account...")
            try:
                request = ProtoOAAccountAuthReq()
                request.ctidTraderAccountId = self.account_id
                request.accessToken = access_token
                deferred = self.client.send(request)
                deferred.addCallbacks(self._on_account_auth_success, self._on_auth_error)
            except Exception as e:
                error_msg = f"Failed to send account auth request: {e}"
                logger.error(f"[Reactor Thread] ✗ {error_msg}", exc_info=True)
                self._auth_error = error_msg

    def _on_account_auth_success(self, message):
        """Callback when account authentication succeeds - runs on reactor thread"""
        # Extract the actual message from the wrapper
        try:
            if CTRADER_AVAILABLE and Protobuf:
                extracted = Protobuf.extract(message)
                logger.info(f"[Reactor Thread] Account auth response type: {type(extracted).__name__}")
            else:
                extracted = message
        except Exception as e:
            logger.error(f"[Reactor Thread] Failed to extract message: {e}")
            extracted = message

        # Check for errors in the response
        if hasattr(extracted, 'errorCode') and extracted.errorCode:
            error_msg = f"Account auth error code: {extracted.errorCode}"
            logger.error(f"[Reactor Thread] ✗ {error_msg}")
            self._auth_error = error_msg
            return

        logger.info(f"[CONNECTION] ✅ Authenticated! Account ID: {self.account_id}")
        self.authenticated = True
        self._auth_error = None
        self._last_message_time = datetime.utcnow()  # Reset message time on auth
        # Now we can start trading operations

    def _on_auth_error(self, failure):
        """Error callback for authentication failures - runs on reactor thread"""
        # Try to extract more details from the failure
        error_msg = f"Authentication error: {failure}"
        logger.error(f"[Reactor Thread] ✗ {error_msg}")

        # Try to extract the actual error message if it's a protobuf error response
        try:
            if CTRADER_AVAILABLE and Protobuf and hasattr(failure, 'value'):
                extracted = Protobuf.extract(failure.value)
                if hasattr(extracted, 'errorCode'):
                    logger.error(f"[Reactor Thread]   Error code: {extracted.errorCode}")
                if hasattr(extracted, 'description'):
                    logger.error(f"[Reactor Thread]   Description: {extracted.description}")
        except Exception:
            pass

        if hasattr(failure, 'value'):
            logger.error(f"[Reactor Thread]   Error details: {failure.value}")
        if hasattr(failure, 'getTraceback'):
            try:
                tb = failure.getTraceback()
                if tb:
                    logger.error(f"[Reactor Thread]   Traceback: {tb}")
            except Exception:
                pass
        self._auth_error = str(failure)

    def _on_disconnected(self, client: Client, reason):
        """Callback when client disconnects - runs on reactor thread"""
        logger.error(f"[CONNECTION] ❌ DISCONNECTED from cTrader: {reason}")
        logger.error(f"[CONNECTION] ❌ Market data streaming has stopped!")
        self._connection_error = f"Disconnected: {reason}"
        was_connected = self.connected
        self.connected = False
        self.authenticated = False

        # Trigger reconnection if not shutting down
        if not self._shutdown_requested and not self._reconnecting:
            logger.info(f"[CONNECTION] 🔄 Will attempt to reconnect (was_connected={was_connected})...")
            # Schedule reconnection on asyncio loop
            try:
                # Try to get the running loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()

                if loop and loop.is_running():
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self._attempt_reconnect()))
                else:
                    logger.error(f"[CONNECTION] ❌ No running event loop - cannot schedule reconnection!")
            except Exception as e:
                logger.error(f"[CONNECTION] ❌ Failed to schedule reconnection: {e}", exc_info=True)
        elif self._shutdown_requested:
            logger.info(f"[CONNECTION] Shutdown requested - not reconnecting")
        elif self._reconnecting:
            logger.info(f"[CONNECTION] Already reconnecting - skipping")

    def _on_message_received(self, client: Client, message):
        """Callback for all received messages - runs on reactor thread"""
        try:
            # Track last message time for connection health monitoring
            self._last_message_time = datetime.utcnow()

            # Use Protobuf.extract() to get the actual message from the wrapper
            # This is the correct way to handle cTrader Open API messages
            if not CTRADER_AVAILABLE or not Protobuf:
                logger.warning("[Reactor Thread] Protobuf helper not available")
                return

            try:
                extracted = Protobuf.extract(message)
                message_type = type(extracted).__name__
                logger.debug(f"[Reactor Thread] Message received: {message_type}")
            except Exception as e:
                logger.debug(f"[Reactor Thread] Could not extract message: {e}")
                return

            # Handle different message types based on class name
            if message_type == "ProtoOASpotEvent":
                self._handle_spot_event(extracted)
            elif message_type == "ProtoOAExecutionEvent":
                self._handle_execution_event(extracted)
            elif message_type == "ProtoOASubscribeSpotsRes":
                self._handle_subscribe_spots_res(extracted)
            elif message_type == "ProtoOASubscribeLiveTrendbarRes":
                logger.debug("[Reactor Thread] Live trendbar subscription confirmed")
            elif message_type in ("ProtoOAGetAccountListByAccessTokenRes", "ProtoOAGetAccountListRes"):
                # Already handled in _on_account_list callback
                pass
            elif message_type == "ProtoOAAccountAuthRes":
                # Already handled in _on_account_auth_success callback
                pass
            elif message_type == "ProtoOAApplicationAuthRes":
                # Already handled in _on_application_auth_success callback
                pass
            elif message_type == "ProtoOAOrderErrorEvent":
                self._handle_order_error(extracted)
            elif message_type in ("ProtoOAReconcileRes", "ProtoOAGetPositionsRes"):
                self._handle_get_positions_res(extracted)
            elif message_type in ("ProtoOASymbolsListRes", "ProtoOASymbolByIdRes"):
                self._handle_symbols_list_res(extracted)
            elif message_type == "ProtoOAGetTickDataRes":
                self._handle_get_tick_data_res(extracted)
            elif message_type == "ProtoOAGetTrendbarsRes":
                # Handled via deferred callback in fetch_historical_data
                logger.debug(f"[Reactor Thread] Received trendbar response with {len(extracted.trendbar) if hasattr(extracted, 'trendbar') else 0} bars")
            elif message_type == "ProtoOAErrorRes":
                # Handle error responses - dump all fields for debugging
                error_code = getattr(extracted, 'errorCode', 'UNKNOWN')
                description = getattr(extracted, 'description', 'No description')
                # Log the full protobuf message for debugging
                logger.error(f"[Reactor Thread] ✗ cTrader API Error: {error_code} - {description}")
                logger.error(f"[Reactor Thread] Full error response: {extracted}")
                self._auth_error = f"API Error: {error_code} - {description}"
                # Resolve any pending order futures on generic API error
                resolved_key = None
                for key, (future, loop, _) in self._pending_order_futures.items():
                    if not future.done():
                        logger.error(f"[ORDER] Resolving pending order future as FAILED: {error_code} - {description}")
                        loop.call_soon_threadsafe(future.set_result, "")
                        resolved_key = key
                        break
                if resolved_key:
                    del self._pending_order_futures[resolved_key]
            elif message_type == "ProtoHeartbeatEvent":
                # Heartbeat - log at debug level for monitoring
                logger.debug("[Reactor Thread] Heartbeat received - connection is alive")
            else:
                logger.debug(f"[Reactor Thread] Unhandled message type: {message_type}")

        except Exception as e:
            logger.error(f"[Reactor Thread] Error handling message: {e}", exc_info=True)

    def _handle_spot_event(self, event):  # type: ignore
        """Handle spot price updates"""
        try:
            symbol_id = event.symbolId

            # Convert prices from cTrader integer format to actual values
            raw_bid = event.bid
            raw_ask = event.ask
            bid = self._convert_price(raw_bid, symbol_id)
            ask = self._convert_price(raw_ask, symbol_id)

            # Log conversion details on first spot for debugging
            digits = self._symbol_digits.get(symbol_id, 5)
            logger.debug(f"Spot price conversion: raw_bid={raw_bid} -> {bid:.5f}, raw_ask={raw_ask} -> {ask:.5f} (digits={digits})")

            # Cache latest bid/ask per symbol so position P&L can be computed in real-time.
            # Written on the reactor thread; read from the asyncio thread — safe under Python GIL.
            if bid > 0 and ask > 0:
                self._last_spot_prices[symbol_id] = {
                    "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
                }

            # Extract volume from live trendbar data if present
            # ProtoOASpotEvent includes trendbar field when subscribed via ProtoOASubscribeLiveTrendbarReq
            tick_volume = 0
            if hasattr(event, 'trendbar') and event.trendbar:
                for tb in event.trendbar:
                    vol = getattr(tb, 'volume', 0)
                    period = getattr(tb, 'period', None)
                    if vol > 0:
                        tick_volume = max(tick_volume, vol)
                    # Cache the latest volume per period for this symbol
                    if period is not None:
                        if symbol_id not in self._live_trendbar_volumes:
                            self._live_trendbar_volumes[symbol_id] = {}
                        self._live_trendbar_volumes[symbol_id][period] = vol
                logger.debug(f"[cTrader] Spot event has {len(event.trendbar)} trendbar(s), volume={tick_volume}")

            # Find asset by symbol_id
            asset = None
            for cached_asset, cached_symbol_id in self._symbol_cache.items():
                if cached_symbol_id == symbol_id:
                    asset = cached_asset
                    break

            if asset and asset in self._data_callbacks:
                mid_price = (bid + ask) / 2.0

                # Validate prices - reject obviously bad data
                if bid <= 0 or ask <= 0 or mid_price <= 0:
                    logger.error(f"[cTrader] Rejecting invalid prices for {asset}: bid={bid}, ask={ask}, mid={mid_price}")
                    return

                # Check for unreasonable spread (>1% is suspicious for forex)
                spread_pct = abs(ask - bid) / bid * 100 if bid > 0 else 0
                if spread_pct > 1:
                    logger.warning(f"[cTrader] Unusually large spread for {asset}: {spread_pct:.2f}% (bid={bid:.5f}, ask={ask:.5f})")

                tick_data = {
                    "type": "tick",
                    "tick_type": 0,  # MID
                    "tick_name": "MID",
                    "price": mid_price,
                    "bid": bid,
                    "ask": ask,
                    "size": tick_volume,
                    "timestamp": datetime.utcnow()
                }
                # Fan out to ALL registered callbacks for this asset
                callbacks = self._data_callbacks[asset]
                callback_ids = self._data_callback_ids.get(asset, [])
                for i, callback in enumerate(callbacks):
                    try:
                        callback(tick_data)
                    except Exception as cb_error:
                        cb_id = callback_ids[i] if i < len(callback_ids) else f"callback_{i}"
                        logger.error(f"Error in callback {cb_id} for {asset}: {cb_error}")
                logger.debug(f"Spot update for {asset}: mid={mid_price:.5f}, bid={bid:.5f}, ask={ask:.5f}, vol={tick_volume} (sent to {len(callbacks)} callbacks)")

        except Exception as e:
            logger.error(f"Error handling spot event: {e}", exc_info=True)

    def _handle_execution_event(self, event):  # type: ignore
        """Handle order execution events (runs on reactor thread)"""
        try:
            order = event.order
            order_id = str(order.orderId)
            execution_type = getattr(event, 'executionType', None)

            logger.info(f"[ORDER] Execution event: order_id={order_id}, executionType={execution_type}, orderStatus={order.orderStatus}")

            # Map cTrader order status to our format
            status_map = {
                ProtoOAOrderStatus.ORDER_STATUS_ACCEPTED: "ACCEPTED",
                ProtoOAOrderStatus.ORDER_STATUS_FILLED: "FILLED",
                ProtoOAOrderStatus.ORDER_STATUS_REJECTED: "REJECTED",
                ProtoOAOrderStatus.ORDER_STATUS_EXPIRED: "EXPIRED",
                ProtoOAOrderStatus.ORDER_STATUS_CANCELLED: "CANCELLED",
            }
            status = status_map.get(order.orderStatus, "UNKNOWN")

            # Convert execution price if present
            symbol_id = getattr(order, 'tradeData', {}).symbolId if hasattr(order, 'tradeData') else None
            if symbol_id is None:
                symbol_id = getattr(order, 'symbolId', None)

            avg_fill_price = None
            last_fill_price = None
            if hasattr(order, 'executionPrice') and order.executionPrice:
                # executionPrice is proto double (actual price) — no conversion needed
                converted_price = order.executionPrice
                avg_fill_price = converted_price
                last_fill_price = converted_price

            # Resolve any pending order future (from place_order)
            # Check all pending futures - match by order_id from the execution event
            resolved_key = None
            for key, (future, loop, order_status_callback) in self._pending_order_futures.items():
                if not future.done():
                    logger.info(f"[ORDER] Resolving pending order future: order_id={order_id}")
                    # Register the order status callback now that we have the order_id
                    if order_status_callback:
                        self._order_callbacks[order_id] = order_status_callback
                    loop.call_soon_threadsafe(future.set_result, order_id)
                    resolved_key = key
                    break
            if resolved_key:
                del self._pending_order_futures[resolved_key]

            # Fire existing order callbacks (for status updates on already-tracked orders)
            if order_id in self._order_callbacks:
                callback = self._order_callbacks[order_id]
                callback({
                    "order_id": order_id,
                    "status": status,
                    "filled": order.filledVolume if hasattr(order, 'filledVolume') else 0.0,
                    "remaining": order.volume - (order.filledVolume if hasattr(order, 'filledVolume') else 0.0),
                    "avg_fill_price": avg_fill_price,
                    "last_fill_price": last_fill_price
                })

        except Exception as e:
            logger.error(f"Error handling execution event: {e}", exc_info=True)

    def _handle_subscribe_spots_res(self, response):  # type: ignore
        """Handle subscribe spots response"""
        # ProtoOASubscribeSpotsRes only has payloadType and ctidTraderAccountId
        logger.debug(f"Subscribe spots response received for account: {response.ctidTraderAccountId}")

    def _handle_order_error(self, response):  # type: ignore
        """Handle order error events (runs on reactor thread)"""
        error_code = getattr(response, 'errorCode', 'UNKNOWN')
        description = getattr(response, 'description', 'No description')
        order_id = getattr(response, 'orderId', None)
        logger.error(f"[ORDER] Order error: {error_code} - {description} (orderId: {order_id})")

        # Resolve any pending order future with empty string (signals failure)
        resolved_key = None
        for key, (future, loop, _) in self._pending_order_futures.items():
            if not future.done():
                logger.error(f"[ORDER] Resolving pending order future as FAILED: {error_code} - {description}")
                loop.call_soon_threadsafe(future.set_result, "")
                resolved_key = key
                break
        if resolved_key:
            del self._pending_order_futures[resolved_key]

    def _handle_get_positions_res(self, response):  # type: ignore
        """Handle pushed get-positions response (ProtoOAReconcileRes / ProtoOAGetPositionsRes).

        NOTE: get_positions() sends an explicit ProtoOAReconcileReq and processes the
        response itself via a deferred callback — that path is authoritative.  This handler
        runs on the reactor thread for any push events carrying the same message types.
        We only update the cache here if the explicit request is NOT already in flight,
        to avoid overwriting a fresher result.
        """
        positions_list = []
        for position in response.position:
            # ProtoOAPosition nests the symbol under tradeData, NOT at the top level.
            if not hasattr(position, 'tradeData'):
                continue
            symbol_id = position.tradeData.symbolId
            asset = self._get_symbol_name(symbol_id)
            if not asset:
                continue

            # ProtoOAPosition.price is proto double (actual price already) — no conversion needed.
            entry_price = position.price if hasattr(position, 'price') else 0.0

            trade_side = position.tradeData.tradeSide if hasattr(position.tradeData, 'tradeSide') else 1
            position_type = "LONG" if trade_side == 1 else "SHORT"
            direction = 1.0 if position_type == "LONG" else -1.0
            volume_in_units = position.tradeData.volume / 100 if hasattr(position.tradeData, 'volume') else 0
            quantity = volume_in_units if position_type == "LONG" else -volume_in_units

            # Use live spot cache for current price; fall back to entry price if unavailable.
            spot = self._last_spot_prices.get(symbol_id)
            if spot and entry_price > 0:
                current_price = spot["bid"] if position_type == "LONG" else spot["ask"]
            else:
                current_price = entry_price

            gross_pnl = (current_price - entry_price) * direction * volume_in_units
            swap_commission = (
                (position.swap if hasattr(position, 'swap') else 0) +
                (position.commission if hasattr(position, 'commission') else 0)
            ) / 100.0
            unrealized_pnl = gross_pnl + swap_commission
            notional = entry_price * volume_in_units
            unrealized_pnl_pct = (unrealized_pnl / notional * 100) if notional > 0 else 0

            # SL/TP are optional double fields (actual price, no unit conversion needed).
            # They are 0.0 (falsy) when not set on the position.
            stop_loss = getattr(position, 'stopLoss', None) or None
            take_profit = getattr(position, 'takeProfit', None) or None

            positions_list.append({
                "asset": asset,
                "quantity": quantity,
                "position_type": position_type,
                "avg_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })

        self._positions_cache = positions_list

    def _handle_symbols_list_res(self, response):  # type: ignore
        """Handle symbols list response (light or full symbols)."""
        try:
            if hasattr(response, 'symbol'):
                for symbol in response.symbol:
                    # Light symbols have symbolName; full symbols (ProtoOASymbol) don't.
                    symbol_name = getattr(symbol, 'symbolName', None)
                    if symbol_name:
                        symbol_asset = self._convert_symbol_to_asset(symbol_name)
                        self._symbol_cache[symbol_asset] = symbol.symbolId
                        self._symbol_id_to_name[symbol.symbolId] = symbol_asset

                    # Store digits for price conversion if present.
                    digits = getattr(symbol, 'digits', None)
                    if digits is not None:
                        self._symbol_digits[symbol.symbolId] = digits
                        self._symbol_digits_verified.add(symbol.symbolId)
                    elif symbol.symbolId not in self._symbol_digits:
                        self._symbol_digits[symbol.symbolId] = 5  # Fallback for forex

                    logger.debug(f"Cached symbol: {symbol.symbolId} (digits={self._symbol_digits.get(symbol.symbolId)})")
        except Exception as e:
            logger.error(f"Error handling symbols list: {e}", exc_info=True)

    def _handle_get_tick_data_res(self, response):  # type: ignore
        """Handle get tick data response"""
        try:
            # Find callback for this request (we'll use a simple approach)
            # In a more sophisticated implementation, you'd track request IDs
            bars = []
            if hasattr(response, 'tickData'):
                # Try to get symbol_id from the response to determine digits
                symbol_id = getattr(response, 'symbolId', None)
                digits = self._symbol_digits.get(symbol_id, 5) if symbol_id else 5
                conversion_factor = 10 ** digits

                for tick in response.tickData:
                    # Convert tick prices from integer format
                    ask_price = tick.ask / conversion_factor
                    bars.append({
                        "date": datetime.fromtimestamp(tick.timestamp / 1000),
                        "open": ask_price,  # Use ask as open
                        "high": ask_price,
                        "low": ask_price,
                        "close": ask_price,
                        "volume": 0
                    })

            # Call the first available callback (simplified - in production use request tracking)
            if self._historical_data_callbacks:
                callback_key = list(self._historical_data_callbacks.keys())[0]
                callback = self._historical_data_callbacks.pop(callback_key)
                context = self._historical_data_context.pop(callback_key, {})
                callback(bars, context)

        except Exception as e:
            logger.error(f"Error handling tick data response: {e}", exc_info=True)

    def _on_error(self, failure):
        """Error callback - runs on reactor thread"""
        logger.error(f"[Reactor Thread] ✗ cTrader error: {failure}")
        if hasattr(failure, 'value'):
            logger.error(f"[Reactor Thread]   Error details: {failure.value}")
        if hasattr(failure, 'getTraceback'):
            logger.error(f"[Reactor Thread]   Traceback: {failure.getTraceback()}")

    async def connect(self) -> bool:
        """Connect to cTrader"""
        if not CTRADER_AVAILABLE:
            logger.error(
                "ctrader-open-api module not available. Please install: pip install ctrader-open-api"
            )
            return False

        # Reset state
        self._connection_error = None
        self._auth_error = None
        self.connected = False
        self.authenticated = False

        try:
            # Validate configuration
            logger.info("=" * 60)
            logger.info("Connecting to cTrader Open API...")
            logger.info("=" * 60)

            # Check required environment variables
            missing_vars = []
            if not config.CTRADER_CLIENT_ID:
                missing_vars.append("CTRADER_CLIENT_ID")
            if not config.CTRADER_CLIENT_SECRET:
                missing_vars.append("CTRADER_CLIENT_SECRET")

            if missing_vars:
                logger.error(f"✗ Missing required environment variables: {', '.join(missing_vars)}")
                logger.error("  Please set these variables before starting the application.")
                return False

            # Access token is optional at connect time but required for trading
            if not config.CTRADER_ACCESS_TOKEN:
                logger.warning(
                    "⚠️ CTRADER_ACCESS_TOKEN is not set. App will authenticate but "
                    "account access and trading will fail. Generate one with: "
                    "python -m live_trading.scripts.get_ctrader_token"
                )

            # Determine host (live or demo)
            environment = config.CTRADER_ENVIRONMENT.upper()
            if environment == "LIVE":
                host = EndPoints.PROTOBUF_LIVE_HOST
            else:
                host = EndPoints.PROTOBUF_DEMO_HOST

            port = EndPoints.PROTOBUF_PORT

            logger.info(f"Environment: {environment}")
            logger.info(f"Host: {host}:{port}")
            logger.info(f"Client ID: {config.CTRADER_CLIENT_ID[:8]}...")
            logger.info(f"Access Token: {config.CTRADER_ACCESS_TOKEN[:8] if config.CTRADER_ACCESS_TOKEN else 'NOT SET'}...")

            # Start reactor FIRST (must be running before startService)
            self._start_reactor()

            if not reactor.running:
                logger.error("✗ Twisted reactor failed to start")
                return False

            # Create client - must be done on reactor thread
            logger.info("Creating cTrader client on reactor thread...")

            # Use an event to signal when client is created
            client_created = threading.Event()
            client_error = [None]  # Use list to allow modification in closure

            def create_client_on_reactor():
                """Create and start client on the reactor thread"""
                try:
                    logger.info(f"[Reactor Thread] Creating Client({host}, {port}, TcpProtocol)")
                    self.client = Client(host, port, TcpProtocol)

                    # Set callbacks
                    logger.info("[Reactor Thread] Setting callbacks...")
                    self.client.setConnectedCallback(self._on_connected)
                    self.client.setDisconnectedCallback(self._on_disconnected)
                    self.client.setMessageReceivedCallback(self._on_message_received)

                    # Start client service
                    logger.info("[Reactor Thread] Starting client service...")
                    self.client.startService()
                    logger.info("[Reactor Thread] Client service started")

                    client_created.set()
                except Exception as e:
                    logger.error(f"[Reactor Thread] Error creating client: {e}", exc_info=True)
                    client_error[0] = str(e)
                    client_created.set()

            # Schedule client creation on the reactor thread
            # This is CRITICAL - Twisted operations must run on the reactor thread
            reactor.callFromThread(create_client_on_reactor)

            # Wait for client to be created
            logger.info("Waiting for client to be created on reactor thread...")
            if not client_created.wait(timeout=10.0):
                logger.error("✗ Timeout waiting for client creation on reactor thread")
                return False

            if client_error[0]:
                logger.error(f"✗ Error creating client: {client_error[0]}")
                return False

            logger.info("✓ Client created and service started on reactor thread")

            # Wait for connection (with timeout)
            max_wait = 300
            waited = 0
            while not self.connected and waited < max_wait:
                # Check for connection errors
                if self._connection_error:
                    logger.error(f"✗ Connection failed: {self._connection_error}")
                    return False
                await asyncio.sleep(0.5)
                waited += 0.5
                if waited % 5 == 0:
                    logger.info(f"  Still waiting for connection... ({waited}s)")

            if not self.connected:
                error_msg = self._connection_error or "Connection timeout - no response from server"
                logger.error(f"✗ Failed to connect to cTrader: {error_msg}")
                logger.error("  Possible causes:")
                logger.error("  - Network connectivity issues")
                logger.error("  - Firewall blocking the connection")
                logger.error("  - cTrader server is down")
                logger.error(f"  - Check if {host}:{port} is reachable")
                return False

            # Wait for authentication
            logger.info("Waiting for authentication to complete...")
            max_wait = 300
            waited = 0
            while not self.authenticated and not self._auth_error and waited < max_wait:
                await asyncio.sleep(0.5)
                waited += 0.5
                if waited % 5 == 0:
                    logger.info(f"  Still waiting for authentication... ({waited}s)")

            if self._auth_error:
                logger.error(f"✗ Authentication failed: {self._auth_error}")
                return False

            if not self.authenticated:
                logger.error("✗ Authentication timeout - no response from server")
                logger.error("  Possible causes:")
                logger.error("  - Invalid CTRADER_CLIENT_ID or CTRADER_CLIENT_SECRET")
                logger.error("  - Invalid or expired CTRADER_ACCESS_TOKEN")
                logger.error("  - No trading accounts linked to this access token")
                return False

            logger.info("=" * 60)
            logger.info("✓ Successfully connected and authenticated with cTrader")
            logger.info(f"  Account ID: {self.account_id}")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"✗ Error connecting to cTrader: {e}", exc_info=True)
            return False

    async def disconnect(self):
        """Disconnect from cTrader"""
        logger.info("[CONNECTION] 🔌 Disconnecting from cTrader (graceful shutdown)...")

        # Mark shutdown to prevent reconnection attempts
        self._shutdown_requested = True

        # Stop connection monitor
        if self._connection_monitor_task:
            self._connection_monitor_task.cancel()
            try:
                await self._connection_monitor_task
            except asyncio.CancelledError:
                pass
            self._connection_monitor_task = None

        # Mark as disconnected immediately to prevent new operations
        self.connected = False
        self.authenticated = False

        # Quick unsubscribe from market data (don't wait too long)
        for asset in list(self._data_subscriptions.keys()):
            try:
                # Use a short timeout for each unsubscribe
                await asyncio.wait_for(self.unsubscribe_market_data(asset), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout unsubscribing from {asset}, continuing...")
            except Exception as e:
                logger.debug(f"Error unsubscribing from {asset}: {e}")

        # Stop the reactor (which will also stop the client)
        self._stop_reactor()

        logger.info("[CONNECTION] ✅ Disconnected from cTrader (graceful)")

    async def _attempt_reconnect(self):
        """Attempt to reconnect to cTrader"""
        if self._shutdown_requested:
            logger.info(f"[CONNECTION] Shutdown requested - aborting reconnection")
            return

        if self._reconnecting:
            logger.debug(f"[CONNECTION] Already reconnecting - skipping duplicate attempt")
            return

        self._reconnecting = True
        self._reconnect_attempts += 1

        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(f"[CONNECTION] ❌ Max reconnect attempts ({self._max_reconnect_attempts}) reached!")
            logger.error(f"[CONNECTION] ❌ CRITICAL: Trading system is OFFLINE - manual intervention required!")
            self._reconnecting = False
            return

        delay = self._reconnect_delay * self._reconnect_attempts
        logger.warning(f"[CONNECTION] 🔄 Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {delay}s...")

        # Wait before reconnecting (exponential backoff)
        await asyncio.sleep(delay)

        try:
            # Store current subscriptions before reconnecting
            # Deep copy the callback lists and IDs
            saved_subscriptions = dict(self._data_subscriptions)
            saved_callbacks = {asset: list(callbacks) for asset, callbacks in self._data_callbacks.items()}
            saved_callback_ids = {asset: list(ids) for asset, ids in self._data_callback_ids.items()}
            saved_trendbar_subs = {asset: set(periods) for asset, periods in self._trendbar_subscriptions.items()}

            # Stop existing reactor
            self._stop_reactor()
            await asyncio.sleep(1)  # Give it time to clean up

            # Reset state
            self.connected = False
            self.authenticated = False
            self._connection_error = None
            self._auth_error = None
            # Clear subscription tracking (will be rebuilt)
            self._data_subscriptions.clear()
            self._data_callbacks.clear()
            self._data_callback_ids.clear()
            self._trendbar_subscriptions.clear()
            self._live_trendbar_volumes.clear()

            # Attempt to reconnect
            logger.info(f"[CONNECTION] 🔄 Connecting to cTrader...")
            success = await self.connect()

            if success:
                logger.info(f"[CONNECTION] ✅ RECONNECTED SUCCESSFULLY!")
                self._reconnect_attempts = 0

                # Restore all subscriptions and callbacks
                num_subscriptions = len(saved_callbacks)
                logger.info(f"[CONNECTION] 🔄 Restoring {num_subscriptions} market data subscriptions...")

                for asset, callbacks in saved_callbacks.items():
                    callback_ids = saved_callback_ids.get(asset, [])
                    for i, callback in enumerate(callbacks):
                        callback_id = callback_ids[i] if i < len(callback_ids) else f"restored_{i}"
                        try:
                            await self.subscribe_market_data(asset, callback, callback_id)
                            logger.info(f"[CONNECTION] ✅ Restored subscription: {asset} ({callback_id})")
                        except Exception as sub_error:
                            logger.error(f"[CONNECTION] ❌ Failed to restore subscription {asset}: {sub_error}")

                # Restore live trendbar subscriptions
                for asset, periods in saved_trendbar_subs.items():
                    # Convert period IDs back to bar size strings
                    reverse_map = {v: k for k, v in self.TRENDBAR_PERIOD_MAP.items()}
                    bar_sizes = [reverse_map[p] for p in periods if p in reverse_map]
                    if bar_sizes:
                        try:
                            await self.subscribe_live_trendbars(asset, bar_sizes)
                            logger.info(f"[CONNECTION] ✅ Restored trendbar subscriptions: {asset} ({bar_sizes})")
                        except Exception as tb_error:
                            logger.error(f"[CONNECTION] ❌ Failed to restore trendbar subscription {asset}: {tb_error}")

                logger.info(f"[CONNECTION] ✅ All subscriptions restored - trading resumed")
            else:
                logger.warning(f"[CONNECTION] ⚠️ Reconnection failed, scheduling retry...")
                # Schedule another attempt
                self._reconnecting = False  # Reset before scheduling next attempt
                asyncio.create_task(self._attempt_reconnect())
                return

        except Exception as e:
            logger.error(f"[CONNECTION] ❌ Reconnection error: {e}", exc_info=True)
            self._reconnecting = False  # Reset before scheduling next attempt
            asyncio.create_task(self._attempt_reconnect())
            return
        finally:
            self._reconnecting = False

    async def start_connection_monitor(self):
        """Start monitoring connection health"""
        if self._connection_monitor_task:
            return

        async def monitor_loop():
            """Monitor connection and attempt reconnect if stale"""
            STALE_THRESHOLD = 60  # seconds without messages (heartbeats should come every ~20s)
            RECONNECT_THRESHOLD = 120  # seconds - force reconnect if no messages
            CHECK_INTERVAL = 30  # seconds between checks
            STATUS_LOG_INTERVAL = 300  # Log subscription details every 5 minutes
            last_status_log = datetime.utcnow()

            while not self._shutdown_requested:
                try:
                    await asyncio.sleep(CHECK_INTERVAL)

                    # Calculate time since last message
                    seconds_since_message = None
                    if self._last_message_time:
                        seconds_since_message = (datetime.utcnow() - self._last_message_time).total_seconds()

                    # Log status periodically
                    if (datetime.utcnow() - last_status_log).total_seconds() > STATUS_LOG_INTERVAL:
                        # Build detailed subscription info
                        sub_details = []
                        for asset, callbacks in self._data_callbacks.items():
                            callback_ids = self._data_callback_ids.get(asset, [])
                            sub_details.append(f"{asset}({len(callbacks)} callbacks: {callback_ids})")

                        status_emoji = "✅" if (self.connected and self.authenticated) else "❌"
                        logger.info(f"[CONNECTION] {status_emoji} Status: connected={self.connected}, "
                                    f"authenticated={self.authenticated}, "
                                    f"last_msg={seconds_since_message:.0f}s ago" if seconds_since_message else "no messages yet")
                        if sub_details:
                            logger.info(f"[CONNECTION] 📊 Active subscriptions: {', '.join(sub_details)}")
                        else:
                            logger.warning(f"[CONNECTION] ⚠️ No active market data subscriptions")
                        last_status_log = datetime.utcnow()

                    if not self.connected or not self.authenticated:
                        logger.error(f"[CONNECTION] ❌ Not connected/authenticated - triggering reconnect")
                        if not self._reconnecting:
                            asyncio.create_task(self._attempt_reconnect())
                        continue

                    # Check for stale connection
                    if seconds_since_message is not None:
                        if seconds_since_message > RECONNECT_THRESHOLD:
                            logger.error(f"[CONNECTION] ❌ No messages for {seconds_since_message:.0f}s - forcing reconnect!")
                            if not self._reconnecting:
                                asyncio.create_task(self._attempt_reconnect())
                        elif seconds_since_message > STALE_THRESHOLD:
                            logger.warning(f"[CONNECTION] ⚠️ Connection may be stale ({seconds_since_message:.0f}s since last message)")
                    else:
                        logger.warning("[CONNECTION] ⚠️ No messages received yet after connection")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[CONNECTION] Monitor error: {e}", exc_info=True)

        self._connection_monitor_task = asyncio.create_task(monitor_loop())
        logger.info("[CONNECTION] 🔍 Connection health monitor started")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        # Build subscription details with callback counts
        subscription_details = {}
        for asset in self._data_subscriptions.keys():
            callback_ids = self._data_callback_ids.get(asset, [])
            subscription_details[asset] = {
                "symbol_id": self._data_subscriptions[asset],
                "callback_count": len(self._data_callbacks.get(asset, [])),
                "callback_ids": callback_ids
            }

        status = {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "account_id": self.account_id,
            "reconnecting": self._reconnecting,
            "reconnect_attempts": self._reconnect_attempts,
            "last_message_time": self._last_message_time.isoformat() if self._last_message_time else None,
            "subscriptions": list(self._data_subscriptions.keys()),
            "subscription_details": subscription_details,
            "connection_error": self._connection_error,
            "auth_error": self._auth_error
        }

        if self._last_message_time:
            status["seconds_since_message"] = (datetime.utcnow() - self._last_message_time).total_seconds()

        return status

    async def _get_symbol_id(self, asset: str) -> Optional[int]:
        """Get symbol ID for an asset"""
        if asset in self._symbol_cache:
            return self._symbol_cache[asset]

        if not self.client or not self.authenticated:
            return None

        # Request symbol list and find our symbol
        symbol_name = self._convert_asset_to_symbol(asset)
        logger.info(f"Requesting symbol list to find {asset} (cTrader: {symbol_name})")

        # Request symbol list - must be sent from reactor thread
        request = ProtoOASymbolsListReq()
        request.ctidTraderAccountId = self.account_id

        try:
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            def on_success(response):
                try:
                    extracted = Protobuf.extract(response)
                    if hasattr(extracted, 'symbol'):
                        found_match = False
                        sample_symbols = []
                        for symbol in extracted.symbol:
                            symbol_name = symbol.symbolName
                            symbol_asset = self._convert_symbol_to_asset(symbol_name)

                            # Cache the symbol with its converted asset name
                            self._symbol_cache[symbol_asset] = symbol.symbolId
                            self._symbol_id_to_name[symbol.symbolId] = symbol_asset

                            # Also cache with original symbol name for direct lookup
                            self._symbol_cache[symbol_name] = symbol.symbolId

                            # Check if this symbol matches what we're looking for
                            if self._symbol_matches_asset(symbol_name, asset):
                                self._symbol_cache[asset] = symbol.symbolId
                                self._symbol_id_to_name[symbol.symbolId] = asset
                                found_match = True
                                logger.info(f"Matched symbol: {symbol_name} -> {asset} (ID: {symbol.symbolId})")

                            # Collect sample symbols for debugging
                            if len(sample_symbols) < 10:
                                sample_symbols.append(symbol_name)

                        logger.info(f"Cached {len(extracted.symbol)} symbols. Sample: {sample_symbols}")
                        if not found_match:
                            logger.warning(f"No exact match found for {asset} in symbol list")

                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, True)
                except Exception as e:
                    logger.error(f"Error processing symbol list: {e}", exc_info=True)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

            def on_error(failure):
                logger.error(f"Symbol list request failed: {failure}")
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, False)

            def send_request():
                try:
                    if self.client:
                        d = self.client.send(request)
                        d.addCallbacks(on_success, on_error)
                    else:
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, False)
                except Exception as e:
                    logger.error(f"Error sending symbol list request: {e}", exc_info=True)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

            reactor.callFromThread(send_request)

            # Wait for response with timeout
            try:
                await asyncio.wait_for(future, timeout=15.0)
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for symbol list")
                return None

            # Check if symbol is now in cache
            if asset in self._symbol_cache:
                logger.info(f"Found symbol {asset} -> ID {self._symbol_cache[asset]}")
                return self._symbol_cache[asset]

            logger.error(f"Symbol {asset} not found in symbol list (checked: {symbol_name})")
            return None

        except Exception as e:
            logger.error(f"Error getting symbol ID for {asset}: {e}", exc_info=True)
            return None

    def _get_symbol_name(self, symbol_id: int) -> str:
        """Get asset name from symbol ID (reverse lookup)"""
        return self._symbol_id_to_name.get(symbol_id, f"SYMBOL_{symbol_id}")

    async def _ensure_symbol_digits(self, symbol_id: int) -> None:
        """
        Fetch full symbol details to get the accurate 'digits' value.

        The light symbol list (ProtoOASymbolsListReq) does not include digits,
        so we must request it via ProtoOASymbolByIdReq. Results are cached.
        """
        if symbol_id in self._symbol_digits_verified:
            return

        if not self.client or not self.authenticated:
            return

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def on_success(response):
            try:
                extracted = Protobuf.extract(response)
                if hasattr(extracted, 'symbol') and extracted.symbol:
                    sym = extracted.symbol[0]
                    digits = getattr(sym, 'digits', None)
                    if digits is not None:
                        old = self._symbol_digits.get(symbol_id)
                        self._symbol_digits[symbol_id] = digits
                        self._symbol_digits_verified.add(symbol_id)
                        asset = self._symbol_id_to_name.get(symbol_id, symbol_id)
                        if old != digits:
                            logger.info(
                                f"[SYMBOL] Updated digits for {asset} (ID {symbol_id}): "
                                f"{old} -> {digits}"
                            )
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, True)
            except Exception as e:
                logger.error(f"Error processing symbol details: {e}", exc_info=True)
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, False)

        def on_error(failure):
            logger.warning(f"Failed to fetch symbol details for {symbol_id}: {failure}")
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, False)

        def send_request():
            try:
                if self.client:
                    request = ProtoOASymbolByIdReq()
                    request.ctidTraderAccountId = self.account_id
                    request.symbolId.append(symbol_id)
                    d = self.client.send(request)
                    d.addCallbacks(on_success, on_error)
                else:
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)
            except Exception as e:
                logger.error(f"Error sending symbol-by-id request: {e}", exc_info=True)
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, False)

        reactor.callFromThread(send_request)

        try:
            await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching symbol details for {symbol_id}")

    async def subscribe_market_data(
        self,
        asset: str,
        callback: Callable[[Dict[str, Any]], None],
        callback_id: str = None,
        bar_sizes: list = None
    ) -> bool:
        """
        Subscribe to real-time market data.

        Supports multiple callbacks per asset (e.g., multiple operations on same asset).
        The callback_id helps track and remove specific callbacks later.

        Args:
            asset: Asset symbol (e.g., "EUR-USD")
            callback: Callback function for tick data
            callback_id: Unique ID for this callback
            bar_sizes: Optional list of bar sizes to subscribe live trendbars for
                      (enables volume data in tick events)
        """
        if not self.connected or not self.authenticated or not self.client:
            logger.error("Not connected to cTrader")
            return False

        # Generate callback_id if not provided
        if callback_id is None:
            callback_id = f"cb_{id(callback)}"

        # Check if we already have a subscription for this asset
        is_new_subscription = asset not in self._data_subscriptions

        # Initialize lists if needed
        if asset not in self._data_callbacks:
            self._data_callbacks[asset] = []
            self._data_callback_ids[asset] = []

        # Add callback to list (if not already present with same id)
        if callback_id not in self._data_callback_ids[asset]:
            self._data_callbacks[asset].append(callback)
            self._data_callback_ids[asset].append(callback_id)
            logger.info(f"Added callback '{callback_id}' for {asset} (total callbacks: {len(self._data_callbacks[asset])})")
        else:
            # Update existing callback
            idx = self._data_callback_ids[asset].index(callback_id)
            self._data_callbacks[asset][idx] = callback
            logger.info(f"Updated callback '{callback_id}' for {asset}")

        # Only subscribe to cTrader if this is a NEW asset subscription
        if is_new_subscription:
            # Get symbol ID
            symbol_id = await self._get_symbol_id(asset)
            if symbol_id is None:
                logger.error(f"Could not find symbol ID for {asset}")
                return False

            self._data_subscriptions[asset] = symbol_id

            # Fetch accurate digits BEFORE the first spot event arrives so that
            # _convert_price uses the real value (not the forex default of 5).
            # Crypto instruments (ETH-USD, BTC-USD) typically have more digits than
            # the 5-digit forex default, which would cause 10x–1000x price errors.
            await self._ensure_symbol_digits(symbol_id)

            # Subscribe to spots - must be sent from reactor thread
            request = ProtoOASubscribeSpotsReq()
            request.ctidTraderAccountId = self.account_id
            request.symbolId.append(symbol_id)

            try:
                loop = asyncio.get_event_loop()
                future = loop.create_future()

                def on_success(res):
                    logger.info(f"[CONNECTION] ✅ Subscribed to market data: {asset} (symbol_id: {symbol_id})")
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, True)

                def on_error(failure):
                    logger.error(f"[CONNECTION] ❌ Failed to subscribe to market data: {asset} - {failure}")
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

                def send_request():
                    if self.client:
                        d = self.client.send(request)
                        d.addCallbacks(on_success, on_error)
                    else:
                        loop.call_soon_threadsafe(future.set_result, False)

                reactor.callFromThread(send_request)
                logger.debug(f"[CONNECTION] Requesting cTrader subscription for {asset} (symbol_id: {symbol_id})")

                # Wait with timeout
                try:
                    spot_result = await asyncio.wait_for(future, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.error(f"[CONNECTION] ❌ Timeout subscribing to market data: {asset}")
                    return False

                # Also subscribe to live trendbars for volume data
                if spot_result and bar_sizes:
                    await self.subscribe_live_trendbars(asset, bar_sizes)
                elif spot_result:
                    # Default: subscribe to M1 and H1 for basic volume data
                    await self.subscribe_live_trendbars(asset, ["1 min", "1 hour"])

                return spot_result

            except Exception as e:
                logger.error(f"[CONNECTION] ❌ Error subscribing to market data: {asset} - {e}", exc_info=True)
                return False
        else:
            # Already subscribed to cTrader, just added callback
            # But check if we need trendbar subscriptions for new bar sizes
            if bar_sizes:
                await self.subscribe_live_trendbars(asset, bar_sizes)
            logger.debug(f"[CONNECTION] Reusing existing subscription for {asset} (callback: {callback_id})")
            return True

    # Period map for live trendbar subscriptions (bar_size -> ProtoOATrendbarPeriod)
    TRENDBAR_PERIOD_MAP = {
        "1 min": 1,    # M1
        "5 mins": 5,   # M5
        "15 mins": 7,  # M15
        "30 mins": 8,  # M30
        "1 hour": 9,   # H1
        "4 hours": 10, # H4
        "1 day": 12,   # D1
        "1 week": 13,  # W1
    }

    async def subscribe_live_trendbars(self, asset: str, bar_sizes: list = None):
        """
        Subscribe to live trendbar updates for an asset.

        This enables volume data in ProtoOASpotEvent responses.
        Call this after subscribe_market_data to get volume with each tick.

        Args:
            asset: Asset symbol (e.g., "EUR-USD")
            bar_sizes: List of bar sizes to subscribe (e.g., ["1 hour", "15 mins"]).
                      If None, subscribes to common periods (M1, H1).
        """
        if not self.connected or not self.authenticated or not self.client:
            logger.warning(f"Cannot subscribe to live trendbars for {asset} - not connected")
            return

        symbol_id = await self._get_symbol_id(asset)
        if symbol_id is None:
            logger.error(f"Cannot subscribe to live trendbars - symbol not found: {asset}")
            return

        # Default to common periods if none specified
        if bar_sizes is None:
            bar_sizes = ["1 min", "1 hour"]

        if asset not in self._trendbar_subscriptions:
            self._trendbar_subscriptions[asset] = set()

        for bar_size in bar_sizes:
            period = self.TRENDBAR_PERIOD_MAP.get(bar_size)
            if period is None:
                logger.warning(f"Unknown bar size for trendbar subscription: {bar_size}")
                continue

            if period in self._trendbar_subscriptions[asset]:
                logger.debug(f"Already subscribed to live trendbar {bar_size} for {asset}")
                continue

            try:
                request = ProtoOASubscribeLiveTrendbarReq()
                request.ctidTraderAccountId = self.account_id
                request.symbolId = symbol_id
                request.period = period

                loop = asyncio.get_event_loop()
                future = loop.create_future()

                def on_success(res, _bs=bar_size, _p=period):
                    logger.info(f"[CONNECTION] ✅ Subscribed to live trendbar: {asset} ({_bs}, period={_p})")
                    self._trendbar_subscriptions.setdefault(asset, set()).add(_p)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, True)

                def on_error(failure, _bs=bar_size):
                    logger.error(f"[CONNECTION] ❌ Failed to subscribe to live trendbar {_bs} for {asset}: {failure}")
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

                def send_request():
                    if self.client:
                        d = self.client.send(request)
                        d.addCallbacks(on_success, on_error)
                    else:
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, False)

                reactor.callFromThread(send_request)

                try:
                    await asyncio.wait_for(future, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout subscribing to live trendbar {bar_size} for {asset}")

            except Exception as e:
                logger.error(f"Error subscribing to live trendbar {bar_size} for {asset}: {e}")

    async def unsubscribe_market_data(self, asset: str, callback_id: str = None):
        """
        Unsubscribe from market data.

        If callback_id is provided, removes only that callback.
        Only unsubscribes from cTrader when the last callback is removed.
        If callback_id is None, removes ALL callbacks for the asset.
        """
        if asset not in self._data_subscriptions:
            return

        # If callback_id provided, remove only that callback
        if callback_id is not None and asset in self._data_callback_ids:
            if callback_id in self._data_callback_ids[asset]:
                idx = self._data_callback_ids[asset].index(callback_id)
                self._data_callbacks[asset].pop(idx)
                self._data_callback_ids[asset].pop(idx)
                logger.info(f"Removed callback '{callback_id}' for {asset} (remaining: {len(self._data_callbacks[asset])})")

                # If there are still callbacks, don't unsubscribe from cTrader
                if len(self._data_callbacks[asset]) > 0:
                    return
            else:
                logger.warning(f"Callback '{callback_id}' not found for {asset}")
                return

        # Unsubscribe from cTrader (last callback removed or removing all)
        symbol_id = self._data_subscriptions[asset]

        if self.client and reactor.running:
            request = ProtoOAUnsubscribeSpotsReq()
            request.ctidTraderAccountId = self.account_id
            request.symbolId.append(symbol_id)

            try:
                loop = asyncio.get_event_loop()
                future = loop.create_future()

                def on_complete(res):
                    logger.debug(f"Unsubscribed from cTrader market data for {asset}")
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, True)

                def on_error(failure):
                    logger.debug(f"Error unsubscribing from {asset}: {failure}")
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

                def send_request():
                    if self.client:
                        d = self.client.send(request)
                        d.addCallbacks(on_complete, on_error)
                    else:
                        loop.call_soon_threadsafe(future.set_result, False)

                reactor.callFromThread(send_request)

                # Short timeout for unsubscribe
                try:
                    await asyncio.wait_for(future, timeout=2.0)
                except asyncio.TimeoutError:
                    pass  # Don't block on unsubscribe timeout

            except Exception as e:
                logger.debug(f"Error unsubscribing from market data: {e}")

        # Clean up all tracking data for this asset
        del self._data_subscriptions[asset]
        if asset in self._data_callbacks:
            del self._data_callbacks[asset]
        if asset in self._data_callback_ids:
            del self._data_callback_ids[asset]
        logger.info(f"✓ Fully unsubscribed from market data for {asset}")

    def register_order_callback(self, broker_order_id: str, callback: Callable):
        """Register a callback for order status updates"""
        self._order_callbacks[broker_order_id] = callback

    def _convert_quantity_to_volume(self, quantity: float, symbol_id: Optional[int] = None) -> int:
        """
        Convert a lot-based quantity to cTrader volume units.

        cTrader uses volume in 'cents' (1 lot = 100,000 units).
        Example: 0.01 lots = 1,000 volume units
                 0.10 lots = 10,000 volume units
                 1.00 lots = 100,000 volume units
        """
        UNITS_PER_LOT = 100_000
        MAX_VOLUME = 10_000_000  # cTrader max = 100 lots
        volume = int(round(quantity * UNITS_PER_LOT))
        if volume < 1000:
            logger.warning(f"[ORDER] Volume {volume} too small (min 1000 = 0.01 lots), using minimum")
            volume = 1000
        if volume > MAX_VOLUME:
            logger.warning(
                f"[ORDER] Volume {volume} ({quantity:.2f} lots) exceeds max {MAX_VOLUME} "
                f"(100 lots), clamping. Check position sizing logic."
            )
            volume = MAX_VOLUME
        return volume

    async def place_order(
        self,
        asset: str,
        action: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_status_callback: Optional[Callable] = None  # Optional parameter, not in base class
    ) -> str:
        """Place an order via cTrader Open API.

        Note: cTrader does NOT return the order result through the Deferred from
        client.send(). Instead, the result arrives as a ProtoOAExecutionEvent
        through the general message handler. We track the pending future and
        resolve it from _handle_execution_event.
        """
        if not self.connected or not self.authenticated or not self.client:
            logger.error("[ORDER] Not connected to cTrader")
            return ""

        # Get symbol ID
        symbol_id = await self._get_symbol_id(asset)
        if symbol_id is None:
            logger.error(f"[ORDER] Could not find symbol ID for {asset}")
            return ""

        # Fetch accurate digits for price rounding
        await self._ensure_symbol_digits(symbol_id)

        # Create order request
        request = ProtoOANewOrderReq()
        request.ctidTraderAccountId = self.account_id
        request.symbolId = symbol_id

        # Set trade side
        if action == "BUY":
            request.tradeSide = ProtoOATradeSide.BUY
        else:
            request.tradeSide = ProtoOATradeSide.SELL

        # Set order type
        if order_type == "MARKET":
            request.orderType = ProtoOAOrderType.MARKET
        elif order_type == "LIMIT":
            request.orderType = ProtoOAOrderType.LIMIT
            if price:
                request.limitPrice = self._round_price_for_symbol(price, symbol_id)
        elif order_type == "STOP":
            request.orderType = ProtoOAOrderType.STOP
            if price:
                request.stopPrice = self._round_price_for_symbol(price, symbol_id)
        else:
            logger.error(f"[ORDER] Unsupported order type: {order_type}")
            return ""

        # Convert quantity (lots) to cTrader volume units (1 lot = 100,000 units)
        volume = self._convert_quantity_to_volume(quantity, symbol_id)
        request.volume = volume
        logger.info(f"[ORDER] Volume conversion: {quantity} lots -> {volume} volume units")

        # Add stop loss and take profit (cTrader expects actual price values)
        if stop_loss:
            request.stopLoss = self._round_price_for_symbol(stop_loss, symbol_id)
        if take_profit:
            request.takeProfit = self._round_price_for_symbol(take_profit, symbol_id)

        digits = self._symbol_digits.get(symbol_id, 5)
        # Log the full request details for debugging
        logger.info(
            f"[ORDER] Request details: "
            f"accountId={self.account_id}, "
            f"symbolId={symbol_id} (digits={digits}), "
            f"tradeSide={action}, "
            f"orderType={order_type}, "
            f"volume={volume} (from {quantity} lots), "
            f"stopLoss={request.stopLoss if stop_loss else None} (raw={stop_loss}), "
            f"takeProfit={request.takeProfit if take_profit else None} (raw={take_profit}), "
            f"price={price}"
        )
        logger.info(f"[ORDER] Full protobuf request: {request}")

        try:
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            # Register the pending future BEFORE sending the request
            # so _handle_execution_event can resolve it when the response arrives
            request_key = f"order_{id(future)}"
            self._pending_order_futures[request_key] = (future, loop, order_status_callback)
            logger.info(f"[ORDER] Registered pending order future: {request_key}")

            def on_send_success(response):
                """Called when the request message is acknowledged (NOT the order result)"""
                logger.info(f"[ORDER] Request sent successfully, waiting for execution event...")
                # The actual order result will come via ProtoOAExecutionEvent
                # which is handled by _handle_execution_event

            def on_send_error(failure):
                """Called if sending the request itself fails"""
                logger.error(f"[ORDER] Failed to send order request: {failure}")
                # Clean up and resolve future
                if request_key in self._pending_order_futures:
                    del self._pending_order_futures[request_key]
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, "")

            def send_request():
                if self.client:
                    d = self.client.send(request)
                    d.addCallbacks(on_send_success, on_send_error)
                else:
                    if request_key in self._pending_order_futures:
                        del self._pending_order_futures[request_key]
                    loop.call_soon_threadsafe(future.set_result, "")

            reactor.callFromThread(send_request)

            # Wait for the execution event to resolve the future (or timeout)
            try:
                result = await asyncio.wait_for(future, timeout=30.0)
                return result
            except asyncio.TimeoutError:
                logger.error(f"[ORDER] Timeout (30s) waiting for order execution event for {action} {asset}")
                # Clean up the pending future
                if request_key in self._pending_order_futures:
                    del self._pending_order_futures[request_key]
                return ""

        except Exception as e:
            logger.error(f"[ORDER] Error placing order: {e}", exc_info=True)
            return ""

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order"""
        if not self.client:
            return False

        try:
            order_id = int(broker_order_id)

            request = ProtoOACancelOrderReq()
            request.ctidTraderAccountId = self.account_id
            request.orderId = order_id

            loop = asyncio.get_event_loop()
            future = loop.create_future()

            def on_success(response):
                try:
                    extracted = Protobuf.extract(response)
                    if hasattr(extracted, 'errorCode') and extracted.errorCode:
                        logger.error(f"Order cancellation error: {extracted.errorCode}")
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, False)
                    else:
                        logger.info(f"Cancelled order: {order_id}")
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, True)
                except Exception as e:
                    logger.error(f"Error processing cancel response: {e}", exc_info=True)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

            def on_error(failure):
                logger.error(f"Order cancellation failed: {failure}")
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, False)

            def send_request():
                if self.client:
                    d = self.client.send(request)
                    d.addCallbacks(on_success, on_error)
                else:
                    loop.call_soon_threadsafe(future.set_result, False)

            reactor.callFromThread(send_request)

            # Wait for response with timeout
            try:
                return await asyncio.wait_for(future, timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for order cancellation response")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}", exc_info=True)
            return False

    async def get_positions(self) -> list[Dict[str, Any]]:
        """Get current positions using ProtoOAReconcileReq"""
        if not self.connected or not self.authenticated or not self.client:
            return []

        try:
            # Use ProtoOAReconcileReq to get positions and orders
            request = ProtoOAReconcileReq()
            request.ctidTraderAccountId = self.account_id

            loop = asyncio.get_event_loop()
            future = loop.create_future()

            def on_success(response):
                try:
                    extracted = Protobuf.extract(response)
                    logger.debug(f"Reconcile response: {extracted}")

                    positions = []
                    if hasattr(extracted, 'position'):
                        for pos in extracted.position:
                            symbol_id = pos.tradeData.symbolId if hasattr(pos, 'tradeData') else None
                            asset = self._get_symbol_name(symbol_id) if symbol_id else "UNKNOWN"

                            # ProtoOAPosition.price is proto double (actual price already) — no conversion needed.
                            entry_price = pos.price if hasattr(pos, 'price') else 0.0

                            side = "BUY" if hasattr(pos, 'tradeData') and pos.tradeData.tradeSide == 1 else "SELL"
                            position_type = "LONG" if side == "BUY" else "SHORT"
                            direction = 1.0 if position_type == "LONG" else -1.0
                            # volume in base-currency units (tradeData.volume uses scale factor 100)
                            volume_in_units = pos.tradeData.volume / 100 if hasattr(pos, 'tradeData') else 0
                            quantity = volume_in_units if side == "BUY" else -volume_in_units

                            # Current market price from live spot cache.
                            # LONG closes at bid; SHORT closes at ask.
                            spot = self._last_spot_prices.get(symbol_id)
                            if spot and entry_price > 0:
                                current_price = spot["bid"] if position_type == "LONG" else spot["ask"]
                            else:
                                current_price = entry_price  # fallback: spot not yet received

                            # Gross P&L = price move × direction × size (in quote currency).
                            # This matches what cTrader shows as "Gross P&L".
                            gross_pnl = (current_price - entry_price) * direction * volume_in_units

                            # Funding costs are stored in cTrader's money units (cents) → /100
                            swap_commission = (
                                (pos.swap if hasattr(pos, 'swap') else 0) +
                                (pos.commission if hasattr(pos, 'commission') else 0)
                            ) / 100.0

                            unrealized_pnl = gross_pnl + swap_commission

                            # P&L% = return on notional (entry value of the position)
                            notional = entry_price * volume_in_units
                            unrealized_pnl_pct = (unrealized_pnl / notional * 100) if notional > 0 else 0

                            # SL/TP are optional double (actual price, no conversion needed).
                            stop_loss = getattr(pos, 'stopLoss', None) or None
                            take_profit = getattr(pos, 'takeProfit', None) or None

                            positions.append({
                                "position_id": str(pos.positionId),
                                "asset": asset,
                                "quantity": quantity,
                                "position_type": position_type,
                                "avg_price": entry_price,
                                "current_price": current_price,
                                "unrealized_pnl": unrealized_pnl,
                                "unrealized_pnl_pct": unrealized_pnl_pct,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                            })

                    # Also cache open (pending) orders from the reconcile response
                    open_orders = []
                    if hasattr(extracted, 'order'):
                        for ord_ in extracted.order:
                            symbol_id = ord_.tradeData.symbolId if hasattr(ord_, 'tradeData') else None
                            open_orders.append({
                                "broker_order_id": str(ord_.orderId),
                                "asset": self._get_symbol_name(symbol_id) if symbol_id else "UNKNOWN",
                                "status": str(ord_.orderStatus),
                            })
                    self._open_orders_cache = open_orders

                    self._positions_cache = positions
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, positions)
                except Exception as e:
                    logger.error(f"Error processing reconcile response: {e}", exc_info=True)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, [])

            def on_error(failure):
                logger.error(f"Reconcile request failed: {failure}")
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, [])

            def send_request():
                if self.client:
                    d = self.client.send(request)
                    d.addCallbacks(on_success, on_error)

            reactor.callFromThread(send_request)

            # Wait for response with timeout
            try:
                return await asyncio.wait_for(future, timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for positions")
                return self._positions_cache.copy()

        except Exception as e:
            logger.error(f"Error getting positions: {e}", exc_info=True)
            return []

    async def get_open_broker_orders(self) -> List[Dict[str, Any]]:
        """Return the pending orders captured during the last get_positions() reconcile call.

        Call get_positions() first to refresh the cache, then call this method to
        retrieve the open orders that were returned in the same ProtoOAReconcileReq
        response.
        """
        return list(self._open_orders_cache)

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information using ProtoOATraderReq"""
        if not self.connected or not self.authenticated or not self.client:
            return {}

        try:
            # Use ProtoOATraderReq to get trader/account info
            request = ProtoOATraderReq()
            request.ctidTraderAccountId = self.account_id

            loop = asyncio.get_event_loop()
            future = loop.create_future()

            def on_success(response):
                try:
                    extracted = Protobuf.extract(response)
                    logger.debug(f"Trader info response: {extracted}")

                    if hasattr(extracted, 'trader'):
                        trader = extracted.trader
                        # Balance is in cents, convert to actual value
                        balance = trader.balance / 100 if hasattr(trader, 'balance') else 0

                        # Compute floating P&L from cached positions (populated by get_positions / spot events)
                        floating_pnl = sum(p.get('unrealized_pnl', 0.0) for p in self._positions_cache)
                        equity = balance + floating_pnl

                        account_info = {
                            "account_id": str(self.account_id),
                            "currency": str(trader.depositAssetId) if hasattr(trader, 'depositAssetId') else "USD",
                            "balance": balance,
                            "equity": equity,
                            "margin_used": 0.0,
                            "margin_available": equity,
                            "margin_level": 0.0,
                            "unrealized_pnl": floating_pnl,
                            "open_trade_count": len(self._positions_cache),
                            "open_position_count": len(self._positions_cache),
                            "broker_name": trader.brokerName if hasattr(trader, 'brokerName') else "cTrader",
                            "leverage": trader.leverageInCents / 100 if hasattr(trader, 'leverageInCents') else 1,
                        }
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, account_info)
                    else:
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, {})
                except Exception as e:
                    logger.error(f"Error processing trader response: {e}", exc_info=True)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, {})

            def on_error(failure):
                logger.error(f"Trader request failed: {failure}")
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, {})

            def send_request():
                if self.client:
                    d = self.client.send(request)
                    d.addCallbacks(on_success, on_error)

            reactor.callFromThread(send_request)

            # Wait for response with timeout
            try:
                return await asyncio.wait_for(future, timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for account info")
                return {}

        except Exception as e:
            logger.error(f"Error getting account info: {e}", exc_info=True)
            return {}

    async def fetch_historical_data(
        self,
        asset: str,
        bar_size: str,
        interval: str,
        callback: Callable[[list, Dict], None],
        context: Optional[Dict] = None
    ) -> bool:
        """
        Fetch historical OHLC bar data for an asset using ProtoOAGetTrendbarsReq.

        Args:
            asset: Asset symbol (e.g., "USD-CAD")
            bar_size: Bar size (e.g., "1 hour", "15 mins")
            interval: Time period (e.g., "1 Y", "6 M")
            callback: Callback function(bars: list, context: dict) called when data is complete
            context: Optional context dict passed to callback

        Returns:
            True if request was sent successfully
        """
        logger.info(f"fetch_historical_data called: asset={asset}, bar_size={bar_size}, interval={interval}")
        logger.info(f"Broker state: connected={self.connected}, authenticated={self.authenticated}, client={self.client is not None}")

        if not self.connected or not self.authenticated or not self.client:
            logger.error(f"Not connected to cTrader: connected={self.connected}, authenticated={self.authenticated}, client={self.client is not None}")
            return False

        # Get symbol ID
        logger.debug(f"Getting symbol ID for {asset}...")
        symbol_id = await self._get_symbol_id(asset)
        if symbol_id is None:
            logger.error(f"Could not find symbol ID for {asset}")
            return False
        logger.debug(f"Got symbol ID for {asset}: {symbol_id}")

        # Convert bar_size to cTrader trendbar period (ProtoOATrendbarPeriod enum values)
        # M1=1, M2=2, M3=3, M4=4, M5=5, M10=6, M15=7, M30=8, H1=9, H4=10, H12=11, D1=12, W1=13, MN1=14
        period_map = {
            "1 min": 1,    # M1
            "5 mins": 5,   # M5
            "15 mins": 7,  # M15
            "30 mins": 8,  # M30
            "1 hour": 9,   # H1
            "4 hours": 10, # H4
            "1 day": 12,   # D1
            "1 week": 13,  # W1
        }

        period = period_map.get(bar_size)
        if period is None:
            logger.error(f"Unsupported bar size: {bar_size}")
            return False

        # Parse interval (e.g., "1 Y" = 1 year, "6 M" = 6 months)
        from datetime import timedelta
        import re

        interval_match = re.match(r"(\d+)\s*([YMD])", interval.upper())
        if interval_match:
            value = int(interval_match.group(1))
            unit = interval_match.group(2)

            if unit == "Y":
                days = value * 365
            elif unit == "M":
                days = value * 30
            else:  # D
                days = value

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
        else:
            # Default to 1 year
            start_time = datetime.utcnow() - timedelta(days=365)
            end_time = datetime.utcnow()

        try:
            # Use ProtoOAGetTrendbarsReq for OHLC bar data
            request = ProtoOAGetTrendbarsReq()
            request.ctidTraderAccountId = self.account_id
            request.symbolId = symbol_id
            request.period = period
            request.fromTimestamp = int(start_time.timestamp() * 1000)  # Convert to milliseconds
            request.toTimestamp = int(end_time.timestamp() * 1000)

            loop = asyncio.get_event_loop()
            future = loop.create_future()

            def on_success(response):
                try:
                    extracted = Protobuf.extract(response)
                    logger.debug(f"Trendbar response for {asset}: {type(extracted).__name__}")

                    bars = []
                    if hasattr(extracted, 'trendbar') and extracted.trendbar:
                        # cTrader trendbar prices use the same fixed 1/100000 scale as spot events.
                        # 'digits' is used only for rounding, not as the divisor.
                        digits = self._symbol_digits.get(symbol_id, 5)
                        conversion_factor = 100000  # always 1/100000 regardless of digits

                        anomaly_count = 0
                        for idx, bar in enumerate(extracted.trendbar):
                            # Convert trendbar to OHLCV format
                            # Trendbars use delta values from the low price
                            # utcTimestampInMinutes is the bar timestamp in minutes since epoch
                            timestamp_seconds = bar.utcTimestampInMinutes * 60 if hasattr(bar, 'utcTimestampInMinutes') else 0

                            # Get raw values for debugging
                            raw_low = bar.low if hasattr(bar, 'low') else 0
                            raw_delta_open = bar.deltaOpen if hasattr(bar, 'deltaOpen') else 0
                            raw_delta_high = bar.deltaHigh if hasattr(bar, 'deltaHigh') else 0
                            raw_delta_close = bar.deltaClose if hasattr(bar, 'deltaClose') else 0

                            # Convert: low is in pipettes, deltas are ALSO in pipettes
                            low = raw_low / conversion_factor
                            open_price = low + (raw_delta_open / conversion_factor)
                            high = low + (raw_delta_high / conversion_factor)
                            close = low + (raw_delta_close / conversion_factor)
                            volume = bar.volume if hasattr(bar, 'volume') else 0

                            # Validate OHLC consistency
                            is_anomaly = False
                            anomaly_reason = []

                            # Check for impossible values (high < low, etc.)
                            if high < low:
                                anomaly_reason.append(f"high({high:.5f}) < low({low:.5f})")
                                is_anomaly = True
                            if open_price < low or open_price > high:
                                anomaly_reason.append(f"open({open_price:.5f}) outside range [{low:.5f}, {high:.5f}]")
                                is_anomaly = True
                            if close < low or close > high:
                                anomaly_reason.append(f"close({close:.5f}) outside range [{low:.5f}, {high:.5f}]")
                                is_anomaly = True

                            # Check for very large bars (> 10% move) which might indicate conversion issues
                            if low > 0:
                                bar_range_pct = (high - low) / low * 100
                                if bar_range_pct > 10:
                                    anomaly_reason.append(f"bar_range={bar_range_pct:.2f}%")
                                    is_anomaly = True

                            if is_anomaly:
                                anomaly_count += 1
                                if anomaly_count <= 5:  # Only log first 5 anomalies
                                    logger.warning(
                                        f"OHLC anomaly for {asset} bar {idx}: {', '.join(anomaly_reason)}. "
                                        f"Raw: low={raw_low}, dO={raw_delta_open}, dH={raw_delta_high}, dC={raw_delta_close}. "
                                        f"Digits={digits}, factor={conversion_factor}"
                                    )

                            bars.append({
                                "date": float(timestamp_seconds),  # Unix timestamp as float for compatibility
                                "open": open_price,
                                "high": high,
                                "low": low,
                                "close": close,
                                "volume": volume
                            })

                        if anomaly_count > 0:
                            logger.warning(f"Found {anomaly_count} anomalous bars out of {len(bars)} for {asset}")
                        logger.info(f"Received {len(bars)} historical bars for {asset} (digits={digits})")
                    else:
                        logger.warning(f"No trendbars in response for {asset}")

                    # Call the callback with the bars - must be called from asyncio thread
                    if callback:
                        def run_callback():
                            try:
                                callback(bars, context or {})
                            except Exception as e:
                                logger.error(f"Error in historical data callback: {e}", exc_info=True)
                        loop.call_soon_threadsafe(run_callback)

                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, True)

                except Exception as e:
                    logger.error(f"Error processing trendbar response: {e}", exc_info=True)
                    if callback:
                        def run_error_callback():
                            try:
                                callback([], context or {})
                            except Exception:
                                pass
                        loop.call_soon_threadsafe(run_error_callback)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

            def on_error(failure):
                logger.error(f"Trendbar request failed: {failure}")
                if callback:
                    def run_error_callback():
                        try:
                            callback([], context or {})
                        except Exception:
                            pass
                    loop.call_soon_threadsafe(run_error_callback)
                if not future.done():
                    loop.call_soon_threadsafe(future.set_result, False)

            def send_request():
                try:
                    if self.client:
                        logger.debug(f"[Reactor Thread] Sending trendbar request for {asset}, symbolId={symbol_id}, period={period}")
                        d = self.client.send(request)
                        d.addCallbacks(on_success, on_error)
                        logger.debug(f"[Reactor Thread] Trendbar request sent for {asset}")
                    else:
                        logger.error("[Reactor Thread] Client is None, cannot send trendbar request")
                        if not future.done():
                            loop.call_soon_threadsafe(future.set_result, False)
                except Exception as e:
                    logger.error(f"[Reactor Thread] Error sending trendbar request: {e}", exc_info=True)
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, False)

            logger.info(f"Requesting historical data for {asset}: bar_size={bar_size}, interval={interval}, symbolId={symbol_id}")
            reactor.callFromThread(send_request)

            # Wait for response with timeout
            try:
                await asyncio.wait_for(future, timeout=30.0)
                return True
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for historical data for {asset}")
                if callback:
                    callback([], context or {})
                return False

        except Exception as e:
            logger.error(f"Error requesting historical data: {e}", exc_info=True)
            if callback:
                callback([], context or {})
            return False

