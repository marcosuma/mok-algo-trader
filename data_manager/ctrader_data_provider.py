"""
cTrader Data Provider

Fetches historical market data from cTrader Open API.
"""

import os
import asyncio
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List
import re

from data_manager.data_provider import BaseDataProvider, Asset


# Check for ctrader-open-api availability
try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiMessages_pb2 import *
    from twisted.internet import reactor
    CTRADER_AVAILABLE = True
except ImportError:
    CTRADER_AVAILABLE = False


class CTraderDataProvider(BaseDataProvider):
    """
    Data provider for cTrader Open API.

    Requires cTrader account credentials (client_id, client_secret, access_token).
    """

    # cTrader-specific interval limits (similar to IBKR)
    BAR_SIZE_INTERVAL_LIMITS = {
        "1 min": "1 M",
        "5 mins": "1 Y",
        "15 mins": "1 Y",
        "1 hour": "1 Y",
        "4 hours": "1 Y",
        "1 day": "10 Y",
        "1 week": "10 Y",
    }

    # cTrader bar period mapping (ProtoOATrendbarPeriod enum values)
    # M1=1,M2=2,M3=3,M4=4,M5=5,M10=6,M15=7,M30=8,H1=9,H4=10,H12=11,D1=12,W1=13,MN1=14
    BAR_SIZE_TO_PERIOD = {
        "1 min": 1,     # M1
        "5 mins": 5,    # M5
        "15 mins": 7,   # M15
        "30 mins": 8,   # M30
        "1 hour": 9,    # H1
        "4 hours": 10,  # H4
        "1 day": 12,    # D1
        "1 week": 13,   # W1
    }

    # Bar duration in minutes (used to calculate chunk sizes for pagination)
    BAR_DURATION_MINUTES = {
        "1 min": 1,
        "5 mins": 5,
        "15 mins": 15,
        "30 mins": 30,
        "1 hour": 60,
        "4 hours": 240,
        "1 day": 1440,
        "1 week": 10080,
    }

    # cTrader API hard limit per request
    MAX_BARS_PER_REQUEST = 4900

    def __init__(
        self,
        data_base_dir: str = "data",
        client_id: str = None,
        client_secret: str = None,
        access_token: str = None,
        environment: str = "demo"  # "demo" or "live"
    ):
        super().__init__(data_base_dir)

        if not CTRADER_AVAILABLE:
            raise ImportError(
                "ctrader-open-api is not installed. "
                "Install it with: pip install ctrader-open-api twisted service_identity"
            )

        # Get credentials from args or environment
        self.client_id = client_id or os.environ.get("CTRADER_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("CTRADER_CLIENT_SECRET")
        self.access_token = access_token or os.environ.get("CTRADER_ACCESS_TOKEN")
        self.environment = environment.lower()

        # Don't raise here - defer to connect() for better error messages
        self._credentials_valid = all([self.client_id, self.client_secret, self.access_token])

        self.client: Optional[Client] = None
        self.account_id: Optional[int] = None
        self._reactor_thread: Optional[threading.Thread] = None
        self._symbol_cache: Dict[str, int] = {}  # asset -> symbol_id
        self._symbol_digits: Dict[int, int] = {}  # symbol_id -> digits
        self._authenticated = False
        self._auth_event = threading.Event()
        self._symbols_loaded = threading.Event()

    def _start_reactor(self):
        """Start Twisted reactor in a separate thread."""
        def run_reactor():
            try:
                reactor.run(installSignalHandlers=0)
            except Exception as e:
                print(f"[cTrader] Reactor error: {e}")

        if not reactor.running:
            self._reactor_thread = threading.Thread(target=run_reactor, daemon=True, name="CTraderReactor")
            self._reactor_thread.start()

            # Wait for reactor to start
            max_wait = 5.0
            waited = 0
            while not reactor.running and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

    def connect(self) -> bool:
        """Connect to cTrader Open API."""
        if self.connected and self._authenticated:
            return True

        # Check credentials before attempting connection
        if not self._credentials_valid:
            print("=" * 80)
            print("cTrader Credentials Required")
            print("=" * 80)
            print("\nTo use cTrader as a data source, you need to set these environment variables:")
            print("\n  export CTRADER_CLIENT_ID='your_client_id'")
            print("  export CTRADER_CLIENT_SECRET='your_client_secret'")
            print("  export CTRADER_ACCESS_TOKEN='your_access_token'")
            print("\nTo get these credentials:")
            print("  1. Go to https://openapi.ctrader.com/")
            print("  2. Register/login to your cTrader account")
            print("  3. Create an application to get Client ID and Secret")
            print("  4. Generate an access token for your trading account")
            print("\nAlternatively, use the helper script:")
            print("  python live_trading/scripts/get_ctrader_token.py")
            print("=" * 80)
            return False

        try:
            # Get endpoint based on environment
            host = EndPoints.PROTOBUF_LIVE_HOST if self.environment == "live" else EndPoints.PROTOBUF_DEMO_HOST
            port = EndPoints.PROTOBUF_PORT

            self.client = Client(host, port, TcpProtocol)

            # Start reactor if needed
            self._start_reactor()

            # Connect and authenticate
            self._auth_event.clear()
            self._symbols_loaded.clear()

            def on_connected(client):
                print(f"✓ Connected to cTrader ({self.environment})")
                # Authenticate application
                request = ProtoOAApplicationAuthReq()
                request.clientId = self.client_id
                request.clientSecret = self.client_secret
                self.client.send(request)

            def on_disconnected(client, reason):
                print(f"✗ Disconnected from cTrader: {reason}")
                self.connected = False
                self._authenticated = False

            def on_message(client, message):
                try:
                    extracted = Protobuf.extract(message)
                    msg_type = type(extracted).__name__

                    if msg_type == "ProtoOAApplicationAuthRes":
                        self._on_app_auth(extracted)
                    elif msg_type == "ProtoOAGetAccountListByAccessTokenRes":
                        self._on_account_list(extracted)
                    elif msg_type == "ProtoOAAccountAuthRes":
                        self._on_account_auth(extracted)
                    elif msg_type in ("ProtoOASymbolsListRes", "ProtoOASymbolByIdRes"):
                        self._on_symbols_list(extracted)
                    elif msg_type == "ProtoOAErrorRes":
                        error_code = getattr(extracted, 'errorCode', 'UNKNOWN')
                        description = getattr(extracted, 'description', '')
                        print(f"✗ cTrader error: {error_code} - {description}")
                except Exception as e:
                    print(f"[cTrader] Message handling error: {e}")

            # Set up callbacks
            self.client.setConnectedCallback(on_connected)
            self.client.setDisconnectedCallback(on_disconnected)
            self.client.setMessageReceivedCallback(on_message)

            # Start service (must be called from reactor thread)
            def start_service():
                self.client.startService()

            reactor.callFromThread(start_service)

            # Wait for authentication
            if not self._auth_event.wait(timeout=30):
                print("✗ cTrader authentication timeout")
                return False

            # Wait for symbols to load
            if not self._symbols_loaded.wait(timeout=30):
                print("⚠ Symbols not loaded (may affect data fetching)")

            self.connected = True
            return True

        except Exception as e:
            print(f"✗ Failed to connect to cTrader: {e}")
            return False

    def _on_app_auth(self, response):
        """Handle application auth response."""
        print("✓ cTrader application authenticated")
        # Request account list
        request = ProtoOAGetAccountListByAccessTokenReq()
        request.accessToken = self.access_token

        def send():
            self.client.send(request)
        reactor.callFromThread(send)

    def _on_account_list(self, response):
        """Handle account list response."""
        if hasattr(response, 'ctidTraderAccount') and response.ctidTraderAccount:
            account = response.ctidTraderAccount[0]
            self.account_id = account.ctidTraderAccountId
            print(f"✓ Using account: {self.account_id}")

            # Authenticate account
            request = ProtoOAAccountAuthReq()
            request.ctidTraderAccountId = self.account_id
            request.accessToken = self.access_token

            def send():
                self.client.send(request)
            reactor.callFromThread(send)
        else:
            print("✗ No trading accounts found")
            self._auth_event.set()

    def _on_account_auth(self, response):
        """Handle account auth response."""
        print("✓ cTrader account authenticated")
        self._authenticated = True
        self._auth_event.set()

        # Request symbols list
        request = ProtoOASymbolsListReq()
        request.ctidTraderAccountId = self.account_id

        def send():
            self.client.send(request)
        reactor.callFromThread(send)

    def _on_symbols_list(self, response):
        """Handle symbols list response."""
        if hasattr(response, 'symbol'):
            for symbol in response.symbol:
                # Convert symbol name to our format
                symbol_name = symbol.symbolName.upper()
                # Remove common suffixes
                for suffix in [".FX", ".PRO", ".ECN", ".M", ".C"]:
                    if symbol_name.endswith(suffix):
                        symbol_name = symbol_name[:-len(suffix)]
                        break

                # Convert to our format (EUR-USD)
                clean_name = symbol_name.replace("/", "").replace(".", "").replace("_", "")
                if len(clean_name) == 6 and clean_name.isalpha():
                    asset_name = f"{clean_name[:3]}-{clean_name[3:]}"
                    self._symbol_cache[asset_name] = symbol.symbolId
                    self._symbol_digits[symbol.symbolId] = getattr(symbol, 'digits', 5)

            print(f"✓ Loaded {len(self._symbol_cache)} symbols")

        self._symbols_loaded.set()

    def disconnect(self):
        """Disconnect from cTrader."""
        if self.client:
            try:
                def stop():
                    self.client.stopService()
                reactor.callFromThread(stop)
                time.sleep(0.5)
            except Exception:
                pass
            self.client = None

        self.connected = False
        self._authenticated = False

    def _convert_asset_name(self, asset: Asset) -> str:
        """Convert Asset to lookup key."""
        return asset.pair  # e.g., "EUR-USD"

    def _get_symbol_id(self, asset: Asset) -> Optional[int]:
        """Get cTrader symbol ID for an asset."""
        asset_name = self._convert_asset_name(asset)
        return self._symbol_cache.get(asset_name)

    def fetch_historical_data(
        self,
        asset: Asset,
        bar_size: str,
        interval: str,
        callback: Optional[Callable[[pd.DataFrame, Asset, str], None]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from cTrader.

        Args:
            asset: Asset to fetch data for
            bar_size: Bar size (e.g., "1 hour", "5 mins")
            interval: Time interval (e.g., "6 M", "1 Y")
            callback: Optional callback for async operation

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected or not self._authenticated:
            raise RuntimeError("Not connected to cTrader. Call connect() first.")

        symbol_id = self._get_symbol_id(asset)
        if not symbol_id:
            print(f"✗ Symbol not found for {asset.pair}")
            print(f"  Available symbols: {list(self._symbol_cache.keys())[:10]}...")
            return None

        # Get period
        period = self.BAR_SIZE_TO_PERIOD.get(bar_size)
        if not period:
            print(f"✗ Unsupported bar size: {bar_size}")
            return None

        # Parse interval
        match = re.match(r"(\d+)\s*([YMWD])", interval.upper())
        if match:
            value = int(match.group(1))
            unit = match.group(2)

            if unit == "Y":
                days = value * 365
            elif unit == "M":
                days = value * 30
            elif unit == "W":
                days = value * 7
            else:  # D
                days = value

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
        else:
            # Default to 6 months
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=180)

        csv_path = self.get_csv_path(asset, bar_size, interval)
        folder = self.get_asset_folder(asset)
        os.makedirs(folder, exist_ok=True)

        # Calculate chunk size to stay within MAX_BARS_PER_REQUEST
        bar_minutes = self.BAR_DURATION_MINUTES.get(bar_size, 60)
        chunk_delta = timedelta(minutes=self.MAX_BARS_PER_REQUEST * bar_minutes)

        digits = self._symbol_digits.get(symbol_id, 5)

        print(f"  Fetching {bar_size} data for {asset.pair} "
              f"({start_time.date()} → {end_time.date()})...")

        all_dfs: List[pd.DataFrame] = []
        chunk_start = start_time
        chunk_num = 0

        while chunk_start < end_time:
            chunk_end = min(chunk_start + chunk_delta, end_time)
            chunk_df = self._fetch_chunk(
                symbol_id=symbol_id,
                period=period,
                from_ts_ms=int(chunk_start.timestamp() * 1000),
                to_ts_ms=int(chunk_end.timestamp() * 1000),
                digits=digits,
            )
            if chunk_df is not None and len(chunk_df) > 0:
                all_dfs.append(chunk_df)
                chunk_num += 1
                if chunk_delta < timedelta(days=365):
                    # Only print chunk progress for small bar sizes with many chunks
                    print(f"    chunk {chunk_num}: {len(chunk_df)} bars "
                          f"({chunk_start.date()} → {chunk_end.date()})")
            chunk_start = chunk_end
            if chunk_start < end_time:
                time.sleep(0.3)  # avoid rate-limiting between chunks

        if not all_dfs:
            print(f"  ⚠ No bars received for {asset.pair}")
            return None

        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {len(df)} bars to {csv_path}")

        if callback:
            callback(df, asset, csv_path)

        return df

    def _fetch_chunk(
        self,
        symbol_id: int,
        period: int,
        from_ts_ms: int,
        to_ts_ms: int,
        digits: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of trendbar data (up to MAX_BARS_PER_REQUEST bars)."""
        result_event = threading.Event()
        result_data = [None]

        def on_success(response):
            try:
                extracted = Protobuf.extract(response)
                bars = []

                if hasattr(extracted, 'trendbar') and extracted.trendbar:
                    conversion_factor = 10 ** digits

                    anomaly_count = 0
                    for idx, bar in enumerate(extracted.trendbar):
                        timestamp = bar.utcTimestampInMinutes * 60 if hasattr(bar, 'utcTimestampInMinutes') else 0

                        raw_low = bar.low if hasattr(bar, 'low') else 0
                        raw_delta_open = bar.deltaOpen if hasattr(bar, 'deltaOpen') else 0
                        raw_delta_high = bar.deltaHigh if hasattr(bar, 'deltaHigh') else 0
                        raw_delta_close = bar.deltaClose if hasattr(bar, 'deltaClose') else 0

                        low = raw_low / conversion_factor
                        open_price = low + (raw_delta_open / conversion_factor)
                        high = low + (raw_delta_high / conversion_factor)
                        close = low + (raw_delta_close / conversion_factor)
                        volume = bar.volume if hasattr(bar, 'volume') else 0

                        # Validate OHLC consistency
                        is_anomaly = False
                        anomaly_reason = []

                        if high < low:
                            anomaly_reason.append(f"high({high:.5f}) < low({low:.5f})")
                            is_anomaly = True
                        if open_price < low or open_price > high:
                            anomaly_reason.append(f"open({open_price:.5f}) outside range")
                            is_anomaly = True
                        if close < low or close > high:
                            anomaly_reason.append(f"close({close:.5f}) outside range")
                            is_anomaly = True

                        if low > 0:
                            bar_range_pct = (high - low) / low * 100
                            if bar_range_pct > 10:
                                anomaly_reason.append(f"bar_range={bar_range_pct:.2f}%")
                                is_anomaly = True

                        if is_anomaly:
                            anomaly_count += 1
                            if anomaly_count <= 5:
                                print(f"  ⚠ OHLC anomaly bar {idx}: {', '.join(anomaly_reason)}. "
                                      f"Raw: low={raw_low}, dO={raw_delta_open}, "
                                      f"dH={raw_delta_high}, dC={raw_delta_close}")

                        bars.append({
                            "date": datetime.utcfromtimestamp(timestamp).strftime("%Y%m%d %H:%M:%S"),
                            "open": open_price,
                            "high": high,
                            "low": low,
                            "close": close,
                            "volume": volume
                        })

                    if anomaly_count > 0:
                        print(f"  ⚠ {anomaly_count} anomalous bars in chunk")

                    if bars:
                        result_data[0] = pd.DataFrame(bars)

            except Exception as e:
                print(f"  ✗ Error processing chunk: {e}")
            finally:
                result_event.set()

        def on_error(failure):
            print(f"  ✗ Chunk request failed: {failure}")
            result_event.set()

        request = ProtoOAGetTrendbarsReq()
        request.ctidTraderAccountId = self.account_id
        request.symbolId = symbol_id
        request.period = period
        request.fromTimestamp = from_ts_ms
        request.toTimestamp = to_ts_ms

        def send_request():
            d = self.client.send(request)
            d.addCallbacks(on_success, on_error)

        reactor.callFromThread(send_request)
        result_event.wait(timeout=120)

        return result_data[0]
