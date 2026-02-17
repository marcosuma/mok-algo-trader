#!/usr/bin/env python3
"""
Trading Bot CLI

A command-line interface for backtesting trading strategies on historical market data.

Data Sources:
- local: Use existing CSV files (default for backtesting, no broker needed)
- ibkr: Fetch from Interactive Brokers TWS/Gateway
- ctrader: Fetch from cTrader Open API

For backtesting, use 'local' mode with pre-downloaded data.
For fetching new data, use 'ibkr' or 'ctrader' depending on your broker.
"""
import argparse
import os
import json
import threading
import collections
import time
import shlex
import sys
import signal
import atexit

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    # Load from .env file in current directory or parent directories
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass

import pandas as pd

from technical_indicators.technical_indicators import TechnicalIndicators
from plot.plot import Plot
from support_resistance.support_resistance_v1 import SupportResistanceV1
from local_extrema.plot_local_extrema import PlotLocalExtrema
from forex_strategies.marsi_strategy import MARSIStrategy
from forex_strategies.rsi_strategy import RSIStrategy
from forex_strategies.hammer_shooting_star import HammerShootingStar
from forex_strategies.backtesting_strategy import ForexBacktestingStrategy

# Data provider imports (lazy loading for broker-specific providers)
from data_manager.data_provider import (
    DataProviderType,
    Asset,
    create_data_provider,
    get_provider_type_from_string,
    LocalDataProvider,
)

# Note: ML trainers are imported lazily inside their commands to avoid
# forcing TensorFlow/Keras dependencies when not needed.
# Other strategies are imported lazily in cmd_fetch_process_plot to avoid
# unnecessary imports when not used.

# IBKR-specific imports (lazy loaded when needed)
# from ib_api_client.ib_api_client import IBApiClient
# from ibapi.contract import Contract
# import request_historical_data.request_historical_data as rhd
# import request_historical_data.callback as rhd_callback


# Global counter for unique client IDs (for IBKR backward compatibility)
_client_id_counter = 1000

# Track all active data providers for cleanup on exit
_active_providers = []
_active_providers_lock = threading.Lock()

# Legacy: Track all active IBKR clients for cleanup on exit
_active_clients = []
_active_clients_lock = threading.Lock()


def _create_data_provider(provider_type_str: str = "local", **kwargs):
    """
    Create a data provider based on the specified type.

    Args:
        provider_type_str: Provider type string ("local", "ibkr", "ctrader")
        **kwargs: Provider-specific arguments

    Returns:
        Configured data provider instance
    """
    global _active_providers

    provider_type = get_provider_type_from_string(provider_type_str)
    provider = create_data_provider(provider_type, **kwargs)

    # Track for cleanup
    with _active_providers_lock:
        _active_providers.append(provider)

    return provider


def _cleanup_all_providers():
    """Cleanup all active data providers."""
    global _active_providers

    with _active_providers_lock:
        for provider in _active_providers:
            try:
                provider.disconnect()
            except Exception:
                pass
        _active_providers.clear()

def _connect_ib():
    """Connect to IBKR with a unique client ID."""
    global _client_id_counter, _active_clients
    callbackFnMap = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    contextMap = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    client = IBApiClient(callbackFnMap, contextMap)

    # Generate unique client ID (thread-safe)
    with _active_clients_lock:
        client_id = _client_id_counter
        _client_id_counter += 1
        if _client_id_counter > 9999:  # Reset if we get too high
            _client_id_counter = 1000

    client.connect('127.0.0.1', 7497, client_id)

    def run_loop():
        try:
            client.run()
        except Exception as e:
            # Thread might be interrupted during shutdown
            pass

    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()

    # Track this client for cleanup
    with _active_clients_lock:
        _active_clients.append(client)

    for _ in range(60):
        if isinstance(getattr(client, 'nextorderId', None), int):
            return client, callbackFnMap, contextMap
        time.sleep(1)
    raise RuntimeError("Could not connect to IB within 60s")


def _disconnect_ib(client):
    """Disconnect from IBKR."""
    global _active_clients
    if client is None:
        return

    try:
        # Try to disconnect - IBKR client.disconnect() is safe to call even if not connected
        client.disconnect()
        # Give it a moment to disconnect
        time.sleep(0.1)
    except Exception:
        # Ignore disconnect errors - connection may already be closed
        pass
    finally:
        # Always remove from active clients list
        with _active_clients_lock:
            if client in _active_clients:
                _active_clients.remove(client)


def _cleanup_all_clients():
    """Disconnect all active IBKR clients and data providers."""
    global _active_clients

    # Cleanup data providers first
    _cleanup_all_providers()

    # Cleanup IBKR clients (legacy)
    with _active_clients_lock:
        clients_to_disconnect = list(_active_clients)
        _active_clients.clear()

    for client in clients_to_disconnect:
        try:
            _disconnect_ib(client)
        except:
            pass


def _parse_contracts(contracts_data):
    """
    Parse contracts from JSON data, handling both old string format and new object format.
    Filters by enabled flag (defaults to True for backward compatibility).

    Args:
        contracts_data: List of contracts (strings or objects with 'contract' and 'enabled' keys)

    Returns:
        List of contract strings that are enabled
    """
    enabled_contracts = []
    for item in contracts_data:
        if isinstance(item, str):
            # Old format: just a string, treat as enabled
            enabled_contracts.append(item)
        elif isinstance(item, dict):
            # New format: object with 'contract' and optional 'enabled' flag
            if item.get("enabled", True):  # Default to True if not specified
                enabled_contracts.append(item["contract"])
        else:
            print(f"⚠ Skipping invalid contract format: {item}")
    return enabled_contracts


def cmd_fetch_process_plot(args: argparse.Namespace):
    """
    Fetch historical data, compute indicators, run strategies, and plot results.

    Supports multiple data sources:
    - local: Use existing CSV files (default, no broker connection needed)
    - ibkr: Fetch from Interactive Brokers
    - ctrader: Fetch from cTrader
    """
    data_source = getattr(args, 'data_source', 'local')
    provider = None

    # Legacy IBKR variables for backward compatibility
    client = None
    callbackFnMap = None
    contextMap = None

    try:
        # Create data provider based on source
        if data_source == 'local':
            provider = _create_data_provider('local')
            provider.connect()
            print("Using local data files (no broker connection)")
        elif data_source == 'ibkr':
            # Use IBKR provider (or fall back to legacy code)
            provider = _create_data_provider('ibkr')
            if not provider.connect():
                print("✗ Failed to connect to IBKR")
                return
        elif data_source == 'ctrader':
            provider = _create_data_provider('ctrader')
            if not provider.connect():
                print("✗ Failed to connect to cTrader")
                return

        with open("contracts.json") as f:
            data = json.load(f)

        # Parse and filter contracts by enabled flag
        enabled_contracts = _parse_contracts(data["contracts"])

        plots_queue = []

        def process_contract_data(df, technical_indicators, file_to_save, asset, candlestick_data):
            """Process downloaded data with indicators and strategies."""
            try:
                print(f"Processing data for {asset.pair}...")
                df = technical_indicators.execute(df)

                # Strategy markers functions to collect
                strategy_markers_fns = []

                # Optional local extrema markers
                local_extrema_markers_fn = None
                if getattr(args, "with_local_extrema", True):
                    extrema_plotter = PlotLocalExtrema()
                    local_extrema_markers_fn = extrema_plotter.execute(df)

                # Check if any strategies/models are explicitly enabled
                has_any_strategy = (
                    args.use_support_resistance_v1 or
                    args.use_support_resistance or
                    args.use_lstm or
                    args.use_svm_trainer
                )

                # Support/Resistance V1: default enabled if no strategies/models specified
                use_support_resistance_v1 = args.use_support_resistance_v1 or (not has_any_strategy)

                # Support/Resistance V1
                srv1_markers_fn = None
                y_lines = None
                if use_support_resistance_v1:
                    srv1 = SupportResistanceV1(plots_queue, file_to_save)
                    srv1_markers_fn, y_lines = srv1.execute(df)
                    if srv1_markers_fn:
                        strategy_markers_fns.append(srv1_markers_fn)

                # Support/Resistance (alternative)
                if args.use_support_resistance:
                    from support_resistance.support_resistance import SupportResistance
                    sr = SupportResistance(candlestick_data, plots_queue, file_to_save)
                    sr.process_data_with_file(df)

                # Create a contract-like object for plotting
                class ContractLike:
                    def __init__(self, symbol, currency):
                        self.symbol = symbol
                        self.currency = currency

                contract_obj = ContractLike(asset.symbol, asset.currency)

                # Plot with all strategy markers
                primary_markers = strategy_markers_fns[0] if strategy_markers_fns else None
                Plot(df, plots_queue, contract_obj).plot(
                    primary_markers,
                    srv1_markers_fn,
                    local_extrema_markers_fn,
                )
                print(f"✓ Completed processing {asset.pair}")
            except Exception as e:
                print(f"✗ Error processing {asset.pair}: {e}")
                import traceback
                traceback.print_exc()

        # Bar size to interval mapping
        BAR_SIZE_INTERVAL_LIMITS = provider.BAR_SIZE_INTERVAL_LIMITS

        for _contract in enabled_contracts:
            candlestick_data = []
            fields = str.split(_contract, ",")

            # Create Asset object
            asset = Asset(
                symbol=fields[0],
                currency=fields[1],
                sec_type=fields[2] if len(fields) > 2 else "CASH",
                exchange=fields[3] if len(fields) > 3 else "IDEALPRO"
            )

            bar_size = args.bar_size
            # Use bar-size-specific interval if user didn't specify one
            if args.interval == '6 M' and bar_size in BAR_SIZE_INTERVAL_LIMITS:
                interval = BAR_SIZE_INTERVAL_LIMITS[bar_size]
            else:
                interval = args.interval

            # Get file path
            file_to_save = provider.get_csv_path(asset, bar_size, interval)
            os.makedirs(os.path.dirname(file_to_save), exist_ok=True)

            technical_indicators = TechnicalIndicators(candlestick_data, file_to_save)

            if not os.path.exists(file_to_save) or args.refresh:
                if data_source == 'local':
                    print(f"⚠ No local data found for {asset.pair} ({bar_size})")
                    print(f"  Expected: {file_to_save}")
                    print(f"  Use --data-source ibkr or --data-source ctrader to download data")
                    continue

                print(f"Fetching data for {asset.pair} ({bar_size})...")
                df = provider.fetch_historical_data(asset, bar_size, interval)

                if df is not None and len(df) > 0:
                    process_contract_data(df, technical_indicators, file_to_save, asset, candlestick_data)
                else:
                    print(f"⚠ No data received for {asset.pair}")
            else:
                print(f"Loading existing data for {asset.pair}...")
                df = pd.read_csv(file_to_save, index_col=[0])
                process_contract_data(df, technical_indicators, file_to_save, asset, candlestick_data)

        # Display plots
        num_plots = len(plots_queue)
        if num_plots > 0:
            print(f"\nDisplaying {num_plots} plot(s)...")
            plot_count = 0
            while plots_queue:
                try:
                    plot_fn = plots_queue.pop(0)
                    plot_count += 1
                    print(f"  Displaying plot {plot_count}/{num_plots}...")
                    plot_fn()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error displaying plot: {e}")
                    import traceback
                    traceback.print_exc()
            print("All plots displayed.")
        else:
            print("No plots to display.")

    finally:
        # Cleanup
        if provider:
            provider.disconnect()
        print("Command completed.")


def cmd_train_extrema_predictor(args: argparse.Namespace):
    """Train ML model to predict local extrema (buy/sell points)."""
    import glob
    try:
        from machine_learning.extrema_predictor import ExtremaPredictor
    except ImportError as e:
        error_msg = str(e)
        if "libomp" in error_msg or "OpenMP" in error_msg:
            print("=" * 80)
            print("ERROR: XGBoost requires OpenMP runtime library")
            print("=" * 80)
            print("\nTo fix this on macOS, run:")
            print("  brew install libomp")
            print("\nAfter installation, you may need to:")
            print("  1. Restart your terminal, OR")
            print("  2. Set the library path:")
            print("     export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH")
            print("\nIf the issue persists, try reinstalling xgboost:")
            print("  pip uninstall xgboost && pip install xgboost")
            print("=" * 80)
        raise

    print("=" * 80)
    print("Training Local Extrema Predictor")
    print("=" * 80)

    # Determine input files
    csv_files = []

    if args.input:
        # Single file specified
        if os.path.exists(args.input):
            csv_files = [args.input]
        else:
            print(f"✗ Error: File not found: {args.input}")
            return
    elif args.asset:
        # Asset specified - find all CSV files for that asset
        asset_dir = os.path.join("data", args.asset)
        if not os.path.exists(asset_dir):
            print(f"✗ Error: Asset directory not found: {asset_dir}")
            return

        # Find all CSV files in the asset directory
        pattern = os.path.join(asset_dir, "*.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"✗ Error: No CSV files found in {asset_dir}")
            return

        print(f"Found {len(csv_files)} CSV file(s) for asset {args.asset}")
    else:
        # Default: use all CSV files from all assets
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"✗ Error: Data directory not found: {data_dir}")
            return

        pattern = os.path.join(data_dir, "*", "*.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"✗ Error: No CSV files found in {data_dir}")
            return

        print(f"Found {len(csv_files)} CSV file(s) across all assets")

    # Load and combine data from all files
    print(f"\nLoading data from {len(csv_files)} file(s)...")
    all_dataframes = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=[0])
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Check if local_extrema column exists
            if "local_extrema" not in df.columns:
                print(f"  ⚠ Skipping {os.path.basename(csv_file)}: missing 'local_extrema' column")
                print(f"    (Run TechnicalIndicators first to calculate local extrema)")
                continue

            # Check if technical indicators exist
            required_indicators = ["SMA_50", "RSI_14", "macd"]
            missing = [ind for ind in required_indicators if ind not in df.columns]
            if missing:
                print(f"  ⚠ Skipping {os.path.basename(csv_file)}: missing indicators: {missing}")
                continue

            # Add source file info
            df["source_file"] = os.path.basename(csv_file)
            all_dataframes.append(df)
            print(f"  ✓ Loaded {os.path.basename(csv_file)}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ Error loading {os.path.basename(csv_file)}: {e}")
            continue

    if not all_dataframes:
        print("\n✗ No valid data files found. Make sure:")
        print("  1. CSV files have 'local_extrema' column (run TechnicalIndicators)")
        print("  2. CSV files have technical indicators (SMA, RSI, MACD, etc.)")
        return

    # Combine all dataframes
    print(f"\nCombining {len(all_dataframes)} dataset(s)...")
    combined_df = pd.concat(all_dataframes, ignore_index=False)
    combined_df = combined_df.sort_index()  # Sort by date

    print(f"Total combined data: {len(combined_df)} bars")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    # Initialize predictor
    predictor = ExtremaPredictor(
        lookback_bars=args.lookback_bars,
        use_technical_indicators=not args.no_indicators,
        model_type=args.model_type,
        use_feature_selection=not args.no_feature_selection,
    )

    # Train model
    print(f"\nTraining model with lookback={args.lookback_bars} bars...")
    try:
        metrics = predictor.train(
            combined_df,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state,
            model_name=args.model_name,
            retrain=args.retrain,
        )

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Model saved to: {os.path.join(predictor.model_dir, args.model_name)}.pkl")
        print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")

        # Note about realistic expectations
        print("\n" + "=" * 80)
        print("📊 Performance Note:")
        print("=" * 80)
        print("Predicting local extrema is inherently challenging because:")
        print("  • Markets contain significant noise")
        print("  • Extrema are rare events (typically 5-15% of bars)")
        print("  • The 'None' class dominates the dataset")
        print("\n60%+ accuracy is actually quite good for this task!")
        print("Consider using the model's probability outputs to filter for")
        print("high-confidence predictions rather than using all predictions.")
        print("=" * 80)

        print("\nPer-class Performance:")

        # Only print metrics for classes that were actually present in the test set
        available_classes = list(metrics['precision'].keys())
        for class_name in available_classes:
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision'][class_name]:.4f}")
            print(f"    Recall: {metrics['recall'][class_name]:.4f}")
            print(f"    F1-Score: {metrics['f1_score'][class_name]:.4f}")

        # Show which classes were missing (if any)
        all_possible_classes = ["None", "LOCAL_MIN", "LOCAL_MAX"]
        missing_classes = [cls for cls in all_possible_classes if cls not in available_classes]
        if missing_classes:
            print(f"\nNote: The following classes were not present in the test set: {', '.join(missing_classes)}")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return


def cmd_download_and_process_data(args: argparse.Namespace):
    """
    Phase 1 & 2: Download historical data and process with technical indicators.

    Supports multiple data sources:
    - ibkr: Download from Interactive Brokers (default)
    - ctrader: Download from cTrader
    """
    from data_manager.indicators_processor import IndicatorsProcessor
    import time

    data_source = getattr(args, 'data_source', 'ibkr')
    provider = None

    print("=" * 80)
    print(f"PHASE 1: Downloading Historical Data from {data_source.upper()}")
    print("=" * 80)

    # Create and connect data provider
    try:
        provider = _create_data_provider(data_source)
        if not provider.connect():
            print(f"✗ Failed to connect to {data_source.upper()}")
            if data_source == 'ibkr':
                print("Make sure IBKR TWS/Gateway is running and API is enabled on port 7497")
            elif data_source == 'ctrader':
                print("Make sure CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, and CTRADER_ACCESS_TOKEN are set")
            return
    except Exception as e:
        print(f"✗ {e}")
        return

    # Parse bar sizes (strip whitespace)
    bar_sizes = [bs.strip() for bs in args.bar_sizes.split(",")] if args.bar_sizes else provider.DEFAULT_BAR_SIZES

    # Add 1-minute data if requested
    if args.include_1min and bar_sizes and "1 min" not in bar_sizes:
        bar_sizes.append("1 min")
        print("⚠ Including 1-minute data (may be limited by broker API, may take longer)")

    # Load contracts
    with open(args.contracts_file) as f:
        contracts_data = json.load(f)

    enabled_contracts = _parse_contracts(contracts_data["contracts"])

    download_results = {}

    try:
        for _contract in enabled_contracts:
            fields = str.split(_contract, ",")
            asset = Asset(
                symbol=fields[0],
                currency=fields[1],
                sec_type=fields[2] if len(fields) > 2 else "CASH",
                exchange=fields[3] if len(fields) > 3 else "IDEALPRO"
            )

            print(f"\n📊 Downloading data for {asset.pair}...")
            contract_results = {}

            for bar_size in bar_sizes:
                # Get appropriate interval for this bar size
                interval = args.interval or provider.get_interval_for_bar_size(bar_size)

                # Check if data exists
                if not args.force_refresh and provider.data_exists(asset, bar_size, interval):
                    print(f"  ✓ {bar_size} data already exists. Skipping.")
                    contract_results[bar_size] = True
                    continue

                # Download data
                print(f"  Downloading {bar_size} data...")
                try:
                    df = provider.fetch_historical_data(asset, bar_size, interval)
                    if df is not None and len(df) > 0:
                        print(f"  ✓ Saved {len(df)} bars")
                        contract_results[bar_size] = True
                    else:
                        print(f"  ⚠ No data received")
                        contract_results[bar_size] = False
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    contract_results[bar_size] = False

                # Small delay between requests to avoid rate limiting
                time.sleep(0.5)

            download_results[asset.pair] = contract_results

        print("\n" + "=" * 80)
        print("PHASE 2: Processing Technical Indicators")
        print("=" * 80)

        # Process indicators for all files
        processor = IndicatorsProcessor(data_base_dir=provider.data_base_dir)
        process_results = processor.process_all_contracts()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total_contracts = len(download_results)
        total_csvs = sum(len(results) for results in process_results.values())
        successful_csvs = sum(
            sum(1 for success in results.values() if success)
            for results in process_results.values()
        )

        print(f"Contracts processed: {total_contracts}")
        print(f"CSV files processed: {successful_csvs}/{total_csvs}")
        print("\n✓ Data download and indicator processing complete!")

    finally:
        # Always disconnect when done
        if provider:
            provider.disconnect()
            time.sleep(0.2)


def cmd_test_forex_strategies(args: argparse.Namespace):
    """Test forex strategies on historical data."""
    from forex_strategies.strategy_tester import StrategyTester
    from forex_strategies.strategy_registry import get_all_strategies, filter_strategies, get_strategy_names
    from forex_strategies.multi_timeframe_strategy import AdaptiveMultiTimeframeStrategy
    from forex_strategies.buy_and_hold_strategy import BuyAndHoldStrategy
    from plot.plot import Plot
    import collections
    import glob
    import re
    from datetime import datetime

    data_dir = "data"

    # Get strategies to test
    if hasattr(args, 'strategies') and args.strategies:
        # Parse comma-separated strategy names
        strategy_names = [s.strip() for s in args.strategies.split(',')]
        strategies_to_test = filter_strategies(strategy_names)
        print(f"\nTesting {len(strategies_to_test)} specified strategy(ies): {', '.join(sorted(strategies_to_test.keys()))}")
    else:
        # Test all strategies (except BuyAndHoldStrategy which is only used for baseline)
        all_strategies = get_all_strategies()
        # Exclude BuyAndHoldStrategy from testing (it's only for baseline comparison)
        strategies_to_test = {k: v for k, v in all_strategies.items() if k != 'BuyAndHoldStrategy'}
        print(f"\nTesting all {len(strategies_to_test)} available strategies: {', '.join(sorted(strategies_to_test.keys()))}")

    if not strategies_to_test:
        print("Error: No strategies found to test.")
        return

    # Initialize markdown report content
    report_lines = []
    def add_to_report(line=""):
        """Add a line to the markdown report."""
        report_lines.append(line)

    def print_and_report(message="", report_only=False):
        """Print to console and add to report."""
        if not report_only:
            print(message)
        add_to_report(message)

    # Validate arguments
    has_input = hasattr(args, 'input') and args.input
    has_asset = hasattr(args, 'asset') and args.asset
    has_bar_size = hasattr(args, 'bar_size') and args.bar_size

    if has_input and (has_asset or has_bar_size):
        print("Error: Cannot use both --input and --asset/--bar-size. Use one or the other.")
        return

    # Helper function to extract bar size from filename
    def extract_bar_size(filename):
        """Extract bar size from filename like 'data-USD-CASH-IDEALPRO-CAD-1 Y-1 hour.csv'"""
        # Remove .csv extension
        name = filename.replace('.csv', '')
        # Split by '-' and get the last part (bar size)
        parts = name.split('-')
        if len(parts) >= 2:
            return parts[-1]
        return None

    # Helper function to get all assets
    def get_all_assets():
        """Get all asset folders in data directory"""
        if not os.path.exists(data_dir):
            return []
        return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and "-" in d]

    # Helper function to get all bar sizes for an asset
    def get_bar_sizes_for_asset(asset_name):
        """Get all available bar sizes for an asset"""
        asset_folder = os.path.join(data_dir, asset_name)
        if not os.path.exists(asset_folder):
            return []

        csv_files = [f for f in os.listdir(asset_folder) if f.endswith('.csv')]
        bar_sizes = {}
        for csv_file in csv_files:
            bar_size = extract_bar_size(csv_file)
            if bar_size:
                bar_sizes[bar_size] = os.path.join(asset_folder, csv_file)
        return bar_sizes

    # Define bar size ordering for multi-timeframe analysis
    BAR_SIZE_ORDER = ["5 mins", "15 mins", "1 hour", "4 hours", "1 day", "1 week"]

    # Determine what to test
    if has_input:
        # Legacy mode: single file
        test_configs = []
        csv_path = args.input
        contract_name = None

        # Try to extract contract name from path
        if os.path.sep in csv_path:
            path_parts = csv_path.split(os.path.sep)
            if "data" in path_parts:
                data_idx = path_parts.index("data")
                if data_idx + 1 < len(path_parts):
                    potential_contract = path_parts[data_idx + 1]
                    if "-" in potential_contract and os.path.isdir(os.path.join("data", potential_contract)):
                        contract_name = potential_contract

        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return

        test_configs.append({
            'asset': contract_name or 'Unknown',
            'csv_path': csv_path,
            'bar_size': extract_bar_size(os.path.basename(csv_path)) or 'Unknown',
            'contract_name': contract_name
        })
    elif has_asset:
        # Single asset mode
        if not has_bar_size:
            print("Error: --bar-size is required when using --asset")
            return

        contract_name = args.asset
        bar_size = args.bar_size

        contract_folder = os.path.join(data_dir, contract_name)
        if not os.path.exists(contract_folder):
            print(f"Error: Asset folder not found: {contract_folder}")
            print(f"  Available assets: {', '.join(get_all_assets())}")
            return

        # Find CSV file matching the pattern
        pattern = os.path.join(contract_folder, f"*-{bar_size}.csv")
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"Error: No CSV file found for asset '{contract_name}' with bar size '{bar_size}'")
            bar_sizes = get_bar_sizes_for_asset(contract_name)
            print(f"  Available bar sizes: {', '.join(bar_sizes.keys())}")
            return

        if len(matching_files) > 1:
            matching_files.sort(key=lambda x: (os.path.getmtime(x), len(x)), reverse=True)
            print(f"Info: Multiple files found for bar size '{bar_size}', using: {os.path.basename(matching_files[0])}")

        test_configs = [{
            'asset': contract_name,
            'csv_path': matching_files[0],
            'bar_size': bar_size,
            'contract_name': contract_name
        }]
    else:
        # Default mode: test all assets with all bar sizes
        print("No specific asset provided. Testing all assets with multi-timeframe strategy...")
        test_configs = []

        all_assets = get_all_assets()
        if not all_assets:
            print(f"Error: No asset folders found in {data_dir}")
            return

        print(f"Found {len(all_assets)} assets: {', '.join(all_assets)}")

        for asset in all_assets:
            bar_sizes = get_bar_sizes_for_asset(asset)
            if bar_sizes:
                # For each asset, we'll test each bar size
                for bar_size, csv_path in bar_sizes.items():
                    test_configs.append({
                        'asset': asset,
                        'csv_path': csv_path,
                        'bar_size': bar_size,
                        'contract_name': asset
                    })

    if not test_configs:
        print("No test configurations found.")
        return

    # Generate report header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_to_report(f"# Forex Multi-Timeframe Strategy Test Results")
    add_to_report(f"")
    add_to_report(f"**Generated:** {timestamp}")
    add_to_report(f"**Total Configurations:** {len(test_configs)}")
    add_to_report(f"**Initial Capital:** ${args.cash:,.2f}")
    add_to_report(f"**Commission Rate:** {args.commission:.4f} ({args.commission*100:.2f}%)")
    add_to_report(f"")
    add_to_report(f"> ⚠️ **Note:** Different bar sizes may have different data ranges (date spans).")
    add_to_report(f"> Buy & Hold returns are calculated for each bar size's specific date range,")
    add_to_report(f"> so they may differ for the same asset across different bar sizes.")
    add_to_report(f"")
    add_to_report(f"---")
    add_to_report(f"")

    print(f"\n{'='*80}")
    print(f"Testing {len(test_configs)} configuration(s)")
    print(f"{'='*80}\n")

    # Store all results across all configurations
    all_results = []

    # Track assets being tested
    assets_tested = set()

    # Process each configuration
    for idx, config in enumerate(test_configs, 1):
        asset = config['asset']
        csv_path = config['csv_path']
        bar_size = config['bar_size']
        contract_name = config['contract_name']

        # Add asset header to report if this is the first time we see it
        if asset not in assets_tested:
            assets_tested.add(asset)
            add_to_report(f"## Asset: {asset}")
            add_to_report(f"")

        print(f"\n[{idx}/{len(test_configs)}] Testing {asset} - {bar_size}")
        add_to_report(f"### Bar Size: {bar_size}")
        add_to_report(f"")
        add_to_report(f"**File:** `{os.path.basename(csv_path)}`")
        add_to_report(f"")

        # Load data
        try:
            df = pd.read_csv(csv_path, index_col=[0])
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Get date range for the data
            first_date = df.index[0]
            last_date = df.index[-1]
            date_range_str = f"{first_date.strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')}"

            add_to_report(f"**Data Points:** {len(df)} bars")
            add_to_report(f"**Date Range:** {date_range_str}")
            add_to_report(f"")
            print(f"  Data range: {date_range_str} ({len(df)} bars)")
        except Exception as e:
            error_msg = f"  ✗ Error loading data: {e}"
            print(error_msg)
            add_to_report(f"❌ **Error:** {e}")
            add_to_report(f"")
            continue

        # Calculate Buy & Hold baseline for comparison
        # Use manual calculation as primary method (more reliable)
        buy_hold_return = 0.0
        try:
            print(f"  Calculating Buy & Hold baseline...")

            # Manual calculation: Buy at first bar, sell at last bar
            # Account for commission on both buy and sell
            if len(df) > 0 and "close" in df.columns:
                first_close = df["close"].iloc[0]
                last_close = df["close"].iloc[-1]

                if pd.notna(first_close) and pd.notna(last_close) and first_close > 0:
                    # Calculate return accounting for commission:
                    # Start with initial_cash
                    # Buy: can buy initial_cash / (first_close * (1 + commission)) units
                    # Sell: receive units * last_close * (1 - commission)
                    # Return = (final_value - initial_cash) / initial_cash * 100

                    initial_cash = args.cash
                    commission = args.commission

                    # Units we can buy (accounting for buy commission)
                    units = initial_cash / (first_close * (1 + commission))

                    # Final value after selling (accounting for sell commission)
                    final_value = units * last_close * (1 - commission)

                    # Return percentage
                    buy_hold_return = ((final_value - initial_cash) / initial_cash) * 100

                    # Also calculate gross return for reference
                    gross_return = ((last_close - first_close) / first_close) * 100

                    print(f"    Buy & Hold Return: {buy_hold_return:.2f}% (${first_close:.2f} → ${last_close:.2f})")
                    print(f"      Gross return: {gross_return:.2f}%, Net (with commission): {buy_hold_return:.2f}%")
                    add_to_report(f"**📊 Buy & Hold Baseline:** {buy_hold_return:.2f}% return (${first_close:.2f} → ${last_close:.2f})")
                    add_to_report(f"")
                else:
                    # Fallback to backtesting framework
                    print(f"    Using backtesting framework for Buy & Hold...")
                    bh_strategy = BuyAndHoldStrategy(
                        initial_cash=args.cash,
                        commission=args.commission,
                    )
                    tester = StrategyTester([bh_strategy])
                    bh_results = tester.test_all(df)

                    if len(bh_results) > 0:
                        buy_hold_return = bh_results.iloc[0]['Return [%]']
                        print(f"    Buy & Hold Return: {buy_hold_return:.2f}% (from backtesting)")
                        add_to_report(f"**📊 Buy & Hold Baseline:** {buy_hold_return:.2f}% return")
                        add_to_report(f"")
                    else:
                        print(f"    ⚠ Could not calculate Buy & Hold baseline")
                        add_to_report(f"⚠️ **Could not calculate Buy & Hold baseline**")
                        add_to_report(f"")
            else:
                print(f"    ⚠ Missing 'close' column for Buy & Hold calculation")
                add_to_report(f"⚠️ **Missing 'close' column for Buy & Hold calculation**")
                add_to_report(f"")
        except Exception as e:
            print(f"    ⚠ Error calculating Buy & Hold: {e}")
            import traceback
            traceback.print_exc()
            add_to_report(f"⚠️ **Error calculating Buy & Hold:** {e}")
            add_to_report(f"")

        # Get available bar sizes for this asset for multi-timeframe analysis
        available_bar_sizes = get_bar_sizes_for_asset(asset)

        # Helper function to test a single strategy
        def test_single_strategy(strategy_name, strategy_class, strategy_kwargs=None):
            """Test a single strategy and return results."""
            if strategy_kwargs is None:
                strategy_kwargs = {
                    'initial_cash': args.cash,
                    'commission': args.commission,
                    'spread': getattr(args, 'spread', 0.0001),
                }

            try:
                # Filter kwargs to only include parameters the strategy accepts
                import inspect
                sig = inspect.signature(strategy_class.__init__)
                accepted_params = set(sig.parameters.keys()) - {'self'}
                # If strategy accepts **kwargs it will handle anything
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if not has_var_keyword:
                    strategy_kwargs = {
                        k: v for k, v in strategy_kwargs.items()
                        if k in accepted_params
                    }
                strategy_instance = strategy_class(**strategy_kwargs)
                tester = StrategyTester([strategy_instance])
                results_df = tester.test_all(df)

                if len(results_df) > 0:
                    result = results_df.iloc[0].to_dict()
                    result['Strategy'] = strategy_name
                    result['Asset'] = asset
                    result['Bar_Size'] = bar_size
                    result['Config'] = f"{asset} ({bar_size}) - {strategy_name}"
                    result['Buy_Hold_Return'] = buy_hold_return
                    result['Excess_Return'] = result['Return [%]'] - buy_hold_return
                    return result, None
                else:
                    return None, "No results"
            except Exception as e:
                return None, str(e)

        # Test all strategies dynamically (except AdaptiveMultiTimeframeStrategy which is handled separately)
        for strategy_name, strategy_class in sorted(strategies_to_test.items()):
            # Skip AdaptiveMultiTimeframeStrategy - it's handled separately below
            if strategy_name == 'AdaptiveMultiTimeframeStrategy':
                continue

            add_to_report(f"#### Strategy: {strategy_name}")
            add_to_report(f"")

            print(f"  Testing {strategy_name}...")
            result, error = test_single_strategy(strategy_name, strategy_class)

            if result:
                all_results.append(result)
                excess_return = result['Excess_Return']
                comparison_symbol = "📈" if excess_return > 0 else "📉"
                print(f"    ✓ {strategy_name}: Return={result['Return [%]']:.2f}%, Sharpe={result['Sharpe Ratio']:.2f}")
                print(f"      {comparison_symbol} vs Buy & Hold: {excess_return:+.2f}%")

                add_to_report(f"**Results:**")
                add_to_report(f"- Return: {result['Return [%]']:.2f}%")
                add_to_report(f"- Excess Return vs Buy & Hold: {comparison_symbol} **{excess_return:+.2f}%**")
                add_to_report(f"- Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
                add_to_report(f"- Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
                add_to_report(f"- Win Rate: {result['Win Rate [%]']:.2f}%")
                add_to_report(f"- Number of Trades: {int(result['# Trades'])}")
            elif error:
                print(f"    ✗ {strategy_name}: {error}")
                add_to_report(f"❌ **Error:** {error}")
            else:
                print(f"    ✗ {strategy_name}: No results")
                add_to_report(f"❌ **No results**")

            add_to_report(f"")

        # Test AdaptiveMultiTimeframeStrategy separately (requires special handling with multiple timeframes)
        if 'AdaptiveMultiTimeframeStrategy' in strategies_to_test:
            add_to_report(f"#### Strategy: AdaptiveMultiTimeframeStrategy")
            add_to_report(f"")

            # Determine higher and lower timeframes based on current bar size
            # Find index of current bar size in the order
            try:
                current_idx = BAR_SIZE_ORDER.index(bar_size)
            except ValueError:
                warning_msg = f"  ⚠ Warning: Bar size '{bar_size}' not in standard order, skipping multi-timeframe"
                print(warning_msg)
                add_to_report(f"⚠️ **Warning:** Bar size not in standard order, skipping multi-timeframe")
                add_to_report(f"")
                # Continue to next config, not next strategy
                continue

            # Generate all valid multi-timeframe combinations
            # Lower timeframe = current bar size
            # Higher timeframes = all bar sizes larger than current
            higher_timeframe_options = []
            for i in range(current_idx + 1, len(BAR_SIZE_ORDER)):
                if BAR_SIZE_ORDER[i] in available_bar_sizes:
                    higher_timeframe_options.append(BAR_SIZE_ORDER[i])

            if not higher_timeframe_options:
                warning_msg = f"  ⚠ No higher timeframes available for {bar_size}, skipping multi-timeframe"
                print(warning_msg)
                add_to_report(f"⚠️ **No higher timeframes available for multi-timeframe testing**")
                add_to_report(f"")
                continue

            add_to_report(f"**Testing against higher timeframes:** {', '.join(higher_timeframe_options)}")
            add_to_report(f"")
            add_to_report(f"| Higher TF | Return (%) | vs B&H | Sharpe | Max DD (%) | Win Rate (%) | # Trades |")
            add_to_report(f"|-----------|------------|--------|--------|------------|--------------|----------|")

            # Test all combinations of higher timeframes
            # For each higher timeframe, create a strategy
            for higher_tf in higher_timeframe_options:
                strategy_name = f"MultiTF_{asset}_{bar_size}_vs_{higher_tf}"

                try:
                    strategy = AdaptiveMultiTimeframeStrategy(
                        initial_cash=args.cash,
                        commission=args.commission,
                        higher_timeframes=[higher_tf],
                        lower_timeframes=[bar_size],
                        data_dir=data_dir,
                        contract_name=contract_name,
                    )

                    # Test the strategy
                    tester = StrategyTester([strategy])
                    results_df = tester.test_all(df)

                    if len(results_df) > 0:
                        # Add metadata to results
                        result = results_df.iloc[0].to_dict()
                        result['Strategy'] = 'AdaptiveMultiTimeframeStrategy'
                        result['Asset'] = asset
                        result['Lower_TF'] = bar_size
                        result['Higher_TF'] = higher_tf
                        result['Config'] = f"{asset} ({bar_size} vs {higher_tf}) - MultiTF"
                        result['Buy_Hold_Return'] = buy_hold_return
                        result['Excess_Return'] = result['Return [%]'] - buy_hold_return
                        all_results.append(result)

                        excess_return = result['Excess_Return']
                        comparison_symbol = "📈" if excess_return > 0 else "📉"
                        print(f"    ✓ {bar_size} vs {higher_tf}: Return={result['Return [%]']:.2f}%, Sharpe={result['Sharpe Ratio']:.2f}")
                        print(f"      {comparison_symbol} vs Buy & Hold: {excess_return:+.2f}%")

                        # Add to markdown table
                        add_to_report(
                            f"| {higher_tf} | "
                            f"{result['Return [%]']:.2f} | "
                            f"{comparison_symbol} {excess_return:+.2f} | "
                            f"{result['Sharpe Ratio']:.2f} | "
                            f"{result['Max. Drawdown [%]']:.2f} | "
                            f"{result['Win Rate [%]']:.2f} | "
                            f"{int(result['# Trades'])} |"
                        )
                    else:
                        print(f"    ✗ {bar_size} vs {higher_tf}: No results")
                        add_to_report(f"| {higher_tf} | ❌ No results | - | - | - | - | - |")

                except Exception as e:
                    error_msg = f"    ✗ {bar_size} vs {higher_tf}: Error - {e}"
                    print(error_msg)
                    add_to_report(f"| {higher_tf} | ❌ Error | - | - | - | - | - |")
                    continue

            add_to_report(f"")

    # Display consolidated results
    if not all_results:
        msg = "\n⚠ No successful test results."
        print(msg)
        add_to_report(f"")
        add_to_report(f"## ⚠️ No Successful Results")
        add_to_report(f"")
        add_to_report(f"No configurations produced valid test results.")

        # Save report even if no results
        report_filename = f"strategy_test_results/strategy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to: {report_filename}")
        return

    # Convert to DataFrame for better display
    results_df = pd.DataFrame(all_results)

    # Sort by Return descending
    results_df = results_df.sort_values('Return [%]', ascending=False)

    print(f"\n{'='*100}")
    print("STRATEGY TEST RESULTS (All Configurations)")
    print(f"{'='*100}")

    # Add summary section to report
    add_to_report(f"")
    add_to_report(f"---")
    add_to_report(f"")
    add_to_report(f"## 📊 Summary of All Results")
    add_to_report(f"")
    add_to_report(f"**Total Successful Tests:** {len(results_df)}")
    add_to_report(f"")

    # Display key columns in console
    display_cols = ['Strategy', 'Config', 'Return [%]', 'Excess_Return', 'Sharpe Ratio', 'Max. Drawdown [%]', 'Win Rate [%]', '# Trades']
    available_cols = [col for col in display_cols if col in results_df.columns]

    # Rename column for better display
    display_df = results_df[available_cols].copy()
    if 'Excess_Return' in display_df.columns:
        display_df = display_df.rename(columns={'Excess_Return': 'vs B&H (%)'})

    print(display_df.to_string(index=False))
    print(f"{'='*100}")

    # Display summary statistics
    if 'Excess_Return' in results_df.columns:
        outperforming = (results_df['Excess_Return'] > 0).sum()
        underperforming = (results_df['Excess_Return'] <= 0).sum()
        avg_excess = results_df['Excess_Return'].mean()

        print(f"\n📊 Performance vs Buy & Hold:")
        print(f"   Outperforming: {outperforming} / {len(results_df)} ({100*outperforming/len(results_df):.1f}%)")
        print(f"   Average Excess Return: {avg_excess:+.2f}%")
        print(f"   Best: {results_df['Excess_Return'].max():+.2f}%  |  Worst: {results_df['Excess_Return'].min():+.2f}%")

    # Add full results table to report
    add_to_report(f"### All Configurations (Sorted by Return)")
    add_to_report(f"")
    add_to_report(f"| Rank | Strategy | Configuration | Return (%) | vs B&H | Sharpe | Max DD (%) | Win Rate (%) | # Trades |")
    add_to_report(f"|------|----------|---------------|------------|--------|--------|------------|--------------|----------|")

    for idx, row in results_df.iterrows():
        rank = results_df.index.get_loc(idx) + 1
        # Determine strategy short name based on actual strategy type
        strategy_name = row.get('Strategy', '')
        if strategy_name == 'AdaptiveMultiIndicatorStrategy':
            strategy_short = 'AMIS'
        elif strategy_name == 'AdaptiveMultiTimeframeStrategy':
            strategy_short = 'MultiTF'
        else:
            # Other strategies (MomentumStrategy, RSIStrategy, etc.)
            strategy_short = strategy_name.replace('Strategy', '')[:8]
        excess_return = row.get('Excess_Return', 0.0)
        comparison_symbol = "📈" if excess_return > 0 else "📉"
        add_to_report(
            f"| {rank} | "
            f"{strategy_short} | "
            f"{row['Config']} | "
            f"{row['Return [%]']:.2f} | "
            f"{comparison_symbol} {excess_return:+.2f} | "
            f"{row['Sharpe Ratio']:.2f} | "
            f"{row['Max. Drawdown [%]']:.2f} | "
            f"{row['Win Rate [%]']:.2f} | "
            f"{int(row['# Trades'])} |"
        )

    add_to_report(f"")

    # Add summary statistics about performance vs Buy & Hold
    if 'Excess_Return' in results_df.columns:
        outperforming = (results_df['Excess_Return'] > 0).sum()
        underperforming = (results_df['Excess_Return'] <= 0).sum()
        avg_excess = results_df['Excess_Return'].mean()
        max_excess = results_df['Excess_Return'].max()
        min_excess = results_df['Excess_Return'].min()

        add_to_report(f"### 📊 Performance vs Buy & Hold")
        add_to_report(f"")
        add_to_report(f"- **Outperforming Buy & Hold:** {outperforming} / {len(results_df)} ({100*outperforming/len(results_df):.1f}%)")
        add_to_report(f"- **Average Excess Return:** {avg_excess:+.2f}%")
        add_to_report(f"- **Best Excess Return:** {max_excess:+.2f}%")
        add_to_report(f"- **Worst Excess Return:** {min_excess:+.2f}%")
        add_to_report(f"")

    # Find and display top 5 configurations
    top_n = min(5, len(results_df))
    print(f"\n🏆 Top {top_n} Configurations:")

    add_to_report(f"## 🏆 Top {top_n} Configurations")
    add_to_report(f"")

    for i in range(top_n):
        row = results_df.iloc[i]
        excess_return = row.get('Excess_Return', 0.0)
        comparison_symbol = "📈" if excess_return > 0 else "📉"

        print(f"\n{i+1}. {row['Config']}")
        print(f"   Strategy: {row.get('Strategy', 'N/A')}")
        print(f"   Return: {row['Return [%]']:.2f}%")
        print(f"   vs Buy & Hold: {comparison_symbol} {excess_return:+.2f}%")
        print(f"   Sharpe Ratio: {row['Sharpe Ratio']:.2f}")
        print(f"   Max Drawdown: {row['Max. Drawdown [%]']:.2f}%")
        print(f"   Win Rate: {row['Win Rate [%]']:.2f}%")
        print(f"   # Trades: {int(row['# Trades'])}")

        add_to_report(f"### {i+1}. {row['Config']}")
        add_to_report(f"")
        add_to_report(f"- **Strategy:** {row.get('Strategy', 'N/A')}")
        add_to_report(f"- **Return:** {row['Return [%]']:.2f}%")
        add_to_report(f"- **Buy & Hold Return:** {row.get('Buy_Hold_Return', 0.0):.2f}%")
        add_to_report(f"- **Excess Return vs Buy & Hold:** {comparison_symbol} **{excess_return:+.2f}%**")
        add_to_report(f"- **Sharpe Ratio:** {row['Sharpe Ratio']:.2f}")
        add_to_report(f"- **Max Drawdown:** {row['Max. Drawdown [%]']:.2f}%")
        add_to_report(f"- **Win Rate:** {row['Win Rate [%]']:.2f}%")
        add_to_report(f"- **Number of Trades:** {int(row['# Trades'])}")
        add_to_report(f"- **Asset:** {row['Asset']}")

        # Add timeframe info based on strategy type
        # Only AdaptiveMultiTimeframeStrategy has Lower_TF/Higher_TF fields
        # All other strategies (including AdaptiveMultiIndicatorStrategy) have Bar_Size
        if row.get('Strategy') == 'AdaptiveMultiTimeframeStrategy':
            add_to_report(f"- **Lower Timeframe:** {row.get('Lower_TF', 'N/A')}")
            add_to_report(f"- **Higher Timeframe:** {row.get('Higher_TF', 'N/A')}")
        else:
            # All other strategies use Bar_Size
            add_to_report(f"- **Bar Size:** {row.get('Bar_Size', 'N/A')}")

        add_to_report(f"")

    # Plot best strategy if requested
    if args.plot and len(all_results) > 0:
        best_result = all_results[0]
        best_asset = best_result['Asset']
        best_strategy_type = best_result.get('Strategy', 'Unknown')

        print(f"\nGenerating plot for best configuration: {best_result['Config']}...")

        # Determine the bar size based on strategy type
        # Only AdaptiveMultiTimeframeStrategy uses Lower_TF, all others use Bar_Size
        if best_strategy_type == 'AdaptiveMultiTimeframeStrategy':
            best_bar_size = best_result['Lower_TF']
        else:
            # All other strategies (including AdaptiveMultiIndicatorStrategy) use Bar_Size
            best_bar_size = best_result['Bar_Size']

        # Load the data for the best configuration
        bar_sizes = get_bar_sizes_for_asset(best_asset)
        best_csv_path = bar_sizes.get(best_bar_size)

        if best_csv_path and os.path.exists(best_csv_path):
            try:
                df_best = pd.read_csv(best_csv_path, index_col=[0])
                if not isinstance(df_best.index, pd.DatetimeIndex):
                    df_best.index = pd.to_datetime(df_best.index)

                # Recreate the best strategy based on type
                if best_strategy_type == 'AdaptiveMultiIndicatorStrategy':
                    best_strategy = AdaptiveMultiIndicatorStrategy(
                        initial_cash=args.cash,
                        commission=args.commission,
                    )
                else:  # MultiTimeframeStrategy
                    best_higher_tf = best_result['Higher_TF']
                    best_strategy = AdaptiveMultiTimeframeStrategy(
                        initial_cash=args.cash,
                        commission=args.commission,
                        higher_timeframes=[best_higher_tf],
                        lower_timeframes=[best_bar_size],
                        data_dir=data_dir,
                        contract_name=best_asset,
                    )

                # Generate signals
                df_with_signals = best_strategy.generate_signals(df_best.copy())

                # Create backtesting strategy wrapper
                from forex_strategies.backtesting_strategy import ForexBacktestingStrategy
                backtest_strategy = ForexBacktestingStrategy(best_strategy)
                backtest_strategy.run(df_with_signals)

                # Plot
                plotter = Plot(
                    candlestickData=df_with_signals,
                    contract=best_asset,
                    strategy=backtest_strategy,
                )
                plotter.show()

            except Exception as e:
                print(f"Error generating plot: {e}")
        else:
            print(f"Could not find data file for best configuration")

    # Save markdown report
    os.makedirs('strategy_test_results', exist_ok=True)
    report_filename = f"strategy_test_results/strategy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_filename, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*100}")
    print(f"📄 Full report saved to: {report_filename}")
    print(f"{'='*100}")


def _get_csv_files_for_training(input_path, asset_name):
    """Helper to get CSV files for training."""
    import glob

    csv_files = []

    if input_path:
        if os.path.exists(input_path):
            csv_files = [input_path]
        else:
            print(f"✗ Error: File not found: {input_path}")
            return None
    elif asset_name:
        asset_dir = os.path.join("data", asset_name)
        if not os.path.exists(asset_dir):
            print(f"✗ Error: Asset directory not found: {asset_dir}")
            return None
        pattern = os.path.join(asset_dir, "*.csv")
        csv_files = glob.glob(pattern)
        if not csv_files:
            print(f"✗ Error: No CSV files found in {asset_dir}")
            return None
        print(f"Found {len(csv_files)} CSV file(s) for asset {asset_name}")
    else:
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"✗ Error: Data directory not found: {data_dir}")
            return None
        pattern = os.path.join(data_dir, "*", "*.csv")
        csv_files = glob.glob(pattern)
        if not csv_files:
            print(f"✗ Error: No CSV files found in {data_dir}")
            return None
        print(f"Found {len(csv_files)} CSV file(s) across all assets")

    return csv_files


def _load_and_combine_data(csv_files):
    """Helper to load and combine CSV files."""
    import pandas as pd

    print(f"\nLoading data from {len(csv_files)} file(s)...")
    all_dataframes = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=[0])
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if "close" not in df.columns:
                print(f"  ⚠ Skipping {os.path.basename(csv_file)}: missing 'close' column")
                continue

            all_dataframes.append(df)
            print(f"  ✓ Loaded {os.path.basename(csv_file)}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ Error loading {os.path.basename(csv_file)}: {e}")
            continue

    if not all_dataframes:
        print("\n✗ No valid data files found.")
        return None

    combined_df = pd.concat(all_dataframes, ignore_index=False)
    combined_df = combined_df.sort_index()

    print(f"Total combined data: {len(combined_df)} bars")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    return combined_df


def cmd_train_price_direction_predictor(args: argparse.Namespace):
    """Train ML model to predict price direction (up/down)."""
    from machine_learning.price_direction_predictor import PriceDirectionPredictor

    print("=" * 80)
    print("Training Price Direction Predictor")
    print("=" * 80)

    csv_files = _get_csv_files_for_training(args.input, args.asset)
    if not csv_files:
        return

    combined_df = _load_and_combine_data(csv_files)
    if combined_df is None:
        return

    predictor = PriceDirectionPredictor(
        lookback_bars=args.lookback_bars,
        prediction_horizon=args.prediction_horizon,
        model_type=args.model_type,
    )

    print(f"\nTraining model...")
    try:
        metrics = predictor.train(
            combined_df,
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
            model_name=args.model_name,
            retrain=args.retrain,
        )

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Model saved to: {os.path.join(predictor.model_dir, args.model_name)}.pkl")
        print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (Up): {metrics['precision_up']:.4f}")
        print(f"Precision (Down): {metrics['precision_down']:.4f}")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


def cmd_train_volatility_predictor(args: argparse.Namespace):
    """Train ML model to predict volatility."""
    from machine_learning.volatility_predictor import VolatilityPredictor

    print("=" * 80)
    print("Training Volatility Predictor")
    print("=" * 80)

    csv_files = _get_csv_files_for_training(args.input, args.asset)
    if not csv_files:
        return

    combined_df = _load_and_combine_data(csv_files)
    if combined_df is None:
        return

    predictor = VolatilityPredictor(
        lookback_bars=args.lookback_bars,
        prediction_horizon=args.prediction_horizon,
    )

    print(f"\nTraining model...")
    try:
        metrics = predictor.train(
            combined_df,
            test_size=0.2,
            model_name=args.model_name,
            retrain=args.retrain,
        )

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


def cmd_train_trend_predictor(args: argparse.Namespace):
    """Train ML model to predict trend continuation vs reversal."""
    from machine_learning.trend_predictor import TrendPredictor

    print("=" * 80)
    print("Training Trend Predictor")
    print("=" * 80)

    csv_files = _get_csv_files_for_training(args.input, args.asset)
    if not csv_files:
        return

    combined_df = _load_and_combine_data(csv_files)
    if combined_df is None:
        return

    predictor = TrendPredictor(
        lookback_bars=args.lookback_bars,
        prediction_horizon=args.prediction_horizon,
    )

    print(f"\nTraining model...")
    try:
        metrics = predictor.train(
            combined_df,
            test_size=0.2,
            model_name=args.model_name,
            retrain=args.retrain,
        )

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


def _create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Trading Bot CLI", add_help=False)
    sub = parser.add_subparsers(dest='command', required=False)

    p0 = sub.add_parser('fetch-process-plot', help='Fetch historical data, compute indicators, run models/support-resistance, plot')
    p0.add_argument('--data-source', default='local', choices=['local', 'ibkr', 'ctrader'],
                   help='Data source: local (use existing CSV files), ibkr (Interactive Brokers), ctrader (cTrader API). Default: local')
    p0.add_argument('--interval', default='6 M', help='Time interval for historical data (e.g., "6 M"). If "6 M" is used, will auto-select optimal interval based on bar size.')
    p0.add_argument('--bar-size', default='1 hour', help='Bar size (e.g., "1 hour", "15 mins")')
    p0.add_argument('--refresh', action='store_true', help='Ignore cached CSVs and re-fetch from data source')
    # Strategy / model selection flags (no forex strategies here)
    p0.add_argument('--use-support-resistance-v1', action='store_true', help='Run Support/Resistance V1 (default: enabled if no strategies/models specified)')
    p0.add_argument('--use-support-resistance', action='store_true', help='Run Support/Resistance (alternative implementation)')
    p0.add_argument('--use-lstm', action='store_true', help='Run LSTM model trainer')
    p0.add_argument('--use-svm-trainer', action='store_true', help='Run SVM model trainer')
    p0.add_argument('--no-local-extrema', action='store_false', dest='with_local_extrema', default=True,
                   help='Disable plotting LOCAL_MAX/LOCAL_MIN markers (default: enabled)')

    p0.set_defaults(func=cmd_fetch_process_plot)


    p_extrema = sub.add_parser('train-extrema-predictor', help='Train ML model to predict local extrema (buy/sell points)')
    p_extrema.add_argument('--input', help='Path to single CSV file (optional, if not provided uses all available data)')
    p_extrema.add_argument('--asset', help='Asset name (folder in data/, e.g., "USD-CAD"). If provided, uses all CSV files for that asset.')
    p_extrema.add_argument('--lookback-bars', type=int, default=10, help='Number of previous bars to use as features (default: 10)')
    p_extrema.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for testing (default: 0.2)')
    p_extrema.add_argument('--validation-size', type=float, default=0.1, help='Fraction of training data for validation (default: 0.1)')
    p_extrema.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    p_extrema.add_argument('--model-name', default='extrema_predictor', help='Name for the saved model (default: extrema_predictor)')
    p_extrema.add_argument('--model-type', choices=['xgboost', 'lightgbm', 'ensemble'], default='lightgbm', help='Model type to use (default: lightgbm). ensemble combines both XGBoost and LightGBM.')
    p_extrema.add_argument('--retrain', action='store_true', help='Force retraining even if model exists')
    p_extrema.add_argument('--no-indicators', action='store_true', help='Disable technical indicators as features (use only OHLCV)')
    p_extrema.add_argument('--no-feature-selection', action='store_true', help='Disable automatic feature selection (use all features)')
    p_extrema.set_defaults(func=cmd_train_extrema_predictor)

    p_price_dir = sub.add_parser('train-price-direction-predictor', help='Train ML model to predict price direction (up/down) - often more accurate than extrema prediction')
    p_price_dir.add_argument('--input', help='Path to single CSV file (optional)')
    p_price_dir.add_argument('--asset', help='Asset name (folder in data/)')
    p_price_dir.add_argument('--lookback-bars', type=int, default=10, help='Number of previous bars to use as features')
    p_price_dir.add_argument('--prediction-horizon', type=int, default=1, help='Number of bars ahead to predict (default: 1)')
    p_price_dir.add_argument('--model-type', choices=['xgboost', 'lightgbm'], default='lightgbm', help='Model type to use')
    p_price_dir.add_argument('--model-name', default='price_direction_predictor', help='Name for the saved model')
    p_price_dir.add_argument('--retrain', action='store_true', help='Force retraining')
    p_price_dir.set_defaults(func=cmd_train_price_direction_predictor)

    p_volatility = sub.add_parser('train-volatility-predictor', help='Train ML model to predict volatility (high/low)')
    p_volatility.add_argument('--input', help='Path to single CSV file (optional)')
    p_volatility.add_argument('--asset', help='Asset name (folder in data/)')
    p_volatility.add_argument('--lookback-bars', type=int, default=10, help='Number of previous bars to use')
    p_volatility.add_argument('--prediction-horizon', type=int, default=5, help='Number of bars ahead to predict')
    p_volatility.add_argument('--model-name', default='volatility_predictor', help='Name for the saved model')
    p_volatility.add_argument('--retrain', action='store_true', help='Force retraining')
    p_volatility.set_defaults(func=cmd_train_volatility_predictor)

    p_trend = sub.add_parser('train-trend-predictor', help='Train ML model to predict trend continuation vs reversal')
    p_trend.add_argument('--input', help='Path to single CSV file (optional)')
    p_trend.add_argument('--asset', help='Asset name (folder in data/)')
    p_trend.add_argument('--lookback-bars', type=int, default=10, help='Number of previous bars to use')
    p_trend.add_argument('--prediction-horizon', type=int, default=5, help='Number of bars ahead to predict')
    p_trend.add_argument('--model-name', default='trend_predictor', help='Name for the saved model')
    p_trend.add_argument('--retrain', action='store_true', help='Force retraining')
    p_trend.set_defaults(func=cmd_train_trend_predictor)

    p_download = sub.add_parser('download-and-process', help='Phase 1 & 2: Download historical data and process with indicators')
    p_download.add_argument('--data-source', default='ibkr', choices=['ibkr', 'ctrader'],
                           help='Data source for downloading: ibkr (Interactive Brokers), ctrader (cTrader API). Default: ibkr')
    p_download.add_argument('--contracts-file', default='contracts.json', help='Path to contracts.json file')
    p_download.add_argument('--interval', default=None, help='Time interval for historical data (e.g., "6 M"). If not specified, uses broker-optimized intervals per bar size')
    p_download.add_argument('--bar-sizes', default='1 week,1 day,4 hours,1 hour,15 mins,5 mins', help='Comma-separated bar sizes (default: "1 week,1 day,4 hours,1 hour,15 mins,5 mins"). Note: 1 min data may be limited by broker API')
    p_download.add_argument('--include-1min', action='store_true', help='Include 1-minute bar data (limited by broker API, slower to download)')
    p_download.add_argument('--force-refresh', action='store_true', help='Force re-download even if data exists')
    p_download.set_defaults(func=cmd_download_and_process_data)

    p3 = sub.add_parser('test-forex-strategies', help='Test forex strategies. By default, tests all available strategies on all assets with all bar size combinations.')
    p3.add_argument('--input', help='Path to CSV file with historical data (legacy mode)')
    p3.add_argument('--asset', help='Asset name (folder name in data/, e.g., "USD-CAD"). If not provided, tests all assets.')
    p3.add_argument('--bar-size', help='Bar size (e.g., "1 hour", "15 mins", "1 day"). If not provided with --asset, tests all bar sizes.')
    p3.add_argument('--strategies', help='Comma-separated list of strategy names to test (e.g., "AdaptiveMultiIndicatorStrategy,MomentumStrategy"). If not provided, tests all available strategies.')
    p3.add_argument('--cash', type=float, default=10000, help='Initial capital (default: 10000)')
    p3.add_argument('--commission', type=float, default=0.0002, help='Commission rate (default: 0.0002 = 0.02%%)')
    p3.add_argument('--spread', type=float, default=0.0001, help='Spread in price units (default: 0.0001 = 1 pip for forex). Set 0 to disable.')
    p3.add_argument('--plot', action='store_true', help='Plot the best performing configuration')
    p3.set_defaults(func=cmd_test_forex_strategies)

    return parser


def _execute_command(cmd_line, parser):
    """Parse and execute a command line string."""
    if not cmd_line.strip():
        return

    # Handle special commands
    cmd_parts = cmd_line.strip().split()
    if not cmd_parts:
        return

    cmd = cmd_parts[0].lower()

    if cmd in ['exit', 'quit', 'q']:
        print("Goodbye!")
        _cleanup_all_clients()
        # Use os._exit to force exit even if threads are blocking
        os._exit(0)
    elif cmd in ['help', 'h', '?']:
        parser.print_help()
        print("\n" + "=" * 80)
        print("Available Commands:")
        print("=" * 80)
        print("  fetch-process-plot      - Fetch data, compute indicators, run strategies, plot")
        print("  download-and-process   - Download historical data and process with indicators")
        print("  train-extrema-predictor - Train ML model to predict local extrema (buy/sell points)")
        print("  test-forex-strategies  - Test multi-timeframe strategies on all assets/bar sizes")
        print("                           (By default tests all combinations. Use --asset and --bar-size to limit)")
        print("\nSpecial Commands:")
        print("  help, h, ?             - Show this help message")
        print("  exit, quit, q          - Exit the interactive shell")
        print("  clear                  - Clear the screen")
        print("\nUse 'COMMAND --help' for detailed help on any command.")
        print("=" * 80)
        return
    elif cmd == 'clear':
        os.system('clear' if os.name != 'nt' else 'cls')
        return

    # Parse the command using argparse
    try:
        args = parser.parse_args(shlex.split(cmd_line))
        if args.command is None:
            print("Error: No command specified. Type 'help' for available commands.")
            return

        # Execute the command
        if hasattr(args, 'func'):
            args.func(args)
        else:
            print(f"Error: Command '{args.command}' has no handler.")
    except SystemExit as e:
        # argparse calls sys.exit() on help or error
        # Exit code 0 means help was shown (success), non-zero means error
        # We catch both to prevent exiting the interactive shell
        pass
    except Exception as e:
        print(f"Error executing command: {e}")
        import traceback
        traceback.print_exc()


def _exit_handler(signum=None, frame=None):
    """Handle exit signals and cleanup."""
    print("\n\nShutting down...")
    _cleanup_all_clients()
    # Force exit without waiting for threads
    os._exit(0)


def interactive_shell():
    """Run the interactive shell."""
    # Register cleanup handlers
    atexit.register(_cleanup_all_clients)
    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)

    parser = _create_parser()

    print("=" * 80)
    print("Trading Bot - Interactive Shell")
    print("=" * 80)
    print("Type 'help' for available commands or 'exit' to quit.")
    print("=" * 80)
    print()

    while True:
        try:
            # Get user input
            cmd_line = input("trading-bot> ").strip()

            # Execute command
            _execute_command(cmd_line, parser)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue with commands.")
        except EOFError:
            print("\n\nGoodbye!")
            _exit_handler()


def main():
    """Main entry point - supports both interactive and single-command modes."""
    # Check if running in non-interactive mode (command provided as arguments)
    if len(sys.argv) > 1:
        # Single command mode (original behavior)
        parser = _create_parser()
        # Add help argument for non-interactive mode
        parser.add_argument('--help', action='help', default=argparse.SUPPRESS,
                          help='Show this help message and exit')
        try:
            args = parser.parse_args()
            if args.command is None:
                parser.print_help()
                sys.exit(1)
            if hasattr(args, 'func'):
                args.func(args)
            else:
                print(f"Error: Command '{args.command}' has no handler.")
                sys.exit(1)
        except SystemExit as e:
            # Let argparse handle help/error messages and exit codes
            sys.exit(e.code if e.code else 0)
    else:
        # Interactive mode
        interactive_shell()


if __name__ == '__main__':
    main()



