# Trading Bot

A comprehensive Python framework for quantitative research, backtesting, and live trading. Supports multiple brokers (IBKR, OANDA), real-time data collection, technical analysis, machine learning predictions, and automated strategy execution.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities

- **Data Management**: Download and cache historical OHLCV data from IBKR
- **Technical Indicators**: RSI, MACD, EMA/SMA, ATR, ADX, Bollinger Bands, Local Extrema
- **Pattern Detection**: Chart patterns (Head & Shoulders, Triangles, etc.) and candlestick patterns
- **Strategy Backtesting**: Comprehensive backtesting framework with multiple strategies
- **Machine Learning**: ML predictors for price direction, volatility, trend, and extrema
- **Live Trading**: Real-time trading system with multi-timeframe support
- **Visualization**: Interactive charts with Matplotlib and Plotly

### Trading Strategies

- **Momentum Strategies**: Trend following with MACD, EMA crossovers
- **Mean Reversion**: RSI, Bollinger Bands mean reversion
- **Breakout Strategies**: Support/Resistance and ATR-based breakouts
- **Multi-Timeframe**: Higher timeframe trend confirmation
- **Pattern-Based**: Chart pattern and triangle detection strategies
- **Adaptive**: Multi-indicator adaptive strategies

See [Strategy Documentation](docs/strategies.md) for details.

### Brokers

- **IBKR**: Market data and order execution (paper and live)
- **OANDA**: Order execution (paper and live)

See [Broker Configuration](docs/brokers.md) for setup instructions.

## Quick Start

### Prerequisites

- Python 3.10+
- IBKR TWS/Gateway (for IBKR data/trading)
- MongoDB (for live trading system)
- Node.js 18+ (for frontend, optional)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd trading-bot
```

2. **Create virtual environment**:
```bash
python3 -m venv .venv-cli
source .venv-cli/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install system dependencies** (macOS):
```bash
brew install ta-lib libomp
```

4. **Install Python dependencies**:
```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e ./IBJts/source/pythonclient  # IBKR API
```

5. **Configure environment**:
Create a `.env` file in the project root:
```env
# Optional: For OANDA trading
OANDA_ACCESS_TOKEN=your_token
OANDA_ACCOUNT_ID=your_account_id

# Optional: For live trading system
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=trading_bot
```

6. **Start IBKR TWS/Gateway**:
- Enable API: Configure → Global Configuration → API → Settings
- Socket port: 7497 (paper) or 7496 (live)
- Add 127.0.0.1 to Trusted IPs

### Basic Usage

**Interactive Mode** (recommended):
```bash
python cli.py
```

Then in the shell:
```
trading-bot> help
trading-bot> download-and-process
trading-bot> test-forex-strategies --asset USD-CAD
trading-bot> exit
```

**Single Command Mode**:
```bash
# Download and process data
python cli.py download-and-process

# Test strategies
python cli.py test-forex-strategies --asset USD-CAD --bar-size "1 hour"

# Train ML models
python cli.py train-extrema-predictor --asset USD-CAD
```

## Architecture

The system consists of two main components:

### 1. Backtesting & Research System

- **CLI Interface**: Interactive shell for data management, strategy testing, and ML training
- **Data Pipeline**: IBKR → CSV cache → Technical Indicators → Strategies → Backtesting
- **Strategy Framework**: Extensible strategy base classes with backtesting integration
- **ML Pipeline**: Feature engineering, model training, and prediction

See [Architecture Documentation](docs/architecture.md) for details.

### 2. Live Trading System

- **Trading Engine**: Orchestrates real-time trading operations
- **Data Manager**: Real-time data collection and bar aggregation
- **Order Manager**: Order placement, position tracking, risk management
- **REST API**: FastAPI-based API for operation management
- **React Frontend**: Web interface for monitoring and control

See [Live Trading Documentation](docs/live-trading/README.md) for details.

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### Getting Started
- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Quick Start Guide](docs/quick-start.md) - Get up and running quickly
- [Configuration](docs/configuration.md) - Environment variables and settings

### Core Concepts
- [Architecture Overview](docs/architecture.md) - System architecture and design
- [Data Management](docs/data-management.md) - Data download, processing, and storage
- [Technical Indicators](docs/technical-indicators.md) - Available indicators and usage
- [Strategies](docs/strategies.md) - Strategy development and backtesting

### Live Trading
- [Live Trading System](docs/live-trading/README.md) - Overview and setup
- [Live Trading Architecture](docs/live-trading/architecture.md) - Detailed architecture
- [Real-Time Data Flow](docs/live-trading/data-flow.md) - How real-time data works
- [Broker Integration](docs/live-trading/brokers.md) - IBKR and OANDA setup
- [Frontend Guide](docs/live-trading/frontend.md) - React frontend usage

### Advanced Topics
- [Machine Learning](docs/machine-learning.md) - ML model training and usage
- [Pattern Detection](docs/patterns-triangles.md) - Chart pattern and triangle detection
- [Alternative Data Sources](docs/live-trading/alternative-data-sources.md) - Free data alternatives
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### Reference
- [CLI Reference](docs/cli-reference.md) - Complete command reference
- [API Reference](docs/live-trading/api-reference.md) - REST API documentation
- [Strategy Examples](docs/strategy-examples.md) - Example strategies

## Installation

See [Installation Guide](docs/installation.md) for detailed instructions.

### System Dependencies

**macOS**:
```bash
brew install ta-lib libomp
```

**Linux**:
```bash
sudo apt-get install ta-lib libomp-dev  # Ubuntu/Debian
```

**Windows**:
Download TA-Lib from https://ta-lib.org/install/

## Usage

### Data Management

Download and process historical data:
```bash
python cli.py download-and-process
python cli.py download-and-process --include-1min --force-refresh
```

### Strategy Testing

Test all strategies on all assets:
```bash
python cli.py test-forex-strategies
```

Test specific strategies on specific assets:
```bash
python cli.py test-forex-strategies --asset USD-CAD --bar-size "1 hour" --strategies "MomentumStrategy,RSIStrategy"
```

### Machine Learning

Train extrema predictor:
```bash
python cli.py train-extrema-predictor --asset USD-CAD --lookback-bars 20
```

Train other predictors:
```bash
python cli.py train-price-direction-predictor --asset USD-CAD
python cli.py train-volatility-predictor --asset USD-CAD
python cli.py train-trend-predictor --asset USD-CAD
```

### Walk-Forward Analysis

Evaluate which strategies and parameter configs consistently beat Buy & Hold
across rolling out-of-sample windows (avoids overfitting to a single backtest):

```bash
# Place an OHLCV CSV in data/XAU-USD/ first, then:
python walk_forward_analyzer.py --symbol XAUUSD --data data/XAU-USD/

# Custom window sizes
python walk_forward_analyzer.py --symbol XAUUSD --data data/XAU-USD/ \
  --warmup 200 --test-bars 500 --step 500

# Explicit file
python walk_forward_analyzer.py --symbol XAUUSD --data-file data/XAU-USD/xauusd_h1.csv
```

Results are written to `strategy_test_results/XAUUSD_walkforward_YYYYMMDD_HHMM.md`.
The report ranks ~50 strategy/param combos by consistency score (% windows beating
B&H × mean Sharpe), plus per-window detail for the top 3 configs.

**Data format**: CSV with columns `date, open, high, low, close, volume` (technical
indicators are auto-computed if not already present).

### Live Trading

**Start in interactive mode** (foreground):
```bash
python -m live_trading.main
```

**Start as a background daemon** (recommended for production):
```bash
python -m live_trading.cli start    # Start daemon
python -m live_trading.cli status   # Check status
python -m live_trading.cli logs -f  # Follow logs (like tail -f)
python -m live_trading.cli stop     # Stop daemon
```

**Access the system:**
- API: `http://localhost:8000`
- Frontend: `http://localhost:3000` (if running separately)
- Logs Viewer: `http://localhost:3000/logs` (frontend)

See [Live Trading Documentation](docs/live-trading/README.md) for details.

## Project Structure

```
trading-bot/
├── cli.py                      # Main CLI entry point
├── config.py                   # Global configuration
├── contracts.json              # Instrument definitions
├── requirements.txt            # Python dependencies
├── docs/                       # Documentation
├── data/                       # Cached historical data (CSV)
├── forex_strategies/          # Trading strategies
├── technical_indicators/      # Technical indicator implementations
├── machine_learning/         # ML models and predictors
├── patterns/                  # Chart pattern detection
├── triangles/                 # Triangle pattern detection
├── live_trading/              # Live trading system
│   ├── api/                   # FastAPI REST API
│   ├── brokers/              # Broker adapters (IBKR, OANDA, cTrader)
│   ├── daemon/               # Background process management
│   ├── data/                 # Data collection and aggregation
│   ├── engine/               # Trading engine
│   ├── logging/              # Modular logging system
│   ├── models/                # MongoDB models
│   ├── orders/               # Order management
│   ├── strategies/           # Strategy adapters
│   ├── cli.py                # Daemon control CLI
│   └── frontend/             # React frontend
├── ib_api_client/            # IBKR API client wrapper
├── request_historical_data/  # Historical data requests
└── data_manager/             # Data download and processing
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENSE file for details.

## Disclaimer

This project is for educational and research purposes. Trading involves significant risk. Use at your own risk and verify behavior on paper trading/sandbox environments before any live deployment.
