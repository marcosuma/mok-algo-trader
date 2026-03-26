"""
Configuration management for live trading system.
"""
import os
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    import warnings
    # Suppress dotenv parsing warnings - they're non-fatal
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*python-dotenv.*")
        try:
            load_dotenv()
        except Exception as e:
            # Log but don't fail - environment variables can be set directly
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not load .env file: {e}. Using environment variables directly.")
except ImportError:
    pass


class Config:
    """Application configuration"""

    # MongoDB
    # Supports both local MongoDB and MongoDB Atlas (mongodb+srv://)
    # Example local: mongodb://localhost:27017
    # Example Atlas: mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
    # Note: For Atlas, ensure your IP is whitelisted in Network Access settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "trading_bot")
    MONGODB_CONNECT_TIMEOUT_MS: int = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "30000"))  # 30 seconds for Atlas

    # Broker Configuration
    BROKER_TYPE: str = os.getenv("BROKER_TYPE", "IBKR").upper()  # IBKR, OANDA, PEPPERSTONE, or CTRADER

    # IBKR Configuration
    IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = int(os.getenv("IBKR_PORT", "7497"))  # 7497 for paper, 7496 for live
    IBKR_ACCOUNT_TYPE: str = os.getenv("IBKR_ACCOUNT_TYPE", "PAPER")  # PAPER or LIVE

    # OANDA Configuration
    OANDA_API_KEY: Optional[str] = os.getenv("OANDA_API_KEY")
    OANDA_ACCOUNT_ID: Optional[str] = os.getenv("OANDA_ACCOUNT_ID")
    OANDA_ENVIRONMENT: str = os.getenv("OANDA_ENVIRONMENT", "PRACTICE")  # PRACTICE or LIVE

    # Pepperstone Configuration (via MetaTrader5)
    PEPPERSTONE_LOGIN: Optional[str] = os.getenv("PEPPERSTONE_LOGIN")
    PEPPERSTONE_PASSWORD: Optional[str] = os.getenv("PEPPERSTONE_PASSWORD")
    PEPPERSTONE_SERVER: Optional[str] = os.getenv("PEPPERSTONE_SERVER")  # e.g., "Pepperstone-Demo" or "Pepperstone-Live"

    # cTrader Configuration
    CTRADER_CLIENT_ID: Optional[str] = os.getenv("CTRADER_CLIENT_ID")
    CTRADER_CLIENT_SECRET: Optional[str] = os.getenv("CTRADER_CLIENT_SECRET")
    CTRADER_ACCESS_TOKEN: Optional[str] = os.getenv("CTRADER_ACCESS_TOKEN")
    CTRADER_ENVIRONMENT: str = os.getenv("CTRADER_ENVIRONMENT", "DEMO")  # DEMO or LIVE

    # Telegram Notifications
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # API Server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")

    # Redis (for Celery, optional)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Risk Management Defaults
    DEFAULT_STOP_LOSS_TYPE: str = os.getenv("DEFAULT_STOP_LOSS_TYPE", "ATR")
    DEFAULT_STOP_LOSS_VALUE: float = float(os.getenv("DEFAULT_STOP_LOSS_VALUE", "1.5"))
    DEFAULT_TAKE_PROFIT_TYPE: str = os.getenv("DEFAULT_TAKE_PROFIT_TYPE", "RISK_REWARD")
    DEFAULT_TAKE_PROFIT_VALUE: float = float(os.getenv("DEFAULT_TAKE_PROFIT_VALUE", "2.0"))
    DEFAULT_CRASH_RECOVERY_MODE: str = os.getenv("DEFAULT_CRASH_RECOVERY_MODE", "CLOSE_ALL")
    DEFAULT_EMERGENCY_STOP_LOSS_PCT: float = float(os.getenv("DEFAULT_EMERGENCY_STOP_LOSS_PCT", "0.05"))
    DEFAULT_DATA_RETENTION_BARS: int = int(os.getenv("DEFAULT_DATA_RETENTION_BARS", "1000"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    JOURNAL_RETENTION_DAYS: int = int(os.getenv("JOURNAL_RETENTION_DAYS", "90"))

    # GCP (for cloud deployment)
    GCP_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID")
    GCP_ZONE: str = os.getenv("GCP_ZONE", "us-central1-a")


config = Config()

