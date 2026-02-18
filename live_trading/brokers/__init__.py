"""
Broker Adapters for IBKR, OANDA, Pepperstone, and cTrader.
"""
from .base_broker import BaseBroker

# Try to import IBKR broker, but make it optional
try:
    from .ibkr_broker import IBKRBroker
except ImportError:
    IBKRBroker = None

# Try to import OANDA broker, but make it optional
try:
    from .oanda_broker import OANDABroker
except ImportError:
    OANDABroker = None

# Try to import Pepperstone broker, but make it optional
try:
    from .pepperstone_broker import PepperstoneBroker
except ImportError:
    PepperstoneBroker = None

# Try to import cTrader broker, but make it optional
try:
    from .ctrader_broker import CTraderBroker
except (ImportError, NameError):
    CTraderBroker = None

# Build __all__ list based on what's available
__all__ = ["BaseBroker"]
if IBKRBroker is not None:
    __all__.append("IBKRBroker")
if OANDABroker is not None:
    __all__.append("OANDABroker")
if PepperstoneBroker is not None:
    __all__.append("PepperstoneBroker")
if CTraderBroker is not None:
    __all__.append("CTraderBroker")

