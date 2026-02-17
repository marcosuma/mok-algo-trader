"""
Strategy Registry - Dynamically discovers and registers all trading strategies.

This module automatically discovers all strategy classes in the forex_strategies
package and provides a registry for easy access.
"""
import importlib
import inspect
import os
from typing import Dict, List, Type, Optional
from forex_strategies.base_strategy import BaseForexStrategy


# Strategy registry: maps strategy name to (class, module_path)
_STRATEGY_REGISTRY: Dict[str, tuple] = {}


def _discover_strategies():
    """Discover all strategy classes in the forex_strategies package."""
    global _STRATEGY_REGISTRY

    if _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY

    # Get the package directory
    package_dir = os.path.dirname(__file__)

    # Import all strategy modules
    strategy_modules = [
        'adaptive_multi_indicator_strategy',
        'breakout_strategy',
        'buy_and_hold_strategy',
        'hammer_shooting_star',
        'market_structure_strategy',
        'marsi_strategy',
        'mean_reversion_strategy',
        'momentum_strategy',
        'multi_timeframe_strategy',
        'pattern_strategy',
        'pattern_triangle_strategy',
        'rsi_strategy',
        'triangle_strategy',
    ]

    for module_name in strategy_modules:
        try:
            module = importlib.import_module(f'forex_strategies.{module_name}')

            # Find all classes that inherit from BaseForexStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseForexStrategy) and
                    obj != BaseForexStrategy and
                    obj.__module__ == module.__name__):
                    # Register the strategy
                    strategy_name = name
                    _STRATEGY_REGISTRY[strategy_name] = (obj, module_name)
        except ImportError as e:
            # Skip modules that can't be imported
            print(f"Warning: Could not import {module_name}: {e}")
            continue

    return _STRATEGY_REGISTRY


def get_all_strategies() -> Dict[str, Type[BaseForexStrategy]]:
    """
    Get all discovered strategy classes.

    Returns:
        Dictionary mapping strategy name to strategy class
    """
    registry = _discover_strategies()
    return {name: cls for name, (cls, _) in registry.items()}


def get_strategy_names() -> List[str]:
    """
    Get list of all strategy names.

    Returns:
        List of strategy class names
    """
    return list(_discover_strategies().keys())


def get_strategy(name: str) -> Optional[Type[BaseForexStrategy]]:
    """
    Get a strategy class by name.

    Args:
        name: Strategy class name

    Returns:
        Strategy class or None if not found
    """
    registry = _discover_strategies()
    if name in registry:
        return registry[name][0]
    return None


def filter_strategies(strategy_names: Optional[List[str]] = None) -> Dict[str, Type[BaseForexStrategy]]:
    """
    Filter strategies by name list.

    Args:
        strategy_names: List of strategy names to include. If None, returns all.

    Returns:
        Dictionary of filtered strategies
    """
    all_strategies = get_all_strategies()

    if strategy_names is None:
        return all_strategies

    # Filter by provided names
    filtered = {}
    for name in strategy_names:
        if name in all_strategies:
            filtered[name] = all_strategies[name]
        else:
            print(f"Warning: Strategy '{name}' not found. Available strategies: {', '.join(all_strategies.keys())}")

    return filtered

