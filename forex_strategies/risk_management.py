"""
Shared risk management calculations used by both backtesting and live trading.

These pure functions encapsulate SL/TP computation so both systems use identical
logic, eliminating one of the biggest sources of backtest-vs-live divergence.
"""
from typing import Optional, Tuple


def calculate_stop_loss(
    entry_price: float,
    position_type: str,
    sl_type: str,
    sl_value: float,
    atr_value: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate stop-loss price.

    Args:
        entry_price: Estimated entry price.
        position_type: ``'LONG'`` or ``'SHORT'``.
        sl_type: One of ``'ATR'``, ``'PERCENTAGE'``, ``'FIXED'``, ``'NONE'``.
        sl_value: Multiplier for ATR, fraction for PERCENTAGE, distance for FIXED.
        atr_value: Current ATR (required when *sl_type* is ``'ATR'``).

    Returns:
        Absolute stop-loss price, or ``None`` when *sl_type* is ``'NONE'``
        or when insufficient data is available (e.g. missing ATR).
    """
    if sl_type == "NONE":
        return None

    if sl_type == "ATR":
        if atr_value is None or atr_value == 0.0:
            return None
        stop_distance = sl_value * atr_value
    elif sl_type == "PERCENTAGE":
        stop_distance = sl_value * entry_price
    elif sl_type == "FIXED":
        stop_distance = sl_value
    else:
        raise ValueError(f"Unknown stop_loss_type: {sl_type}")

    if position_type == "LONG":
        return entry_price - stop_distance
    return entry_price + stop_distance


def calculate_take_profit(
    entry_price: float,
    stop_loss_price: Optional[float],
    position_type: str,
    tp_type: str,
    tp_value: float,
    atr_value: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate take-profit price.

    Args:
        entry_price: Estimated entry price.
        stop_loss_price: Previously calculated stop-loss (needed for
            ``RISK_REWARD``).
        position_type: ``'LONG'`` or ``'SHORT'``.
        tp_type: One of ``'RISK_REWARD'``, ``'ATR'``, ``'PERCENTAGE'``,
            ``'FIXED'``, ``'NONE'``.
        tp_value: Ratio for RISK_REWARD, multiplier for ATR, fraction for
            PERCENTAGE, absolute price for FIXED.
        atr_value: Current ATR (required when *tp_type* is ``'ATR'``).

    Returns:
        Absolute take-profit price, or ``None`` when *tp_type* is ``'NONE'``
        or when insufficient data is available.
    """
    if tp_type == "NONE":
        return None

    if tp_type == "RISK_REWARD":
        if stop_loss_price is None:
            return None
        if position_type == "LONG":
            risk_distance = entry_price - stop_loss_price
        else:
            risk_distance = stop_loss_price - entry_price
        profit_distance = risk_distance * tp_value
    elif tp_type == "ATR":
        if atr_value is None or atr_value == 0.0:
            return None
        profit_distance = tp_value * atr_value
    elif tp_type == "PERCENTAGE":
        profit_distance = tp_value * entry_price
    elif tp_type == "FIXED":
        return tp_value
    else:
        raise ValueError(f"Unknown take_profit_type: {tp_type}")

    if position_type == "LONG":
        return entry_price + profit_distance
    return entry_price - profit_distance


def compute_sl_tp(
    entry_price: float,
    position_type: str,
    sl_type: str,
    sl_value: float,
    tp_type: str,
    tp_value: float,
    atr_value: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Convenience wrapper that returns ``(stop_loss, take_profit)``."""
    sl = calculate_stop_loss(entry_price, position_type, sl_type, sl_value, atr_value)
    tp = calculate_take_profit(entry_price, sl, position_type, tp_type, tp_value, atr_value)
    return sl, tp
