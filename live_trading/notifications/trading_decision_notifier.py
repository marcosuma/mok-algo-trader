"""Hourly grouped Telegram notifications for trading decisions."""
import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple, Union

import requests

from live_trading.notifications.trading_events import (
    BlockedDecision,
    HoldDecision,
    OrderPlacedDecision,
    PositionClosedDecision,
    SkippedDecision,
)

logger = logging.getLogger(__name__)

TradingDecision = Union[
    SkippedDecision, HoldDecision, BlockedDecision, OrderPlacedDecision, PositionClosedDecision
]


def _default_send(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    if not response.ok:
        raise RuntimeError(f"Telegram API error {response.status_code}: {response.text}")


def _format_decision(decision: TradingDecision) -> str:
    ts = decision.timestamp.strftime("%H:%M")
    if isinstance(decision, SkippedDecision):
        return f"{ts} ⚫ SKIPPED [{decision.reason}]"
    if isinstance(decision, HoldDecision):
        return f"{ts} ⚪ HOLD @ {decision.price:.5f}"
    if isinstance(decision, BlockedDecision):
        emoji = "🟢" if decision.signal_type == "BUY" else "🔴"
        return f"{ts} ⛔ {emoji} {decision.signal_type} blocked [{decision.reason}] @ {decision.price:.5f}"
    if isinstance(decision, OrderPlacedDecision):
        sl = f"{decision.stop_loss:.5f}" if decision.stop_loss else "N/A"
        tp = f"{decision.take_profit:.5f}" if decision.take_profit else "N/A"
        emoji = "🟢" if decision.signal_type == "BUY" else "🔴"
        return (
            f"{ts} {emoji} {decision.signal_type} @ {decision.price:.5f} "
            f"→ SL {sl} TP {tp} ({decision.lot_size:.2f} lots)"
        )
    if isinstance(decision, PositionClosedDecision):
        pnl_sign = "+" if decision.pnl >= 0 else ""
        return (
            f"{ts} 🏁 {decision.side} closed @ {decision.close_price:.5f} "
            f"[{decision.close_reason}] P/L: {pnl_sign}{decision.pnl:.2f}"
        )
    return str(decision)


class TradingDecisionNotifier:
    """Collects trading decisions from all operations and sends hourly grouped Telegram messages."""

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        environment: str,
        _send_fn: Callable = _default_send,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._environment = environment
        self._send_fn = _send_fn
        self._enabled = bool(bot_token and chat_id)

        self._buffer: List[TradingDecision] = []
        self._buffer_lock = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None

        if not self._enabled:
            logger.warning(
                "[TradingNotifier] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "trading decision notifications are disabled."
            )

    def record(self, decision: TradingDecision) -> None:
        """Record a trading decision. Thread-safe."""
        if not self._enabled:
            return
        with self._buffer_lock:
            self._buffer.append(decision)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule the hourly flush loop on the given event loop."""
        if not self._enabled:
            return
        self._flush_task = loop.create_task(self._hourly_flush_loop())
        logger.info("[TradingNotifier] Hourly flush loop started")

    def stop(self) -> None:
        """Cancel the flush loop."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

    async def _hourly_flush_loop(self) -> None:
        while True:
            now = datetime.now(timezone.utc)
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            sleep_secs = (next_hour - now).total_seconds()
            await asyncio.sleep(sleep_secs)

            window_start = next_hour - timedelta(hours=1)

            with self._buffer_lock:
                decisions = list(self._buffer)
                self._buffer.clear()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._send_summary, decisions, window_start, next_hour)

    def _send_summary(
        self,
        decisions: List[TradingDecision],
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        start_str = window_start.strftime("%H:%M")
        end_str = window_end.strftime("%H:%M")
        header = f"📊 Trading Summary — {self._environment} [{start_str}–{end_str} UTC]"

        if not decisions:
            text = f"{header}\n\nNo decisions this hour."
        else:
            groups: Dict[Tuple[str, str], List[TradingDecision]] = {}
            for d in decisions:
                key = (d.asset, d.strategy_name)
                groups.setdefault(key, []).append(d)

            lines = [header, ""]
            for (asset, strategy), group_decisions in groups.items():
                lines.append(f"{asset} | {strategy}")
                for d in sorted(group_decisions, key=lambda x: x.timestamp):
                    lines.append(f"• {_format_decision(d)}")
                lines.append("")

            text = "\n".join(lines).rstrip()

        try:
            self._send_fn(self._token, self._chat_id, text)
            logger.info(f"[TradingNotifier] Sent hourly summary ({len(decisions)} decisions)")
        except Exception:
            logger.warning("[TradingNotifier] Failed to send hourly summary", exc_info=True)
