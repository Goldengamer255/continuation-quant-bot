"""Risk management: position sizing, exposure caps and the kill-switch.

Why risk code is its own module
-------------------------------
Signals decide *what* to trade; risk decides *how much* and *whether we are
allowed to trade at all*. Keeping it separate means:

* It has **veto power**. Every order must pass ``evaluate_entry``; there is
  no code path that sends an entry without asking.
* It is **pure and deterministic** — no network, no clock (time is passed in
  as ``now``). That makes it trivially unit-testable, and it lets the
  backtester use the *same object* as the live engine. A backtest that sizes
  positions differently from production is measuring a different strategy.

Position sizing: fixed-fractional risk
--------------------------------------
Decide how much to *lose* if wrong, then derive the size::

    quantity = (equity * risk_per_trade) / |entry_price - stop_price|

With equity 10,000, 1% risk and a stop 50 away: quantity = 100 / 50 = 2
units. If the stop is hit the loss is 2 * 50 = 100 = 1% of equity, whatever
the asset's price or volatility. Because the stop distance is ``1 * ATR``,
positions automatically shrink when the market is wild and grow when it is
calm, keeping the *risk* per trade constant ("volatility targeting").

Why 1% and not something larger: losses compound geometrically. Ten losses
in a row at 1% is a 9.6% drawdown; at 5% it is 40%, which needs a 67% gain
to recover. Strategies with a ~35% win rate *will* see long losing streaks.

Why the notional cap matters: the formula knows nothing about leverage. If
ATR is 0.5% of price, risking 1% implies a position worth 200% of equity.
On a spot account that is impossible, so ``max_position_pct`` shrinks the
order. The realised risk is then *below* 1% — the cap only ever reduces risk.

The kill-switch (circuit breaker)
---------------------------------
Two conditions halt all new orders and tell the engine to flatten:

1. **Daily drawdown** — equity falls ``max_daily_drawdown`` below the day's
   high-water mark. Measuring from the intraday peak (not the day's open) is
   stricter: giving back a morning's gains also counts.
2. **Consecutive API errors** — if the exchange connection is failing we
   cannot trust our view of orders and positions, so we stop adding risk.

The thresholds come from config but are clamped by **hard-coded ceilings**
below. A config typo (0.3 instead of 0.03) or a tempted operator can make the
bot *safer* but can never loosen it past the ceiling without a code change
that shows up in review.

The halt is a **latch**: once tripped it stays tripped. A drawdown halt may
auto-clear at the next UTC day (configurable); an API-error or internal-error
halt requires a human to restart the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from core.config import RiskConfig

logger = logging.getLogger(__name__)

# Hard ceilings. Config may tighten these, never loosen them.
HARD_MAX_DAILY_DRAWDOWN = 0.03
HARD_MAX_RISK_PER_TRADE = 0.02


class HaltReason(StrEnum):
    DAILY_DRAWDOWN = "daily_drawdown"
    API_ERRORS = "api_errors"
    INTERNAL_ERROR = "internal_error"
    MANUAL = "manual"


@dataclass(frozen=True)
class OrderDecision:
    """Result of a pre-trade risk check."""

    approved: bool
    quantity: float = 0.0
    reason: str = "ok"


def atr_position_size(
    equity: float, risk_fraction: float, entry_price: float, stop_price: float
) -> float:
    """Units to trade so that hitting the stop loses ``risk_fraction`` of equity."""
    stop_distance = abs(entry_price - stop_price)
    if equity <= 0 or stop_distance <= 0:
        return 0.0
    return (equity * risk_fraction) / stop_distance


class RiskManager:
    """Stateful pre-trade gatekeeper and kill-switch."""

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.risk_per_trade = min(cfg.risk_per_trade, HARD_MAX_RISK_PER_TRADE)
        self.max_daily_drawdown = min(cfg.max_daily_drawdown, HARD_MAX_DAILY_DRAWDOWN)
        if self.risk_per_trade < cfg.risk_per_trade:
            logger.warning("risk_per_trade clamped to hard ceiling %s", HARD_MAX_RISK_PER_TRADE)
        if self.max_daily_drawdown < cfg.max_daily_drawdown:
            logger.warning("max_daily_drawdown clamped to hard ceiling %s", HARD_MAX_DAILY_DRAWDOWN)

        self._day: object | None = None
        self._day_peak: float = 0.0
        self.daily_drawdown: float = 0.0
        self._consecutive_errors: int = 0
        self._halt_reason: HaltReason | None = None

    # ------------------------------------------------------------------ state
    @property
    def halted(self) -> bool:
        return self._halt_reason is not None

    @property
    def halt_reason(self) -> HaltReason | None:
        return self._halt_reason

    def trip(self, reason: HaltReason, detail: str = "") -> None:
        """Engage the kill-switch (idempotent: the first reason is kept)."""
        if self._halt_reason is None:
            self._halt_reason = reason
            logger.critical("kill_switch_tripped", extra={"reason": reason.value, "detail": detail})

    def reset(self) -> None:
        """Manual re-arm. Never called automatically for non-drawdown halts."""
        self._halt_reason = None
        self._consecutive_errors = 0

    # ----------------------------------------------------------- equity / PnL
    def update_equity(self, equity: float, now: pd.Timestamp) -> None:
        """Feed the latest account equity; trips the switch on a drawdown breach."""
        day = now.date()
        if day != self._day:  # new UTC day: reset the daily high-water mark
            self._day = day
            self._day_peak = equity
            if self._halt_reason is HaltReason.DAILY_DRAWDOWN and self.cfg.auto_resume_next_day:
                logger.warning("daily drawdown halt cleared at start of %s", day)
                self._halt_reason = None

        self._day_peak = max(self._day_peak, equity)
        self.daily_drawdown = 1.0 - equity / self._day_peak if self._day_peak > 0 else 0.0
        if self.daily_drawdown >= self.max_daily_drawdown:
            self.trip(HaltReason.DAILY_DRAWDOWN, f"drawdown={self.daily_drawdown:.4f}")

    # ------------------------------------------------------------- API health
    def record_api_error(self) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.cfg.max_consecutive_api_errors:
            self.trip(HaltReason.API_ERRORS, f"{self._consecutive_errors} consecutive errors")

    def record_api_success(self) -> None:
        # Only *consecutive* failures matter: one success proves the link is alive.
        self._consecutive_errors = 0

    # --------------------------------------------------------- pre-trade check
    def evaluate_entry(
        self,
        *,
        equity: float,
        entry_price: float,
        stop_price: float,
        open_positions: int,
        gross_exposure: float,
    ) -> OrderDecision:
        """Size a proposed entry and apply every limit. Returns the approved quantity.

        Checks run from cheapest/most fundamental to most specific. Caps
        *shrink* an order where that is safe (notional limits) and *reject*
        it where shrinking makes no sense (halted, too many positions).
        """
        if self.halted:
            return OrderDecision(False, reason="halted")
        if open_positions >= self.cfg.max_open_positions:
            return OrderDecision(False, reason="max_open_positions")

        quantity = atr_position_size(equity, self.risk_per_trade, entry_price, stop_price)
        if quantity <= 0:
            return OrderDecision(False, reason="invalid_stop")

        # Cap 1: single-position notional.
        max_notional = self.cfg.max_position_pct * equity
        # Cap 2: total exposure across all positions and pending orders.
        exposure_room = self.cfg.max_gross_exposure_pct * equity - gross_exposure
        allowed_notional = min(quantity * entry_price, max_notional, exposure_room)

        if allowed_notional < max(self.cfg.min_order_notional, 1e-12):
            reason = "exposure_cap" if exposure_room < max_notional else "below_min_notional"
            return OrderDecision(False, reason=reason)
        return OrderDecision(True, quantity=allowed_notional / entry_price)
