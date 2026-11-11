"""Risk limit tests: sizing, caps, and the kill-switch."""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import RiskConfig
from execution.risk_manager import (
    HARD_MAX_DAILY_DRAWDOWN,
    HARD_MAX_RISK_PER_TRADE,
    HaltReason,
    RiskManager,
    atr_position_size,
)

T0 = pd.Timestamp("2024-03-01 09:00", tz="UTC")
# Caps opened wide so individual tests exercise one limit at a time.
LOOSE = RiskConfig(max_position_pct=100.0, max_gross_exposure_pct=100.0)


def entry(risk: RiskManager, **overrides):
    args = {
        "equity": 10_000.0,
        "entry_price": 100.0,
        "stop_price": 98.0,
        "open_positions": 0,
        "gross_exposure": 0.0,
    }
    return risk.evaluate_entry(**(args | overrides))


def test_position_size_risks_exactly_the_configured_fraction():
    qty = atr_position_size(10_000.0, 0.01, entry_price=100.0, stop_price=98.0)
    assert qty == pytest.approx(50.0)
    assert qty * (100.0 - 98.0) == pytest.approx(100.0)  # loss at the stop = 1% of equity


def test_position_size_is_symmetric_for_shorts_and_safe_on_bad_input():
    assert atr_position_size(10_000.0, 0.01, 100.0, 102.0) == pytest.approx(50.0)
    assert atr_position_size(10_000.0, 0.01, 100.0, 100.0) == 0.0
    assert atr_position_size(0.0, 0.01, 100.0, 98.0) == 0.0


def test_wider_stop_means_smaller_position():
    risk = RiskManager(LOOSE)
    tight = entry(risk, stop_price=99.0).quantity
    wide = entry(risk, stop_price=96.0).quantity
    assert tight == pytest.approx(4 * wide)


def test_position_notional_cap_shrinks_the_order():
    # Uncapped: 1% risk / 0.5 stop = 200 units = 20,000 notional (2x equity).
    risk = RiskManager(RiskConfig(max_position_pct=0.5, max_gross_exposure_pct=10.0))
    decision = entry(risk, stop_price=99.5)
    assert decision.approved
    assert decision.quantity * 100.0 == pytest.approx(5_000.0)


def test_gross_exposure_cap():
    risk = RiskManager(RiskConfig(max_position_pct=1.0, max_gross_exposure_pct=1.0))
    partly_used = entry(risk, stop_price=99.5, gross_exposure=8_000.0)
    assert partly_used.quantity * 100.0 == pytest.approx(2_000.0)
    full = entry(risk, stop_price=99.5, gross_exposure=10_000.0)
    assert not full.approved and full.reason == "exposure_cap"


def test_max_open_positions():
    risk = RiskManager(RiskConfig(max_open_positions=2))
    assert entry(risk, open_positions=1).approved
    blocked = entry(risk, open_positions=2)
    assert not blocked.approved and blocked.reason == "max_open_positions"


def test_kill_switch_trips_at_three_percent_daily_drawdown():
    risk = RiskManager(RiskConfig())
    risk.update_equity(10_000.0, T0)
    risk.update_equity(9_750.0, T0 + pd.Timedelta(hours=1))  # -2.5%: still trading
    assert not risk.halted
    risk.update_equity(9_690.0, T0 + pd.Timedelta(hours=2))  # -3.1%
    assert risk.halted and risk.halt_reason is HaltReason.DAILY_DRAWDOWN
    assert not entry(risk).approved


def test_drawdown_is_measured_from_the_intraday_peak():
    risk = RiskManager(RiskConfig())
    risk.update_equity(10_000.0, T0)
    risk.update_equity(10_500.0, T0 + pd.Timedelta(hours=1))
    # Still above the day's open, but 3.3% off the peak.
    risk.update_equity(10_150.0, T0 + pd.Timedelta(hours=2))
    assert risk.halted


def test_config_cannot_loosen_the_hard_ceilings():
    risk = RiskManager(RiskConfig(max_daily_drawdown=0.30, risk_per_trade=0.25))
    assert risk.max_daily_drawdown == HARD_MAX_DAILY_DRAWDOWN
    assert risk.risk_per_trade == HARD_MAX_RISK_PER_TRADE
    risk.update_equity(10_000.0, T0)
    risk.update_equity(9_600.0, T0 + pd.Timedelta(hours=1))  # -4%: config said 30% was fine
    assert risk.halted


def test_config_can_tighten_the_limits():
    risk = RiskManager(RiskConfig(max_daily_drawdown=0.01))
    risk.update_equity(10_000.0, T0)
    risk.update_equity(9_890.0, T0 + pd.Timedelta(hours=1))
    assert risk.halted


def test_drawdown_halt_latches_then_clears_next_day_only_if_configured():
    auto = RiskManager(RiskConfig(auto_resume_next_day=True))
    manual = RiskManager(RiskConfig(auto_resume_next_day=False))
    for risk in (auto, manual):
        risk.update_equity(10_000.0, T0)
        risk.update_equity(9_600.0, T0 + pd.Timedelta(hours=1))
        # Recovering later the same day must NOT re-enable trading.
        risk.update_equity(10_000.0, T0 + pd.Timedelta(hours=2))
        assert risk.halted
        risk.update_equity(9_600.0, T0 + pd.Timedelta(days=1))
    assert not auto.halted
    assert manual.halted


def test_consecutive_api_errors_trip_the_switch_and_never_auto_clear():
    risk = RiskManager(RiskConfig(max_consecutive_api_errors=3))
    risk.record_api_error()
    risk.record_api_error()
    risk.record_api_success()  # a success resets the streak
    risk.record_api_error()
    risk.record_api_error()
    assert not risk.halted
    risk.record_api_error()
    assert risk.halt_reason is HaltReason.API_ERRORS

    risk.update_equity(10_000.0, T0)
    risk.update_equity(10_000.0, T0 + pd.Timedelta(days=1))
    assert risk.halted, "API-error halts require a manual reset"
    risk.reset()
    assert not risk.halted
