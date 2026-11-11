"""Backtester tests: cost accounting, portfolio rules and metric maths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import (
    max_drawdown,
    profit_factor,
    run_backtest,
    sharpe_ratio,
    sortino_ratio,
    threshold_sweep,
)

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def signal(start_hour, outcome="pt", prob=0.9, symbol="BTC/USDT", hold=2, entry=100.0, atr=1.0):
    """One labelled setup: long, entered one bar after the setup."""
    time = T0 + start_hour * HOUR
    exit_price = {"pt": entry + 2 * atr, "sl": entry - atr, "time": entry + 0.5, "no_fill": np.nan}
    filled = outcome != "no_fill"
    end = time + (1 + hold) * HOUR
    return {
        "symbol": symbol,
        "time": time,
        "side": 1,
        "atr": atr,
        "trigger": entry,
        "outcome": outcome,
        "entry_time": time + HOUR if filled else pd.NaT,
        "entry_price": entry if filled else np.nan,
        "exit_time": end if filled else pd.NaT,
        "exit_price": exit_price[outcome],
        "release_time": end,
        "bars_held": hold if filled else np.nan,
        "prob": prob,
    }


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_fees_and_slippage_are_deducted_on_both_sides(settings):
    result = run_backtest(frame(signal(0, "pt")), settings)
    trade = result.trades.iloc[0]

    # Sizing: 1% of 10,000 / 1.0 stop = 100 units = 10,000 notional, capped at
    # 50% of equity -> 50 units.
    assert trade["qty"] == pytest.approx(50.0)
    assert trade["gross_pnl"] == pytest.approx(100.0)

    entry_fill = 100.0 * 1.0005  # 5 bps adverse slippage on the taker entry
    fees = 50 * entry_fill * 0.001 + 50 * 102.0 * 0.001  # target exit: maker fee, no slippage
    expected_net = 50 * (102.0 - entry_fill) - fees
    assert trade["net_pnl"] == pytest.approx(expected_net)
    assert trade["net_pnl"] < trade["gross_pnl"]
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0 + expected_net)


def test_stop_exit_pays_slippage_too(settings):
    trade = run_backtest(frame(signal(0, "sl")), settings).trades.iloc[0]
    assert trade["exit_fill"] == pytest.approx(99.0 * (1 - 0.0005))
    # A full stop-out costs more than 1R once costs are included.
    assert trade["r_net"] < -1.0


def test_only_signals_above_threshold_are_traded(settings):
    signals = frame(signal(0, prob=0.90), signal(10, prob=0.40), signal(20, prob=0.50))
    assert len(run_backtest(signals, settings, threshold=0.5).trades) == 1  # strictly greater
    assert len(run_backtest(signals, settings, threshold=0.3).trades) == 3


def test_one_position_per_symbol(settings):
    signals = frame(signal(0, hold=5), signal(2, hold=5), signal(3, symbol="ETH/USDT", hold=5))
    result = run_backtest(signals, settings)
    assert sorted(result.trades["symbol"]) == ["BTC/USDT", "ETH/USDT"]
    assert result.skipped["symbol_busy"] == 1


def test_unfilled_order_blocks_the_symbol_until_it_expires(settings):
    signals = frame(signal(0, "no_fill", hold=2), signal(1), signal(10))
    result = run_backtest(signals, settings)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["setup_time"] == T0 + 10 * HOUR
    assert result.skipped == {"order_not_filled": 1, "symbol_busy": 1}


def test_daily_kill_switch_stops_new_trades(settings):
    # Ten back-to-back losers on one day. Each loses ~0.55% of equity, so the
    # 3% daily limit is breached part-way through and the rest are refused.
    losers = [signal(i * 2, "sl", hold=1) for i in range(10)]
    next_day = signal(30, "pt")
    result = run_backtest(frame(*losers, next_day), settings)

    same_day = result.trades[result.trades["setup_time"] < T0 + pd.Timedelta(days=1)]
    assert 0 < len(same_day) < 10
    assert result.skipped["halted"] == 10 - len(same_day)
    assert result.equity_curve.min() > 10_000.0 * (1 - 0.04)
    # Trading resumes on the next UTC day (auto_resume_next_day: true).
    assert (result.trades["setup_time"] == T0 + 30 * HOUR).any()


def test_position_size_compounds_with_equity(settings):
    result = run_backtest(frame(signal(0, "pt"), signal(10, "pt")), settings)
    first, second = result.trades.iloc[0], result.trades.iloc[1]
    assert second["qty"] > first["qty"]


def test_metrics_keys_and_sweep(settings):
    signals = frame(*[signal(i * 30, "pt" if i % 3 == 0 else "sl") for i in range(12)])
    metrics = run_backtest(signals, settings).metrics
    for key in ("sharpe", "sortino", "max_drawdown", "profit_factor", "win_rate"):
        assert key in metrics
    assert metrics["win_rate"] == pytest.approx(4 / 12)
    assert metrics["max_drawdown"] < 0

    sweep = threshold_sweep(signals, settings, [0.5, 0.95])
    assert list(sweep["trades"]) == [12, 0]


def test_no_trades_is_handled(settings):
    result = run_backtest(frame(signal(0, prob=0.1)), settings)
    assert result.metrics == {"trades": 0}


# ------------------------------------------------------------- metric maths
def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(-0.25)  # 120 -> 90


def test_profit_factor():
    assert profit_factor(pd.Series([10.0, -5.0, 20.0, -5.0])) == pytest.approx(3.0)
    assert profit_factor(pd.Series([10.0, 5.0])) == float("inf")


def test_sharpe_and_sortino():
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(365)
    assert sharpe_ratio(returns, 365) == pytest.approx(expected_sharpe)

    downside = np.sqrt(np.mean(np.array([0.0, -0.01, 0.0, 0.0]) ** 2))
    assert sortino_ratio(returns, 365) == pytest.approx(returns.mean() / downside * np.sqrt(365))
    # Sortino ignores upside volatility, so it is the larger of the two here.
    assert sortino_ratio(returns, 365) > sharpe_ratio(returns, 365)
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01]), 365))
