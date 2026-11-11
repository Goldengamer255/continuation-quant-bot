"""Event-driven backtester with fees, slippage and live risk controls.

What a backtest is (and is not)
-------------------------------
A backtest replays history and asks "what would this exact rulebook have
earned?". It is only as honest as its assumptions. The ones made here:

* **Out-of-sample signals only.** Input is the walk-forward predictions from
  ``train.py``: every probability was produced by a model that had not seen
  that trade. Backtesting on in-sample predictions measures memorisation.
* **Trade paths come from the labeler.** Entry fill, exit bar and exit price
  were already simulated by ``labeling.simulate_trade`` (stop-limit entry,
  pessimistic stop-first barriers). The backtester does not re-invent fills;
  it adds *money*: sizing, costs, portfolio constraints and equity.
* **Costs on every fill.**
    - Entry is a triggered stop-limit: it crosses the spread -> taker fee
      plus adverse slippage.
    - Stop-loss and time exits are market orders -> taker fee plus slippage.
    - Profit target is a resting limit order -> maker fee, no slippage.
  At 0.1% per side plus 5 bps slippage a round trip costs ~0.25-0.3% of
  notional. When the stop is 0.5% away, that is over half an R per trade:
  costs, not the model, decide whether a short-horizon strategy lives.
* **The live risk manager is in the loop.** Sizing, exposure caps and the
  daily kill-switch are the *same* ``RiskManager`` class the engine uses.
* **One order per symbol.** A symbol with an open position *or a pending
  entry order* is busy, exactly as in the live engine. Unfilled orders
  therefore matter: they block later setups until they expire.

Known simplifications (state these before an interviewer finds them)
--------------------------------------------------------------------
* Equity is marked at trade close (realised), not bar-by-bar. Maximum
  drawdown is therefore measured between closes and understates intra-trade
  drawdown, and the simulated kill-switch reacts to realised losses only.
* Slippage is a constant number of basis points. Real slippage grows with
  order size and during breakouts (precisely when this strategy trades).
* No funding costs, no partial fills, no exchange outages.

Metrics primer
--------------
* **Sharpe ratio** = mean(return) / std(return), annualised by
  ``sqrt(periods_per_year)``. Return per unit of total volatility.
* **Sortino ratio** = mean(return) / downside deviation. Same idea but only
  penalises *downside* volatility; upside surprises are not "risk".
* **Maximum drawdown** = worst peak-to-trough fall in equity. The number
  that determines whether you (or an investor) would have kept going.
* **Profit factor** = gross profit / gross loss. Above 1 is profitable.
* **Win rate** = share of winning trades. Meaningless alone: with a 2:1
  payoff, 35% wins is profitable and with 1:2 payoff, 60% wins is not.
* **Expectancy (R)** = average net result per trade in units of risk.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import Settings, load_settings, resolve_path
from execution.risk_manager import RiskManager
from research.labeling import OUTCOME_NO_FILL, OUTCOME_TARGET
from research.train import PREDICTIONS_FILE


# ---------------------------------------------------------------------------
# Metric functions (pure, individually tested)
# ---------------------------------------------------------------------------
def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate."""
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Annualised Sortino ratio: mean return over downside deviation (target 0)."""
    if len(returns) < 2:
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2))
    if downside == 0:
        return float("nan")
    return float(returns.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a negative fraction (-0.25 = -25%)."""
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class _Slot:
    """Something occupying a symbol: an open position or a pending entry order."""

    symbol: str
    release_time: pd.Timestamp
    notional: float
    pnl: float = 0.0
    trade: dict | None = None


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict[str, float]
    skipped: dict[str, int] = field(default_factory=dict)


def run_backtest(
    signals: pd.DataFrame, settings: Settings, threshold: float | None = None
) -> BacktestResult:
    """Simulate trading the out-of-sample signals.

    The loop walks setups in chronological order. Before each one it realises
    every position that closed earlier (so sizing uses up-to-date equity and
    the kill-switch has seen the latest losses), then asks the risk manager
    whether a new order is allowed and how large it may be.
    """
    threshold = settings.model.probability_threshold if threshold is None else threshold
    costs, lab = settings.costs, settings.labeling
    slip = costs.slippage

    signals = signals.sort_values("time", kind="stable")
    candidates = signals[signals["prob"] > threshold]

    risk = RiskManager(settings.risk)
    equity = settings.backtest.initial_equity
    start_time = signals["time"].min() if len(signals) else pd.Timestamp("1970-01-01", tz="UTC")
    equity_points: list[tuple[pd.Timestamp, float]] = [(start_time, equity)]
    busy: dict[str, _Slot] = {}
    trades: list[dict] = []
    skipped: Counter[str] = Counter()

    def release(until: pd.Timestamp | None) -> None:
        """Realise every slot whose trade/order ended at or before ``until``."""
        nonlocal equity
        due = [s for s in busy.values() if until is None or s.release_time <= until]
        for slot in sorted(due, key=lambda s: s.release_time):
            del busy[slot.symbol]
            if slot.trade is None:  # an entry order that expired unfilled
                continue
            equity += slot.pnl
            slot.trade["equity_after"] = equity
            trades.append(slot.trade)
            equity_points.append((slot.release_time, equity))
            risk.update_equity(equity, slot.release_time)

    for row in candidates.itertuples(index=False):
        release(row.time)
        risk.update_equity(equity, row.time)

        if row.symbol in busy:
            skipped["symbol_busy"] += 1
            continue

        side = int(row.side)
        planned_stop = row.trigger - side * lab.stop_loss_atr * row.atr
        decision = risk.evaluate_entry(
            equity=equity,
            entry_price=row.trigger,
            stop_price=planned_stop,
            open_positions=len(busy),
            gross_exposure=sum(s.notional for s in busy.values()),
        )
        if not decision.approved:
            skipped[decision.reason] += 1
            continue
        qty = decision.quantity

        if row.outcome == OUTCOME_NO_FILL:
            # The order rests until it expires; it ties up the symbol and exposure.
            busy[row.symbol] = _Slot(row.symbol, row.release_time, qty * row.trigger)
            skipped["order_not_filled"] += 1
            continue

        # ---- fills and costs ------------------------------------------------
        entry_fill = row.entry_price * (1.0 + side * slip)  # pay up to get in
        if row.outcome == OUTCOME_TARGET:
            exit_fill, exit_fee_rate = row.exit_price, costs.maker_fee
        else:
            exit_fill, exit_fee_rate = row.exit_price * (1.0 - side * slip), costs.taker_fee
        fees = qty * entry_fill * costs.taker_fee + qty * exit_fill * exit_fee_rate
        gross_pnl = side * qty * (row.exit_price - row.entry_price)
        net_pnl = side * qty * (exit_fill - entry_fill) - fees
        risk_amount = qty * lab.stop_loss_atr * row.atr

        trade = {
            "symbol": row.symbol,
            "side": side,
            "setup_time": row.time,
            "entry_time": row.entry_time,
            "exit_time": row.exit_time,
            "outcome": row.outcome,
            "prob": row.prob,
            "qty": qty,
            "entry_fill": entry_fill,
            "exit_fill": exit_fill,
            "fees": fees,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "r_net": net_pnl / risk_amount,
            "bars_held": row.bars_held,
        }
        busy[row.symbol] = _Slot(row.symbol, row.release_time, qty * row.trigger, net_pnl, trade)

    release(None)  # close out whatever is still open at the end of the data

    trades_df = pd.DataFrame(trades)
    curve = pd.Series(dict(equity_points)).sort_index()  # dict keeps the last value per timestamp
    curve.name = "equity"
    metrics = compute_metrics(trades_df, curve, settings)
    return BacktestResult(trades_df, curve, metrics, dict(skipped))


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, settings: Settings) -> dict:
    """Headline performance statistics from the trade list and equity curve."""
    ppy = settings.backtest.periods_per_year
    initial = settings.backtest.initial_equity
    metrics: dict[str, float] = {"trades": len(trades)}
    if trades.empty:
        return metrics

    # Daily returns include flat days (ffill): days without trades are part of
    # the track record and must not be dropped, or Sharpe is overstated.
    daily = equity.resample("1D").last().ffill()
    daily_returns = daily.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]) / pd.Timedelta(days=365), 1e-9)
    final = float(equity.iloc[-1])

    metrics.update(
        {
            "total_return": final / initial - 1.0,
            "cagr": (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0,
            "sharpe": sharpe_ratio(daily_returns, ppy),
            "sortino": sortino_ratio(daily_returns, ppy),
            "max_drawdown": max_drawdown(equity),
            "profit_factor": profit_factor(trades["net_pnl"]),
            "win_rate": float((trades["net_pnl"] > 0).mean()),
            "expectancy_r": float(trades["r_net"].mean()),
            "avg_bars_held": float(trades["bars_held"].mean()),
            "total_fees": float(trades["fees"].sum()),
            "gross_pnl": float(trades["gross_pnl"].sum()),
            "net_pnl": float(trades["net_pnl"].sum()),
        }
    )
    return metrics


def threshold_sweep(
    signals: pd.DataFrame, settings: Settings, thresholds: list[float]
) -> pd.DataFrame:
    """Re-run the backtest across probability thresholds.

    Use this to *understand* the selectivity/frequency trade-off — not to pick
    the best row and report it. Choosing a threshold by its backtest result
    is fitting a parameter on the test set; an honest process fixes the
    threshold beforehand or selects it on a separate validation period.
    """
    rows = []
    for thr in thresholds:
        m = run_backtest(signals, settings, threshold=thr).metrics
        rows.append({"threshold": thr} | m)
    cols = ["threshold", "trades", "win_rate", "expectancy_r", "profit_factor", "sharpe",
            "max_drawdown", "total_return"]  # fmt: skip
    return pd.DataFrame(rows).reindex(columns=cols)


def print_report(result: BacktestResult, threshold: float) -> None:
    m = result.metrics
    print(f"\n=== Backtest (probability > {threshold:.2f}, out-of-sample signals) ===")
    if not m.get("trades"):
        print("No trades were taken at this threshold.")
    else:
        lines = [
            ("Trades", f"{m['trades']:d}"),
            ("Win rate", f"{m['win_rate']:.1%}"),
            ("Profit factor", f"{m['profit_factor']:.2f}"),
            ("Expectancy", f"{m['expectancy_r']:+.3f} R per trade"),
            ("Sharpe ratio", f"{m['sharpe']:.2f}"),
            ("Sortino ratio", f"{m['sortino']:.2f}"),
            ("Maximum drawdown", f"{m['max_drawdown']:.1%}"),
            ("Total return", f"{m['total_return']:+.1%}"),
            ("CAGR", f"{m['cagr']:+.1%}"),
            ("Gross PnL", f"{m['gross_pnl']:+,.2f}"),
            ("Fees paid", f"{m['total_fees']:,.2f}"),
            ("Net PnL", f"{m['net_pnl']:+,.2f}"),
            ("Avg bars held", f"{m['avg_bars_held']:.1f}"),
        ]
        for name, value in lines:
            print(f"  {name:<18} {value}")
    if result.skipped:
        print("  Setups not traded:", dict(sorted(result.skipped.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the out-of-sample model signals")
    parser.add_argument("--config", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="also show a threshold sweep")
    args = parser.parse_args()

    settings = load_settings(args.config)
    path = resolve_path(settings.model.artifact_dir) / PREDICTIONS_FILE
    if not path.exists():
        raise SystemExit("No predictions found. Run `python -m research.train` first.")
    signals = pd.read_parquet(path)

    threshold = args.threshold or settings.model.probability_threshold
    print_report(run_backtest(signals, settings, threshold), threshold)
    if args.sweep:
        pd.set_option("display.width", 200)
        sweep = threshold_sweep(signals, settings, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
        print("\n=== Threshold sweep (diagnostic only) ===")
        print(sweep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
