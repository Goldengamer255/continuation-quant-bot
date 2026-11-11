"""End-to-end tests: research pipeline, then the live engine replayed on paper."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from data.synthetic import generate_synthetic_ohlcv
from execution.broker import PaperBroker, ReplayFeed
from execution.engine import Phase, TradingEngine
from execution.risk_manager import HaltReason, RiskManager
from research.backtest import run_backtest
from research.dataset import build_dataset
from research.labeling import entry_levels
from research.train import load_model, save_model, train_final_model, walk_forward_validate

SYMBOL = "BTC/USDT"


class StubModel:
    """Model that returns a fixed win probability for every setup."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack([1 - p, p])


def replay(settings, bars, model, risk=None):
    """Run the real engine over historical bars with a paper broker."""
    risk = risk or RiskManager(settings.risk)
    broker = PaperBroker(settings.backtest.initial_equity, settings.costs)
    feed = ReplayFeed({SYMBOL: bars}, warmup=settings.execution.warmup_bars)
    engine = TradingEngine(settings, feed, broker, risk, model)
    asyncio.run(engine.run(install_signal_handlers=False))
    return engine, broker, risk


def with_risk(settings, **overrides):
    return settings.model_copy(update={"risk": settings.risk.model_copy(update=overrides)})


@pytest.fixture(scope="module")
def replay_bars():
    return generate_synthetic_ohlcv(n_bars=1600, seed=21)


# ------------------------------------------------------------------ research
def test_research_pipeline_end_to_end(settings, tmp_path):
    bars = generate_synthetic_ohlcv(n_bars=12_000, seed=5)
    dataset = build_dataset({SYMBOL: bars}, settings)
    result = walk_forward_validate(dataset, settings)

    assert len(result.fold_metrics) >= 1
    for key in ("precision", "recall", "log_loss"):
        assert key in result.overall
    # Out-of-sample rows only: nothing from the initial training block.
    assert result.predictions["time"].min() > dataset["time"].min()

    backtest = run_backtest(result.predictions, settings, threshold=0.3)
    assert backtest.metrics["trades"] > 0
    assert backtest.metrics["total_fees"] > 0
    assert backtest.metrics["net_pnl"] < backtest.metrics["gross_pnl"]

    # Model persistence and the train/serve guard.
    local = settings.model_copy(
        update={"model": settings.model.model_copy(update={"weights_dir": str(tmp_path)})}
    )
    save_model(train_final_model(dataset, local), local, n_samples=len(dataset))
    assert load_model(local) is not None

    changed = local.model_copy(
        update={"features": local.features.model_copy(update={"impulse_window": 20})}
    )
    with pytest.raises(RuntimeError):
        load_model(changed)  # trained on a 12-bar impulse, now configured for 20


# -------------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def always_trade(settings, replay_bars):
    """One shared replay where the model approves every setup."""
    return replay(settings, replay_bars, StubModel(1.0))


def test_engine_trades_and_keeps_state_consistent(always_trade):
    engine, broker, _ = always_trade

    entries = [f for f in broker.fills if f["type"] == "stop_limit"]
    exits = [f for f in broker.fills if f["type"] in ("stop", "limit", "market")]
    assert len(entries) > 3
    assert len(entries) - len(exits) in (0, 1)  # at most one position still open at the end

    state = engine._state[SYMBOL]
    position = broker.positions[SYMBOL]
    if state.phase is Phase.OPEN:
        # An open position is always protected by a resting stop.
        assert position == pytest.approx(state.side * state.qty)
        assert state.stop_order is not None and state.stop_order in broker.open_orders(SYMBOL)
    else:
        assert position == 0.0
        assert broker.open_orders(SYMBOL) == []  # pending entries are cancelled on shutdown


def test_entries_are_stop_limit_orders_beyond_the_consolidation(settings, always_trade):
    _, broker, _ = always_trade
    entry_orders = [o for o in broker.orders.values() if o.type == "stop_limit"]
    assert entry_orders
    assert not [o for o in broker.orders.values() if o.type == "market" and not o.reduce_only]
    for order in entry_orders:
        assert order.side == "buy"  # allow_short is false
        assert order.price > order.trigger_price  # limit sits above the trigger
    # Trigger/limit spacing equals limit_offset_atr * ATR, as defined in research.
    trigger, limit = entry_levels(1, 100.0, 99.0, 1.0, settings.labeling)
    assert limit - trigger == pytest.approx(settings.labeling.limit_offset_atr)


def test_every_position_respects_the_time_barrier(settings, always_trade):
    _, broker, _ = always_trade
    bar = settings.bar_timedelta
    open_time = None
    for fill in broker.fills:
        if fill["type"] == "stop_limit":
            open_time = fill["time"]
        elif open_time is not None:
            held_bars = (fill["time"] - open_time) / bar + 1
            assert held_bars <= settings.labeling.max_holding_bars
            open_time = None


def test_model_veto_means_no_orders(settings, replay_bars):
    _, broker, _ = replay(settings, replay_bars, StubModel(0.0))
    assert broker.orders == {}
    assert broker.cash == settings.backtest.initial_equity


def test_position_notional_never_exceeds_the_cap(settings, always_trade):
    _, broker, _ = always_trade
    cap = settings.risk.max_position_pct
    for fill in (f for f in broker.fills if f["type"] == "stop_limit"):
        # Allow headroom for equity moving between sizing and fill.
        assert fill["qty"] * fill["price"] <= cap * settings.backtest.initial_equity * 1.25


def test_drawdown_kill_switch_flattens_and_stops_trading(settings, replay_bars):
    # A limit so tight that the first trade's costs breach it.
    tight = with_risk(settings, max_daily_drawdown=0.0002, auto_resume_next_day=False)
    engine, broker, risk = replay(tight, replay_bars, StubModel(1.0))

    assert risk.halt_reason is HaltReason.DAILY_DRAWDOWN
    assert broker.positions[SYMBOL] == 0.0  # flattened
    assert broker.open_orders() == []  # nothing left resting
    assert engine._state[SYMBOL].phase is Phase.FLAT
    entries = [f for f in broker.fills if f["type"] == "stop_limit"]
    assert len(entries) == 1  # no new entries after the halt


def test_engine_places_no_orders_while_halted(settings, replay_bars):
    risk = RiskManager(with_risk(settings, max_consecutive_api_errors=1).risk)
    risk.record_api_error()  # simulate the exchange connection failing
    assert risk.halt_reason is HaltReason.API_ERRORS
    _, broker, _ = replay(settings, replay_bars, StubModel(1.0), risk=risk)
    assert broker.orders == {}
