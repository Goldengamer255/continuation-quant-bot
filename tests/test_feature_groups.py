"""Tests for the optional feature groups and the volatility-regime setup filter."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from data.synthetic import generate_synthetic_ohlcv
from execution.broker import PaperBroker, ReplayFeed
from execution.engine import TradingEngine
from execution.risk_manager import RiskManager
from research.dataset import build_dataset
from research.features import (
    MODEL_FEATURES,
    build_feature_matrix,
    compute_features,
    detect_setups,
    feature_columns,
)


@pytest.fixture(scope="module")
def flow_bars(bars):
    """Synthetic bars with the two extra order-flow columns."""
    rng = np.random.default_rng(1)
    out = bars.copy()
    out["taker_buy_volume"] = out["volume"] * rng.uniform(0.3, 0.7, len(out))
    out["trades"] = rng.integers(50, 500, len(out)).astype(float)
    return out


@pytest.fixture(scope="module")
def context_bars(bars):
    return generate_synthetic_ohlcv(n_bars=len(bars), seed=99)


@pytest.fixture(scope="module")
def all_groups(settings):
    return settings.features.model_copy(
        update={"use_order_flow": True, "use_market_context": True, "use_regime": True}
    )


def test_feature_columns_follow_the_flags(settings, all_groups):
    assert feature_columns(settings.features) == list(MODEL_FEATURES)
    cols = feature_columns(all_groups)
    assert cols[: len(MODEL_FEATURES)] == list(MODEL_FEATURES)
    assert {"cons_flow_imbalance", "rel_strength", "vol_regime"} <= set(cols)
    assert len(cols) == len(set(cols))


def test_optional_groups_have_no_lookahead(flow_bars, context_bars, all_groups):
    """Same truncation test as the base features, with every optional group on.

    The context series is truncated too: at time t only context bars <= t exist.
    """
    full = compute_features(flow_bars, all_groups, context=context_bars)
    for cut in (1200, 2500):
        part = compute_features(flow_bars.iloc[:cut], all_groups, context=context_bars.iloc[:cut])
        pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-12)


def test_future_context_cannot_change_past_features(flow_bars, context_bars, all_groups):
    cut = 2000
    tampered = context_bars.copy()
    tampered.iloc[cut:, :4] *= 5.0
    a = compute_features(flow_bars, all_groups, context=context_bars).iloc[:cut]
    b = compute_features(flow_bars, all_groups, context=tampered).iloc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)


def test_matrix_columns_match_feature_columns(flow_bars, context_bars, all_groups, settings):
    features = compute_features(flow_bars, all_groups, context=context_bars)
    events = detect_setups(features, settings.setup)
    x = build_feature_matrix(features, events)
    assert list(x.columns) == feature_columns(all_groups)
    assert not x.isna().any().any()


def test_flow_imbalance_is_bounded_and_signed(flow_bars, all_groups, context_bars):
    features = compute_features(flow_bars, all_groups, context=context_bars)
    imbalance = features["cons_flow_imbalance"].dropna()
    assert imbalance.between(-1, 1).all()

    all_buys = flow_bars.copy()
    all_buys["taker_buy_volume"] = all_buys["volume"]
    one_sided = compute_features(all_buys, all_groups, context=context_bars)
    np.testing.assert_allclose(one_sided["cons_flow_imbalance"].dropna(), 1.0)


def test_missing_context_is_an_error(flow_bars, all_groups):
    with pytest.raises(ValueError):
        compute_features(flow_bars, all_groups)


def test_dataset_requires_the_context_symbol(flow_bars, settings, all_groups):
    with_context = settings.model_copy(update={"features": all_groups})
    with pytest.raises(ValueError):
        build_dataset({"ETH/USDT": flow_bars}, with_context)  # BTC/USDT bars not supplied


def test_volatility_filter_only_removes_setups(bars, settings):
    features = compute_features(bars, settings.features)
    everything = detect_setups(features, settings.setup)
    atr_pct = (features["atr"] / features["close"]).loc[everything.index]
    cap = float(atr_pct.median())

    filtered = detect_setups(features, settings.setup.model_copy(update={"max_atr_pct": cap}))
    assert 0 < len(filtered) < len(everything)
    assert filtered.index.isin(everything.index).all()
    assert (features["atr"] / features["close"]).loc[filtered.index].max() <= cap
    # Surviving setups are unchanged, not recomputed.
    pd.testing.assert_frame_equal(filtered, everything.loc[filtered.index])


class _AlwaysYes:
    def predict_proba(self, x):
        return np.column_stack([np.zeros(len(x)), np.ones(len(x))])


def test_engine_refuses_order_flow_features_without_a_flow_feed(settings, bars, all_groups):
    """A model trained on order-flow inputs must not run on a feed that lacks them."""
    with_flow = settings.model_copy(update={"features": all_groups})
    broker = PaperBroker(10_000.0, settings.costs)
    feed = ReplayFeed({"BTC/USDT": bars}, warmup=600)  # plain OHLCV only
    assert not feed.supports_flow
    with pytest.raises(NotImplementedError):
        TradingEngine(with_flow, feed, broker, RiskManager(settings.risk), model=None)


def test_engine_trades_with_all_feature_groups(settings, flow_bars, context_bars, all_groups):
    """Full replay with order flow, market context and regime features on.

    BTC/USDT is the context symbol and is NOT traded here, so it is streamed
    as a data-only symbol.
    """
    rng = np.random.default_rng(2)
    context = context_bars.copy()
    context["taker_buy_volume"] = context["volume"] * rng.uniform(0.3, 0.7, len(context))
    context["trades"] = 100.0
    cfg = settings.model_copy(
        update={
            "features": all_groups,
            "market": settings.market.model_copy(update={"symbols": ["ETH/USDT"]}),
        }
    )
    feed = ReplayFeed({"ETH/USDT": flow_bars, "BTC/USDT": context}, warmup=1500)
    assert feed.supports_flow
    broker = PaperBroker(10_000.0, cfg.costs)
    engine = TradingEngine(cfg, feed, broker, RiskManager(cfg.risk), _AlwaysYes())
    asyncio.run(engine.run(install_signal_handlers=False))

    assert [f for f in broker.fills if f["type"] == "stop_limit"], "expected at least one entry"
    assert {f["symbol"] for f in broker.fills} == {"ETH/USDT"}  # context symbol is never traded
    # Both buffers advanced to the end and carry the order-flow columns.
    assert engine._buffers["ETH/USDT"].index[-1] == flow_bars.index[-1]
    assert engine._buffers["BTC/USDT"].index[-1] == context.index[-1]
    assert "taker_buy_volume" in engine._buffers["ETH/USDT"].columns


def test_replay_feed_delivers_bars_in_time_order(bars):
    """Two symbols must interleave by timestamp, not race ahead of each other."""
    feed = ReplayFeed({"A": bars.iloc[:60], "B": bars.iloc[:60]}, warmup=10)
    seen: list[tuple[pd.Timestamp, str]] = []

    async def consume(symbol, slow):
        async for bar in feed.stream_closed_bars(symbol, "1h"):
            for _ in range(slow):  # one consumer is much slower than the other
                await asyncio.sleep(0)
            seen.append((bar.timestamp, symbol))

    async def main():
        await asyncio.gather(consume("A", 0), consume("B", 7))

    asyncio.run(main())
    stamps = [ts for ts, _ in seen]
    assert stamps == sorted(stamps)
    assert len(seen) == 100


def test_score_latest_bar_matches_research(settings, flow_bars, context_bars, all_groups):
    """The engine's live scoring path must produce the same feature row as research."""
    cfg = settings.model_copy(update={"features": all_groups})
    feats = compute_features(flow_bars, all_groups, context=context_bars)
    events = detect_setups(feats, cfg.setup)
    last_setup = events.index[-1]
    upto = flow_bars.loc[:last_setup]

    captured = {}

    class Capture:
        def predict_proba(self, x):
            captured["x"] = x
            return np.array([[0.3, 0.7]])

    feed = ReplayFeed(
        {"ETH/USDT": flow_bars, "BTC/USDT": context_bars.assign(taker_buy_volume=1.0, trades=1.0)},
        warmup=10,
    )
    cfg = cfg.model_copy(update={"market": cfg.market.model_copy(update={"symbols": ["ETH/USDT"]})})
    engine = TradingEngine(
        cfg, feed, PaperBroker(10_000.0, cfg.costs), RiskManager(cfg.risk), Capture()
    )
    engine._buffers = {
        "ETH/USDT": upto[engine._columns],
        "BTC/USDT": context_bars.loc[:last_setup].assign(taker_buy_volume=1.0, trades=1.0),
    }
    event, probability = engine._score_latest_bar("ETH/USDT")
    assert probability == 0.7
    expected = build_feature_matrix(feats, events.loc[[last_setup]])
    pd.testing.assert_frame_equal(captured["x"], expected, check_exact=False, rtol=1e-9)
