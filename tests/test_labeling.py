"""Triple-barrier labeling tests on hand-built price paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import LabelingConfig
from research.dataset import build_symbol_dataset
from research.labeling import (
    apply_triple_barrier,
    average_uniqueness,
    barrier_levels,
    entry_levels,
    simulate_trade,
)

CFG = LabelingConfig(
    profit_target_atr=2.0,
    stop_loss_atr=1.0,
    max_holding_bars=4,
    entry_buffer_atr=0.0,
    limit_offset_atr=0.5,
    entry_valid_bars=2,
)
# Long setup at bar 0 with ATR = 1: trigger 100, limit 100.5, stop 99, target 102.
SETUP_BAR = (99.5, 99.9, 99.0, 99.8)  # (open, high, low, close)
QUIET_BAR = (100.2, 100.4, 100.0, 100.2)


def run(bars, side=1, trigger=100.0, limit=100.5):
    o, h, low, c = (np.array(col, dtype=float) for col in zip(*bars, strict=True))
    return simulate_trade(o, h, low, c, 0, side, trigger, limit, 1.0, CFG)


def test_levels():
    assert entry_levels(1, 100.0, 98.0, 1.0, CFG) == (100.0, 100.5)
    assert entry_levels(-1, 100.0, 98.0, 1.0, CFG) == (98.0, 97.5)
    assert barrier_levels(1, 100.0, 1.0, CFG) == (99.0, 102.0)
    assert barrier_levels(-1, 100.0, 1.0, CFG) == (101.0, 98.0)


def test_profit_target_hit_first():
    path = run([SETUP_BAR, (99.8, 100.4, 99.7, 100.3), (100.3, 102.5, 100.2, 102.2)])
    assert (path.outcome, path.entry_pos, path.entry_price) == ("pt", 1, 100.0)
    assert (path.exit_pos, path.exit_price) == (2, 102.0)


def test_stop_loss_hit_first():
    path = run([SETUP_BAR, (99.8, 100.4, 99.7, 100.3), (100.3, 100.5, 98.9, 99.0)])
    assert (path.outcome, path.exit_price) == ("sl", 99.0)


def test_both_barriers_in_one_bar_is_scored_as_a_loss():
    # Bar 2 touches 102.5 AND 98.9. We cannot know the order, so assume the worst.
    path = run([SETUP_BAR, (99.8, 100.4, 99.7, 100.3), (100.3, 102.5, 98.9, 101.0)])
    assert path.outcome == "sl"


def test_entry_bar_wick_before_the_fill_is_not_a_stop_out():
    # Bar 1 dips to 98.5 (below the 99 stop) and then breaks out through 100.
    # The dip happened before we were filled, so the trade is still alive.
    path = run([SETUP_BAR, (99.8, 100.4, 98.5, 100.3), (100.3, 102.5, 100.2, 102.2)])
    assert (path.outcome, path.entry_pos, path.exit_pos) == ("pt", 1, 2)


def test_entry_bar_close_through_stop_is_a_stop_out():
    # Bar 1 breaks out through 100 and then closes at 98.8: stopped after the fill.
    path = run([SETUP_BAR, (99.8, 100.4, 98.5, 98.8), QUIET_BAR])
    assert (path.outcome, path.exit_pos, path.exit_price) == ("sl", 1, 99.0)


def test_entry_at_the_open_uses_the_full_bar_range():
    # Bar 1 gaps to 100.2 (inside the limit), so we hold for the whole bar.
    # Stop is 99.2; the low of 99.0 is a real stop-out even though it closes higher.
    path = run([SETUP_BAR, (100.2, 100.4, 99.0, 100.1), QUIET_BAR])
    assert (path.outcome, path.entry_price, path.exit_pos) == ("sl", 100.2, 1)


def test_gap_through_stop_fills_at_the_worse_open():
    path = run([SETUP_BAR, (99.8, 100.4, 99.7, 100.3), (98.5, 98.8, 98.0, 98.2)])
    assert (path.outcome, path.exit_price) == ("sl", 98.5)


def test_time_barrier_exits_at_close():
    path = run([SETUP_BAR, (99.8, 100.4, 99.7, 100.3)] + [QUIET_BAR] * 5)
    # Entry on bar 1, max_holding_bars=4 -> exit at the close of bar 4.
    assert (path.outcome, path.exit_pos, path.exit_price) == ("time", 4, 100.2)


def test_no_fill_when_trigger_is_never_reached():
    below = (99.5, 99.9, 99.2, 99.6)
    path = run([SETUP_BAR, below, below, below])
    assert path.outcome == "no_fill"
    assert path.release_pos == 2  # order expires after entry_valid_bars


def test_gap_beyond_limit_is_not_chased():
    # Bar 1 opens above the limit (no fill); bar 2 opens back inside it (fill at open).
    path = run(
        [SETUP_BAR, (101.0, 101.2, 100.8, 101.0), (100.2, 100.4, 100.0, 100.2)] + [QUIET_BAR] * 4
    )
    assert (path.entry_pos, path.entry_price) == (2, 100.2)


def test_unresolved_trade_at_end_of_data_is_incomplete():
    path = run([SETUP_BAR, (99.8, 100.4, 99.7, 100.3), QUIET_BAR])
    assert path.outcome == "incomplete"


def test_short_side_is_a_mirror_image():
    setup = (100.5, 101.0, 100.1, 100.2)
    # Short: trigger 100, limit 99.5, stop 101, target 98.
    win = run([setup, (100.2, 100.3, 99.6, 99.7), (99.7, 99.8, 97.5, 97.8)], side=-1, limit=99.5)
    assert (win.outcome, win.entry_price, win.exit_price) == ("pt", 100.0, 98.0)
    loss = run([setup, (100.2, 100.3, 99.6, 99.7), (99.7, 101.2, 99.6, 101.0)], side=-1, limit=99.5)
    assert (loss.outcome, loss.exit_price) == ("sl", 101.0)


def test_labels_ignore_the_setup_bar_and_everything_before_it():
    """The label for a setup at bar t must depend only on bars after t."""
    bars = [SETUP_BAR, (99.8, 100.4, 99.7, 100.3), (100.3, 102.5, 100.2, 102.2)]
    baseline = run(bars)
    tampered = run([(1.0, 1000.0, 0.5, 500.0), *bars[1:]])
    assert baseline == tampered


def test_apply_triple_barrier_frame():
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    rows = [SETUP_BAR, (99.8, 100.4, 99.7, 100.3), (100.3, 102.5, 100.2, 102.2)]
    bars = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    events = pd.DataFrame(
        {"side": [1], "atr": [1.0], "cons_high": [100.0], "cons_low": [99.0]}, index=index[:1]
    )
    labels = apply_triple_barrier(bars, events, CFG)
    row = labels.iloc[0]
    assert row["label"] == 1.0
    assert row["r_multiple"] == pytest.approx(2.0)
    assert row["entry_time"] == index[1]
    assert row["t1"] == index[2]
    # The outcome is known strictly after the setup bar.
    assert row["t1"] > labels.index[0]


def test_average_uniqueness():
    # Label A spans bars 0-1, label B spans bars 1-2, label C spans bar 5 alone.
    weights = average_uniqueness(np.array([0, 1, 5]), np.array([1, 2, 5]), n_bars=10)
    np.testing.assert_allclose(weights, [0.75, 0.75, 1.0])


def test_dataset_labels_resolve_after_their_setup(bars, settings):
    dataset = build_symbol_dataset("TEST/USDT", bars, settings)
    filled = dataset[dataset["label"].notna()]
    assert len(filled) > 20
    assert (filled["entry_time"] > filled["time"]).all()
    assert (filled["t1"] >= filled["entry_time"]).all()
    assert set(filled["label"].unique()) <= {0.0, 1.0}
    assert (filled["weight"] > 0).all() and (filled["weight"] <= 1).all()
