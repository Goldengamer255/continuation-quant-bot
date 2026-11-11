"""Leakage tests for the validation scheme, plus a placebo test on noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic import generate_synthetic_ohlcv
from research.dataset import build_dataset
from research.train import PurgedWalkForwardSplit, calibration_table, walk_forward_validate

HOUR = pd.Timedelta(hours=1)


def make_samples(n=200, label_span_bars=10):
    times = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))
    return times, times + label_span_bars * HOUR


def test_train_always_precedes_test_and_labels_are_purged():
    times, t1 = make_samples()
    embargo = 5 * HOUR
    folds = list(PurgedWalkForwardSplit(4, embargo, 0.4).split(times, t1))
    assert len(folds) == 4
    for fold in folds:
        test_start = times.iloc[fold.test_idx].min()
        # 1. Chronology: every training setup is earlier than every test setup.
        assert times.iloc[fold.train_idx].max() < test_start
        # 2. Purge + embargo: every training LABEL resolved before the gap.
        assert (t1.iloc[fold.train_idx] + embargo < test_start).all()
        # 3. No sample is in both sets.
        assert not set(fold.train_idx) & set(fold.test_idx)


def test_purge_removes_exactly_the_overlapping_samples():
    times, t1 = make_samples(n=100, label_span_bars=10)
    fold = next(PurgedWalkForwardSplit(1, 0 * HOUR, 0.5).split(times, t1))
    # Test starts at sample 50. Samples 40..49 have labels ending at 50..59,
    # i.e. inside the test period, so they must be dropped. Sample 39 ends at 49.
    assert fold.train_idx.max() == 39
    with_embargo = next(PurgedWalkForwardSplit(1, 5 * HOUR, 0.5).split(times, t1))
    assert with_embargo.train_idx.max() == 34


def test_folds_are_consecutive_and_cover_the_out_of_sample_region_once():
    times, t1 = make_samples()
    folds = list(PurgedWalkForwardSplit(4, HOUR, 0.4).split(times, t1))
    all_test = np.concatenate([f.test_idx for f in folds])
    assert sorted(all_test) == list(range(80, 200))
    # Expanding window: each fold trains on more history than the last.
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)


def test_unlabeled_rows_never_enter_training():
    times, t1 = make_samples()
    t1.iloc[::3] = pd.NaT  # setups whose entry order never filled
    for fold in PurgedWalkForwardSplit(3, HOUR, 0.4).split(times, t1):
        assert t1.iloc[fold.train_idx].notna().all()


def test_same_timestamp_setups_stay_in_the_same_fold():
    times, t1 = make_samples(n=100)
    times = pd.concat([times, times]).sort_values(kind="stable").reset_index(drop=True)
    t1 = times + 10 * HOUR
    for fold in PurgedWalkForwardSplit(3, HOUR, 0.4).split(times, t1):
        assert not set(times.iloc[fold.train_idx]) & set(times.iloc[fold.test_idx])


def test_unsorted_input_is_rejected():
    times, t1 = make_samples()
    with pytest.raises(ValueError):
        list(PurgedWalkForwardSplit(2, HOUR).split(times.iloc[::-1], t1.iloc[::-1]))


@pytest.fixture(scope="module")
def noise_result(settings):
    """Run the full research pipeline on a pure random walk (no bursts)."""
    bars = generate_synthetic_ohlcv(n_bars=20_000, seed=3, burst_prob=0.0, vol_clustering=False)
    dataset = build_dataset({"NOISE/USDT": bars}, settings)
    return walk_forward_validate(dataset, settings)


def test_placebo_no_skill_on_a_random_walk(noise_result):
    """The most important test in the repo.

    A random walk has no predictable direction. If the pipeline reported a
    strong out-of-sample edge here, it could only come from leaked future
    information. A leak-free pipeline must score close to a coin flip (AUC
    0.5) and must not beat the no-skill baseline by a meaningful margin.
    """
    overall = noise_result.overall
    assert overall["n"] > 100
    assert 0.40 < overall["auc"] < 0.60
    assert overall["log_loss"] > overall["baseline_log_loss"] - 0.02


def test_predictions_exist_only_for_out_of_sample_rows(noise_result):
    preds = noise_result.predictions
    assert preds["prob"].between(0, 1).all()
    assert (preds["fold"] >= 1).all()
    assert not calibration_table(preds).empty
    # Folds are chronological.
    assert preds.groupby("fold")["time"].min().is_monotonic_increasing
