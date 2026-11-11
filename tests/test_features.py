"""Look-ahead (data leakage) tests for the feature pipeline.

The core idea: if a feature at bar t only uses data up to t, then deleting or
changing everything *after* t cannot change it. These tests do exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.features import (
    MODEL_FEATURES,
    build_feature_matrix,
    compute_features,
    detect_setups,
)


def test_features_do_not_change_when_the_future_is_removed(bars, settings):
    full = compute_features(bars, settings.features)
    for cut in (800, 1500, 2600):
        truncated = compute_features(bars.iloc[:cut], settings.features)
        pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_exact=False, rtol=1e-12)


def test_features_do_not_change_when_the_future_is_altered(bars, settings):
    cut = 2000
    tampered = bars.copy()
    # Multiply every future price by 10: any forward-looking feature would react.
    tampered.iloc[cut:, :4] *= 10.0
    original = compute_features(bars, settings.features).iloc[:cut]
    altered = compute_features(tampered, settings.features).iloc[:cut]
    pd.testing.assert_frame_equal(original, altered, check_exact=False, rtol=1e-12)


def test_setups_are_identical_on_truncated_history(bars, settings):
    cut = 2200
    full = detect_setups(compute_features(bars, settings.features), settings.setup)
    part = detect_setups(compute_features(bars.iloc[:cut], settings.features), settings.setup)
    expected = full[full.index < bars.index[cut]]
    pd.testing.assert_frame_equal(expected, part, check_exact=False, rtol=1e-12)


def test_live_window_matches_full_history(bars, settings):
    """Train/serve skew check.

    The live engine computes features on a sliding buffer of recent bars, not
    on the full history. Recursive indicators (EMA/ATR) depend on where they
    start, so the buffer must be long enough for that start-up effect to
    vanish. This asserts the configured warm-up is sufficient.
    """
    warmup = settings.execution.warmup_bars
    full = compute_features(bars, settings.features)
    window = compute_features(bars.iloc[-warmup:], settings.features)
    cols = [c for c in MODEL_FEATURES if c in full.columns] + ["atr"]
    np.testing.assert_allclose(
        window[cols].iloc[-1].to_numpy(), full[cols].iloc[-1].to_numpy(), rtol=1e-3
    )


def test_relative_volume_baseline_excludes_current_bar(settings):
    """A lone volume spike must register at full size, not be averaged into its own baseline."""
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    flat = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}, index=index
    )
    flat.iloc[150, flat.columns.get_loc("volume")] = 50.0
    features = compute_features(flat, settings.features)
    assert features["rel_volume"].iloc[150] == 5.0


def test_feature_matrix_shape_and_direction_normalisation(bars, settings):
    features = compute_features(bars, settings.features)
    events = detect_setups(features, settings.setup)
    assert len(events) > 0, "synthetic data should contain setups"

    x = build_feature_matrix(features, events)
    assert list(x.columns) == list(MODEL_FEATURES)
    assert not x.isna().any().any()
    # Every setup is defined by an impulse in the trade direction, so after
    # multiplying by side the impulse must always be positive.
    assert (x["impulse_atr"] >= settings.setup.min_impulse_atr).all()
