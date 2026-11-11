"""Assemble the modelling dataset: bars -> features -> setups -> labels -> weights.

This is the single place where the research stages are chained, so
``train.py`` and the tests are guaranteed to build data the same way.

Multiple symbols are *pooled* into one dataset (one row per setup, with a
``symbol`` column). Pooling gives the model more examples of the same
pattern; because every feature is normalised by ATR or expressed as a ratio,
a setup on BTC and one on ETH are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import Settings
from research.features import build_feature_matrix, compute_features, detect_setups
from research.labeling import FILLED_OUTCOMES, apply_triple_barrier, average_uniqueness


def build_symbol_dataset(
    symbol: str, bars: pd.DataFrame, settings: Settings, context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the labelled setup table for one symbol."""
    features = compute_features(bars, settings.features, context=context)
    events = detect_setups(features, settings.setup)
    labels = apply_triple_barrier(bars, events, settings.labeling)
    if labels.empty:
        return pd.DataFrame()

    x = build_feature_matrix(features, events.loc[labels.index])
    dataset = labels.join(x)

    # Sample weights only for rows that became trades (and so have a label).
    filled = dataset["outcome"].isin(FILLED_OUTCOMES).to_numpy()
    weights = np.zeros(len(dataset))
    if filled.any():
        weights[filled] = average_uniqueness(
            dataset.loc[filled, "event_pos"].to_numpy(),
            dataset.loc[filled, "release_pos"].to_numpy(),
            n_bars=len(bars),
        )
    dataset["weight"] = weights

    dataset = dataset.reset_index()
    dataset.insert(0, "symbol", symbol)
    return dataset


def build_dataset(bars_by_symbol: dict[str, pd.DataFrame], settings: Settings) -> pd.DataFrame:
    """Pool all symbols into one chronologically sorted dataset."""
    context = None
    if settings.features.use_market_context:
        if settings.features.context_symbol not in bars_by_symbol:
            raise ValueError(f"context symbol {settings.features.context_symbol} has no bars")
        context = bars_by_symbol[settings.features.context_symbol]
    parts = [
        build_symbol_dataset(sym, bars, settings, context) for sym, bars in bars_by_symbol.items()
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise ValueError("No setups found. Loosen the `setup` thresholds or add more history.")
    dataset = pd.concat(parts, ignore_index=True)
    # Stable sort keeps a deterministic order for setups sharing a timestamp.
    return dataset.sort_values("time", kind="stable").reset_index(drop=True)
