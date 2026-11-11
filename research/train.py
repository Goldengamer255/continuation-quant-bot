"""Model training with purged walk-forward validation (LightGBM).

Concept primer
--------------
**Why not ordinary k-fold cross-validation?** K-fold shuffles rows into
folds, so the model trains on 2024 data and is "tested" on 2023 data. In
markets that is cheating twice over: you never get to train on the future,
and neighbouring rows are near-duplicates (overlapping windows), so a
shuffled test row has close relatives in the training set.

**Walk-forward validation** imitates real life. Train on everything up to
date D, test on the period after D, then roll D forward and repeat::

    fold 1: [==== train ====]..gap..[test]
    fold 2: [======= train =======]..gap..[test]
    fold 3: [========== train ==========]..gap..[test]

Every prediction is made by a model that had only seen the past. Stitching
the test predictions together gives an **out-of-sample (OOS)** track record
covering most of the history, which is what the backtester consumes.

**Purging.** A training sample's *label* is not known at its setup time; it
is known at ``t1``, when a barrier was hit, up to ``max_holding_bars`` later.
A sample that starts before the test period but whose trade ends inside it
has a label built from test-period prices. Training on it leaks the test
period into the model. Purging removes every training sample whose label
window ``[time, t1]`` reaches into the test period.

**Embargo.** An extra safety gap of ``embargo_bars`` between the end of
training labels and the start of testing. Features are built on rolling
windows and market states are serially correlated, so samples immediately
either side of the boundary are highly similar; the embargo keeps the test
set from being graded on near-copies of training rows.

The rule implemented: a sample may be used for training fold *k* only if
``t1 + embargo < test_start``.

Model choices
-------------
* **LightGBM** (gradient-boosted decision trees): the standard for small,
  noisy tabular data. Handles non-linear interactions (e.g. "tight range
  *and* volume decay"), needs no feature scaling, and is fast.
* **Deliberately weak learner**: shallow trees, few leaves, large
  ``min_child_samples``, strong L2. Financial data has a very low
  signal-to-noise ratio; a flexible model memorises noise.
* **No class re-weighting.** The base win rate is ~1/3. Re-balancing classes
  would inflate predicted probabilities, and we *trade on* a probability
  threshold — so the probabilities must be honest (calibrated). The
  calibration table printed after training checks this directly.
* **Sample weights = label uniqueness** (see ``labeling.py``).
* **Metrics.** *Precision* (of the trades we take, how many win) is what
  drives PnL. *Recall* (share of winners we caught) measures opportunity
  cost. *Log-loss* scores the probabilities themselves and is compared with
  a no-skill baseline that always predicts the training win rate: a model
  that cannot beat that baseline has learned nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score

from core.config import Settings, load_settings, resolve_path
from research.dataset import build_dataset
from research.features import feature_columns

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pkl"
META_FILE = "model_meta.json"
PREDICTIONS_FILE = "oos_predictions.parquet"
FOLD_METRICS_FILE = "fold_metrics.csv"


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    number: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward splitter with purging and an embargo.

    The first ``initial_train_fraction`` of samples is training-only. The
    remainder is cut into ``n_splits`` consecutive test blocks. For each test
    block the training set is every *earlier* sample whose label resolved at
    least ``embargo`` before the block starts.
    """

    def __init__(
        self, n_splits: int, embargo: pd.Timedelta, initial_train_fraction: float = 0.4
    ) -> None:
        self.n_splits = n_splits
        self.embargo = embargo
        self.initial_train_fraction = initial_train_fraction

    def split(self, times: pd.Series, t1: pd.Series) -> Iterator[Fold]:
        """Yield folds. ``times`` must be sorted ascending; ``t1`` is label end time.

        Rows with ``t1 = NaT`` (setups that never became trades) are never
        placed in a training set, because they have no label.
        """
        times = pd.Series(times).reset_index(drop=True)
        t1 = pd.Series(t1).reset_index(drop=True)
        if not times.is_monotonic_increasing:
            raise ValueError("samples must be sorted by time before splitting")

        n = len(times)
        first_test = int(n * self.initial_train_fraction)
        edges = np.linspace(first_test, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            start, stop = edges[k], edges[k + 1]
            if start >= stop:
                continue
            test_start = times.iloc[start]
            # Boundaries are defined on *timestamps*, so setups that share a
            # timestamp (different symbols) always land in the same fold.
            in_test = times >= test_start
            if stop < n:
                in_test &= times < times.iloc[stop]

            # Purge + embargo. NaT comparisons are False, which drops unlabeled rows.
            in_train = (times < test_start) & (t1 + self.embargo < test_start)

            yield Fold(
                number=k + 1,
                train_idx=np.flatnonzero(in_train.to_numpy()),
                test_idx=np.flatnonzero(in_test.to_numpy()),
                test_start=test_start,
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConstantModel:
    """Fallback 'model' that always predicts the training base rate.

    Used when a training fold contains a single class (nothing to learn).
    """

    def __init__(self, positive_rate: float) -> None:
        self.positive_rate = positive_rate

    def predict_proba(self, x) -> np.ndarray:
        p = np.full(len(x), self.positive_rate)
        return np.column_stack([1.0 - p, p])


def fit_model(x: pd.DataFrame, y: pd.Series, weights: pd.Series, params: dict):
    """Fit a LightGBM classifier (or the constant fallback)."""
    if y.nunique() < 2:
        return ConstantModel(float(y.mean()))
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(objective="binary", verbose=-1, **params)
    model.fit(x, y, sample_weight=weights)
    return model


def classification_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float, baseline_rate: float
) -> dict[str, float]:
    """Precision / recall at the trading threshold, plus probability quality."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    pred = (prob > threshold).astype(int)
    baseline = np.full(len(y_true), np.clip(baseline_rate, 1e-6, 1 - 1e-6))
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": len(y_true),
        "win_rate": float(y_true.mean()),
        "signals": int(pred.sum()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(y_true, baseline, labels=[0, 1])),
        "auc": float(roc_auc_score(y_true, prob)) if both_classes else float("nan"),
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # dataset rows from all test folds + `prob`, `fold`
    fold_metrics: pd.DataFrame
    overall: dict[str, float]


def walk_forward_validate(dataset: pd.DataFrame, settings: Settings) -> WalkForwardResult:
    """Run purged walk-forward validation and collect out-of-sample probabilities.

    Every setup in a test fold gets a probability — including setups whose
    entry order never filled — because at decision time the model cannot know
    whether the order will fill. Metrics are computed on filled trades only.
    """
    cfg = settings.model
    ds = dataset.reset_index(drop=True)
    features = feature_columns(settings.features)
    has_label = ds["label"].notna().to_numpy()

    splitter = PurgedWalkForwardSplit(
        cfg.n_splits, cfg.embargo_bars * settings.bar_timedelta, cfg.initial_train_fraction
    )
    prob = np.full(len(ds), np.nan)
    fold_id = np.zeros(len(ds), dtype=int)
    rows = []

    for fold in splitter.split(ds["time"], ds["t1"]):
        train = fold.train_idx
        if len(train) < cfg.min_train_samples:
            logger.warning("fold %d skipped: only %d training samples", fold.number, len(train))
            continue
        y_train = ds.loc[train, "label"].astype(int)
        model = fit_model(
            ds.loc[train, features], y_train, ds.loc[train, "weight"], cfg.lgbm_params
        )

        test = fold.test_idx
        prob[test] = model.predict_proba(ds.loc[test, features])[:, 1]
        fold_id[test] = fold.number

        scored = test[has_label[test]]
        if len(scored):
            metrics = classification_metrics(
                ds.loc[scored, "label"].to_numpy(),
                prob[scored],
                cfg.probability_threshold,
                baseline_rate=float(y_train.mean()),
            )
            rows.append({"fold": fold.number, "test_start": fold.test_start, "n_train": len(train)}
                        | metrics)  # fmt: skip

    predictions = ds.assign(prob=prob, fold=fold_id)
    predictions = predictions[predictions["prob"].notna()].reset_index(drop=True)
    fold_metrics = pd.DataFrame(rows)

    scored = predictions[predictions["label"].notna()]
    overall: dict[str, float] = {}
    if len(scored):
        # Pooled baseline: weight each fold's baseline log-loss by its size.
        overall = classification_metrics(
            scored["label"].to_numpy(),
            scored["prob"].to_numpy(),
            cfg.probability_threshold,
            baseline_rate=float(scored["label"].mean()),
        )
        if len(fold_metrics):
            overall["baseline_log_loss"] = float(
                np.average(fold_metrics["baseline_log_loss"], weights=fold_metrics["n"])
            )
    return WalkForwardResult(predictions, fold_metrics, overall)


def calibration_table(predictions: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Compare predicted probability with the realised win rate, by bucket.

    A calibrated model's ``mean_prob`` and ``win_rate`` columns agree. If the
    0.6 bucket only wins 35% of the time, a 0.6 threshold is meaningless.
    """
    scored = predictions[predictions["label"].notna()]
    if scored.empty:
        return pd.DataFrame()
    buckets = pd.qcut(scored["prob"], q=bins, duplicates="drop")
    table = scored.groupby(buckets, observed=True).agg(
        n=("label", "size"), mean_prob=("prob", "mean"), win_rate=("label", "mean")
    )
    return table.reset_index(names="prob_bucket")


# ---------------------------------------------------------------------------
# Final model + persistence
# ---------------------------------------------------------------------------
def train_final_model(dataset: pd.DataFrame, settings: Settings):
    """Fit on *all* labelled data for deployment.

    Walk-forward estimates how well the *procedure* generalises. Once that is
    accepted, the production model is refit on everything, because more data
    is better and there is nothing left to hold out for.
    """
    labelled = dataset[dataset["label"].notna()]
    return fit_model(
        labelled[feature_columns(settings.features)],
        labelled["label"].astype(int),
        labelled["weight"],
        settings.model.lgbm_params,
    )


def save_model(model, settings: Settings, n_samples: int) -> None:
    """Persist weights plus the metadata the live engine validates on start-up."""
    weights_dir = resolve_path(settings.model.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, weights_dir / MODEL_FILE)
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_samples": n_samples,
        "features": feature_columns(settings.features),
        "signature": settings.signature(),
    }
    (weights_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(settings: Settings):
    """Load the model and refuse to proceed if it does not match the current config.

    Guards against train/serve skew: a model trained with a 12-bar impulse
    window must not be fed features computed with a 14-bar one.
    """
    weights_dir = resolve_path(settings.model.weights_dir)
    model_path, meta_path = weights_dir / MODEL_FILE, weights_dir / META_FILE
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("No trained model found. Run `python -m research.train` first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["features"] != feature_columns(settings.features):
        raise RuntimeError("Model feature list does not match the current code. Retrain.")
    if meta["signature"] != json.loads(json.dumps(settings.signature())):
        raise RuntimeError("Model was trained with different feature/label settings. Retrain.")
    # joblib/pickle can execute code on load: only ever load files you produced.
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_bars(settings: Settings, synthetic: bool, refresh: bool) -> dict[str, pd.DataFrame]:
    if synthetic:
        from data.synthetic import generate_synthetic_ohlcv

        n_bars = int(pd.Timedelta(days=settings.data.history_days) / settings.bar_timedelta)
        return {
            symbol: generate_synthetic_ohlcv(
                n_bars=n_bars, seed=7 + i, timeframe=settings.market.timeframe
            )
            for i, symbol in enumerate(settings.market.symbols)
        }
    from data.fetcher import load_ohlcv

    return {symbol: load_ohlcv(symbol, settings, refresh) for symbol in settings.market.symbols}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the continuation model (walk-forward)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    parser.add_argument("--refresh", action="store_true", help="re-download market data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings(args.config)

    dataset = build_dataset(load_bars(settings, args.synthetic, args.refresh), settings)
    labelled = int(dataset["label"].notna().sum())
    print(f"\nSetups: {len(dataset)}   filled trades: {labelled}   "
          f"base win rate: {dataset['label'].mean():.3f}")  # fmt: skip

    result = walk_forward_validate(dataset, settings)
    pd.set_option("display.width", 200)
    print("\n=== Walk-forward folds (out-of-sample) ===")
    print(result.fold_metrics.to_string(index=False, float_format="{:.4f}".format))
    print("\n=== Pooled out-of-sample ===")
    for key, value in result.overall.items():
        print(f"  {key:<18} {value:.4f}")
    print("\n=== Calibration (predicted vs realised) ===")
    print(calibration_table(result.predictions).round(3).to_string(index=False))

    artifact_dir = resolve_path(settings.model.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(artifact_dir / PREDICTIONS_FILE)
    result.fold_metrics.to_csv(artifact_dir / FOLD_METRICS_FILE, index=False)

    save_model(train_final_model(dataset, settings), settings, n_samples=labelled)
    print(f"\nSaved OOS predictions to {artifact_dir} and the final model to "
          f"{resolve_path(settings.model.weights_dir)}")  # fmt: skip


if __name__ == "__main__":
    main()
