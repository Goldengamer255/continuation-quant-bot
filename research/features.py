"""Continuation feature engineering and candidate-setup detection.

The trading idea
----------------
A *continuation* is a three-act pattern:

1. **Impulse** — price moves sharply in one direction on heavy volume
   (someone large is buying/selling aggressively).
2. **Consolidation** — price pauses in a tight range while volume dries up
   (the aggressor rests; nobody is willing to push it back).
3. **Continuation** — price breaks out of the pause in the impulse direction.

Each feature below measures one part of that anatomy.

Indicator primer
----------------
* **ROC (rate of change)**: ``close[t] / close[t-N] - 1`` — the N-bar return.
* **True Range**: ``max(high-low, |high-prev_close|, |low-prev_close|)`` —
  a bar's range *including* any gap from the previous close.
* **ATR (average true range)**: a smoothed mean of True Range. It is the
  standard "how much does this thing move per bar" yardstick. Dividing a
  price distance by ATR makes it comparable across assets and across calm vs
  wild regimes, which is why almost every feature here is in ATR units.
* **EMA**: exponentially weighted moving average; recent bars weigh more.
* **Bollinger Bands**: moving average +/- k standard deviations. Their
  *width* is a volatility measure; a narrow width is a "squeeze".

How look-ahead bias is prevented (the key interview answer)
-----------------------------------------------------------
Look-ahead bias = a feature at time *t* contains information that was not
available until after *t*. It makes backtests look brilliant and live
trading lose money. Three structural rules prevent it here:

1. **Only backward-looking operators** — ``rolling``, ``ewm`` and positive
   ``shift``. There is no ``shift(-k)``, no centred window, and no statistic
   fitted on the whole series (e.g. a global z-score, which would leak the
   future mean/std into every row).
2. **A row is "known" at its bar's close**, and every consumer acts on the
   *next* bar. Labels (``labeling.py``) start at bar ``t+1``.
3. **It is tested, not assumed.** ``tests/test_features.py`` recomputes
   features on a truncated history and asserts that the past rows are
   bit-for-bit identical. If any feature peeked forward, cutting off the
   future would change it.

Train/serve consistency
-----------------------
The live engine imports *these same functions*. A second, re-implemented
"live" feature path is the classic source of train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import FeatureConfig, SetupConfig

# Features whose sign depends on direction. They are multiplied by the trade
# side (+1 long / -1 short) so the model sees "in the direction of the trade"
# rather than "up". One model can then learn from longs and shorts alike.
SIGNED_FEATURES: tuple[str, ...] = (
    "roc_n",
    "impulse_roc",
    "impulse_atr",
    "ema_fast_dist_atr",
    "ema_slow_dist_atr",
    "ema_spread_atr",
)

# The exact, ordered column list the model is trained on and served with.
MODEL_FEATURES: tuple[str, ...] = (
    # --- impulse strength
    "roc_n",
    "impulse_roc",
    "impulse_atr",
    "rel_volume",
    "impulse_rel_volume",
    # --- consolidation tightness
    "atr_compression",
    "bb_width",
    "bb_width_contraction",
    "cons_range_atr",
    "retrace_frac",
    "close_pos_in_range",
    # --- volume decay
    "volume_decay_ratio",
    "volume_slope",
    # --- trend regime
    "ema_fast_dist_atr",
    "ema_slow_dist_atr",
    "ema_spread_atr",
)

# ---- Optional feature groups ------------------------------------------------
# The base set above is built only from one symbol's own candles. These groups
# add *different information*, each switched on by a FeatureConfig flag:
#
# * Order flow - who was the aggressor. "Taker buy volume" is volume where
#   the buyer crossed the spread. An impulse driven by aggressive buying, and
#   a pause where sellers are not hitting bids, is a different animal from the
#   same candle shapes produced by passive flow.
# * Market context - crypto is highly correlated. A long setup on an altcoin
#   while BTC is breaking down is fighting the tide.
# * Regime - very long trend, whether volatility is high or low relative to
#   its own history, and time of day/week (liquidity differs by session).
ORDER_FLOW_FEATURES: tuple[str, ...] = (
    "cons_flow_imbalance",
    "impulse_flow_imbalance",
    "trade_size_ratio",
)
CONTEXT_FEATURES: tuple[str, ...] = (
    "ctx_roc",
    "ctx_ema_spread_atr",
    "rel_strength",
    "ctx_atr_compression",
)
REGIME_FEATURES: tuple[str, ...] = ("trend_long_atr", "vol_regime", "hour", "day_of_week")
ALL_FEATURES: tuple[str, ...] = (
    MODEL_FEATURES + ORDER_FLOW_FEATURES + CONTEXT_FEATURES + REGIME_FEATURES
)
ALL_SIGNED_FEATURES: tuple[str, ...] = SIGNED_FEATURES + (
    "cons_flow_imbalance",
    "impulse_flow_imbalance",
    "ctx_roc",
    "ctx_ema_spread_atr",
    "rel_strength",
    "trend_long_atr",
)


def feature_columns(cfg: FeatureConfig) -> list[str]:
    """The ordered model input columns for a given feature configuration."""
    cols = list(MODEL_FEATURES)
    if cfg.use_order_flow:
        cols += ORDER_FLOW_FEATURES
    if cfg.use_market_context:
        cols += CONTEXT_FEATURES
    if cfg.use_regime:
        cols += REGIME_FEATURES
    return cols


# ---------------------------------------------------------------------------
# Indicator building blocks
# ---------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range: the bar's range extended to include the gap from the prior close."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range with Wilder smoothing (an EMA with ``alpha = 1/window``).

    ``adjust=False`` makes this a pure recursion from the first bar, so the
    value at *t* depends only on bars ``<= t``.
    """
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Ordinary-least-squares slope of ``series`` against time over a trailing window.

    For evenly spaced x the OLS slope is a fixed weighted sum of y:
    ``slope = sum((x - mean(x)) * y) / sum((x - mean(x))**2)``, so the weights
    are computed once and applied as a dot product.
    """
    x = np.arange(window, dtype=float) - (window - 1) / 2.0
    weights = x / (x**2).sum()
    return series.rolling(window).apply(lambda y: float(np.dot(weights, y)), raw=True)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_features(
    df: pd.DataFrame, cfg: FeatureConfig, context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute per-bar features from OHLCV. Row *t* uses bars ``<= t`` only.

    ``context`` is the OHLCV frame of ``cfg.context_symbol`` and is required
    when ``cfg.use_market_context`` is on. It is aligned on timestamp, and a
    context bar with the same timestamp closes at the same instant, so no
    future information crosses over.

    Window geometry at bar *t* (C = consolidation_window, I = impulse_window)::

        |<------ impulse: I bars ------>|<--- consolidation: C bars --->|
        t-C-I                          t-C                              t

    Returns a frame aligned to ``df.index`` containing both model features and
    helper columns (``atr``, ``cons_high``, ``cons_low`` ...) that the setup
    detector and labeler need.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    n_cons, n_imp = cfg.consolidation_window, cfg.impulse_window

    out = pd.DataFrame(index=df.index)
    atr_main = atr(df, cfg.atr_window)
    out["close"] = close
    out["atr"] = atr_main

    # ---- Impulse strength ---------------------------------------------------
    # Plain N-bar ROC ending now, and the ROC of the impulse leg itself (the I
    # bars that ended C bars ago, i.e. *before* the consolidation).
    impulse_end = close.shift(n_cons)
    impulse_start = close.shift(n_cons + n_imp)
    out["roc_n"] = close / close.shift(n_imp) - 1.0
    out["impulse_end_close"] = impulse_end
    out["impulse_move"] = impulse_end - impulse_start
    out["impulse_roc"] = impulse_end / impulse_start - 1.0
    out["impulse_atr"] = out["impulse_move"] / atr_main  # impulse size in ATRs

    # Relative volume: current volume vs its recent norm. The baseline is
    # shifted by one bar so the spike is not diluted by being part of its own
    # average (still causal: shift(1) only looks further back).
    volume_baseline = volume.shift(1).rolling(cfg.volume_window).mean()
    rel_volume = volume / volume_baseline
    out["rel_volume"] = rel_volume
    # Peak relative volume *during the impulse leg*: was the move "real"?
    out["impulse_rel_volume"] = rel_volume.shift(n_cons).rolling(n_imp).max()

    # ---- Consolidation tightness --------------------------------------------
    # ATR compression < 1 means recent bars are smaller than the long-run norm.
    out["atr_compression"] = atr(df, cfg.atr_short_window) / atr(df, cfg.atr_long_window)

    mid = close.rolling(cfg.bb_window).mean()
    std = close.rolling(cfg.bb_window).std(ddof=0)
    bb_width = (2.0 * cfg.bb_std * std) / mid  # (upper - lower) / middle
    out["bb_width"] = bb_width
    # Width relative to its own trailing average: < 1 is a squeeze.
    out["bb_width_contraction"] = bb_width / bb_width.rolling(cfg.bb_squeeze_lookback).mean()

    cons_high = high.rolling(n_cons).max()
    cons_low = low.rolling(n_cons).min()
    out["cons_high"] = cons_high
    out["cons_low"] = cons_low
    out["cons_range_atr"] = (cons_high - cons_low) / atr_main

    # ---- Volume decay -------------------------------------------------------
    # Healthy pauses happen on falling volume: nobody is fighting the move.
    cons_volume = volume.rolling(n_cons).mean()
    impulse_volume = volume.shift(n_cons).rolling(n_imp).mean()
    out["volume_decay_ratio"] = cons_volume / impulse_volume
    # Slope of volume inside the consolidation, as a fraction of mean volume per bar.
    out["volume_slope"] = rolling_slope(volume, n_cons) / cons_volume

    # ---- Trend regime -------------------------------------------------------
    ema_fast = ema(close, cfg.ema_fast)
    ema_slow = ema(close, cfg.ema_slow)
    out["ema_fast_dist_atr"] = (close - ema_fast) / atr_main
    out["ema_slow_dist_atr"] = (close - ema_slow) / atr_main
    out["ema_spread_atr"] = (ema_fast - ema_slow) / atr_main

    # ---- Optional: order flow -----------------------------------------------
    if cfg.use_order_flow:
        taker_buy = df["taker_buy_volume"]
        # Imbalance in [-1, +1]: +1 = every trade was an aggressive buy.
        out["cons_flow_imbalance"] = (
            2.0 * taker_buy.rolling(n_cons).sum() / volume.rolling(n_cons).sum() - 1.0
        )
        out["impulse_flow_imbalance"] = (
            2.0
            * taker_buy.shift(n_cons).rolling(n_imp).sum()
            / volume.shift(n_cons).rolling(n_imp).sum()
            - 1.0
        )
        # Average trade size in the pause vs the impulse: are large players still active?
        trade_size = volume / df["trades"]
        out["trade_size_ratio"] = (
            trade_size.rolling(n_cons).mean() / trade_size.shift(n_cons).rolling(n_imp).mean()
        )

    # ---- Optional: market context -------------------------------------------
    if cfg.use_market_context:
        if context is None:
            raise ValueError("use_market_context is on but no context bars were supplied")
        ctx_close = context["close"]
        # Indicators are computed on the context's own index, then aligned.
        ctx_roc = (ctx_close / ctx_close.shift(n_imp) - 1.0).reindex(df.index)
        ctx_spread = (ema(ctx_close, cfg.ema_fast) - ema(ctx_close, cfg.ema_slow)) / atr(
            context, cfg.atr_window
        )
        ctx_compression = atr(context, cfg.atr_short_window) / atr(context, cfg.atr_long_window)
        out["ctx_roc"] = ctx_roc
        out["ctx_ema_spread_atr"] = ctx_spread.reindex(df.index)
        out["rel_strength"] = out["roc_n"] - ctx_roc  # outperforming the market?
        out["ctx_atr_compression"] = ctx_compression.reindex(df.index)

    # ---- Optional: regime ---------------------------------------------------
    if cfg.use_regime:
        out["trend_long_atr"] = (close - ema(close, cfg.trend_long_span)) / atr_main
        atr_pct = atr_main / close
        out["vol_regime"] = atr_pct / atr_pct.rolling(cfg.vol_regime_lookback).mean()
        out["hour"] = df.index.hour.astype(float)
        out["day_of_week"] = df.index.day_of_week.astype(float)

    # Division by zero (e.g. a zero-volume window) becomes NaN, and NaN rows
    # are never treated as setups.
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Candidate setup detection (rule-based primary signal)
# ---------------------------------------------------------------------------
def detect_setups(features: pd.DataFrame, cfg: SetupConfig) -> pd.DataFrame:
    """Find bars that structurally look like impulse -> tight consolidation.

    Why not let the model score every bar? This is López de Prado's
    **meta-labeling**: a simple, explainable *primary* rule decides direction
    and *when* there is something to trade; the ML model is a *secondary*
    filter deciding whether to take that trade. Benefits:

    * The model answers a narrow question ("is this continuation setup a good
      one?") instead of "predict the market".
    * Class balance is far healthier than labelling every bar.
    * The strategy remains explainable: the model can only veto trades.

    Returns one row per setup bar with ``side`` (+1 long / -1 short) and the
    levels needed to place and manage the order.
    """
    f = features
    move_abs = f["impulse_move"].abs()

    tight = f["cons_range_atr"] <= cfg.max_consolidation_range_atr
    # Retracement: how much of the impulse the consolidation has given back.
    retrace_long = (f["impulse_end_close"] - f["cons_low"]) / move_abs
    retrace_short = (f["cons_high"] - f["impulse_end_close"]) / move_abs

    long_setup = (
        (f["impulse_atr"] >= cfg.min_impulse_atr) & tight & (retrace_long <= cfg.max_retrace)
    )
    short_setup = (
        (f["impulse_atr"] <= -cfg.min_impulse_atr) & tight & (retrace_short <= cfg.max_retrace)
    )

    side = pd.Series(0, index=f.index, dtype="int64")
    side[long_setup] = 1
    if cfg.allow_short:
        side[short_setup] = -1

    # Volatility-regime filter. Quiet markets tend to be followed by volatility
    # *expansion*; a breakout that starts from a quiet base has room to travel
    # several (small) ATRs. In an already-wild market the same pattern is
    # mostly noise. Uses ATR and close as of this bar only.
    if cfg.max_atr_pct is not None:
        side[(f["atr"] / f["close"]) > cfg.max_atr_pct] = 0

    # Require every raw input to be present (drops the warm-up period).
    raw_cols = [c for c in ALL_FEATURES if c in f.columns]
    complete = f[raw_cols + ["atr", "cons_high", "cons_low"]].notna().all(axis=1)

    mask = (side != 0) & complete
    events = pd.DataFrame(
        {
            "side": side[mask],
            "atr": f.loc[mask, "atr"],
            "cons_high": f.loc[mask, "cons_high"],
            "cons_low": f.loc[mask, "cons_low"],
        }
    )
    events.index.name = "time"
    return events


def build_feature_matrix(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model input ``X`` for the given setup events.

    Direction-dependent features are flipped by ``side`` so that, for both
    longs and shorts, a positive value means "in the trade's favour".
    """
    rows = features.loc[events.index]
    side = events["side"].to_numpy()
    is_long = side > 0

    x = pd.DataFrame(index=events.index)
    for col in ALL_FEATURES:
        if col in rows.columns:
            x[col] = rows[col].to_numpy()
    for col in ALL_SIGNED_FEATURES:
        if col in x.columns:
            x[col] = x[col] * side

    move_abs = rows["impulse_move"].abs().to_numpy()
    cons_high, cons_low = rows["cons_high"].to_numpy(), rows["cons_low"].to_numpy()
    impulse_end = rows["impulse_end_close"].to_numpy()

    retrace_long = (impulse_end - cons_low) / move_abs
    retrace_short = (cons_high - impulse_end) / move_abs
    x["retrace_frac"] = np.where(is_long, retrace_long, retrace_short)

    # Where the latest close sits inside the range (1 = pressed against the
    # breakout edge, 0 = at the far side).
    pos = (rows["close"].to_numpy() - cons_low) / (cons_high - cons_low)
    x["close_pos_in_range"] = np.where(is_long, pos, 1.0 - pos)

    return x[[c for c in ALL_FEATURES if c in x.columns]]
