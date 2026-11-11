"""Synthetic OHLCV generator for tests, CI and offline demos.

Why a synthetic generator belongs in a serious repo
---------------------------------------------------
1. **CI must not depend on an exchange being reachable.**
2. **It is a leakage detector.** With ``burst_prob=0`` the series is a random
   walk: there is *nothing* to predict about direction. If the research
   pipeline reports a strong edge on it, the pipeline is leaking future
   information. ``tests/test_train.py`` uses exactly this as a "placebo test".

The process: log-returns with optional volatility clustering (an AR(1)
process on log-volatility, a simple stochastic-volatility model) plus optional
"bursts" — short stretches of directional drift followed by a quiet period —
so that impulse/consolidation shapes actually occur. Nothing here is tuned to
make the strategy profitable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import timeframe_to_timedelta


def generate_synthetic_ohlcv(
    n_bars: int = 6000,
    seed: int = 7,
    start: str = "2022-01-01",
    timeframe: str = "1h",
    base_price: float = 100.0,
    base_vol: float = 0.004,
    vol_clustering: bool = True,
    burst_prob: float = 0.01,
    burst_len: int = 12,
    burst_drift: float = 0.6,
) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame of ``n_bars`` bars.

    Parameters
    ----------
    burst_prob:
        Per-bar probability of starting a directional burst. ``0`` gives a
        driftless random walk (the placebo dataset).
    burst_drift:
        Drift during a burst, in units of that bar's volatility.
    """
    rng = np.random.default_rng(seed)

    # --- volatility path -------------------------------------------------
    log_vol = np.zeros(n_bars)
    if vol_clustering:
        shocks = rng.normal(0.0, 0.1, n_bars)
        for i in range(1, n_bars):
            log_vol[i] = 0.97 * log_vol[i - 1] + shocks[i]
    sigma = base_vol * np.exp(log_vol)

    # --- directional bursts followed by quiet consolidation ---------------
    drift = np.zeros(n_bars)
    quiet = np.ones(n_bars)
    i = 0
    while i < n_bars:
        if burst_prob > 0 and rng.random() < burst_prob:
            direction = rng.choice([-1.0, 1.0])
            drift[i : i + burst_len] = direction * burst_drift * sigma[i : i + burst_len]
            pause = burst_len // 2
            quiet[i + burst_len : i + burst_len + pause] = 0.4
            i += burst_len + pause
        else:
            i += 1

    eff_sigma = sigma * quiet
    returns = drift + eff_sigma * rng.normal(size=n_bars)

    # --- build candles ------------------------------------------------------
    close = base_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[base_price], close[:-1]])  # each bar opens at the prior close
    upper_wick = eff_sigma * np.abs(rng.normal(size=n_bars)) * 0.5
    lower_wick = eff_sigma * np.abs(rng.normal(size=n_bars)) * 0.5
    high = np.maximum(open_, close) * np.exp(upper_wick)
    low = np.minimum(open_, close) * np.exp(-lower_wick)

    # Volume rises with activity (big moves happen on big volume).
    activity = 1.0 + np.abs(returns) / base_vol
    volume = 1000.0 * activity * np.exp(rng.normal(0.0, 0.4, n_bars))

    index = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        periods=n_bars,
        freq=timeframe_to_timedelta(timeframe),
        name="timestamp",
    )
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
