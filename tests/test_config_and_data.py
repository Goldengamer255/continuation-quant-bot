"""Config validation and data-hygiene tests."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from core.config import Credentials, Settings, timeframe_to_timedelta
from data.fetcher import ohlcv_to_frame, validate_ohlcv


def test_example_config_loads(settings):
    assert settings.labeling.profit_target_atr == 2.0
    assert settings.labeling.stop_loss_atr == 1.0
    assert settings.bar_timedelta == pd.Timedelta(hours=1)


def test_unknown_config_key_is_rejected():
    # A typo must fail loudly instead of silently using a default.
    with pytest.raises(ValidationError):
        Settings.model_validate({"labeling": {"stop_los_atr": 1.0}})


def test_shorting_spot_is_rejected():
    with pytest.raises(ValidationError):
        Settings.model_validate({"setup": {"allow_short": True}, "market": {"type": "spot"}})


def test_credentials_never_appear_in_repr():
    creds = Credentials(api_key="KEY123", api_secret="SECRET456")
    assert "KEY123" not in repr(creds)
    assert "SECRET456" not in repr(creds)


def test_timeframe_parsing():
    assert timeframe_to_timedelta("15m") == pd.Timedelta(minutes=15)
    assert timeframe_to_timedelta("1d") == pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        timeframe_to_timedelta("1x")


def test_in_progress_bar_is_dropped():
    """The still-forming bar must never reach research or the live engine."""
    hour = 3_600_000
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    rows = [[t0 + i * hour, 100, 101, 99, 100, 10] for i in range(3)]
    df = ohlcv_to_frame(rows)

    # "now" is 30 minutes into the third bar: only two bars are closed.
    now = pd.Timestamp("2024-01-01 02:30", tz="UTC")
    clean = validate_ohlcv(df, "1h", now=now)
    assert len(clean) == 2
    assert clean.index[-1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")


def test_validation_removes_duplicates_and_bad_candles():
    hour = 3_600_000
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    rows = [
        [t0 + hour, 100, 101, 99, 100, 10],
        [t0, 100, 101, 99, 100, 10],  # out of order
        [t0, 100, 101, 99, 100, 10],  # duplicate
        [t0 + 2 * hour, 100, 99, 98, 100, 10],  # high below open/close: impossible
    ]
    clean = validate_ohlcv(ohlcv_to_frame(rows), "1h", now=pd.Timestamp("2025-01-01", tz="UTC"))
    assert len(clean) == 2
    assert clean.index.is_monotonic_increasing
