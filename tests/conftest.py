"""Shared fixtures. Tests use synthetic data and the public example config only,
so they need no network, no credentials and no local settings.yaml."""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import EXAMPLE_CONFIG, Settings, load_settings
from data.synthetic import generate_synthetic_ohlcv


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(EXAMPLE_CONFIG)


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    return generate_synthetic_ohlcv(n_bars=3000, seed=11)
