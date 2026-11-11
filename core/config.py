"""Typed, validated configuration.

Why this module exists
----------------------
A trading system has dozens of parameters. Three properties matter:

1. **Single source of truth.** Research (``research/``) and live trading
   (``execution/``) read the *same* ``Settings`` object. If the backtest used a
   12-bar impulse window, the live engine cannot accidentally use 14.
2. **Fail loudly.** Every section is a frozen pydantic model with
   ``extra="forbid"``. A typo such as ``stop_los_atr`` raises at start-up
   instead of silently falling back to a default — the kind of bug that
   otherwise costs real money before anyone notices.
3. **Edge and secret isolation.** Tuned parameters live in
   ``config/settings.yaml`` (gitignored); API keys live in ``.env``
   (gitignored). Only placeholder templates are committed. Secrets are never
   part of ``Settings``, so logging or serialising the config can never leak
   a key.

Resolution order for the config file:
``load_settings(path)`` argument -> ``$CQB_CONFIG`` -> ``config/settings.yaml``
-> ``config/settings.example.yaml`` (so CI and a fresh clone still run).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "settings.example.yaml"
LOCAL_CONFIG = REPO_ROOT / "config" / "settings.yaml"

_TIMEFRAME_UNITS = {"m": "min", "h": "h", "d": "D", "w": "W"}


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert an exchange timeframe string (``"15m"``, ``"1h"``, ``"1d"``) to a Timedelta."""
    unit = timeframe[-1]
    if unit not in _TIMEFRAME_UNITS or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return pd.Timedelta(int(timeframe[:-1]), unit=_TIMEFRAME_UNITS[unit])


class _Section(BaseModel):
    """Base for all config sections: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class MarketConfig(_Section):
    exchange: str = "binance"
    type: Literal["spot", "swap"] = "spot"
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"], min_length=1)
    timeframe: str = "1h"
    quote_currency: str = "USDT"
    sandbox: bool = True


class DataConfig(_Section):
    cache_dir: str = "data/cache"
    history_days: int = Field(730, gt=0)


class FeatureConfig(_Section):
    impulse_window: int = Field(12, ge=2)
    consolidation_window: int = Field(6, ge=3)
    volume_window: int = Field(50, ge=5)
    atr_window: int = Field(14, ge=2)
    atr_short_window: int = Field(5, ge=2)
    atr_long_window: int = Field(50, ge=5)
    bb_window: int = Field(20, ge=5)
    bb_std: float = Field(2.0, gt=0)
    bb_squeeze_lookback: int = Field(100, ge=10)
    ema_fast: int = Field(20, ge=2)
    ema_slow: int = Field(50, ge=3)
    # Optional feature groups (off by default; see features.py).
    use_order_flow: bool = False  # needs taker-buy volume and trade counts (Binance klines)
    use_market_context: bool = False  # needs bars for the context symbol
    use_regime: bool = False
    context_symbol: str = "BTC/USDT"
    trend_long_span: int = Field(200, ge=20)
    vol_regime_lookback: int = Field(300, ge=20)

    @model_validator(mode="after")
    def _check_windows(self) -> FeatureConfig:
        if self.atr_short_window >= self.atr_long_window:
            raise ValueError("atr_short_window must be < atr_long_window")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be < ema_slow")
        return self


class SetupConfig(_Section):
    min_impulse_atr: float = Field(2.0, gt=0)
    max_consolidation_range_atr: float = Field(2.0, gt=0)
    max_retrace: float = Field(0.5, gt=0, le=1.0)
    allow_short: bool = False
    # Optional volatility-regime filter: only take setups while ATR / price is
    # at or below this fraction (e.g. 0.0107 = 1.07%). None disables it.
    max_atr_pct: float | None = Field(None, gt=0)


class LabelingConfig(_Section):
    profit_target_atr: float = Field(2.0, gt=0)
    stop_loss_atr: float = Field(1.0, gt=0)
    max_holding_bars: int = Field(24, ge=1)
    entry_buffer_atr: float = Field(0.1, ge=0)
    limit_offset_atr: float = Field(0.25, ge=0)
    entry_valid_bars: int = Field(3, ge=1)


class ModelConfig(_Section):
    probability_threshold: float = Field(0.5, gt=0, lt=1)
    n_splits: int = Field(5, ge=1)
    initial_train_fraction: float = Field(0.4, gt=0, lt=1)
    embargo_bars: int = Field(24, ge=0)
    min_train_samples: int = Field(100, ge=10)
    artifact_dir: str = "artifacts"
    weights_dir: str = "models"
    lgbm_params: dict[str, Any] = Field(default_factory=dict)


class CostConfig(_Section):
    taker_fee: float = Field(0.001, ge=0, lt=0.05)
    maker_fee: float = Field(0.001, ge=-0.01, lt=0.05)  # maker rebates can be negative
    slippage_bps: float = Field(5.0, ge=0, lt=500)
    # Perpetual-futures funding: the fraction of position notional a LONG pays
    # every 8 hours (shorts receive it). 0 for spot. Real funding varies and can
    # flip sign; a constant is a deliberately simple, long-unfriendly estimate.
    funding_rate_8h: float = Field(0.0, ge=-0.01, le=0.01)

    @property
    def slippage(self) -> float:
        """Slippage as a fraction (5 bps -> 0.0005)."""
        return self.slippage_bps / 10_000.0


class BacktestConfig(_Section):
    initial_equity: float = Field(10_000.0, gt=0)
    periods_per_year: int = Field(365, gt=0)


class RiskConfig(_Section):
    risk_per_trade: float = Field(0.01, gt=0, lt=1)
    max_daily_drawdown: float = Field(0.03, gt=0, lt=1)
    max_position_pct: float = Field(0.5, gt=0)
    max_gross_exposure_pct: float = Field(1.0, gt=0)
    max_open_positions: int = Field(3, ge=1)
    max_consecutive_api_errors: int = Field(5, ge=1)
    min_order_notional: float = Field(10.0, ge=0)
    auto_resume_next_day: bool = True


class ExecutionConfig(_Section):
    mode: Literal["paper", "live"] = "paper"
    warmup_bars: int = Field(600, ge=100)
    reconcile_seconds: float = Field(5.0, gt=0)
    context_wait_seconds: float = Field(30.0, ge=0)
    flatten_on_shutdown: bool = False
    cancel_orphans_on_start: bool = True
    max_retries: int = Field(5, ge=1)
    retry_base_delay: float = Field(0.5, gt=0)
    log_level: str = "INFO"
    log_json: bool = True


class Settings(_Section):
    """Root configuration object passed to every component."""

    market: MarketConfig = Field(default_factory=MarketConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    setup: SetupConfig = Field(default_factory=SetupConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def _check_cross_section(self) -> Settings:
        if self.setup.allow_short and self.market.type == "spot":
            raise ValueError("setup.allow_short requires market.type: swap (cannot short spot)")
        return self

    @property
    def bar_timedelta(self) -> pd.Timedelta:
        return timeframe_to_timedelta(self.market.timeframe)

    def signature(self) -> dict[str, Any]:
        """The parameters that define *what the model was trained on*.

        Saved next to the model weights and re-checked by the live engine. If
        someone changes the impulse window after training, the engine refuses
        to start rather than feed the model features it has never seen
        (train/serve skew).
        """
        return {
            "timeframe": self.market.timeframe,
            "features": self.features.model_dump(),
            "setup": self.setup.model_dump(),
            "labeling": self.labeling.model_dump(),
        }


def resolve_path(path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_settings(path: str | Path | None = None) -> Settings:
    """Load and validate settings (see module docstring for resolution order)."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(resolve_path(path))
    elif os.getenv("CQB_CONFIG"):
        candidates.append(resolve_path(os.environ["CQB_CONFIG"]))
    else:
        candidates.extend([LOCAL_CONFIG, EXAMPLE_CONFIG])

    for candidate in candidates:
        if candidate.exists():
            if candidate == EXAMPLE_CONFIG:
                logger.warning("config/settings.yaml not found; using the public example template")
            with candidate.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            return Settings.model_validate(raw)
    raise FileNotFoundError(f"No config file found (looked in: {candidates})")


@dataclass(frozen=True)
class Credentials:
    """Exchange API credentials, loaded from the environment only.

    ``repr=False`` keeps the values out of logs, tracebacks and debugger
    output even if the object is accidentally printed.
    """

    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    api_password: str = field(default="", repr=False)

    @property
    def present(self) -> bool:
        return bool(self.api_key and self.api_secret)


def load_credentials() -> Credentials:
    """Read credentials from ``.env`` / process environment."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    return Credentials(
        api_key=os.getenv("EXCHANGE_API_KEY", ""),
        api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
        api_password=os.getenv("EXCHANGE_API_PASSWORD", ""),
    )


def live_trading_armed() -> bool:
    """Second arming key for live trading (see ``.env.example``)."""
    return os.getenv("CQB_LIVE_TRADING", "NO").strip().upper() == "YES"
