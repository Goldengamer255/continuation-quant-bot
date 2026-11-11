"""Configuration search with a locked holdout.

The problem this solves
-----------------------
"Keep changing the strategy until the backtest is profitable" always succeeds
eventually — by luck. Try 20 random configurations and one will look good at
the 5% level even if none has an edge (the *multiple-testing* problem). A
result found that way does not survive live trading.

The protocol
------------
Time is split once, in advance::

    |<-------------- development --------------->|<----- holdout ----->|
    first bar                              HOLDOUT_START            last bar

1. **Development.** Every candidate configuration is evaluated here with
   purged walk-forward validation and the cost-aware backtest. Iterate as
   much as you like; nothing in this period is sacred.
2. **Selection rule, fixed before looking.** A candidate *passes* only if its
   development out-of-sample results meet all of ``PASS_CRITERIA``. Among
   passing candidates the one with the highest expectancy is selected.
3. **Holdout, used once.** The selected configuration is refit on all
   development data and traded on the holdout. That number is the honest
   estimate. If it fails, the strategy failed; going back to pick a different
   candidate "because the holdout liked it better" turns the holdout into
   training data.
4. **If no candidate passes, the holdout is not touched**, so it stays clean
   for a genuinely new idea later.

Candidates vary only three things, each chosen for a stated reason rather
than by trial and error:

* **Timeframe** — lower gives more trades but a smaller ATR, so fixed
  percentage costs eat more of each trade's risk (R).
* **Stop width** — a wider stop makes costs a
  smaller fraction of R, at the price of fewer, slower trades.
* **Payoff ratio** (target / stop) — 2:1 needs a 33% win rate to break even,
  1:1 needs 50%. A lower ratio wins more often but each win pays less; it is
  not automatically more profitable.

Usage::

    python -m research.experiment dev --ratio 2.0
    python -m research.experiment dev --ratio 1.0
    python -m research.experiment holdout --timeframe 4h --stop 1.5 --ratio 1.0 --threshold 0.55
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from core.config import Settings, load_settings
from data.fetcher import load_ohlcv
from research.backtest import run_backtest
from research.dataset import build_dataset
from research.features import feature_columns
from research.train import classification_metrics, fit_model, walk_forward_validate

logger = logging.getLogger(__name__)

HOLDOUT_START = pd.Timestamp("2026-02-01", tz="UTC")
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT",
]  # fmt: skip
# Wider universe for a follow-up round: the 30 most liquid Binance USDT spot
# pairs (by 24h quote volume on the selection date, stablecoins and wrapped
# tokens excluded) that have a full four years of history. Note the
# survivorship bias: these are coins that are still liquid *today*.
SYMBOLS_WIDE = [
    "BTC/USDT", "ETH/USDT", "ZEC/USDT", "SOL/USDT", "XRP/USDT", "NEAR/USDT",
    "BNB/USDT", "UNI/USDT", "DOGE/USDT", "AVAX/USDT", "TRX/USDT", "ONE/USDT",
    "INJ/USDT", "LINK/USDT", "ADA/USDT", "LTC/USDT", "AAVE/USDT", "QNT/USDT",
    "XLM/USDT", "FET/USDT", "ICP/USDT", "DASH/USDT", "FIL/USDT", "BCH/USDT",
    "HBAR/USDT", "DOT/USDT", "APT/USDT", "DODO/USDT", "SHIB/USDT", "CAKE/USDT",
]  # fmt: skip
UNIVERSES = {"core": SYMBOLS, "wide": SYMBOLS_WIDE}
# Feature sets: "base" is price/volume shape only; the others add new information.
FEATURE_SETS = {
    "base": {},
    "flow": {"use_order_flow": True},
    "context": {"use_market_context": True},
    "regime": {"use_regime": True},
    "all": {"use_order_flow": True, "use_market_context": True, "use_regime": True},
}
# timeframe -> days of history to request
TIMEFRAMES = {"15m": 730, "1h": 1460, "4h": 1460}
STOP_WIDTHS = [1.0, 1.5, 2.0]
# Target distance as a multiple of the stop. 2.0 = "win twice what you risk"
# (break-even win rate 33%); 1.0 = even payoff (break-even 50%).
PAYOFF_RATIOS = [2.0, 1.0]
# Probability thresholds to score, per payoff ratio. They start near the
# break-even win rate 1 / (1 + ratio): below it a trade has negative expectancy
# even if the model is perfectly calibrated. 0.0 means "rule only": every setup
# is traded and the model is ignored, showing what the entry rule is worth alone.
THRESHOLDS = {
    2.0: [0.0, 0.30, 0.35, 0.40, 0.45, 0.50],
    1.0: [0.0, 0.50, 0.55, 0.60, 0.65, 0.70],
}

# Cost scenarios. "futures" approximates a USDT-margined perpetual at base
# fee tier: 0.05% taker, 0.02% maker, tighter slippage on the deeper book, and
# longs paying 0.01% of notional per 8 hours in funding. Prices are still the
# SPOT series, used as a proxy for the perpetual.
COST_SCENARIOS = {
    "spot": {},
    "futures": {
        "taker_fee": 0.0005,
        "maker_fee": 0.0002,
        "slippage_bps": 2.0,
        "funding_rate_8h": 0.0001,
    },
}
# Set from the command line in main().
ACTIVE = {"costs": "spot", "shorts": False}

PASS_CRITERIA = {
    "min_trades": 150,  # enough trades for the averages to mean something
    "min_expectancy_r": 0.05,  # clearly positive after costs, not a rounding error
    "min_profit_factor": 1.10,
    "min_positive_fold_share": 0.6,  # profitable in most periods, not one lucky one
}


def candidate_settings(
    base: Settings,
    timeframe: str,
    stop_atr: float,
    ratio: float = 2.0,
    universe: str = "core",
    feature_set: str = "base",
    max_atr_pct: float | None = None,
) -> Settings:
    """Derive a candidate from the base config; everything not listed here is unchanged."""
    market_update = {"timeframe": timeframe, "symbols": UNIVERSES[universe]}
    setup_update: dict = {"max_atr_pct": max_atr_pct}
    if ACTIVE["shorts"]:  # shorting needs a derivatives market
        market_update["type"] = "swap"
        setup_update["allow_short"] = True
    return base.model_copy(
        update={
            "market": base.market.model_copy(update=market_update),
            "costs": base.costs.model_copy(update=COST_SCENARIOS[ACTIVE["costs"]]),
            "data": base.data.model_copy(update={"history_days": TIMEFRAMES[timeframe]}),
            "features": base.features.model_copy(update=FEATURE_SETS[feature_set]),
            "setup": base.setup.model_copy(update=setup_update),
            "labeling": base.labeling.model_copy(
                update={"stop_loss_atr": stop_atr, "profit_target_atr": ratio * stop_atr}
            ),
        }
    )


def split_dataset(dataset: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split setups into development and holdout, purging the boundary.

    A development setup is dropped if its trade (or pending order) was still
    alive at the holdout start: its outcome contains holdout prices.
    """
    embargo = settings.model.embargo_bars * settings.bar_timedelta
    is_holdout = dataset["time"] >= HOLDOUT_START
    resolved_before = dataset["release_time"] + embargo < HOLDOUT_START
    dev = dataset[~is_holdout & resolved_before].reset_index(drop=True)
    return dev, dataset[is_holdout].reset_index(drop=True)


def cost_in_r(dataset: pd.DataFrame, settings: Settings) -> float:
    """Median round-trip cost as a fraction of the risk taken (1R = stop distance)."""
    filled = dataset[dataset["label"].notna()]
    round_trip = 2 * settings.costs.taker_fee + 2 * settings.costs.slippage
    stop_distance = settings.labeling.stop_loss_atr * filled["atr"]
    return float((round_trip * filled["entry_price"] / stop_distance).median())


def positive_fold_share(trades: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Share of walk-forward folds in which the traded setups made money."""
    if trades.empty:
        return 0.0
    fold_of = predictions.set_index(["symbol", "time"])["fold"]
    keys = pd.MultiIndex.from_frame(trades[["symbol", "setup_time"]])
    by_fold = trades["net_pnl"].groupby(fold_of.reindex(keys).to_numpy()).sum()
    return float((by_fold > 0).sum() / predictions["fold"].nunique())


def evaluate_development(dev: pd.DataFrame, settings: Settings, ratio: float) -> list[dict]:
    """Walk-forward on the development period; one result row per threshold."""
    result = walk_forward_validate(dev, settings)
    rows = []
    for threshold in THRESHOLDS[ratio]:
        backtest = run_backtest(result.predictions, settings, threshold)
        m = backtest.metrics
        rows.append(
            {
                "threshold": threshold,
                "trades": m.get("trades", 0),
                "win_rate": m.get("win_rate", np.nan),
                "expectancy_r": m.get("expectancy_r", np.nan),
                "profit_factor": m.get("profit_factor", np.nan),
                "sharpe": m.get("sharpe", np.nan),
                "max_drawdown": m.get("max_drawdown", np.nan),
                "total_return": m.get("total_return", np.nan),
                "pos_fold_share": positive_fold_share(backtest.trades, result.predictions),
                "auc": result.overall.get("auc", np.nan),
                "log_loss": result.overall.get("log_loss", np.nan),
                "baseline_log_loss": result.overall.get("baseline_log_loss", np.nan),
            }
        )
    return rows


def passes(row: dict) -> bool:
    c = PASS_CRITERIA
    return bool(
        row["trades"] >= c["min_trades"]
        and row["expectancy_r"] >= c["min_expectancy_r"]
        and row["profit_factor"] >= c["min_profit_factor"]
        and row["pos_fold_share"] >= c["min_positive_fold_share"]
    )


def load_candidate(
    base: Settings,
    timeframe: str,
    stop_atr: float,
    ratio: float = 2.0,
    universe: str = "core",
    feature_set: str = "base",
    max_atr_pct: float | None = None,
):
    settings = candidate_settings(
        base, timeframe, stop_atr, ratio, universe, feature_set, max_atr_pct
    )
    bars = {symbol: load_ohlcv(symbol, settings) for symbol in settings.market.symbols}
    dataset = build_dataset(bars, settings)
    return settings, dataset


def run_development(
    base: Settings,
    ratio: float,
    universe: str = "core",
    feature_set: str = "base",
    timeframes: list[str] | None = None,
    stops: list[float] | None = None,
    max_atr_pct: float | None = None,
) -> pd.DataFrame:
    rows = []
    for timeframe in timeframes or TIMEFRAMES:
        for stop_atr in stops or STOP_WIDTHS:
            settings, dataset = load_candidate(
                base, timeframe, stop_atr, ratio, universe, feature_set, max_atr_pct
            )
            dev, _ = split_dataset(dataset, settings)  # the holdout half is discarded unseen
            labelled = dev[dev["label"].notna()]
            info = {
                "timeframe": timeframe,
                "stop_atr": stop_atr,
                "ratio": ratio,
                "dev_setups": len(labelled),
                "base_win": float(labelled["label"].mean()),
                "gross_r": float(labelled["r_multiple"].mean()),
                "cost_r": cost_in_r(dev, settings),
            }
            for row in evaluate_development(dev, settings, ratio):
                rows.append(info | row | {"passes": passes(row)})
            print(
                f"done {timeframe} stop={stop_atr}: {len(labelled)} labelled dev setups", flush=True
            )
    return pd.DataFrame(rows)


def run_holdout(
    base: Settings,
    timeframe: str,
    stop_atr: float,
    threshold: float,
    ratio: float,
    universe: str = "core",
    feature_set: str = "base",
    max_atr_pct: float | None = None,
) -> None:
    """Fit on development, trade the holdout once."""
    settings, dataset = load_candidate(
        base, timeframe, stop_atr, ratio, universe, feature_set, max_atr_pct
    )
    dev, holdout = split_dataset(dataset, settings)
    train = dev[dev["label"].notna()]
    model = fit_model(
        train[feature_columns(settings.features)],
        train["label"].astype(int),
        train["weight"],
        settings.model.lgbm_params,
    )
    signals = holdout.assign(
        prob=model.predict_proba(holdout[feature_columns(settings.features)])[:, 1]
    )
    scored = signals[signals["label"].notna()]
    stats = classification_metrics(
        scored["label"].to_numpy(),
        scored["prob"].to_numpy(),
        threshold,
        float(train["label"].mean()),
    )
    result = run_backtest(signals, settings, threshold)
    print(f"\n=== HOLDOUT {timeframe} stop={stop_atr} ratio={ratio} threshold={threshold} "
          f"({HOLDOUT_START.date()} onward, never used in development) ===")  # fmt: skip
    print("model:", {k: round(v, 4) for k, v in stats.items()})
    print("backtest:", {k: round(float(v), 4) for k, v in result.metrics.items()})
    print("not traded:", result.skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked-holdout configuration search")
    parser.add_argument("phase", choices=["dev", "holdout"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--timeframe", choices=list(TIMEFRAMES))
    parser.add_argument("--stop", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--ratio", type=float, default=2.0, choices=PAYOFF_RATIOS)
    parser.add_argument("--universe", default="core", choices=list(UNIVERSES))
    parser.add_argument("--features", default="base", choices=list(FEATURE_SETS))
    parser.add_argument("--max-atr-pct", type=float, default=None)
    parser.add_argument("--costs", default="spot", choices=list(COST_SCENARIOS))
    parser.add_argument("--shorts", action="store_true", help="also trade short setups")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    base = load_settings(args.config)
    ACTIVE.update(costs=args.costs, shorts=args.shorts)
    if args.phase == "dev":
        table = run_development(
            base,
            args.ratio,
            args.universe,
            args.features,
            [args.timeframe] if args.timeframe else None,
            [args.stop] if args.stop else None,
            args.max_atr_pct,
        )
        pd.set_option("display.width", 250, "display.max_rows", 200)
        print(table.round(3).to_string(index=False))
        print("\nPassing candidates:")
        passing = table[table["passes"]].sort_values("expectancy_r", ascending=False)
        print(passing.round(3).to_string(index=False) if len(passing) else "  none")
    else:
        if None in (args.timeframe, args.stop, args.threshold):
            parser.error("holdout needs --timeframe, --stop and --threshold")
        run_holdout(
            base,
            args.timeframe,
            args.stop,
            args.threshold,
            args.ratio,
            args.universe,
            args.features,
            args.max_atr_pct,
        )


if __name__ == "__main__":
    main()
