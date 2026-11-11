"""Triple-Barrier labeling (López de Prado, *Advances in Financial ML*, ch. 3).

Concept primer
--------------
The naive way to label is "did price go up over the next N bars?". That
ignores how a trade is actually managed: a position with a stop-loss can be
knocked out *before* bar N even if price ends higher. The label would say
"win" for a trade that really lost.

The **triple-barrier method** labels each trade by which of three barriers
price touches first:

* **upper** — profit target, ``+2.0 * ATR`` from entry          -> label 1
* **lower** — stop-loss,     ``-1.0 * ATR`` from entry          -> label 0
* **vertical** — time limit, ``max_holding_bars`` after entry   -> label 0

The label is therefore *path-dependent* and matches the live exit logic
exactly. Barriers are in ATR units so they adapt to volatility: a fixed 1%
stop is tight in a wild market and loose in a quiet one.

With a 2:1 payoff the break-even win rate is ``1 / (1 + 2) = 33.3%`` before
costs — the model needs P(win) above that, not above 50%.

Design decisions
----------------
* **The label simulates the real order, not a fantasy fill.** Live entries
  are stop-limit orders above the consolidation, so the labeler first checks
  whether that order *would have triggered* within ``entry_valid_bars``. A
  setup whose order never fills produces no trade (``no_fill``) and is not
  used for training: the model learns ``P(win | we actually got filled)``,
  which is the only population it is ever applied to.
* **No look-ahead.** The setup is known at the close of bar *t*. The earliest
  possible fill is bar ``t+1``. Barrier distances use ATR as of bar *t*.
* **Pessimistic intrabar assumption.** OHLC bars do not reveal the order of
  the high and the low. If one bar touches both the stop and the target, the
  stop is assumed to come first. Backtests should err against themselves.
* **Entry-bar exception.** When the entry fills intrabar, the bar's adverse
  extreme (the low, for a long) usually printed *before* the breakout, i.e.
  before we held a position. Counting it as a stop-out would be wrong, and on
  random-walk data it produced a large negative bias. On the entry bar a stop
  is therefore only counted if the bar *closes* through the stop. This can
  miss a genuine same-bar stop-out that recovered by the close; resolving
  that exactly needs lower-timeframe data. Fills at the open are unaffected:
  the whole bar is post-entry, so the full low/high is used.
* **Gaps.** If price gaps through the stop, the fill is the (worse) open, not
  the stop price. Targets fill at the target (limit orders do not improve
  in this model — again conservative).
* **Unresolved paths are dropped.** A setup near the end of the data whose
  barriers have not resolved is ``incomplete``; inventing a label for it
  would be fabricating the future.
* **One function, three users.** ``simulate_trade`` is used by the labeler,
  and its output is what the backtester replays. The live engine implements
  the same rules, so label, backtest and execution agree by construction.

Overlapping labels and sample weights
-------------------------------------
Consecutive bars often flag the same setup, so their label windows overlap
and the samples are far from independent. Treating 10 near-duplicates as 10
independent observations makes the model overconfident. Each label is
weighted by its **average uniqueness**: for every bar in a label's lifespan,
take ``1 / (number of labels alive on that bar)``, then average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import LabelingConfig

OUTCOME_TARGET = "pt"
OUTCOME_STOP = "sl"
OUTCOME_TIME = "time"
OUTCOME_NO_FILL = "no_fill"
OUTCOME_INCOMPLETE = "incomplete"
FILLED_OUTCOMES = (OUTCOME_TARGET, OUTCOME_STOP, OUTCOME_TIME)


@dataclass(frozen=True)
class TradePath:
    """Result of simulating one setup forward through the bars."""

    outcome: str
    entry_pos: int = -1
    entry_price: float = float("nan")
    exit_pos: int = -1
    exit_price: float = float("nan")
    release_pos: int = -1  # bar at which the symbol is free again (exit, or order expiry)


def entry_levels(
    side: int, cons_high: float, cons_low: float, atr: float, cfg: LabelingConfig
) -> tuple[float, float]:
    """Stop-limit entry prices for a setup: ``(trigger, limit)``.

    Longs trigger slightly *above* the consolidation high (shorts slightly
    below the low), so a fill only happens once the market has started to
    confirm the breakout. The limit price caps the slippage we will accept:
    if price gaps past it we simply do not chase.
    """
    if side > 0:
        trigger = cons_high + cfg.entry_buffer_atr * atr
        return trigger, trigger + cfg.limit_offset_atr * atr
    trigger = cons_low - cfg.entry_buffer_atr * atr
    return trigger, trigger - cfg.limit_offset_atr * atr


def barrier_levels(
    side: int, entry_price: float, atr: float, cfg: LabelingConfig
) -> tuple[float, float]:
    """Horizontal barriers for a filled trade: ``(stop, target)``."""
    stop = entry_price - side * cfg.stop_loss_atr * atr
    target = entry_price + side * cfg.profit_target_atr * atr
    return stop, target


def simulate_trade(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    t: int,
    side: int,
    trigger: float,
    limit: float,
    atr: float,
    cfg: LabelingConfig,
) -> TradePath:
    """Simulate the stop-limit entry and the triple-barrier exit for one setup.

    ``t`` is the integer position of the setup bar. Only bars ``> t`` are read.
    """
    n = len(open_)

    # ---- 1. Does the stop-limit entry fill? --------------------------------
    last_entry_bar = t + cfg.entry_valid_bars
    entry_pos, entry_price = -1, float("nan")
    for j in range(t + 1, min(last_entry_bar, n - 1) + 1):
        if side > 0:
            if open_[j] > limit:  # gapped beyond our limit: do not chase this bar
                continue
            if open_[j] >= trigger:  # gapped through the trigger but inside the limit
                entry_pos, entry_price = j, open_[j]
                break
            if high[j] >= trigger:  # traded up through the trigger intrabar
                entry_pos, entry_price = j, trigger
                break
        else:
            if open_[j] < limit:
                continue
            if open_[j] <= trigger:
                entry_pos, entry_price = j, open_[j]
                break
            if low[j] <= trigger:
                entry_pos, entry_price = j, trigger
                break

    if entry_pos < 0:
        if last_entry_bar > n - 1:  # ran out of data before the order expired
            return TradePath(OUTCOME_INCOMPLETE)
        return TradePath(OUTCOME_NO_FILL, release_pos=last_entry_bar)

    # ---- 2. Which barrier is touched first? --------------------------------
    stop, target = barrier_levels(side, entry_price, atr, cfg)
    last_hold_bar = entry_pos + cfg.max_holding_bars - 1
    # If we were filled intrabar (price traded *up* to a long trigger), part of
    # the entry bar happened before we were in the trade. Its low/high extreme
    # against us most likely printed before the fill, so it cannot be used as
    # evidence of a stop-out. The close is the only price on that bar known to
    # come after the fill. A fill at the open means the whole bar is ours.
    entered_at_open = open_[entry_pos] == entry_price
    for k in range(entry_pos, min(last_hold_bar, n - 1) + 1):
        gap_allowed = k > entry_pos  # an overnight-style gap can only hurt us after entry
        partial_bar = k == entry_pos and not entered_at_open
        if side > 0:
            adverse = close[k] if partial_bar else low[k]
            if adverse <= stop:  # stop checked first: pessimistic tie-break
                price = min(stop, open_[k]) if gap_allowed else stop
                return TradePath(OUTCOME_STOP, entry_pos, entry_price, k, price, k)
            if high[k] >= target:
                return TradePath(OUTCOME_TARGET, entry_pos, entry_price, k, target, k)
        else:
            adverse = close[k] if partial_bar else high[k]
            if adverse >= stop:
                price = max(stop, open_[k]) if gap_allowed else stop
                return TradePath(OUTCOME_STOP, entry_pos, entry_price, k, price, k)
            if low[k] <= target:
                return TradePath(OUTCOME_TARGET, entry_pos, entry_price, k, target, k)

    # ---- 3. Vertical barrier ------------------------------------------------
    if last_hold_bar > n - 1:
        return TradePath(OUTCOME_INCOMPLETE)
    return TradePath(
        OUTCOME_TIME, entry_pos, entry_price, last_hold_bar, close[last_hold_bar], last_hold_bar
    )


def apply_triple_barrier(
    bars: pd.DataFrame, events: pd.DataFrame, cfg: LabelingConfig
) -> pd.DataFrame:
    """Label every setup in ``events``.

    Returns a frame indexed by setup time. ``label`` is 1.0 (target first),
    0.0 (stop or time first) or NaN for ``no_fill`` rows, which are kept
    because the backtester needs to know the symbol was tied up by a pending
    order. ``incomplete`` setups are dropped entirely.

    ``t1`` is the time the label's outcome became known. The cross-validation
    splitter uses it to purge training samples whose outcome overlaps the test
    period.
    """
    index = bars.index
    o, h, low, c = (bars[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close"))
    positions = index.get_indexer(events.index)
    if (positions < 0).any():
        raise ValueError("events contain timestamps that are not in bars")

    records = []
    for pos, (time, ev) in zip(positions, events.iterrows(), strict=True):
        side, atr = int(ev["side"]), float(ev["atr"])
        trigger, limit = entry_levels(side, ev["cons_high"], ev["cons_low"], atr, cfg)
        path = simulate_trade(o, h, low, c, int(pos), side, trigger, limit, atr, cfg)
        if path.outcome == OUTCOME_INCOMPLETE:
            continue

        record = {
            "time": time,
            "side": side,
            "atr": atr,
            "trigger": trigger,
            "limit": limit,
            "outcome": path.outcome,
            "event_pos": int(pos),
            "release_pos": path.release_pos,
            "release_time": index[path.release_pos],
            "label": np.nan,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
            "stop": np.nan,
            "target": np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "bars_held": np.nan,
            "r_multiple": np.nan,
            "t1": pd.NaT,
        }
        if path.outcome in FILLED_OUTCOMES:
            stop, target = barrier_levels(side, path.entry_price, atr, cfg)
            record.update(
                label=1.0 if path.outcome == OUTCOME_TARGET else 0.0,
                entry_time=index[path.entry_pos],
                entry_price=path.entry_price,
                stop=stop,
                target=target,
                exit_time=index[path.exit_pos],
                exit_price=path.exit_price,
                bars_held=path.exit_pos - path.entry_pos + 1,
                # R-multiple: profit expressed in units of initial risk (1R = stop distance).
                r_multiple=side * (path.exit_price - path.entry_price) / (cfg.stop_loss_atr * atr),
                t1=index[path.exit_pos],
            )
        records.append(record)

    columns = [
        "side", "atr", "trigger", "limit", "outcome", "event_pos", "release_pos",
        "release_time", "label", "entry_time", "entry_price", "stop", "target",
        "exit_time", "exit_price", "bars_held", "r_multiple", "t1",
    ]  # fmt: skip
    if not records:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "time"
        return empty
    labels = pd.DataFrame.from_records(records).set_index("time")[columns]
    for col in ("entry_time", "exit_time", "t1", "release_time"):
        labels[col] = pd.to_datetime(labels[col], utc=True)
    return labels


def average_uniqueness(start_pos: np.ndarray, end_pos: np.ndarray, n_bars: int) -> np.ndarray:
    """Average-uniqueness sample weights for labels spanning ``[start, end]`` bars.

    ``concurrency[b]`` = number of labels alive on bar *b* (built with a
    difference array, O(n)). A label's weight is the mean of
    ``1 / concurrency`` over its own lifespan: 1.0 if it never overlaps
    another label, 0.5 if it always shares its bars with one other, etc.
    """
    start_pos = np.asarray(start_pos, dtype=int)
    end_pos = np.asarray(end_pos, dtype=int)
    delta = np.zeros(n_bars + 1)
    np.add.at(delta, start_pos, 1.0)
    np.add.at(delta, end_pos + 1, -1.0)
    concurrency = np.cumsum(delta)[:n_bars]

    weights = np.empty(len(start_pos))
    for i, (s, e) in enumerate(zip(start_pos, end_pos, strict=True)):
        weights[i] = np.mean(1.0 / concurrency[s : e + 1])
    return weights
