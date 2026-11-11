"""Historical OHLCV fetcher (CCXT) with validation and a local parquet cache.

Concept primer
--------------
*OHLCV* = Open, High, Low, Close, Volume for one time bucket ("bar" /
"candle"). CCXT is a library that gives ~100 crypto exchanges one common API.
Exchanges return at most ~1000 bars per request, so history is fetched in
pages by advancing a ``since`` cursor.

Convention used everywhere in this project: **a bar's timestamp is its OPEN
time (UTC)**. The bar ``2024-01-01 10:00`` on a 1h chart covers 10:00-11:00
and is only *known* at 11:00.

Design choices worth explaining in an interview
-----------------------------------------------
* **The in-progress bar is dropped.** The last bar an exchange returns is
  usually still forming: its close/high/low will change. Training or trading
  on it is look-ahead bias in disguise, because the "final" values you see in
  history were not available at decision time.
* **Validation before caching.** Duplicates, out-of-order rows, zero prices
  and inconsistent candles (high < close) are removed, and gaps are counted.
  Garbage bars create fake volatility, which creates fake signals.
* **Cache as parquet.** Columnar, typed, compressed, and it preserves the
  UTC index exactly. CSV would lose dtypes and timezone. The cache directory
  is gitignored: data is reproducible and often licence-restricted.
* **Public endpoints only.** Fetching history needs no API key, so research
  never touches credentials at all.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from core.config import Settings, load_settings, resolve_path, timeframe_to_timedelta

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
# Extra per-bar fields from Binance's raw kline endpoint (not in CCXT's unified
# OHLCV): volume bought by aggressive (taker) buyers, and the number of trades.
FLOW_COLUMNS = ["taker_buy_volume", "trades"]


def ohlcv_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Convert raw CCXT rows ``[ms, o, h, l, c, v]`` to a UTC-indexed DataFrame."""
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype("float64")


def validate_ohlcv(
    df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Clean a raw OHLCV frame and drop any bar that has not closed yet.

    Parameters
    ----------
    now:
        "Current time" used to decide which bars are closed. Injectable so the
        function is deterministic in tests.
    """
    bar = timeframe_to_timedelta(timeframe)
    now = now if now is not None else pd.Timestamp.now(tz="UTC")

    df = df[~df.index.duplicated(keep="last")].sort_index()

    # A bar opened at T is complete only once T + bar <= now.
    closed = df.index + bar <= now
    df = df[closed]

    prices = df[["open", "high", "low", "close"]]
    consistent = (
        prices.notna().all(axis=1)
        & (prices > 0).all(axis=1)
        & (df["high"] >= prices[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= prices[["open", "close", "high"]].min(axis=1))
        & (df["volume"] >= 0)
    )
    if (~consistent).any():
        logger.warning("dropping %d inconsistent bars", int((~consistent).sum()))
        df = df[consistent]

    if len(df) > 1:
        gaps = int((df.index.to_series().diff().dropna() > bar).sum())
        if gaps:
            # Not fatal (exchange maintenance happens), but rolling windows
            # spanning a gap are slightly off, so make it visible.
            logger.warning("%d gaps detected in %s data", gaps, timeframe)
    return df


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since: pd.Timestamp,
    until: pd.Timestamp | None = None,
    page_limit: int = 1000,
    max_retries: int = 5,
    with_flow: bool = False,
) -> pd.DataFrame:
    """Download closed bars in ``[since, until)`` by paginating the exchange API.

    ``with_flow`` adds ``FLOW_COLUMNS`` (Binance only).
    """
    import ccxt  # imported lazily so research code that only reads the cache has no network dep

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if with_flow:
        if exchange_id != "binance":
            raise ValueError("order-flow columns are only available from Binance klines")
        exchange.load_markets()
    bar_ms = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
    until = until if until is not None else pd.Timestamp.now(tz="UTC")
    cursor, end_ms = int(since.timestamp() * 1000), int(until.timestamp() * 1000)

    rows: list[list[float]] = []
    while cursor < end_ms:
        batch = _fetch_page(exchange, symbol, timeframe, cursor, page_limit, max_retries, with_flow)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + bar_ms
        if next_cursor <= cursor:  # exchange returned nothing new; avoid an infinite loop
            break
        cursor = next_cursor

    if not rows:
        raise RuntimeError(f"No OHLCV data returned for {symbol} on {exchange_id}")
    if with_flow:
        df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS, *FLOW_COLUMNS])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype("float64")
    else:
        df = ohlcv_to_frame(rows)
    return validate_ohlcv(df[df.index < until], timeframe)


def _fetch_page(exchange, symbol, timeframe, since_ms, limit, max_retries, with_flow=False):
    """One page with exponential backoff on transient network errors."""
    import ccxt

    for attempt in range(1, max_retries + 1):
        try:
            if not with_flow:
                return exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            # Raw kline layout: [open_time, o, h, l, c, volume, close_time,
            # quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
            raw = exchange.publicGetKlines(
                {
                    "symbol": exchange.market(symbol)["id"],
                    "interval": timeframe,
                    "startTime": since_ms,
                    "limit": limit,
                }
            )
            return [[int(k[0]), *(float(v) for v in k[1:6]), float(k[9]), float(k[8])] for k in raw]
        except ccxt.NetworkError as exc:  # includes rate-limit and timeout errors
            if attempt == max_retries:
                raise
            delay = 0.5 * 2 ** (attempt - 1)
            logger.warning("fetch retry %d/%d after %s", attempt, max_retries, type(exc).__name__)
            time.sleep(delay)
    return []


def cache_path(settings: Settings, symbol: str) -> Path:
    safe_symbol = symbol.replace("/", "-").replace(":", "_")
    suffix = "_flow" if settings.features.use_order_flow else ""
    name = f"{settings.market.exchange}_{safe_symbol}_{settings.market.timeframe}{suffix}.parquet"
    return resolve_path(settings.data.cache_dir) / name


def load_ohlcv(symbol: str, settings: Settings, refresh: bool = False) -> pd.DataFrame:
    """Return cached bars for ``symbol``, downloading them first if needed."""
    path = cache_path(settings, symbol)
    requested_days = settings.data.history_days
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        span_days = (cached.index[-1] - cached.index[0]).days
        if span_days < requested_days - 30:
            # A cache written for a shorter history is silently reused otherwise.
            logger.warning(
                "%s cache spans %d days but %d were requested; pass refresh=True to extend",
                symbol,
                span_days,
                requested_days,
            )
        return cached

    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=requested_days)
    df = fetch_ohlcv(
        settings.market.exchange,
        symbol,
        settings.market.timeframe,
        since,
        with_flow=settings.features.use_order_flow,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("cached %d bars for %s at %s", len(df), symbol, path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache historical OHLCV data")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings(args.config)
    for symbol in settings.market.symbols:
        df = load_ohlcv(symbol, settings, refresh=args.refresh)
        print(f"{symbol}: {len(df)} bars  {df.index[0]} -> {df.index[-1]}")


if __name__ == "__main__":
    main()
