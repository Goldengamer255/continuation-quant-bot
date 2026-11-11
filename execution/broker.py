"""Broker abstraction: live exchange (CCXT), paper simulator and bar replay.

Why an abstraction layer
------------------------
The engine talks to two small interfaces, never to an exchange library:

* ``MarketFeed`` — "give me history" and "stream closed bars".
* ``Broker`` — "place / cancel / look up an order" and "what is my equity".

Three implementations plug in:

==============  ==========================================  =================
Class           What it is                                  Used for
==============  ==========================================  =================
``CcxtBroker``  real exchange via async CCXT (+ websocket)  live, and as the
                                                            data feed in paper
``PaperBroker`` in-memory matching engine                   paper trading, tests
``ReplayFeed``  plays back a DataFrame of bars              tests, dry runs
==============  ==========================================  =================

Because the engine cannot tell them apart, the exact code path that trades
real money is the one exercised by the test-suite. This is *dependency
inversion*: high-level logic depends on an interface, not on a vendor.

Async, rate limits and retries (interview notes)
------------------------------------------------
* **Why async?** The bot mostly *waits* — for bars, for order
  acknowledgements. ``asyncio`` lets one thread watch several symbols and
  poll order status without blocking on any single slow request.
* **Rate limits.** Exchanges ban clients that exceed request quotas.
  ``enableRateLimit`` makes CCXT throttle itself (a token-bucket per
  exchange), so we stay under the limit by construction.
* **Retry only what is safe to retry.** Network errors, timeouts and
  rate-limit responses are *transient*: retry with **exponential backoff**
  (0.5s, 1s, 2s, ...) plus **jitter** (a random extra) so many clients do not
  retry in lockstep. Errors like ``InsufficientFunds`` or ``InvalidOrder``
  are *permanent*: retrying cannot help, so they fail immediately.
* **Idempotent order placement.** The dangerous case: we send an order, the
  request times out, and we do not know whether the exchange received it.
  Blindly retrying can double the position. Every order therefore carries a
  unique ``clientOrderId`` generated *before* the first attempt; on retry we
  first look for an open order with that id and adopt it instead of sending
  a second one. Exchanges also reject duplicate client ids.
* **Errors feed the kill-switch.** Every failure/success is reported through
  callbacks to ``RiskManager``; enough consecutive failures halt trading.

Venue caveat: stop/trigger order parameters are the least standardised part
of any exchange API. ``CcxtBroker`` uses CCXT's unified ``triggerPrice``
parameter; confirm the behaviour on your venue's **sandbox** before going
live (some venues need extra flags to fetch or cancel trigger orders).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pandas as pd

from core.config import CostConfig, Credentials, Settings, timeframe_to_timedelta
from data.fetcher import FLOW_COLUMNS, OHLCV_COLUMNS, ohlcv_to_frame, validate_ohlcv

logger = logging.getLogger(__name__)

STATUS_OPEN = "open"
STATUS_FILLED = "closed"  # CCXT calls a fully filled order "closed"
STATUS_CANCELED = "canceled"


class BrokerError(RuntimeError):
    """A broker operation failed permanently (after retries, if retryable)."""


@dataclass(frozen=True)
class Bar:
    """One closed OHLCV bar. ``timestamp`` is the bar's open time (UTC)."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Order-flow extras (NaN when the feed does not provide them).
    taker_buy_volume: float = float("nan")
    trades: float = float("nan")


@dataclass
class Order:
    """Venue-neutral order record.

    ``type`` is one of: ``market``, ``limit`` (resting), ``stop`` (stop-market:
    becomes a market order at ``trigger_price``) or ``stop_limit`` (becomes a
    limit order at ``price`` once ``trigger_price`` trades).
    """

    id: str
    symbol: str
    side: str  # "buy" | "sell"
    type: str
    amount: float
    price: float | None = None
    trigger_price: float | None = None
    status: str = STATUS_OPEN
    filled: float = 0.0
    average: float | None = None
    reduce_only: bool = False
    client_id: str | None = None


def new_client_id() -> str:
    """Unique, exchange-safe client order id (the idempotency key)."""
    return "cqb" + uuid.uuid4().hex[:24]


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------
class MarketFeed(ABC):
    #: True if bars carry ``taker_buy_volume`` and ``trades`` (order-flow features).
    supports_flow: bool = False

    @abstractmethod
    async def fetch_history(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...

    @abstractmethod
    def stream_closed_bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        """Yield each bar once, only after it has closed."""

    async def connect(self) -> None:  # noqa: B027 - optional hook
        """Open connections / load market metadata."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections."""


class Broker(ABC):
    #: True if order status must be polled between bars (real exchanges).
    needs_polling: bool = False
    #: True if a stop and a take-profit order can rest at the same time.
    #: Spot venues lock the balance for the first order, so the second is
    #: rejected; the engine then manages the profit target itself.
    native_bracket: bool = True

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: float | None = None,
        trigger_price: float | None = None,
        reduce_only: bool = False,
    ) -> Order: ...

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Order: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> None: ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> None: ...

    @abstractmethod
    async def fetch_equity(self) -> float: ...

    @abstractmethod
    async def fetch_last_price(self, symbol: str) -> float: ...

    async def on_bar(self, symbol: str, bar: Bar) -> None:  # noqa: B027 - optional hook
        """Market-data hook. Only the paper broker needs it (to simulate fills)."""

    async def connect(self) -> None:  # noqa: B027 - optional hook
        """Open connections / load market metadata."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections."""


# ---------------------------------------------------------------------------
# Live exchange
# ---------------------------------------------------------------------------
class CcxtBroker(Broker, MarketFeed):
    """Async CCXT wrapper with rate limiting, retries and idempotent orders."""

    needs_polling = True

    def __init__(
        self,
        settings: Settings,
        credentials: Credentials,
        on_error: Callable[[], None] | None = None,
        on_success: Callable[[], None] | None = None,
        exchange=None,
    ) -> None:
        import ccxt  # lazy: paper/replay users and unit tests do not need it loaded

        self._ccxt = ccxt
        self._settings = settings
        self._on_error = on_error or (lambda: None)
        self._on_success = on_success or (lambda: None)
        self._max_retries = settings.execution.max_retries
        self._base_delay = settings.execution.retry_base_delay
        self.native_bracket = settings.market.type != "spot"
        # Order-flow columns come from Binance's raw spot kline endpoint.
        self._with_flow = settings.features.use_order_flow
        self.supports_flow = settings.market.exchange == "binance" and (
            settings.market.type == "spot"
        )

        if exchange is not None:  # injected fake for tests
            self._ex = exchange
            return

        try:  # ccxt.pro adds websocket `watch_*` methods on top of the async REST client
            import ccxt.pro as ccxt_async
        except ImportError:  # pragma: no cover - depends on the installed ccxt build
            import ccxt.async_support as ccxt_async
        klass = getattr(ccxt_async, settings.market.exchange)
        params: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": settings.market.type},
        }
        if credentials.present:
            params["apiKey"] = credentials.api_key
            params["secret"] = credentials.api_secret
            if credentials.api_password:
                params["password"] = credentials.api_password
        self._ex = klass(params)
        if settings.market.sandbox and credentials.present:
            self._ex.set_sandbox_mode(True)

    # ------------------------------------------------------------ plumbing
    async def connect(self) -> None:
        await self._call("load_markets", self._ex.load_markets)

    async def close(self) -> None:
        await self._ex.close()

    async def _call(self, op: str, fn, *args, **kwargs):
        """Run an exchange call with retry on transient errors only."""
        ccxt = self._ccxt
        for attempt in range(1, self._max_retries + 1):
            try:
                result = await fn(*args, **kwargs)
            except ccxt.NetworkError as exc:
                # NetworkError covers timeouts, DDoS protection and rate-limit responses.
                self._on_error()
                if attempt == self._max_retries:
                    raise BrokerError(f"{op} failed after {attempt} attempts") from exc
                delay = self._base_delay * 2 ** (attempt - 1)
                delay *= 1.0 + random.random() * 0.25  # jitter
                logger.warning(
                    "broker_retry",
                    extra={"op": op, "attempt": attempt, "error": type(exc).__name__},
                )
                await asyncio.sleep(delay)
            except ccxt.BaseError as exc:
                # Permanent: bad order, insufficient funds, auth failure...
                self._on_error()
                raise BrokerError(f"{op} rejected: {type(exc).__name__}") from exc
            else:
                self._on_success()
                return result
        raise BrokerError(f"{op} failed")  # pragma: no cover - loop always returns or raises

    def _to_order(self, raw: dict, type_hint: str | None = None) -> Order:
        status = (raw.get("status") or STATUS_OPEN).lower()
        if status in ("canceled", "cancelled", "expired", "rejected"):
            status = STATUS_CANCELED
        elif status != STATUS_FILLED:
            status = STATUS_OPEN
        return Order(
            id=str(raw["id"]),
            symbol=raw.get("symbol", ""),
            side=raw.get("side", ""),
            type=type_hint or raw.get("type", ""),
            amount=float(raw.get("amount") or 0.0),
            price=raw.get("price"),
            trigger_price=raw.get("triggerPrice") or raw.get("stopPrice"),
            status=status,
            filled=float(raw.get("filled") or 0.0),
            average=raw.get("average"),
            client_id=raw.get("clientOrderId"),
        )

    # -------------------------------------------------------------- trading
    async def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: float | None = None,
        trigger_price: float | None = None,
        reduce_only: bool = False,
    ) -> Order:
        client_id = new_client_id()  # fixed BEFORE the first attempt: the idempotency key
        amount = float(self._ex.amount_to_precision(symbol, amount))
        params: dict = {"clientOrderId": client_id}
        if trigger_price is not None:
            params["triggerPrice"] = float(self._ex.price_to_precision(symbol, trigger_price))
        if price is not None:
            price = float(self._ex.price_to_precision(symbol, price))
        if reduce_only and self._settings.market.type != "spot":
            params["reduceOnly"] = True  # spot has no such flag; position size bounds it instead
        ccxt_type = "limit" if type in ("limit", "stop_limit") else "market"

        attempts = 0

        async def submit():
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                # A previous attempt may have reached the exchange. Adopt it
                # rather than risk a duplicate order.
                for existing in await self._ex.fetch_open_orders(symbol):
                    if existing.get("clientOrderId") == client_id:
                        return existing
            return await self._ex.create_order(symbol, ccxt_type, side, amount, price, params)

        raw = await self._call("create_order", submit)
        order = self._to_order(raw, type_hint=type)
        order.reduce_only = reduce_only
        order.client_id = client_id
        if not order.amount:
            order.amount = amount
        return order

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        return self._to_order(
            await self._call("fetch_order", self._ex.fetch_order, order_id, symbol)
        )

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        async def cancel():
            # OrderNotFound = already filled or cancelled: the goal state is reached.
            with contextlib.suppress(self._ccxt.OrderNotFound):
                await self._ex.cancel_order(order_id, symbol)

        await self._call("cancel_order", cancel)

    async def cancel_all_orders(self, symbol: str) -> None:
        for raw in await self._call("fetch_open_orders", self._ex.fetch_open_orders, symbol):
            await self.cancel_order(str(raw["id"]), symbol)

    async def fetch_last_price(self, symbol: str) -> float:
        ticker = await self._call("fetch_ticker", self._ex.fetch_ticker, symbol)
        return float(ticker["last"])

    async def fetch_equity(self) -> float:
        """Account equity in quote currency (cash plus marked-to-market holdings)."""
        balance = await self._call("fetch_balance", self._ex.fetch_balance)
        total = balance.get("total") or {}
        equity = float(total.get(self._settings.market.quote_currency) or 0.0)
        if self._settings.market.type == "spot":
            for symbol in self._settings.market.symbols:
                held = float(total.get(symbol.split("/")[0]) or 0.0)
                if held:
                    equity += held * await self.fetch_last_price(symbol)
        return equity

    # ------------------------------------------------------------ market data
    async def _raw_klines(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Binance raw klines as ``[ms, o, h, l, c, volume, taker_buy_volume, trades]``."""
        raw = await self._call(
            "klines",
            self._ex.publicGetKlines,
            {"symbol": self._ex.market(symbol)["id"], "interval": timeframe, "limit": limit},
        )
        return [[int(k[0]), *(float(v) for v in k[1:6]), float(k[9]), float(k[8])] for k in raw]

    async def fetch_history(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        now = pd.Timestamp(self._ex.milliseconds(), unit="ms", tz="UTC")
        if self._with_flow:
            rows = await self._raw_klines(symbol, timeframe, min(limit + 1, 1000))
            frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS, *FLOW_COLUMNS])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            frame = frame.set_index("timestamp").astype("float64")
        else:
            rows = await self._call(
                "fetch_ohlcv", self._ex.fetch_ohlcv, symbol, timeframe, None, limit + 1
            )
            frame = ohlcv_to_frame(rows)
        return validate_ohlcv(frame, timeframe, now=now).tail(limit)

    async def stream_closed_bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        """Yield closed bars from a websocket, falling back to REST polling.

        A bar opened at T is "closed" once exchange time reaches T + timeframe.
        The websocket keeps resending the still-forming candle; those updates
        are ignored, so the strategy never sees a half-finished bar.
        """
        bar_ms = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
        # The websocket candle stream has no taker-buy volume, so order-flow
        # mode polls the REST kline endpoint once per bar instead.
        use_ws = bool(getattr(self._ex, "has", {}).get("watchOHLCV")) and not self._with_flow
        last_ts = -1
        while True:
            if use_ws:
                try:
                    candles = await self._ex.watch_ohlcv(symbol, timeframe)
                    self._on_success()
                except self._ccxt.NotSupported:
                    use_ws = False
                    continue
                except self._ccxt.NetworkError:
                    self._on_error()  # ccxt reconnects on the next call
                    await asyncio.sleep(1.0)
                    continue
            else:
                now_ms = self._ex.milliseconds()
                next_close = (now_ms // bar_ms + 1) * bar_ms
                await asyncio.sleep((next_close - now_ms) / 1000.0 + 2.0)
                if self._with_flow:
                    candles = await self._raw_klines(symbol, timeframe, 5)
                else:
                    candles = await self._call(
                        "fetch_ohlcv", self._ex.fetch_ohlcv, symbol, timeframe, None, 5
                    )

            now_ms = self._ex.milliseconds()
            for candle in candles:
                ts = int(candle[0])
                if ts > last_ts and ts + bar_ms <= now_ms:
                    last_ts = ts
                    yield Bar(pd.Timestamp(ts, unit="ms", tz="UTC"), *candle[1:])


# ---------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------
class PaperBroker(Broker):
    """In-memory broker that fills orders against closed bars.

    Fill rules deliberately mirror ``research.labeling.simulate_trade`` and
    the cost model mirrors ``research.backtest``, so paper results are
    comparable with the backtest:

    * ``stop_limit``: triggers when the bar trades through the trigger; fills
      at ``max(open, trigger)`` for buys. Skipped if the bar opens beyond the
      limit price (we do not chase gaps).
    * ``stop``: fills at the trigger, or at the open if the bar gapped through.
    * ``limit``: fills at the limit price when the bar reaches it.
    * Stops are matched before limits (pessimistic when one bar hits both).
    * ``reduce_only`` orders can shrink a position but never flip it.

    One approximation: protective orders are placed after the entry bar has
    closed, so a stop-out *within the entry bar* is not simulated here.
    """

    _MATCH_PRIORITY = {"stop": 0, "stop_limit": 1, "limit": 2, "market": 3}

    def __init__(self, starting_cash: float, costs: CostConfig) -> None:
        self.cash = float(starting_cash)
        self.costs = costs
        self.positions: dict[str, float] = defaultdict(float)
        self.last_price: dict[str, float] = {}
        self.orders: dict[str, Order] = {}
        self.fills: list[dict] = []
        self._next_id = 0
        self._clock: pd.Timestamp | None = None

    # -------------------------------------------------------------- helpers
    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o
            for o in self.orders.values()
            if o.status == STATUS_OPEN and (symbol is None or o.symbol == symbol)
        ]

    def _fill(self, order: Order, price: float, taker: bool) -> None:
        qty = order.amount
        if order.reduce_only:
            position = self.positions[order.symbol]
            closable = max(position, 0.0) if order.side == "sell" else max(-position, 0.0)
            qty = min(qty, closable)
            if qty <= 0:
                order.status = STATUS_CANCELED
                return

        if taker:  # market-style fills cross the spread and slip against us
            price *= 1.0 + self.costs.slippage if order.side == "buy" else 1.0 - self.costs.slippage
        fee = qty * price * (self.costs.taker_fee if taker else self.costs.maker_fee)

        if order.side == "buy":
            self.cash -= qty * price + fee
            self.positions[order.symbol] += qty
        else:
            self.cash += qty * price - fee
            self.positions[order.symbol] -= qty
        if abs(self.positions[order.symbol]) < 1e-12:
            self.positions[order.symbol] = 0.0

        order.status, order.filled, order.average = STATUS_FILLED, qty, price
        self.fills.append(
            {
                "time": self._clock,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "type": order.type,
                "qty": qty,
                "price": price,
                "fee": fee,
            }
        )

    # ------------------------------------------------------------ interface
    async def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: float | None = None,
        trigger_price: float | None = None,
        reduce_only: bool = False,
    ) -> Order:
        if amount <= 0:
            raise BrokerError("order amount must be positive")
        self._next_id += 1
        order = Order(
            id=f"paper-{self._next_id}",
            symbol=symbol,
            side=side,
            type=type,
            amount=float(amount),
            price=price,
            trigger_price=trigger_price,
            reduce_only=reduce_only,
            client_id=new_client_id(),
        )
        self.orders[order.id] = order
        if type == "market":
            if symbol not in self.last_price:
                raise BrokerError("no market price available yet")
            self._fill(order, self.last_price[symbol], taker=True)
        return order

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        if order_id not in self.orders:
            raise BrokerError(f"unknown order {order_id}")
        return self.orders[order_id]

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        order = self.orders.get(order_id)
        if order is not None and order.status == STATUS_OPEN:
            order.status = STATUS_CANCELED

    async def cancel_all_orders(self, symbol: str) -> None:
        for order in self.open_orders(symbol):
            order.status = STATUS_CANCELED

    async def fetch_equity(self) -> float:
        marked = sum(qty * self.last_price.get(sym, 0.0) for sym, qty in self.positions.items())
        return self.cash + marked

    async def fetch_last_price(self, symbol: str) -> float:
        if symbol not in self.last_price:
            raise BrokerError("no market price available yet")
        return self.last_price[symbol]

    async def on_bar(self, symbol: str, bar: Bar) -> None:
        """Match resting orders for ``symbol`` against a newly closed bar."""
        self._clock = bar.timestamp
        resting = sorted(self.open_orders(symbol), key=lambda o: self._MATCH_PRIORITY[o.type])
        for order in resting:
            if order.status != STATUS_OPEN:  # cancelled by an earlier fill in this bar
                continue
            buy = order.side == "buy"
            if order.type == "stop_limit":
                if buy:
                    if bar.open > order.price:
                        continue
                    if bar.open >= order.trigger_price:
                        self._fill(order, bar.open, taker=True)
                    elif bar.high >= order.trigger_price:
                        self._fill(order, order.trigger_price, taker=True)
                else:
                    if bar.open < order.price:
                        continue
                    if bar.open <= order.trigger_price:
                        self._fill(order, bar.open, taker=True)
                    elif bar.low <= order.trigger_price:
                        self._fill(order, order.trigger_price, taker=True)
            elif order.type == "stop":
                if buy and bar.high >= order.trigger_price:
                    self._fill(order, max(bar.open, order.trigger_price), taker=True)
                elif not buy and bar.low <= order.trigger_price:
                    self._fill(order, min(bar.open, order.trigger_price), taker=True)
            elif order.type == "limit" and (
                (buy and bar.low <= order.price) or (not buy and bar.high >= order.price)
            ):
                self._fill(order, order.price, taker=False)
        self.last_price[symbol] = bar.close


# ---------------------------------------------------------------------------
# Replay feed
# ---------------------------------------------------------------------------
class ReplayFeed(MarketFeed):
    """Plays historical bars back as if they were arriving live.

    Everything before ``start`` is served as history; the rest is streamed
    one bar at a time. (``warmup`` is the older way to say the same thing by
    row count.) Lets the full engine run deterministically in a test.

    With several symbols the streams share a **global clock**: a bar stamped T
    is only released once every other stream has delivered all of its bars
    before T. Real markets deliver bars in time order; without this, one
    symbol's task could race years ahead of another's and the engine's shared
    equity and daily risk limits would be evaluated out of order.
    """

    def __init__(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        warmup: int = 0,
        start: pd.Timestamp | None = None,
    ) -> None:
        self._bars = bars_by_symbol
        self._warmup = warmup
        self._start = start
        self._pending: dict[str, pd.Timestamp] = {}
        self.supports_flow = all(
            set(FLOW_COLUMNS) <= set(frame.columns) for frame in bars_by_symbol.values()
        )

    def _split(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = self._bars[symbol]
        if self._start is not None:
            return frame[frame.index < self._start], frame[frame.index >= self._start]
        return frame.iloc[: self._warmup], frame.iloc[self._warmup :]

    async def fetch_history(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        return self._split(symbol)[0].tail(limit)

    async def stream_closed_bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        live = self._split(symbol)[1]
        has_flow = set(FLOW_COLUMNS) <= set(live.columns)
        try:
            for ts, row in live.iterrows():
                self._pending[symbol] = ts
                # Wait until no other stream still owes an earlier bar.
                while min(self._pending.values()) < ts:
                    await asyncio.sleep(0)
                extras = (row["taker_buy_volume"], row["trades"]) if has_flow else ()
                yield Bar(
                    ts, row["open"], row["high"], row["low"], row["close"], row["volume"], *extras
                )
                await asyncio.sleep(0)  # hand control back to the event loop, like real I/O
        finally:
            self._pending.pop(symbol, None)
