"""Paper matching-engine rules and the live wrapper's retry behaviour."""

from __future__ import annotations

import asyncio

import ccxt
import pandas as pd
import pytest

from core.config import CostConfig, Credentials
from execution.broker import (
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_OPEN,
    Bar,
    BrokerError,
    CcxtBroker,
    PaperBroker,
)

SYMBOL = "BTC/USDT"
FREE = CostConfig(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0)
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def bar(o, h, low, c, i=0):
    return Bar(T0 + pd.Timedelta(hours=i), o, h, low, c, 100.0)


def run(coro):
    return asyncio.run(coro)


def test_stop_limit_buy_fills_at_trigger_when_traded_through():
    async def scenario():
        broker = PaperBroker(10_000.0, FREE)
        order = await broker.create_order(SYMBOL, "buy", "stop_limit", 1.0, 100.5, 100.0)
        await broker.on_bar(SYMBOL, bar(99.0, 99.5, 98.5, 99.2))
        assert order.status == STATUS_OPEN  # trigger not reached
        await broker.on_bar(SYMBOL, bar(99.5, 100.8, 99.4, 100.6, i=1))
        return broker, order

    broker, order = run(scenario())
    assert (order.status, order.average) == (STATUS_FILLED, 100.0)
    assert broker.positions[SYMBOL] == 1.0


def test_stop_limit_does_not_chase_a_gap_beyond_the_limit():
    async def scenario():
        broker = PaperBroker(10_000.0, FREE)
        order = await broker.create_order(SYMBOL, "buy", "stop_limit", 1.0, 100.5, 100.0)
        await broker.on_bar(SYMBOL, bar(101.0, 101.5, 100.8, 101.2))  # opens above the limit
        assert order.status == STATUS_OPEN
        await broker.on_bar(SYMBOL, bar(100.2, 100.4, 100.0, 100.3, i=1))  # gap inside the limit
        return order

    order = run(scenario())
    assert (order.status, order.average) == (STATUS_FILLED, 100.2)


def test_stop_sell_gapping_through_fills_at_the_worse_open():
    async def scenario():
        broker = PaperBroker(10_000.0, FREE)
        await broker.on_bar(SYMBOL, bar(100, 100, 100, 100))
        await broker.create_order(SYMBOL, "buy", "market", 1.0)
        stop = await broker.create_order(SYMBOL, "sell", "stop", 1.0, trigger_price=99.0)
        await broker.on_bar(SYMBOL, bar(98.0, 98.5, 97.5, 98.2, i=1))
        return broker, stop

    broker, stop = run(scenario())
    assert stop.average == 98.0
    assert broker.positions[SYMBOL] == 0.0


def test_stop_is_matched_before_target_and_reduce_only_prevents_a_flip():
    """One bar touches both exits: only the stop fills, and the position ends flat."""

    async def scenario():
        broker = PaperBroker(10_000.0, FREE)
        await broker.on_bar(SYMBOL, bar(100, 100, 100, 100))
        await broker.create_order(SYMBOL, "buy", "market", 1.0)
        stop = await broker.create_order(
            SYMBOL, "sell", "stop", 1.0, trigger_price=99.0, reduce_only=True
        )
        target = await broker.create_order(
            SYMBOL, "sell", "limit", 1.0, price=102.0, reduce_only=True
        )
        await broker.on_bar(SYMBOL, bar(100.0, 102.5, 98.5, 101.0, i=1))
        return broker, stop, target

    broker, stop, target = run(scenario())
    assert stop.status == STATUS_FILLED
    assert target.status == STATUS_CANCELED  # nothing left to close
    assert broker.positions[SYMBOL] == 0.0  # not short


def test_fees_and_slippage_reduce_equity():
    async def scenario():
        costs = CostConfig(taker_fee=0.001, maker_fee=0.001, slippage_bps=10.0)
        broker = PaperBroker(10_000.0, costs)
        await broker.on_bar(SYMBOL, bar(100, 100, 100, 100))
        await broker.create_order(SYMBOL, "buy", "market", 10.0)
        await broker.create_order(SYMBOL, "sell", "market", 10.0)
        return await broker.fetch_equity(), broker.fills

    equity, fills = run(scenario())
    buy, sell = 100.0 * 1.001, 100.0 * 0.999
    expected = 10_000.0 - 10 * (buy - sell) - 10 * buy * 0.001 - 10 * sell * 0.001
    assert equity == pytest.approx(expected)
    assert equity < 10_000.0 and len(fills) == 2


def test_cancelled_order_never_fills():
    async def scenario():
        broker = PaperBroker(10_000.0, FREE)
        order = await broker.create_order(SYMBOL, "buy", "stop_limit", 1.0, 100.5, 100.0)
        await broker.cancel_order(order.id, SYMBOL)
        await broker.on_bar(SYMBOL, bar(99.5, 101.0, 99.4, 100.6))
        return broker, order

    broker, order = run(scenario())
    assert order.status == STATUS_CANCELED and broker.positions[SYMBOL] == 0.0


# --------------------------------------------------------------- CcxtBroker
class FakeExchange:
    """Minimal stand-in for a ccxt exchange with scripted failures."""

    has: dict = {}

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0
        self.created = 0
        self.open_orders: list[dict] = []

    async def fetch_ticker(self, symbol):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"last": 123.0}

    def amount_to_precision(self, symbol, amount):
        return str(amount)

    def price_to_precision(self, symbol, price):
        return str(price)

    async def fetch_open_orders(self, symbol):
        return self.open_orders

    async def create_order(self, symbol, type, side, amount, price, params):
        self.created += 1
        raw = {"id": "1", "symbol": symbol, "side": side, "amount": amount, "status": "open",
               "clientOrderId": params["clientOrderId"]}  # fmt: skip
        self.open_orders.append(raw)
        if self.failures:  # the order reached the exchange but the response was lost
            raise self.failures.pop(0)
        return raw


def make_broker(settings, exchange):
    fast = settings.model_copy(
        update={"execution": settings.execution.model_copy(update={"retry_base_delay": 0.001})}
    )
    counts = {"error": 0, "success": 0}
    broker = CcxtBroker(
        fast,
        Credentials(),
        on_error=lambda: counts.__setitem__("error", counts["error"] + 1),
        on_success=lambda: counts.__setitem__("success", counts["success"] + 1),
        exchange=exchange,
    )
    return broker, counts


def test_transient_errors_are_retried_and_reported(settings):
    exchange = FakeExchange([ccxt.RequestTimeout("t"), ccxt.RateLimitExceeded("r")])
    broker, counts = make_broker(settings, exchange)
    assert run(broker.fetch_last_price(SYMBOL)) == 123.0
    assert exchange.calls == 3
    assert counts == {"error": 2, "success": 1}


def test_permanent_errors_fail_immediately(settings):
    exchange = FakeExchange([ccxt.InsufficientFunds("no money")])
    broker, counts = make_broker(settings, exchange)
    with pytest.raises(BrokerError):
        run(broker.fetch_last_price(SYMBOL))
    assert exchange.calls == 1  # no retry: it cannot succeed
    assert counts["error"] == 1


def test_retries_are_bounded(settings):
    exchange = FakeExchange([ccxt.NetworkError("down")] * 50)
    broker, counts = make_broker(settings, exchange)
    with pytest.raises(BrokerError):
        run(broker.fetch_last_price(SYMBOL))
    assert exchange.calls == settings.execution.max_retries


def test_order_retry_after_timeout_does_not_duplicate_the_order(settings):
    """The request timed out AFTER the exchange accepted it. The retry must
    find the existing order by client id instead of sending a second one."""
    exchange = FakeExchange([ccxt.RequestTimeout("lost response")])
    broker, _ = make_broker(settings, exchange)
    order = run(broker.create_order(SYMBOL, "buy", "stop_limit", 1.0, 100.5, 100.0))
    assert exchange.created == 1
    assert order.id == "1" and order.client_id.startswith("cqb")
