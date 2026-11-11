"""Live trading engine: bar listener, signal evaluation and order routing.

What the engine does on every closed bar
----------------------------------------
1. Append the bar to a rolling buffer (ignoring duplicates, refilling gaps).
2. **Reconcile**: ask the broker what happened to our orders. Did the entry
   fill? Did the stop or target fill?
3. Report equity to the risk manager. If the kill-switch is tripped:
   cancel everything, flatten, and place no new orders.
4. Advance the per-symbol **state machine**::

       FLAT --setup + model + risk OK--> PENDING_ENTRY --filled--> IN_POSITION
        ^                                    |                         |
        +------ expired / cancelled ---------+                         |
        +------ stop / target / time barrier / kill-switch ------------+

5. If FLAT: compute features with the *research* functions, check for a
   setup on the bar that just closed, score it with the model, size it with
   the risk manager, and submit a stop-limit entry.

Why stop-limit entries instead of market orders
-----------------------------------------------
A breakout strategy buys exactly when everyone else is buying. A market
order there pays whatever the thin order book asks (slippage). Instead a
**stop-limit** order rests just above the consolidation high:

* the *stop (trigger)* means we only buy once price proves the breakout;
* the *limit* caps the worst price we accept. If price gaps far beyond it
  we miss the trade rather than overpay — a missed trade costs nothing, a
  bad fill has negative expectancy from the first tick.

The order is cancelled after ``entry_valid_bars``: a setup that has not
triggered by then is stale.

Protecting an open position
---------------------------
As soon as the entry fills, a **stop-loss order is placed on the exchange**.
It lives on the exchange's servers, so the position stays protected if this
process crashes or loses its connection. If the protective order cannot be
placed, the position is closed at market immediately: an unprotected
position is never left open. The profit target is a resting limit order
where the venue allows two exit orders at once, otherwise the engine watches
price and exits itself (``Broker.native_bracket``).

Consistency with research
-------------------------
Features, setup detection, entry levels and barrier levels are imported from
``research``; sizing and limits come from the same ``RiskManager`` the
backtester uses. The bar-counting rules for order expiry and the time
barrier match ``labeling.simulate_trade``. Live behaviour should therefore
differ from the backtest only by genuine market effects.

Concurrency model
-----------------
One asyncio task per symbol consumes that symbol's bar stream, and (for real
exchanges) one task polls order status between bars so protective stops go
out within seconds of a fill. A single ``asyncio.Lock`` serialises every
state mutation: correctness matters far more than parallelism when a bar
arrives once an hour.

Graceful shutdown
-----------------
SIGINT/SIGTERM set a stop event. The engine then cancels pending *entry*
orders (so nothing new can fill unattended), optionally flattens, and closes
connections. Exchange-side protective stops are left in place by default.

Known limitations
-----------------
* State is in memory. After a restart the engine does not re-adopt an open
  position; the exchange-side stop still protects it, but manage it by hand.
* Partial fills are only resolved when the entry order completes or expires.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from core.config import (
    Credentials,
    Settings,
    live_trading_armed,
    load_credentials,
    load_settings,
)
from core.logging_setup import setup_logging
from data.fetcher import OHLCV_COLUMNS
from execution.broker import (
    STATUS_CANCELED,
    STATUS_FILLED,
    Bar,
    Broker,
    BrokerError,
    CcxtBroker,
    MarketFeed,
    Order,
    PaperBroker,
)
from execution.risk_manager import HaltReason, RiskManager
from research.features import build_feature_matrix, compute_features, detect_setups
from research.labeling import barrier_levels, entry_levels

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    FLAT = "flat"
    PENDING = "pending_entry"
    OPEN = "in_position"


@dataclass
class SymbolState:
    """Everything the engine remembers about one symbol."""

    phase: Phase = Phase.FLAT
    side: int = 0
    qty: float = 0.0
    atr: float = 0.0
    ref_price: float = 0.0  # planned trigger, then actual fill price
    stop_price: float = 0.0
    target_price: float = 0.0
    entry_order: Order | None = None
    stop_order: Order | None = None
    target_order: Order | None = None
    bars_pending: int = 0
    bars_held: int = 0

    @property
    def exit_side(self) -> str:
        return "sell" if self.side > 0 else "buy"

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        feed: MarketFeed,
        broker: Broker,
        risk: RiskManager,
        model,
    ) -> None:
        f = settings.features
        if f.use_order_flow or f.use_market_context or f.use_regime:
            # The live feed only supplies plain OHLCV for the traded symbols.
            # Running a model trained on extra inputs without them would be
            # train/serve skew, so refuse rather than improvise.
            raise NotImplementedError(
                "Optional feature groups are research-only: the live engine does not "
                "yet supply order-flow or context data."
            )
        self.settings = settings
        self.feed = feed
        self.broker = broker
        self.risk = risk
        self.model = model
        self.symbols = list(settings.market.symbols)
        self._timeframe = settings.market.timeframe
        self._bar_td = settings.bar_timedelta
        self._buffer_size = settings.execution.warmup_bars
        self._state = {symbol: SymbolState() for symbol in self.symbols}
        self._buffers: dict[str, pd.DataFrame] = {}
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._halt_handled = False

    # =================================================================== run
    def request_shutdown(self) -> None:
        self._stop.set()

    async def run(self, install_signal_handlers: bool = True) -> None:
        """Run until the feed ends or a shutdown is requested."""
        if install_signal_handlers:
            self._install_signal_handlers()
        await self._warm_up()

        workers = [asyncio.create_task(self._run_symbol(s), name=f"bars:{s}") for s in self.symbols]
        background = [asyncio.create_task(self._stop.wait(), name="stop")]
        if self.broker.needs_polling:
            background.append(asyncio.create_task(self._reconcile_loop(), name="reconcile"))
        streams = asyncio.gather(*workers)
        try:
            await asyncio.wait({streams, background[0]}, return_when=asyncio.FIRST_COMPLETED)
            if streams.done() and not streams.cancelled():
                streams.result()  # surface a crashed worker
        finally:
            for task in (*workers, *background):
                task.cancel()
            await asyncio.gather(*workers, *background, return_exceptions=True)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await streams
            await self._shutdown()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_shutdown)
            except NotImplementedError:  # Windows event loops lack add_signal_handler
                signal.signal(sig, lambda *_: loop.call_soon_threadsafe(self.request_shutdown))

    async def _warm_up(self) -> None:
        await self.broker.connect()
        if self.feed is not self.broker:
            await self.feed.connect()
        for symbol in self.symbols:
            if self.settings.execution.cancel_orphans_on_start:
                # Orders left over from a previous run are not in our state machine.
                await self.broker.cancel_all_orders(symbol)
            history = await self.feed.fetch_history(symbol, self._timeframe, self._buffer_size)
            self._buffers[symbol] = history[OHLCV_COLUMNS]
            logger.info("warmed_up", extra={"symbol": symbol, "bars": len(history)})

    async def _run_symbol(self, symbol: str) -> None:
        async for bar in self.feed.stream_closed_bars(symbol, self._timeframe):
            if self._stop.is_set():
                break
            try:
                await self._on_bar(symbol, bar)
            except BrokerError as exc:
                # Already counted by the risk manager; state is retried next bar.
                logger.error("bar_processing_failed", extra={"symbol": symbol, "error": str(exc)})
            except Exception:
                # Unknown bug: fail safe. Stop trading and get flat.
                logger.exception("internal_error", extra={"symbol": symbol})
                self.risk.trip(HaltReason.INTERNAL_ERROR, "unhandled exception in bar handler")
                async with self._lock:
                    await self._handle_halt()

    async def _reconcile_loop(self) -> None:
        """Between bars: detect fills quickly so protective orders go out fast."""
        while not self._stop.is_set():
            await asyncio.sleep(self.settings.execution.reconcile_seconds)
            async with self._lock:
                for symbol in self.symbols:
                    try:
                        await self._reconcile_symbol(symbol)
                    except BrokerError as exc:
                        logger.error(
                            "reconcile_failed", extra={"symbol": symbol, "error": str(exc)}
                        )
                if self.risk.halted:
                    await self._handle_halt()

    async def _shutdown(self) -> None:
        logger.info("shutdown_started")
        async with self._lock:
            for symbol, state in self._state.items():
                if state.phase is Phase.PENDING:
                    with contextlib.suppress(BrokerError):
                        await self._cancel_entry(symbol, "shutdown")
            if self.settings.execution.flatten_on_shutdown:
                await self._flatten_all("shutdown")
        with contextlib.suppress(Exception):
            await self.broker.close()
            if self.feed is not self.broker:
                await self.feed.close()
        logger.info("shutdown_complete")

    # ============================================================ bar handler
    async def _on_bar(self, symbol: str, bar: Bar) -> None:
        async with self._lock:
            if not await self._append_bar(symbol, bar):
                return
            await self.broker.on_bar(symbol, bar)  # paper fills; no-op for a real exchange
            await self._reconcile_symbol(symbol)

            now = bar.timestamp + self._bar_td  # the moment this bar closed
            self.risk.update_equity(await self.broker.fetch_equity(), now)
            if self.risk.halted:
                await self._handle_halt()
                return
            self._halt_handled = False

            state = self._state[symbol]
            if state.phase is Phase.OPEN:
                state.bars_held += 1
                if state.bars_held >= self.settings.labeling.max_holding_bars:
                    await self._exit_position(symbol, "time_barrier")
            elif state.phase is Phase.PENDING:
                state.bars_pending += 1
                if state.bars_pending >= self.settings.labeling.entry_valid_bars:
                    await self._cancel_entry(symbol, "expired")

            if state.phase is Phase.FLAT:
                await self._maybe_enter(symbol)

    async def _append_bar(self, symbol: str, bar: Bar) -> bool:
        """Add a bar to the buffer. Returns False for duplicates / stale bars."""
        buffer = self._buffers[symbol]
        if len(buffer) and bar.timestamp <= buffer.index[-1]:
            return False
        if len(buffer) and bar.timestamp > buffer.index[-1] + self._bar_td:
            # Missed bars (disconnect). Rolling features over a hole would be
            # wrong, so rebuild the buffer from the exchange.
            logger.warning("bar_gap_detected", extra={"symbol": symbol})
            history = await self.feed.fetch_history(symbol, self._timeframe, self._buffer_size)
            buffer = history[OHLCV_COLUMNS]
            buffer = buffer[buffer.index < bar.timestamp]
        row = pd.DataFrame(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume]],
            columns=OHLCV_COLUMNS,
            index=pd.DatetimeIndex([bar.timestamp], name="timestamp"),
        )
        self._buffers[symbol] = pd.concat([buffer, row]).tail(self._buffer_size)
        return True

    # ================================================================= entry
    async def _maybe_enter(self, symbol: str) -> None:
        """Evaluate the bar that just closed and submit an entry if warranted."""
        buffer = self._buffers[symbol]
        features = compute_features(buffer, self.settings.features)
        events = detect_setups(features, self.settings.setup)
        if events.empty or events.index[-1] != buffer.index[-1]:
            return  # no setup on the latest bar

        event = events.iloc[[-1]]
        probability = float(self.model.predict_proba(build_feature_matrix(features, event))[0, 1])
        side, atr = int(event["side"].iloc[0]), float(event["atr"].iloc[0])
        log = {"symbol": symbol, "side": side, "probability": round(probability, 4)}
        if probability <= self.settings.model.probability_threshold:
            logger.info("setup_rejected_by_model", extra=log)
            return

        lab = self.settings.labeling
        trigger, limit = entry_levels(
            side, float(event["cons_high"].iloc[0]), float(event["cons_low"].iloc[0]), atr, lab
        )
        planned_stop, _ = barrier_levels(side, trigger, atr, lab)
        busy = [s for s in self._state.values() if s.phase is not Phase.FLAT]
        decision = self.risk.evaluate_entry(
            equity=await self.broker.fetch_equity(),
            entry_price=trigger,
            stop_price=planned_stop,
            open_positions=len(busy),
            gross_exposure=sum(s.notional for s in busy),
        )
        if not decision.approved:
            logger.info("setup_rejected_by_risk", extra=log | {"reason": decision.reason})
            return

        order = await self.broker.create_order(
            symbol,
            "buy" if side > 0 else "sell",
            "stop_limit",
            decision.quantity,
            price=limit,
            trigger_price=trigger,
        )
        self._state[symbol] = SymbolState(
            phase=Phase.PENDING,
            side=side,
            qty=order.amount,
            atr=atr,
            ref_price=trigger,
            entry_order=order,
        )
        logger.info(
            "entry_submitted",
            extra=log | {"qty": order.amount, "trigger": trigger, "limit": limit, "id": order.id},
        )

    async def _cancel_entry(self, symbol: str, reason: str) -> None:
        state = self._state[symbol]
        if state.entry_order is None:
            self._state[symbol] = SymbolState()
            return
        await self.broker.cancel_order(state.entry_order.id, symbol)
        final = await self.broker.fetch_order(state.entry_order.id, symbol)
        if final.filled > 0:  # filled (fully or partly) before the cancel landed
            await self._on_entry_filled(symbol, final)
            return
        self._state[symbol] = SymbolState()
        logger.info("entry_cancelled", extra={"symbol": symbol, "reason": reason})

    # ============================================================= reconcile
    async def _reconcile_symbol(self, symbol: str) -> None:
        """Bring our state in line with what the broker says happened."""
        state = self._state[symbol]

        if state.phase is Phase.PENDING and state.entry_order is not None:
            order = await self.broker.fetch_order(state.entry_order.id, symbol)
            if order.status == STATUS_FILLED and order.filled > 0:
                await self._on_entry_filled(symbol, order)
            elif order.status == STATUS_CANCELED:
                if order.filled > 0:
                    await self._on_entry_filled(symbol, order)
                else:
                    self._state[symbol] = SymbolState()
            return

        if state.phase is not Phase.OPEN:
            return
        if state.stop_order is None:
            await self._exit_position(symbol, "unprotected_position")
            return

        stop = await self.broker.fetch_order(state.stop_order.id, symbol)
        if stop.status == STATUS_FILLED:
            await self._finish_exit(symbol, "stop_loss", stop.average, sibling=state.target_order)
            return
        if stop.status == STATUS_CANCELED:  # cancelled by someone other than us
            state.stop_order = None
            await self._exit_position(symbol, "stop_order_lost")
            return

        if state.target_order is not None:
            target = await self.broker.fetch_order(state.target_order.id, symbol)
            if target.status == STATUS_FILLED:
                await self._finish_exit(
                    symbol, "profit_target", target.average, sibling=state.stop_order
                )
        else:
            # Venue cannot hold two exit orders: watch the target ourselves.
            price = await self.broker.fetch_last_price(symbol)
            if state.side * (price - state.target_price) >= 0:
                await self._exit_position(symbol, "profit_target")

    async def _on_entry_filled(self, symbol: str, order: Order) -> None:
        state = self._state[symbol]
        state.phase = Phase.OPEN
        state.qty = order.filled
        state.ref_price = float(order.average or state.ref_price)
        state.entry_order = None
        state.bars_held = 0
        # Barriers are anchored to the ACTUAL fill, same as in the labeler.
        state.stop_price, state.target_price = barrier_levels(
            state.side, state.ref_price, state.atr, self.settings.labeling
        )
        logger.info(
            "entry_filled",
            extra={
                "symbol": symbol,
                "qty": state.qty,
                "price": state.ref_price,
                "stop": state.stop_price,
                "target": state.target_price,
            },
        )
        try:
            state.stop_order = await self.broker.create_order(
                symbol,
                state.exit_side,
                "stop",
                state.qty,
                trigger_price=state.stop_price,
                reduce_only=True,
            )
            if self.broker.native_bracket:
                state.target_order = await self.broker.create_order(
                    symbol,
                    state.exit_side,
                    "limit",
                    state.qty,
                    price=state.target_price,
                    reduce_only=True,
                )
        except BrokerError:
            # Never hold a position we could not protect.
            logger.critical("protective_order_failed", extra={"symbol": symbol})
            await self._exit_position(symbol, "protective_order_failed")

    # ================================================================== exit
    async def _finish_exit(
        self, symbol: str, reason: str, exit_price: float | None, sibling: Order | None
    ) -> None:
        """An exchange-side exit order filled: cancel its sibling and go flat."""
        state = self._state[symbol]
        if sibling is not None:
            with contextlib.suppress(BrokerError):
                await self.broker.cancel_order(sibling.id, symbol)
        self._log_closed(symbol, state, reason, exit_price)
        self._state[symbol] = SymbolState()

    async def _exit_position(self, symbol: str, reason: str) -> None:
        """Close at market. Cancels resting exits first and accounts for any
        quantity they filled in the meantime, so we never over-sell."""
        state = self._state[symbol]
        remaining = state.qty
        for order in (state.stop_order, state.target_order):
            if order is None:
                continue
            await self.broker.cancel_order(order.id, symbol)
            remaining -= (await self.broker.fetch_order(order.id, symbol)).filled
        exit_price = None
        if remaining > 1e-12:
            fill = await self.broker.create_order(
                symbol, state.exit_side, "market", remaining, reduce_only=True
            )
            exit_price = fill.average
        self._log_closed(symbol, state, reason, exit_price)
        self._state[symbol] = SymbolState()

    @staticmethod
    def _log_closed(symbol: str, state: SymbolState, reason: str, exit_price) -> None:
        pnl = None
        if exit_price is not None:
            pnl = round(state.side * state.qty * (float(exit_price) - state.ref_price), 6)
        logger.info(
            "position_closed",
            extra={
                "symbol": symbol,
                "reason": reason,
                "qty": state.qty,
                "entry": state.ref_price,
                "exit": exit_price,
                "pnl_before_fees": pnl,
                "bars_held": state.bars_held,
            },
        )

    # ============================================================ kill-switch
    async def _handle_halt(self) -> None:
        """Kill-switch response: cancel everything and flatten, once per halt."""
        if self._halt_handled:
            return
        self._halt_handled = True
        reason = self.risk.halt_reason.value if self.risk.halt_reason else "unknown"
        logger.critical("halting_and_flattening", extra={"reason": reason})
        await self._flatten_all(reason)

    async def _flatten_all(self, reason: str) -> None:
        for symbol, state in self._state.items():
            try:
                if state.phase is Phase.PENDING:
                    await self._cancel_entry(symbol, reason)
                if self._state[symbol].phase is Phase.OPEN:
                    await self._exit_position(symbol, reason)
            except BrokerError as exc:
                # Best effort: keep going so one bad symbol cannot block the rest.
                # The exchange-side stop (if placed) still protects the position.
                logger.critical("flatten_failed", extra={"symbol": symbol, "error": str(exc)})


# ============================================================================
# Entry point
# ============================================================================
async def _amain(settings: Settings) -> None:
    from research.train import load_model

    model = load_model(settings)  # refuses to load if config no longer matches training
    risk = RiskManager(settings.risk)

    if settings.execution.mode == "live":
        # Two independent switches must both be on to send real orders.
        if not live_trading_armed():
            raise SystemExit("execution.mode is 'live' but CQB_LIVE_TRADING is not YES. Refusing.")
        credentials = load_credentials()
        if not credentials.present:
            raise SystemExit("Live mode needs EXCHANGE_API_KEY / EXCHANGE_API_SECRET in .env")
        broker = CcxtBroker(settings, credentials, risk.record_api_error, risk.record_api_success)
        feed: MarketFeed = broker
    else:
        # Paper: real public market data, simulated fills, no credentials.
        feed = CcxtBroker(settings, Credentials())
        broker = PaperBroker(settings.backtest.initial_equity, settings.costs)

    logger.info(
        "engine_starting",
        extra={"mode": settings.execution.mode, "symbols": settings.market.symbols},
    )
    await TradingEngine(settings, feed, broker, risk, model).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the continuation trading engine")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.execution.log_level, settings.execution.log_json)
    asyncio.run(_amain(settings))


if __name__ == "__main__":
    main()
