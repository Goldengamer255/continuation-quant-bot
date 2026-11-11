"""Structured (JSON) logging.

Why structured logs
-------------------
A live trading system is debugged after the fact, from logs. Free-text lines
like ``"placed order for BTC at 64000"`` have to be regex-parsed. One JSON
object per line can be loaded straight into pandas / Loki / CloudWatch and
queried: "show every ``entry_filled`` event for BTC/USDT where slippage was
above 5 bps".

Usage::

    logger.info("entry_submitted", extra={"symbol": "BTC/USDT", "qty": 0.01})

The message is the *event name*; everything else goes in ``extra``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else was supplied via ``extra=``.
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON line with UTC timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger once, at process start."""
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s  %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # ccxt is chatty at DEBUG and can echo request details; keep it quiet.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
