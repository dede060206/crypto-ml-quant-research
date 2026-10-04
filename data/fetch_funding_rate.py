"""Retain actual settlement timestamps; never assume a fixed 8h interval."""
import pandas as pd
from .common import cached, combine, cursor_for, request, save

COLUMNS = ["timestamp", "funding_rate"]


def fetch_funding_rate(exchange, symbol, start, end, path):
    old = cached(path, COLUMNS)
    stop = min(end, exchange.milliseconds())
    cursor = cursor_for(old, start)
    while cursor < stop:
        rows = request(lambda: exchange.fetch_funding_rate_history(symbol, since=cursor, limit=1000, params={"endTime": stop - 1}))
        if not rows:
            break
        latest = max(int(row["timestamp"]) for row in rows)
        if latest < cursor:
            raise RuntimeError("Funding pagination made no progress")
        batch = pd.DataFrame([{"timestamp": row["timestamp"], "funding_rate": row["fundingRate"]} for row in rows], columns=COLUMNS)
        batch = batch[(batch.timestamp >= start) & (batch.timestamp < stop)]
        old = combine(old, batch, COLUMNS)
        save(old, path)
        cursor = latest + 1
    save(old, path)
    return old[(old.timestamp >= start) & (old.timestamp < stop)].copy()
