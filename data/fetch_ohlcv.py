"""Historical USD-M perpetual candles; timestamps denote candle opens."""
import pandas as pd
from .common import cached, combine, cursor_for, request, save

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch_ohlcv(exchange, symbol, timeframe, start, end, path):
    old = cached(path, COLUMNS)
    interval = exchange.parse_timeframe(timeframe) * 1000
    # Exclude still-open candles even when the requested end lies in the future.
    stop = min(end, exchange.milliseconds())
    cursor = cursor_for(old, start)
    while cursor < stop:
        rows = request(lambda: exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000, params={"endTime": stop - 1}))
        if not rows:
            break
        latest = max(int(row[0]) for row in rows)
        if latest < cursor:
            raise RuntimeError("OHLCV pagination made no progress")
        batch = pd.DataFrame(rows, columns=COLUMNS)
        batch = batch[(batch.timestamp >= start) & (batch.timestamp + interval <= stop)]
        old = combine(old, batch, COLUMNS)
        save(old, path)  # checkpoint each page so interrupted downloads can resume
        cursor = latest + interval
    result = old[(old.timestamp >= start) & (old.timestamp + interval <= stop)].copy()
    save(old, path)
    return result
