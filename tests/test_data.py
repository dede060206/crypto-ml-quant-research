import pandas as pd
import pytest
from data.common import milliseconds
from data.fetch_ohlcv import fetch_ohlcv
from data.fetch_funding_rate import fetch_funding_rate
from data.db import query_parquet

class Exchange:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.cursors = []
    def parse_timeframe(self, timeframe):
        return 1
    def milliseconds(self):
        return 10000
    def fetch_ohlcv(self, symbol, timeframe, since, **kwargs):
        self.cursors.append(since)
        return next(self.pages, [])
    def fetch_funding_rate_history(self, symbol, since, **kwargs):
        self.cursors.append(since)
        return next(self.pages, [])


def candle(t):
    return [t, 10., 12., 9., 11., 4.]


def test_pagination_dedup_closed_candles_and_resume(tmp_path):
    path = tmp_path / 'prices.parquet'
    exchange = Exchange([[candle(0), candle(1000)], [candle(1000), candle(2000), candle(3000)]])
    result = fetch_ohlcv(exchange, 'BTC', '1s', 0, 3500, path)
    assert result.timestamp.tolist() == [0, 1000, 2000]
    assert exchange.cursors == [0, 2000]
    result = fetch_ohlcv(Exchange([[candle(3000)]]), 'BTC', '1s', 0, 4000, path)
    assert result.timestamp.tolist() == [0, 1000, 2000, 3000]
    assert query_parquet(path, 1000, 3000).timestamp.tolist() == [1000, 2000]


def test_funding_actual_irregular_timestamps(tmp_path):
    rows = [{'timestamp': t, 'fundingRate': -.0001} for t in (0, 1000, 1500, 2000)]
    result = fetch_funding_rate(Exchange([rows]), 'BTC', 0, 2000, tmp_path / 'funding.parquet')
    assert result.timestamp.tolist() == [0, 1000, 1500]


def test_no_progress_raises(tmp_path):
    with pytest.raises(RuntimeError, match='no progress'):
        fetch_ohlcv(Exchange([[candle(0)]]), 'BTC', '1s', 1000, 3000, tmp_path / 'p.parquet')


def test_earlier_start_refetches_and_empty_cache(tmp_path):
    path = tmp_path / 'prices.parquet'
    fetch_ohlcv(Exchange([]), 'BTC', '1s', 1000, 3000, path)
    fetch_ohlcv(Exchange([[candle(1000)]]), 'BTC', '1s', 1000, 2000, path)
    exchange = Exchange([[candle(0), candle(1000)]])
    assert fetch_ohlcv(exchange, 'BTC', '1s', 0, 2000, path).timestamp.tolist() == [0, 1000]
    assert exchange.cursors[0] == 0


def test_utc_requires_timezone():
    assert milliseconds('1970-01-01T08:00:00+08:00') == 0
    with pytest.raises(ValueError):
        milliseconds('2021-01-01')
