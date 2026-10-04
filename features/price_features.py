"""Only completed candles; rolling histories restart after every missing candle."""
import numpy as np
import pandas as pd

INTERVAL_MS = 28_800_000
PRICE_COLUMNS = ['momentum_24h', 'momentum_3d', 'momentum_7d', 'volatility_3d', 'volatility_7d', 'atr_24h_fraction']


def price_features(candles):
    c = candles.sort_values('timestamp').reset_index(drop=True).copy()
    if c.timestamp.duplicated().any():
        raise ValueError('Duplicate candle timestamps')
    if c.empty or not np.isfinite(c[['open','high','low','close','volume']]).all().all():
        raise ValueError('Empty or nonfinite candles')
    if (c[['open','high','low','close']] <= 0).any().any() or (c.volume < 0).any():
        raise ValueError('Invalid price/volume')
    if (c.timestamp % INTERVAL_MS != 0).any():
        raise ValueError('Candles must be aligned to UTC 8h boundaries')
    c['signal_time'] = c.timestamp + INTERVAL_MS - 1
    c['segment'] = (c.timestamp.diff() != INTERVAL_MS).cumsum()
    pieces = []
    for _, part in c.groupby('segment', sort=False):
        part = part.copy()
        log_return = np.log(part.close).diff()
        for periods, label in [(3,'24h'),(9,'3d'),(21,'7d')]:
            part[f'momentum_{label}'] = part.close.pct_change(periods, fill_method=None)
        for periods, label in [(9,'3d'),(21,'7d')]:
            part[f'volatility_{label}'] = log_return.rolling(periods, min_periods=periods).std(ddof=1)
        previous = part.close.shift(1)
        tr = pd.concat([part.high-part.low, (part.high-previous).abs(), (part.low-previous).abs()], axis=1).max(axis=1)
        part['atr_24h_fraction'] = tr.rolling(3,min_periods=3).mean()/part.close
        pieces.append(part)
    return pd.concat(pieces).reset_index(drop=True)
