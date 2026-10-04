"""Fixed-unit, long-only Buy & Hold validation engine, not yet an ML strategy engine."""
import numpy as np
import pandas as pd
from .cost_model import Costs

INTERVAL = 28_800_000


def buy_and_hold(candles, funding=None, capital=10000., fraction=.1, costs=Costs(), strict=True):
    if capital <= 0 or not 0 < fraction <= 1:
        raise ValueError('Invalid capital or fraction')
    candles = candles.sort_values('timestamp').reset_index(drop=True)
    if len(candles) < 2 or candles.timestamp.duplicated().any():
        raise ValueError('Need at least two unique candles')
    if not np.isfinite(candles.close).all() or (candles.close <= 0).any():
        raise ValueError('Invalid prices')
    if strict and not (candles.timestamp.diff().dropna() == INTERVAL).all():
        raise ValueError('Candle gaps: choose a continuous interval')
    times = candles.timestamp.to_numpy(dtype='int64') + INTERVAL - 1
    prices = candles.close.to_numpy(dtype=float)
    entry, exit = int(times[0]), int(times[-1])
    fill = prices[0] * (1 + costs.slippage)
    # Entry fees fit inside the budget; unused collateral earns zero interest.
    units = capital * fraction / (fill * (1 + costs.fee))
    entry_fee = units * fill * costs.fee
    exit_fill = prices[-1] * (1 - costs.slippage)
    exit_fee = units * exit_fill * costs.fee
    funding_cash = np.zeros(len(prices))
    funding_events = 0
    if funding is not None and len(funding):
        funding = funding.sort_values('timestamp').copy()
        if funding.timestamp.duplicated().any() or not np.isfinite(funding.funding_rate).all():
            raise ValueError('Invalid funding observations')
        # Source carries settlement jitter in milliseconds. Use reported timestamps as-is.
        f = funding[(funding.timestamp > entry) & (funding.timestamp <= exit)]
        ts = f.timestamp.to_numpy(dtype='int64')
        previous = np.searchsorted(times, ts, side='right') - 1
        destinations = np.searchsorted(times, ts, side='left')
        if (previous < 0).any():
            raise ValueError('No historical price for funding')
        # Actual mark price is unavailable: latest COMPLETED candle close is a causal proxy.
        amounts = units * prices[previous] * f.funding_rate.to_numpy(dtype=float)
        np.add.at(funding_cash, destinations, amounts)
        funding_events = len(f)
    equity = capital + units * (prices-fill) - entry_fee - np.cumsum(funding_cash)
    equity[-1] -= units * (prices[-1]-exit_fill) + exit_fee
    if (equity <= 0).any():
        raise ValueError('Nonpositive equity: simplified engine cannot model liquidation')
    # Include untouched initial capital so entry cost is reflected in returns/drawdown.
    index = pd.to_datetime(np.r_[entry-1, times], unit='ms', utc=True)
    curve = pd.DataFrame({'equity':np.r_[capital,equity], 'price':np.r_[prices[0],prices], 'funding_cash':np.r_[0,funding_cash]}, index=index)
    curve.index.name = 'time'
    expected_final = capital + units*(exit_fill-fill) - entry_fee-exit_fee-funding_cash.sum()
    np.testing.assert_allclose(curve.equity.iloc[-1], expected_final, rtol=1e-12)
    info = {'units':units, 'entry_fee':entry_fee, 'exit_fee':exit_fee, 'funding_paid_net':float(funding_cash.sum()), 'funding_events':funding_events, 'slippage_cost':float(units*(fill-prices[0]+prices[-1]-exit_fill)), 'accounting_error':float(curve.equity.iloc[-1]-expected_final), 'entry_utc':index[1].isoformat(), 'exit_utc':index[-1].isoformat(), 'trades':1, 'max_exposure_to_equity':float(np.max(units*prices/equity))}
    return curve, info
