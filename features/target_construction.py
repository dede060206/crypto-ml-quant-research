"""24h open-to-open direction label. Future prices never enter the feature list."""
import numpy as np
from backtest.cost_model import Costs
from .price_features import INTERVAL_MS


def cost_threshold(costs=Costs()):
    # Fixed gross price move required to break even for a long round trip.
    return (1+costs.slippage)*(1+costs.fee)/((1-costs.slippage)*(1-costs.fee))-1


def construct_targets(candles, horizon_bars=3, costs=Costs()):
    if not isinstance(horizon_bars,int) or horizon_bars < 1:
        raise ValueError('Horizon must be a positive integer')
    c=candles.sort_values('timestamp').reset_index(drop=True)
    if c.timestamp.duplicated().any():
        raise ValueError('Duplicate candle timestamps')
    result=c[['timestamp']].copy()
    result['entry_time']=c.timestamp.shift(-1)
    result['label_end_time']=c.timestamp.shift(-(horizon_bars+1))
    result['entry_price']=c.open.shift(-1)
    result['exit_price']=c.open.shift(-(horizon_bars+1))
    valid=np.ones(len(c),dtype=bool)
    for offset in range(1,horizon_bars+2):
        valid &= (c.timestamp.shift(-offset)==c.timestamp+offset*INTERVAL_MS).to_numpy()
    result['label_valid']=valid
    result['future_return']=(result.exit_price/result.entry_price-1).where(valid)
    result['cost_threshold']=cost_threshold(costs)
    result['target']=(result.future_return>result.cost_threshold).astype('Int64').where(valid)
    return result
