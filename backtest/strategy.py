"""Non-overlapping long trades, fixed 10% entry equity, causal funding price proxy."""
import numpy as np
import pandas as pd
from .cost_model import Costs
from .engine import INTERVAL


def backtest_predictions(predictions,candles,funding=None,costs=Costs(),threshold=.55,fraction=.1,capital=10000.):
    if not 0<fraction<=1 or not 0<=threshold<=1 or capital<=0:
        raise ValueError('Invalid strategy parameters')
    p=predictions.sort_values('signal_time');c=candles.sort_values('timestamp')
    closes=c.timestamp.to_numpy(dtype='int64')+INTERVAL-1
    prices=c.close.to_numpy(dtype=float)
    start=int(p.entry_time.min());end=int(p.label_end_time.max())
    timeline=np.unique(np.r_[start-1,closes[(closes>=start)&(closes<=end)],p.entry_time.to_numpy(dtype='int64'),p.label_end_time.to_numpy(dtype='int64')])
    changes=np.zeros(len(timeline));trade_ledger=[]
    equity=capital;free_at=start
    for row in p.itertuples():
        if row.probability<threshold or row.entry_time<free_at:continue
        enter=int(row.entry_time);exit=int(row.label_end_time)
        if not row.signal_time<enter or exit-enter!=3*INTERVAL:
            raise ValueError('Invalid signal/holding interval')
        held=c[(c.timestamp>=enter)&(c.timestamp<=exit)]
        if not np.array_equal(held.timestamp.to_numpy(),np.arange(enter,exit+INTERVAL,INTERVAL)):
            raise ValueError('Trade crosses missing candles')
        entry_price=float(row.entry_price);exit_price=float(row.exit_price)
        fill=entry_price*(1+costs.slippage);exit_fill=exit_price*(1-costs.slippage)
        units=equity*fraction/(fill*(1+costs.fee))
        entry_fee=units*fill*costs.fee;exit_fee=units*exit_fill*costs.fee
        ia=int(np.searchsorted(timeline,enter));ib=int(np.searchsorted(timeline,exit))
        # Cumulative PnL during the trade, mark to completed close, then cash at exit.
        ts=timeline[ia:ib+1]
        indices=np.searchsorted(closes,ts,side='right')-1
        mark=prices[indices];mark[0]=entry_price;mark[-1]=exit_price
        path=units*(mark-fill)-entry_fee
        paid=0.;events=0
        if funding is not None:
            f=funding[(funding.timestamp>enter)&(funding.timestamp<=exit)]
            fi=np.searchsorted(closes,f.timestamp.to_numpy(),side='right')-1
            if (fi<0).any():raise ValueError('Missing causal funding price')
            amounts=units*prices[fi]*f.funding_rate.to_numpy()
            payments=np.zeros(len(ts));dest=np.searchsorted(ts,f.timestamp.to_numpy(),side='left')
            np.add.at(payments,dest,amounts);path-=payments.cumsum();paid=float(amounts.sum());events=len(f)
        path[-1]-=units*(exit_price-exit_fill)+exit_fee
        delta=np.diff(np.r_[0,path]);changes[ia:ib+1]+=delta
        net=float(path[-1]);equity+=net;free_at=exit
        if equity<=0:raise ValueError('Nonpositive equity')
        trade_ledger.append({'signal_time':int(row.signal_time),'entry_time':enter,'exit_time':exit,'units':units,'entry_equity':equity-net,'net_pnl':net,'gross_price_pnl':units*(exit_price-entry_price),'fees':entry_fee+exit_fee,'slippage_cost':units*(fill-entry_price+exit_price-exit_fill),'funding_paid_net':paid,'funding_events':events,'probability':float(row.probability)})
    curve=pd.DataFrame({'equity':capital+changes.cumsum()},index=pd.to_datetime(timeline,unit='ms',utc=True));curve.index.name='time'
    trades=pd.DataFrame(trade_ledger)
    np.testing.assert_allclose(curve.equity.iloc[-1],capital+sum(x['net_pnl'] for x in trade_ledger),rtol=1e-10)
    return curve,trades
