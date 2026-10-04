import numpy as np
import pandas as pd
import pytest
from backtest.engine import buy_and_hold, INTERVAL
from backtest.cost_model import Costs
from backtest.metrics import metrics


def candles(prices):
    return pd.DataFrame({'timestamp':np.arange(len(prices))*INTERVAL,'close':prices})


def test_price_identity_and_cost_hand_calculation():
    c = candles([100,120,110])
    curve,info=buy_and_hold(c,capital=1000,fraction=.1,costs=Costs(0,0))
    assert curve.equity.tolist()==[1000,1000,1020,1010]
    assert metrics(curve)['max_drawdown']==pytest.approx(1010/1020-1)
    curve,info=buy_and_hold(c,capital=1000,fraction=.1,costs=Costs(.01,.02))
    q=100/(102*1.01)
    expected=1000+q*(110*.98-102)-q*102*.01-q*110*.98*.01
    assert curve.equity.iloc[-1]==pytest.approx(expected)
    assert info['accounting_error']==pytest.approx(0,abs=1e-10)


def test_funding_sign_boundary_and_causal_price():
    c=candles([100,200,300])
    entry=INTERVAL-1
    f=pd.DataFrame({'timestamp':[entry,INTERVAL+2,2*INTERVAL+3,3*INTERVAL], 'funding_rate':[99,.01,-.01,99]})
    curve,info=buy_and_hold(c,f,capital=1000,fraction=.1,costs=Costs(0,0))
    # q=1, pay 1 using prior close 100, receive 2 using prior close 200.
    assert info['funding_events']==2
    assert info['funding_paid_net']==pytest.approx(-1)
    assert curve.equity.iloc[-1]==pytest.approx(1201)
    changed=c.copy();changed.loc[2,'close']=900
    _,new=buy_and_hold(changed,f,capital=1000,fraction=.1,costs=Costs(0,0))
    assert new['funding_paid_net']==info['funding_paid_net']


def test_gap_refusal_and_flat_metrics():
    c=candles([100,100,100]);c.loc[2,'timestamp']+=INTERVAL
    with pytest.raises(ValueError,match='gaps'):
        buy_and_hold(c)
    curve,_=buy_and_hold(candles([100,100,100]),costs=Costs(0,0))
    m=metrics(curve)
    assert m['total_return']==0 and m['max_drawdown']==0
    assert m['sharpe_daily_365'] is None and m['sortino_daily_365'] is None


def test_daily_sharpe_sortino_and_recovered_drawdown_duration():
    index=pd.date_range('2022-01-01',periods=4,freq='1D',tz='UTC')
    curve=pd.DataFrame({'equity':[100.,110.,99.,120.]},index=index)
    m=metrics(curve)
    daily=np.array([0.,.1,-.1,120/99-1])
    assert m['sharpe_daily_365']==pytest.approx(daily.mean()/daily.std(ddof=1)*np.sqrt(365))
    assert m['sortino_daily_365']==pytest.approx(daily.mean()/np.sqrt((.1**2)/4)*np.sqrt(365))
    assert m['max_drawdown']==pytest.approx(-.1)
    assert m['max_drawdown_duration_days']==2


def test_slippage_increases_cost_on_flat_market():
    c=candles([100,100,100])
    free,_=buy_and_hold(c,costs=Costs(0,0))
    paid,_=buy_and_hold(c,costs=Costs(.0005,.0003))
    worse,_=buy_and_hold(c,costs=Costs(.0005,.001))
    assert worse.equity.iloc[-1]<paid.equity.iloc[-1]<free.equity.iloc[-1]
