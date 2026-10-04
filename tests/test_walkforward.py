import pandas as pd
import numpy as np
from evaluation.walk_forward_split import walk_forward_split
from models.baseline_logreg import baseline_logreg
from backtest.strategy import backtest_predictions
from backtest.cost_model import Costs
from backtest.engine import INTERVAL as I


def test_purge_embargo_calendar_and_unique_test_samples():
    ts=pd.date_range('2021-01-01','2022-04-01',freq='8h',tz='UTC')
    t=ts.as_unit('ms').astype('int64')
    d=pd.DataFrame({'signal_time':t,'label_end_time':t+4*I})
    seen=[]
    for fold in walk_forward_split(d,end='2022-04-01'):
        start=pd.Timestamp(fold['test_start']).value//1_000_000
        assert (d.loc[fold['train'],'label_end_time']<start-I).all()
        assert not set(fold['train'])&set(fold['test'])
        seen.extend(fold['test'])
    assert len(set(seen))==len(seen)


def test_scaler_does_not_fit_test_data():
    m=baseline_logreg();X=np.array([[0],[1],[2],[3]])
    m.fit(X,[0,0,1,1]);mean=m[0].mean_.copy()
    m.predict_proba([[10000]])
    np.testing.assert_array_equal(m[0].mean_,mean)
    assert mean[0]==1.5


def test_strategy_nonoverlap_and_exact_price_pnl():
    c=pd.DataFrame({'timestamp':np.arange(10)*I,'close':np.arange(10)+100.})
    p=pd.DataFrame({'signal_time':[I-1,2*I-1,4*I-1],'entry_time':[I,2*I,4*I],'label_end_time':[4*I,5*I,7*I],'entry_price':[100.,101.,110.],'exit_price':[110.,111.,121.],'probability':[.8,.8,.8]})
    curve,trades=backtest_predictions(p,c,costs=Costs(0,0))
    assert len(trades)==2
    assert trades.net_pnl.tolist()==[100.,101.]
    assert curve.equity.iloc[-1]==10201
    assert (trades.entry_time.iloc[1:].to_numpy()>=trades.exit_time.iloc[:-1].to_numpy()).all()


def test_funding_strategy_sign_uses_past_price_and_respects_boundaries():
    c=pd.DataFrame({'timestamp':np.arange(8)*I,'close':np.arange(8)+100.})
    p=pd.DataFrame({'signal_time':[I-1],'entry_time':[I],'label_end_time':[4*I],'entry_price':[100.],'exit_price':[110.],'probability':[.8]})
    f=pd.DataFrame({'timestamp':[I,I+2,2*I+3,4*I+1],'funding_rate':[99,.01,-.01,99]})
    curve,trades=backtest_predictions(p,c,f,Costs(0,0),capital=1000)
    assert trades.funding_events.iloc[0]==2
    assert abs(trades.funding_paid_net.iloc[0]+.01)<1e-12
    assert abs(curve.equity.iloc[-1]-1010.01)<1e-10
    c.loc[3:,'close']*=10
    _,changed=backtest_predictions(p,c,f,Costs(0,0),capital=1000)
    assert changed.funding_paid_net.iloc[0]==trades.funding_paid_net.iloc[0]
