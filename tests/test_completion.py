import numpy as np
import pandas as pd
import pytest
from data.audit_sol_gaps import merge_daily
from features.build_dataset import build_features
from features.funding_features import funding_columns
from features.price_features import PRICE_COLUMNS
from backtest.strategy import backtest_predictions
from backtest.cost_model import Costs
from tests.test_no_lookahead import fixture


def test_daily_repair_only_missing_real_records():
    c,_=fixture(8);original=c.drop(index=[3,4])
    fixed,patch,common=merge_daily(original,c,c.timestamp.iloc[[3,4]].tolist())
    pd.testing.assert_frame_equal(fixed,c)
    assert len(patch)==2 and len(common)==6
    conflict=c.copy();conflict.loc[0,'close']+=1
    with pytest.raises(ValueError,match='disagreement'):merge_daily(original,conflict,c.timestamp.iloc[[3,4]].tolist())
    with pytest.raises(ValueError,match='not complete'):merge_daily(original,c.drop(index=3),c.timestamp.iloc[[3,4]].tolist())


def test_actual_window_recomputation_is_different_and_causal():
    c,f=fixture(70);windows={'48h':'48h','5d':'5D','14d':'14D'}
    alternate=build_features(c,f,windows=windows);base=build_features(c,f)
    assert 'funding_zscore_48h' in alternate
    assert not np.allclose(alternate.funding_zscore_48h.iloc[10:],base.funding_zscore_24h.iloc[10:])
    cols=PRICE_COLUMNS+funding_columns(windows)
    t=int(alternate.signal_time.iloc[40]);prefix=build_features(c.iloc[:41],f[f.timestamp+60000<=t],windows=windows)
    pd.testing.assert_frame_equal(alternate.loc[:40,cols],prefix[cols],check_exact=True)


def test_strategy_refuses_cross_gap_trade():
    c,_=fixture(10);I=28_800_000;t=int(c.timestamp.iloc[1])
    p=pd.DataFrame({'signal_time':[t-1],'entry_time':[t],'label_end_time':[t+3*I],'entry_price':[100.],'exit_price':[110.],'probability':[.8]})
    with pytest.raises(ValueError,match='missing candles'):backtest_predictions(p,c.drop(index=2),costs=Costs(0,0))
