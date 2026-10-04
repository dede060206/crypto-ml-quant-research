import numpy as np
import pandas as pd
import pytest
from features.build_dataset import build_features, build_dataset, FEATURE_COLUMNS
from features.price_features import INTERVAL_MS as I, price_features
from features.funding_features import align_funding, funding_events
from features.target_construction import construct_targets, cost_threshold


def fixture(n=70):
    ts=1609459200000+np.arange(n)*I
    p=100+np.arange(n)+np.sin(np.arange(n))
    c=pd.DataFrame({'timestamp':ts,'open':p,'high':p+2,'low':p-2,'close':p+1,'volume':10.})
    f=pd.DataFrame({'timestamp':ts+2,'funding_rate':np.sin(np.arange(n))*.0001,'funding_interval_hours':8.})
    return c,f


@pytest.mark.parametrize('cut',[5,21,35,60])
def test_truncation_invariance_for_every_feature(cut):
    c,f=fixture()
    full=build_features(c,f)
    signal=int(full.signal_time.iloc[cut])
    short=build_features(c.iloc[:cut+1],f[f.timestamp+60000<=signal])
    pd.testing.assert_frame_equal(full.loc[:cut,FEATURE_COLUMNS],short[FEATURE_COLUMNS],check_exact=True)


def test_future_mutations_do_not_change_history():
    c,f=fixture();old=build_features(c,f)
    c.loc[36:,['open','high','low','close']]*=5
    f.loc[36:,'funding_rate']=.5
    new=build_features(c,f)
    pd.testing.assert_frame_equal(old.loc[:35,FEATURE_COLUMNS],new.loc[:35,FEATURE_COLUMNS],check_exact=True)


def test_publication_boundary_and_latest_backward_event():
    c,f=fixture();p=price_features(c)
    t=int(f.timestamp.iloc[5]+60000)
    signals=pd.DataFrame({'signal_time':[t-1,t,t+1]})
    joined=align_funding(signals,f)
    assert joined.funding_time.tolist()==[int(f.timestamp.iloc[4]),int(f.timestamp.iloc[5]),int(f.timestamp.iloc[5])]
    assert (joined.available_time<=joined.signal_time).all()


def test_missing_settlement_invalidates_stale_rate():
    c,f=fixture();t=int(f.timestamp.iloc[5]+3*I)
    joined=align_funding(pd.DataFrame({'signal_time':[t]}),f.iloc[:6])
    assert joined.funding_rate.isna().all()


def test_constant_rates_and_timestamp_jitter_are_finite():
    _,f=fixture();f.funding_rate=.0001
    f.loc[3,'timestamp']+=20
    features=funding_events(f)
    assert features.funding_zscore_24h.iloc[3]==0
    assert features.funding_zscore_24h.iloc[4]==0
    assert np.isfinite(features.funding_zscore_7d.iloc[3:]).all()


def test_gap_resets_price_history_and_removes_cross_gap_labels():
    c,f=fixture();c=c.drop(index=30).reset_index(drop=True)
    features=price_features(c)
    assert features.momentum_7d.iloc[30:51].isna().all()
    labels=construct_targets(c)
    assert not labels.label_valid.iloc[26:30].any()
    assert labels.label_valid.iloc[30]
    assert labels.target.iloc[-4:].isna().all()


def test_label_entry_exit_horizon_and_exact_cost_threshold():
    c,f=fixture()
    c.loc[1,'open']=100
    threshold=cost_threshold()
    c.loc[4,'open']=100*(1+threshold+.00001)
    labels=construct_targets(c)
    assert labels.entry_time.iloc[0]==c.timestamp.iloc[1]
    assert labels.label_end_time.iloc[0]-labels.entry_time.iloc[0]==3*I
    assert labels.target.iloc[0]==1
    c.loc[4,'open']=100*(1+threshold-.00001)
    assert construct_targets(c).target.iloc[0]==0
    assert labels.cost_threshold.nunique()==1


def test_allowlist_excludes_future_fields_and_training_eligibility():
    c,f=fixture();d=build_dataset(c,f)
    assert not set(FEATURE_COLUMNS)&{'target','future_return','entry_price','exit_price','label_end_time','entry_time'}
    assert not d.training_eligible.iloc[-4:].any()
    assert (d.loc[d.training_eligible,'signal_time']<d.loc[d.training_eligible,'entry_time']).all()
    assert (d.loc[d.training_eligible,'available_time']<=d.loc[d.training_eligible,'signal_time']).all()
