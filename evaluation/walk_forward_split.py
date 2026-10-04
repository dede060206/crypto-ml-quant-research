"""Calendar walk-forward with actual label interval purge and extra embargo."""
import pandas as pd


def walk_forward_split(data, train_months=12, embargo_hours=8, first_test='2022-01-01', end='2026-09-01'):
    if train_months<1 or embargo_hours<0:
        raise ValueError('Invalid windows')
    times=pd.to_datetime(data.signal_time,unit='ms',utc=True)
    label_ends=pd.to_datetime(data.label_end_time,unit='ms',utc=True)
    for start in pd.date_range(first_test,end,freq='MS',tz='UTC',inclusive='left'):
        stop=start+pd.DateOffset(months=1)
        begin=start-pd.DateOffset(months=train_months)
        candidate=(times>=begin)&(times<start)
        train=candidate&(label_ends<start-pd.Timedelta(hours=embargo_hours))
        test=(times>=start)&(times<stop)
        if not test.any():continue
        if train.sum()<100:raise ValueError('Insufficient training samples')
        yield {'fold':start.strftime('%Y-%m'),'train':data.index[train].to_numpy(),'test':data.index[test].to_numpy(),'train_start':begin.isoformat(),'test_start':start.isoformat(),'test_end':stop.isoformat(),'purged_and_embargoed':int((candidate&~train).sum())}
