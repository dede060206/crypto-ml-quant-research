"""Repair missing 8h candles only when verified 1h OHLCV agrees with existing bars."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .archive import acquire,parse_zip
from .common import save

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'data'/'cache'/'SOL_USDT_USDT'/'ohlcv.parquet'
    original=pd.read_parquet(target);start=int(original.timestamp.min());stop=int(original.timestamp.max())+28_800_000
    absent=sorted(set(range(start,stop,28_800_000))-set(original.timestamp))
    if not absent:
        print('SOL already continuous');return
    months=sorted(set(pd.to_datetime(absent,unit='ms',utc=True).strftime('%Y-%m')))
    records=[];replacements=[]
    for month in months:
        record=acquire(('SOLUSDT','ohlcv',month,'1h'),ROOT/'data'/'raw_archive')
        records.append(record)
        if record['status']!='ok':raise ValueError(f'1h source failed {record}')
        hour=parse_zip(record['path'],'ohlcv');hour.index=pd.to_datetime(hour.timestamp,unit='ms',utc=True)
        group=hour.resample('8h',origin='epoch')
        bar=group.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        counts=group.timestamp.count();bar=bar[counts==8].copy();bar['timestamp']=bar.index.as_unit('ms').astype('int64');bar=bar.reset_index(drop=True)
        existing=original.merge(bar,on='timestamp',suffixes=('_original','_hour'))
        for col in ['open','high','low','close','volume']:
            if not np.allclose(existing[col+'_original'],existing[col+'_hour'],rtol=1e-8,atol=1e-7):
                raise ValueError(f'1h source disagrees with existing 8h {month} {col}; refuse automatic repair')
        replacements.append(bar[bar.timestamp.isin(absent)])
    patches=pd.concat(replacements,ignore_index=True)
    if set(patches.timestamp)!=set(absent):raise ValueError('1h archives do not cover all missing bars')
    backup=target.with_name('ohlcv_original_monthly_8h.parquet')
    if not backup.exists():save(original,backup)
    result=pd.concat([original,patches],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    assert not result.timestamp.duplicated().any() and (result.timestamp.diff().dropna()==28_800_000).all()
    save(result,target)
    audit={'method':'exact UTC 8h aggregation from 8 verified consecutive 1h bars; existing monthly 8h OHLCV all agree','patched_timestamps_ms':absent,'patched_count':len(patches),'source_archives':records}
    (target.parent/'repair_audit.json').write_text(json.dumps(audit,indent=2))
    print('Repaired SOL missing bars:',len(patches))


if __name__=='__main__':main()
