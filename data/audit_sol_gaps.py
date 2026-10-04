"""Audit official daily archives; do not invent prices when the source is missing."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .common import save
from .archive import get_bytes,validate_zip,parse_zip
ROOT=Path(__file__).resolve().parents[1]



def merge_daily(c,daily,absent):
    common=c.merge(daily,on='timestamp',suffixes=('_monthly','_daily'))
    for col in ['open','high','low','close','volume']:
        if not np.allclose(common[col+'_monthly'],common[col+'_daily'],rtol=1e-8,atol=1e-7):raise ValueError(f'Daily/monthly overlap disagreement: {col}')
    patch=daily[daily.timestamp.isin(absent)]
    if len(patch)!=len(absent) or set(patch.timestamp)!=set(absent):raise ValueError('Daily source not complete; refuse repair')
    fixed=pd.concat([c,patch],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if fixed.timestamp.duplicated().any() or not (fixed.timestamp.diff().dropna()==28_800_000).all():raise ValueError('Repair not continuous')
    return fixed,patch,common

def main(offline=False):
    folder=ROOT/'data'/'cache'/'SOL_USDT_USDT'
    c=pd.read_parquet(folder/'ohlcv.parquet')
    absent=sorted(set(range(int(c.timestamp.min()),int(c.timestamp.max())+28_800_000,28_800_000))-set(c.timestamp))
    if not absent:
        print('SOL candles already continuous');return
    dates=set(pd.to_datetime(absent,unit='ms',utc=True).strftime('%Y-%m-%d'))
    for i,t in enumerate(absent):
        if i==0 or t-absent[i-1]!=28_800_000:dates.add((pd.Timestamp(t,unit='ms',tz='UTC')-pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if i==len(absent)-1 or absent[i+1]-t!=28_800_000:dates.add((pd.Timestamp(t,unit='ms',tz='UTC')+pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    days=sorted(dates)
    records=[];frames=[]
    for day in days:
        url=f'https://data.binance.vision/data/futures/um/daily/klines/SOLUSDT/8h/SOLUSDT-8h-{day}.zip'
        path=ROOT/'data'/'raw_archive'/'SOL_daily_audit'/url.rsplit('/',1)[-1]
        checksum=path.with_suffix('.zip.CHECKSUM')
        if not path.exists():
            if offline:raise ValueError(f'Missing local daily archive {day}')
            payload=get_bytes(url)
            if payload is None:
                records.append({'url':url,'status':'not_found'});continue
            check=get_bytes(url+'.CHECKSUM')
            if check is None:raise ValueError('No official checksum')
            validate_zip(payload,check);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);checksum.write_bytes(check)
        digest=validate_zip(path.read_bytes(),checksum.read_bytes())
        f=parse_zip(path,'ohlcv');frames.append(f)
        records.append({'url':url,'sha256':digest,'rows':len(f),'missing_times_present':[int(x) for x in f.timestamp if x in absent]})
    daily=pd.concat(frames,ignore_index=True)
    fixed,patch,common=merge_daily(c,daily,absent)
    backup=folder/'ohlcv_original_monthly_8h.parquet'
    if not backup.exists():save(c,backup)
    save(fixed,folder/'ohlcv.parquet')
    audit={'patched_bars':len(patch),'overlap_reference_rows':len(common),'missing_8h_bars':len(absent),'missing_timestamps_ms':absent,'daily_source_audit':records,'treatment':'Filled from verified actual official daily 8h candles, no interpolation. Compared all overlapping reference rows with monthly OHLCV.'}
    (folder/'gap_audit.json').write_text(json.dumps(audit,indent=2))
    coverage_path=ROOT/'data'/'cache'/'coverage.json'
    coverage=json.loads(coverage_path.read_text())
    part=coverage['assets']['SOL/USDT:USDT']['ohlcv']
    part.update({'rows':len(fixed),'missing_rows':0,'internal_gap_count':0,'gaps':[],'source_repair':'verified daily 8h archives; gap_audit.json preserves original missing timestamps'})
    coverage_path.write_text(json.dumps(coverage,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
