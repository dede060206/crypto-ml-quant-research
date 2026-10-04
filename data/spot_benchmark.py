"""Verified offline BTC spot reference. Spot archives switch to microseconds in 2025."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
from hashlib import sha256
import json,zipfile
import pandas as pd
from .archive import get_bytes,validate_zip
from .common import save

ROOT=Path(__file__).resolve().parents[1]


def acquire_spot(month):
    url=f'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/8h/BTCUSDT-8h-{month}.zip'
    path=ROOT/'data'/'raw_archive'/'BTC_SPOT'/url.rsplit('/',1)[-1]
    checksum=path.with_suffix('.zip.CHECKSUM')
    if not path.exists() or not checksum.exists():
        payload=get_bytes(url);check=get_bytes(url+'.CHECKSUM')
        if payload is None or check is None:raise ValueError(f'Missing spot archive {month}')
        validate_zip(payload,check);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(payload);checksum.write_bytes(check)
    digest=validate_zip(path.read_bytes(),checksum.read_bytes())
    with zipfile.ZipFile(path) as archive:
        raw=archive.read(archive.namelist()[0])
    f=pd.read_csv(BytesIO(raw),header=None)
    if str(f.iloc[0,0])=='open_time':f=f.iloc[1:].copy()
    f=f.iloc[:,:6];f.columns=['timestamp','open','high','low','close','volume'];f=f.apply(pd.to_numeric)
    unit='us' if int(f.timestamp.iloc[0])>10_000_000_000_000 else 'ms'
    if unit=='us':f.timestamp=(f.timestamp//1000).astype('int64')
    return f,{'url':url,'sha256':digest,'timestamp_source_unit':unit,'rows':len(f)}


def main():
    months=pd.date_range('2022-01-01','2026-08-01',freq='MS').strftime('%Y-%m')
    with ThreadPoolExecutor(max_workers=6) as pool:parts=list(pool.map(acquire_spot,months))
    f=pd.concat([x[0] for x in parts],ignore_index=True).sort_values('timestamp')
    if f.timestamp.duplicated().any() or not (f.timestamp.diff().dropna()==28_800_000).all():raise ValueError('Spot reference has gaps/duplicates')
    save(f,ROOT/'data'/'cache'/'BTC_SPOT'/'ohlcv.parquet')
    (ROOT/'data'/'cache'/'BTC_SPOT'/'manifest.json').write_text(json.dumps([x[1] for x in parts],indent=2))
    print('BTC spot reference:',len(f),'candles;',len(parts),'verified monthly archives')


if __name__=='__main__':main()
