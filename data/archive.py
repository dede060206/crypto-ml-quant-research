"""Official monthly USD-M archive import; no trading API or credentials required."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile
import pandas as pd
import requests
from .common import save

BASE = 'https://data.binance.vision/data/futures/um/monthly'


def url_for(symbol, kind, month, timeframe='8h'):
    if kind == 'ohlcv':
        return f'{BASE}/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{month}.zip'
    return f'{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip'


def get_bytes(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def validate_zip(payload, checksum):
    expected = checksum.decode().split()[0].lower()
    actual = sha256(payload).hexdigest()
    if len(expected) != 64 or actual != expected:
        raise ValueError('Archive SHA256 mismatch')
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt ZIP')
    return actual


def parse_zip(path, kind):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(names) != 1:
            raise ValueError('Expected exactly one CSV')
        raw = archive.read(names[0])
    # Historical candles can have either a header or numeric first record.
    header = 0 if raw.splitlines()[0].split(b',')[0] in (b'open_time', b'calc_time') else None
    frame = pd.read_csv(BytesIO(raw), header=header)
    if kind == 'ohlcv':
        if frame.shape[1] != 12:
            raise ValueError('Unexpected candle schema')
        result = frame.iloc[:, :6].copy()
        result.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    else:
        if header is None or not {'calc_time', 'funding_interval_hours', 'last_funding_rate'} <= set(frame.columns):
            raise ValueError('Unexpected funding schema')
        result = frame[['calc_time', 'last_funding_rate', 'funding_interval_hours']].rename(columns={'calc_time':'timestamp', 'last_funding_rate':'funding_rate'})
    result = result.apply(pd.to_numeric, errors='raise')
    if result.isna().any().any() or result.empty:
        raise ValueError('Empty data or null values')
    # Refuse silent precision/unit mistakes; these futures files are milliseconds.
    if not result.timestamp.between(1_000_000_000_000, 10_000_000_000_000).all():
        raise ValueError('Timestamp outside expected millisecond range')
    result['timestamp'] = result.timestamp.astype('int64')
    if kind == 'ohlcv':
        if (result[['open','high','low','close']] <= 0).any().any() or (result.volume < 0).any():
            raise ValueError('Invalid price or volume')
        if ((result.high < result[['open','close','low']].max(axis=1)) | (result.low > result[['open','close','high']].min(axis=1))).any():
            raise ValueError('Invalid OHLC bounds')
    elif (result.funding_interval_hours <= 0).any():
        raise ValueError('Invalid funding interval')
    if result.timestamp.duplicated().any():
        raise ValueError('Duplicate timestamps inside archive')
    return result.sort_values('timestamp').reset_index(drop=True)


def acquire(task, raw_dir, offline=False):
    symbol, kind, month, timeframe = task
    url = url_for(symbol, kind, month, timeframe)
    path = Path(raw_dir) / symbol / kind / url.rsplit('/', 1)[-1]
    record = {'symbol':symbol, 'kind':kind, 'month':month, 'url':url}
    try:
        checksum_path = path.with_suffix('.zip.CHECKSUM')
        if not path.exists() or not checksum_path.exists():
            if offline:
                return {**record, 'status':'missing_local'}
            payload = get_bytes(url)
            if payload is None:
                return {**record, 'status':'not_found'}
            checksum = get_bytes(url + '.CHECKSUM')
            if checksum is None:
                raise ValueError('Missing official checksum')
            validate_zip(payload, checksum)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_bytes(payload)
            temp.replace(path)
            checksum_path.write_bytes(checksum)
        digest = validate_zip(path.read_bytes(), checksum_path.read_bytes())
        frame = parse_zip(path, kind)
        return {**record, 'status':'ok', 'sha256':digest, 'rows':len(frame), 'path':str(path), 'checked_at':pd.Timestamp.now(tz='UTC').isoformat()}
    except (requests.RequestException, ValueError, OSError, zipfile.BadZipFile) as error:
        return {**record, 'status':'error', 'error':str(error)}


def run_archive(config, start, end, offline=False, workers=6):
    root = Path(config['cache_dir'])
    raw = Path(config.get('archive_dir', 'data/raw_archive'))
    now = pd.Timestamp.now(tz='UTC')
    cutoff = int(now.normalize().replace(day=1).value // 1_000_000)
    stop = min(end, cutoff)  # monthly archives only: exclude unpublished current month
    if stop <= start:
        raise ValueError('No completed monthly archive in requested range')
    begin = pd.Timestamp(start, unit='ms', tz='UTC').normalize().replace(day=1)
    months = pd.date_range(begin, pd.Timestamp(stop - 1, unit='ms', tz='UTC'), freq='MS').strftime('%Y-%m').tolist()
    symbols = {}
    for symbol in config['symbols']:
        base, quote = symbol.split('/')
        if quote != 'USDT:USDT' or not base.isalnum():
            raise ValueError('Only USDT linear perpetual symbols supported')
        symbols[base + 'USDT'] = symbol
    tasks = [(symbol, kind, month, config['timeframe']) for symbol in symbols for kind in ('ohlcv','funding') for month in months]
    if config['timeframe'] != '8h':
        raise ValueError('This stage validates 8h candles only')
    root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(lambda task: acquire(task, raw, offline), tasks):
            records.append(record)
            if len(records) % 24 == 0 or record['status'] != 'ok':
                print(f"Archives {len(records)}/{len(tasks)}: {record['symbol']} {record['month']} {record['kind']} {record['status']}", flush=True)
            temp = root / 'manifest.tmp.json'
            temp.write_text(json.dumps(records, indent=2))
            temp.replace(root / 'manifest.json')
    report = {'source':'binance_official_monthly_archive', 'requested_start_ms':start, 'requested_end_ms':end, 'effective_end_exclusive_ms':stop, 'current_month_excluded':end > cutoff, 'assets':{}}
    for archive_symbol, symbol in symbols.items():
        asset = {}
        for kind in ('ohlcv','funding'):
            good = [r for r in records if r['symbol'] == archive_symbol and r['kind'] == kind and r['status'] == 'ok']
            bad = [r for r in records if r['symbol'] == archive_symbol and r['kind'] == kind and r['status'] != 'ok']
            frames = [parse_zip(r['path'], kind) for r in good]
            frame = pd.concat(frames, ignore_index=True).sort_values('timestamp') if frames else pd.DataFrame(columns=['timestamp'])
            frame = frame[(frame.timestamp >= start) & (frame.timestamp < stop)].copy()
            if frame.timestamp.duplicated().any():
                raise ValueError('Duplicate observations across monthly archives')
            if kind == 'ohlcv':
                frame = frame[frame.timestamp + 28_800_000 <= stop]
            output = root / symbol.replace('/','_').replace(':','_') / f'{kind}.parquet'
            if not frame.empty:
                save(frame, output)
            elif output.exists():
                output.unlink()  # avoid stale observations from a previous import
            asset[kind] = {'rows':len(frame), 'first_utc':pd.Timestamp(int(frame.timestamp.min()),unit='ms',tz='UTC').isoformat() if len(frame) else None, 'last_utc':pd.Timestamp(int(frame.timestamp.max()),unit='ms',tz='UTC').isoformat() if len(frame) else None, 'unavailable_months':[{'month':r['month'],'status':r['status']} for r in bad]}
            if kind == 'ohlcv':
                expected = len(range(((start + 28_800_000 - 1)//28_800_000)*28_800_000, stop - 28_800_000 + 1, 28_800_000))
                asset[kind]['expected_rows'] = expected
                asset[kind]['missing_rows'] = expected - len(frame)
                gaps = frame.timestamp.diff() > 28_800_000
                asset[kind]['internal_gap_count'] = int(gaps.sum())
                asset[kind]['gaps'] = [{'previous_open_ms':int(frame.timestamp.iloc[i-1]), 'next_open_ms':int(frame.timestamp.iloc[i]), 'missing_bars':int((frame.timestamp.iloc[i]-frame.timestamp.iloc[i-1])//28_800_000-1)} for i in range(1,len(frame)) if gaps.iloc[i]]
            elif len(frame):
                # Settlement timestamps contain millisecond jitter: do not treat jitter as gaps.
                steps = frame.timestamp.diff().dropna()/3_600_000
                asset[kind]['interval_hours'] = sorted(float(x) for x in frame.funding_interval_hours.unique())
                asset[kind]['unexpected_interval_count'] = int(((steps-frame.funding_interval_hours.iloc[1:].to_numpy()).abs() > 1/60).sum())
        report['assets'][symbol] = asset
    (root / 'coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report
