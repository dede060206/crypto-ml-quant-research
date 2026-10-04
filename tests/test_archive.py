from hashlib import sha256
from io import BytesIO
import zipfile
import pytest
from data.archive import validate_zip, parse_zip, acquire, run_archive
from data.common import milliseconds


def archive(path, text):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('sample.csv', text)
    return path


def test_checksum_rejects_tampering(tmp_path):
    path = archive(tmp_path/'x.zip', 'a,b\n1,2\n')
    payload = path.read_bytes()
    checksum = (sha256(payload).hexdigest() + ' x.zip').encode()
    assert validate_zip(payload, checksum) == sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='SHA256'):
        validate_zip(payload+b'changed', checksum)


@pytest.mark.parametrize('header', ['', 'open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n'])
def test_candle_header_variants(tmp_path, header):
    path = archive(tmp_path/'x.zip', header+'1640995200000,10,12,9,11,4,1641023999999,44,1,2,22,0\n')
    frame = parse_zip(path, 'ohlcv')
    assert frame.timestamp.tolist() == [1640995200000]
    assert frame.close.tolist() == [11]


def test_funding_preserves_settlement_jitter_and_interval(tmp_path):
    path = archive(tmp_path/'x.zip', 'calc_time,funding_interval_hours,last_funding_rate\n1640995200006,8,-0.0001\n')
    frame = parse_zip(path, 'funding')
    assert frame.timestamp.tolist() == [1640995200006]
    assert frame.funding_interval_hours.tolist() == [8]
    assert frame.funding_rate.tolist() == [-.0001]


def test_offline_missing_never_calls_network(tmp_path, monkeypatch):
    monkeypatch.setattr('data.archive.get_bytes', lambda url: pytest.fail('offline network call'))
    assert acquire(('BTCUSDT', 'funding', '2022-01', '8h'), tmp_path, True)['status'] == 'missing_local'


def test_offline_import_exact_range_and_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr('data.archive.get_bytes', lambda url: pytest.fail('offline network call'))
    raw = tmp_path/'raw'
    for kind, text in [('ohlcv','open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n1640995200000,10,12,9,11,4,1641023999999,44,1,2,22,0\n'), ('funding','calc_time,funding_interval_hours,last_funding_rate\n1640995200006,8,-0.0001\n')]:
        from data.archive import url_for
        path = raw/'BTCUSDT'/kind/url_for('BTCUSDT',kind,'2022-01').rsplit('/',1)[-1]
        path.parent.mkdir(parents=True)
        archive(path, text)
        path.with_suffix('.zip.CHECKSUM').write_text(sha256(path.read_bytes()).hexdigest())
    config = {'symbols':['BTC/USDT:USDT'], 'timeframe':'8h', 'cache_dir':str(tmp_path/'cache'), 'archive_dir':str(raw)}
    report = run_archive(config, milliseconds('2022-01-01T00:00:00Z'), milliseconds('2022-01-01T16:00:00Z'), True, 1)
    assert report['assets']['BTC/USDT:USDT']['ohlcv']['missing_rows'] == 1
    assert report['assets']['BTC/USDT:USDT']['funding']['unavailable_months'] == []
