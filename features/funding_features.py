"""Funding event histories first, then backward as-of join at signal time."""
import numpy as np
import pandas as pd

FUNDING_COLUMNS = ['funding_rate', 'funding_rate_8h_equivalent', 'funding_change', 'funding_zscore_24h', 'funding_zscore_3d', 'funding_zscore_7d', 'funding_age_hours']


def funding_columns(windows=None):
    windows=windows or {'24h':'24h','3d':'3D','7d':'7D'}
    return ['funding_rate','funding_rate_8h_equivalent','funding_change']+[f'funding_zscore_{key}' for key in windows]+['funding_age_hours']


def funding_events(funding, publication_lag_ms=60000, windows=None):
    if publication_lag_ms < 0:
        raise ValueError('Publication lag cannot be negative')
    f = funding.sort_values('timestamp').reset_index(drop=True).copy()
    if f.empty or f.timestamp.duplicated().any() or not np.isfinite(f[['funding_rate','funding_interval_hours']]).all().all():
        raise ValueError('Empty/duplicate/nonfinite funding data')
    if (f.funding_interval_hours <= 0).any():
        raise ValueError('Funding interval must be positive')
    f['funding_time'] = f.timestamp.astype('int64')
    f['available_time'] = f.funding_time + publication_lag_ms
    f['funding_rate_8h_equivalent'] = f.funding_rate*8/f.funding_interval_hours
    # Respect actual elapsed time rather than assuming every three events is a day.
    elapsed = f.funding_time.diff()/3_600_000
    discontinuity = (elapsed-f.funding_interval_hours).abs() > 1/60
    f['funding_segment'] = discontinuity.cumsum()
    pieces = []
    for _, part in f.groupby('funding_segment',sort=False):
        part = part.copy()
        rate = part.funding_rate_8h_equivalent
        part['funding_change'] = rate.diff()
        history = pd.Series(rate.to_numpy(),index=pd.to_datetime(part.funding_time,unit='ms',utc=True))
        for label,window in (windows or {'24h':'24h','3d':'3D','7d':'7D'}).items():
            # Exclude current event from reference statistics; current rate is already known.
            rolling = history.rolling(pd.Timedelta(window)+pd.Timedelta(seconds=1),closed='left',min_periods=3)
            std = rolling.std(ddof=1)
            part[f'funding_zscore_{label}'] = ((history-rolling.mean())/std.clip(lower=1e-6)).to_numpy()
        pieces.append(part)
    return pd.concat(pieces).sort_values('available_time').reset_index(drop=True)


def align_funding(signals, funding, publication_lag_ms=60000, windows=None):
    f = funding_events(funding,publication_lag_ms,windows)
    columns=funding_columns(windows)
    right = f[['funding_time','available_time','funding_interval_hours']+[x for x in columns if x!='funding_age_hours']]
    result = pd.merge_asof(signals.sort_values('signal_time'),right,left_on='signal_time',right_on='available_time',direction='backward',allow_exact_matches=True)
    result['funding_age_hours'] = (result.signal_time-result.funding_time)/3_600_000
    # Missing settlements cannot silently leave an old rate in the model indefinitely.
    stale = result.funding_time.isna() | ((result.signal_time-result.funding_time) > result.funding_interval_hours*3_600_000+publication_lag_ms+60000)
    result.loc[stale,columns] = np.nan
    return result
