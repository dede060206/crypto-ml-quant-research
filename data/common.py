"""UTC timestamps, bounded retries and crash-safe local storage."""
from pathlib import Path
import time
import ccxt
import pandas as pd


def milliseconds(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Timestamp must specify a timezone")
    return stamp.tz_convert("UTC").value // 1_000_000


def request(call, attempts=4):
    for attempt in range(attempts):
        try:
            return call()
        except (ccxt.NetworkError, ccxt.RateLimitExceeded):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def save(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def cached(path, columns):
    return pd.read_parquet(path) if Path(path).exists() else pd.DataFrame(columns=columns)


def combine(old, new, columns):
    # An API can return an overlapping boundary; keep one observation per timestamp.
    parts = [part for part in (old, new) if not part.empty]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def cursor_for(old, start):
    # Earlier start requests re-download the range, including any earlier listing gaps.
    if old.empty or int(old.timestamp.min()) > start:
        return start
    return max(start, int(old.timestamp.max()) + 1)
