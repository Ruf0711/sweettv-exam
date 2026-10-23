"""Strict wall-time parsing using zoneinfo, with UTC elapsed-time arithmetic."""

import hashlib
import importlib.resources
from datetime import datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo

import numpy as np
import pandas as pd

LOG_FORMAT = "%Y-%m-%d %H:%M:%S"
SCHEDULE_FORMAT = "%d.%m.%Y %H:%M:%S"
_PATTERNS = {
    LOG_FORMAT: r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
    SCHEDULE_FORMAT: r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}",
}


@lru_cache(maxsize=2048)
def _transition_day(day_number: int, zone_key: str) -> bool:
    zone = ZoneInfo(zone_key)
    day = datetime(1970, 1, 1) + timedelta(days=int(day_number))
    # Kyiv has at most one offset transition per day. Include neighbouring
    # midnights to catch a transition at a date boundary as well.
    offsets = {
        (day + timedelta(hours=h)).replace(tzinfo=zone).utcoffset()
        for h in [-1, 0, 6, 12, 18, 24]
    }
    return len(offsets) > 1


def parse_local_times(values: pd.Series, fmt: str, zone: ZoneInfo) -> np.ndarray:
    """Return int64 UTC epoch seconds, rejecting ambiguous/nonexistent times.

    pandas 1.5 does not reliably enforce ambiguous/nonexistent='raise' for
    ZoneInfo objects. Explicit fold checks on transition dates close that gap.
    No timezone is guessed from the machine's local settings.
    """
    if not isinstance(zone, ZoneInfo):
        raise TypeError("An explicit zoneinfo.ZoneInfo is required")
    text = values.astype(str)
    if not text.str.fullmatch(_PATTERNS[fmt]).all():
        raise ValueError(f"Timestamp does not match exact format {fmt}")
    parsed = pd.DatetimeIndex(pd.to_datetime(text, format=fmt, exact=True, errors="raise"))
    naive = parsed.asi8 // 1_000_000_000
    day_numbers = naive // 86400
    suspect_days = [int(d) for d in np.unique(day_numbers) if _transition_day(int(d), zone.key)]
    if suspect_days:
        mask = np.isin(day_numbers, suspect_days)
        for stamp in parsed[mask].unique():
            wall = stamp.to_pydatetime()
            if wall.replace(tzinfo=zone, fold=0).utcoffset() != wall.replace(tzinfo=zone, fold=1).utcoffset():
                raise ValueError(f"Ambiguous or nonexistent local timestamp: {wall} in {zone.key}")
    localized = parsed.tz_localize(zone, ambiguous="raise", nonexistent="raise")
    return localized.tz_convert("UTC").asi8 // 1_000_000_000


def local_epoch(text: str, zone: ZoneInfo) -> int:
    return int(parse_local_times(pd.Series([text]), LOG_FORMAT, zone)[0])


def utc_iso(seconds: int) -> str:
    return pd.Timestamp(int(seconds), unit="s", tz="UTC").isoformat()


def zone_fingerprint(zone: ZoneInfo) -> dict:
    for directory in TZPATH:
        path = Path(directory) / zone.key
        if path.is_file():
            return {"key": zone.key, "provider": "system_tzpath", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    resource = importlib.resources.files("tzdata.zoneinfo").joinpath(*zone.key.split("/"))
    return {"key": zone.key, "provider": "tzdata_package", "sha256": hashlib.sha256(resource.read_bytes()).hexdigest()}
