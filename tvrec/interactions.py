"""Window-clipped evidence aggregation and independent forecast catalogues."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cache import CacheBuild, code_hash, complete_manifest, verify_cache
from .io import stable_hash
from .resources import LOGGER
from .time_utils import utc_iso

INTERACTION_COLUMNS = ["user_id", "tv_show_id", "seconds", "fragment_count", "view_count", "confidence_seconds", "relevant"]
INTERACTION_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("tv_show_id", pa.int64()), ("seconds", pa.float64()),
    ("fragment_count", pa.int64()), ("view_count", pa.int64()),
    ("confidence_seconds", pa.float64()), ("relevant", pa.bool_()),
])


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    end: int

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("A temporal window must have positive duration")

    def identity(self):
        return {"start_epoch": self.start, "end_exclusive_epoch": self.end, "start_utc": utc_iso(self.start), "end_exclusive_utc": utc_iso(self.end)}


def epoch_column(series):
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.to_numpy(dtype=np.int64)
    return series.astype("int64").to_numpy() // 1_000_000_000


def empty_interactions():
    return pd.DataFrame({
        "user_id": pd.Series(dtype=str), "tv_show_id": pd.Series(dtype="int64"),
        "seconds": pd.Series(dtype="float64"), "fragment_count": pd.Series(dtype="int64"),
        "view_count": pd.Series(dtype="int64"), "confidence_seconds": pd.Series(dtype="float64"),
        "relevant": pd.Series(dtype=bool),
    })


def aggregate_portion(frame, window):
    """A fragment is clipped before duration or support is calculated."""
    starts = epoch_column(frame.start_time)
    ends = epoch_column(frame.end_time)
    seconds = np.minimum(ends, window.end) - np.maximum(starts, window.start)
    keep = (seconds > 0) & (frame.tv_show_id.to_numpy() > 0)
    if not keep.any():
        return empty_interactions().drop(columns=["confidence_seconds", "relevant"])
    denominator = frame.allocation_denominator.to_numpy()[keep]
    if np.any(denominator <= 0):
        raise ValueError("Invalid evidence allocation denominator")
    part = pd.DataFrame({
        "user_id": frame.user_id.to_numpy()[keep], "tv_show_id": frame.tv_show_id.to_numpy()[keep],
        "seconds": seconds[keep] / denominator, "view_id": frame.view_id.to_numpy()[keep],
    })
    return part.groupby(["user_id", "tv_show_id"], sort=False).agg(
        seconds=("seconds", "sum"), fragment_count=("seconds", "size"), view_count=("view_id", "nunique")
    ).reset_index()


def finish_aggregation(parts, relevance_seconds, confidence_cap):
    if not parts:
        return empty_interactions()
    frame = pd.concat(parts, ignore_index=True).groupby(["user_id", "tv_show_id"], sort=True, as_index=False)[["seconds", "fragment_count", "view_count"]].sum()
    frame["tv_show_id"] = frame.tv_show_id.astype("int64")
    frame["fragment_count"] = frame.fragment_count.astype("int64")
    frame["view_count"] = frame.view_count.astype("int64")
    frame["seconds"] = frame.seconds.astype("float64")
    frame["confidence_seconds"] = frame.seconds if confidence_cap is None else frame.seconds.clip(upper=confidence_cap)
    frame["relevant"] = frame.seconds >= relevance_seconds
    return frame[INTERACTION_COLUMNS]


def evidence_batches(path, batch_rows):
    """Keep complete view IDs together, so view_count survives batch boundaries."""
    carry = None
    previous = -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        ids = frame.view_id.to_numpy()
        if ids[0] < previous or np.any(ids[1:] < ids[:-1]):
            raise ValueError("Evidence must be sorted by normalized view_id")
        last = ids == ids[-1]
        carry = frame.loc[last].copy()
        closed = frame.loc[~last]
        if not closed.empty:
            previous = int(closed.view_id.iloc[-1])
            yield closed
    if carry is not None and not carry.empty:
        yield carry


def interaction_stats(frame, cohort=None):
    if cohort is not None:
        frame = frame.loc[frame.user_id.isin(cohort)]
    relevant = frame.loc[frame.relevant]
    return {
        "users_with_positive_seconds": int(frame.user_id.nunique()),
        "programmes_with_positive_seconds": int(frame.tv_show_id.nunique()),
        "user_programme_pairs": len(frame), "credited_seconds": float(frame.seconds.sum()),
        "evidence_fragment_count": int(frame.fragment_count.sum()),
        "user_programme_view_count": int(frame.view_count.sum()),
        "relevant_user_programme_pairs": len(relevant),
        "users_with_threshold_support": int(relevant.user_id.nunique()),
        "programmes_with_threshold_support": int(relevant.tv_show_id.nunique()),
    }


def aggregate_windows(settings, prepared, preparation_key, windows):
    options = settings.values["interactions"]
    outputs, missing = {}, []
    for window in windows:
        identity = {
            "schema_version": 1, "preparation_key": preparation_key, "window": window.identity(),
            "interactions": options, "aggregation_code_hash": code_hash(["interactions.py", "io.py", "cache.py"]),
        }
        destination = settings.artifacts_dir / "windows" / stable_hash(identity)
        if destination.exists():
            verify_cache(destination, identity)
            frame = pq.read_table(destination / "interactions.parquet").to_pandas()
            outputs[window.name] = {"frame": frame, "path": destination, "cache_hit": True, "window": window}
            LOGGER.info("Reusing window cache %s", window.name)
        else:
            missing.append((window, identity, destination))
    if missing:
        parts = {window.name: [] for window, _, _ in missing}
        for batch in evidence_batches(prepared / "evidence.parquet", settings.values["preparation"]["parquet_batch_rows"]):
            for window, _, _ in missing:
                part = aggregate_portion(batch, window)
                if not part.empty:
                    parts[window.name].append(part)
        for window, identity, destination in missing:
            frame = finish_aggregation(parts.pop(window.name), options["relevance_seconds"], options["confidence_seconds_cap"])
            with CacheBuild(destination) as staging:
                pq.write_table(pa.Table.from_pandas(frame, schema=INTERACTION_SCHEMA, preserve_index=False), staging / "interactions.parquet", compression="zstd")
                complete_manifest(staging, identity, ["interactions.parquet"], {"statistics": interaction_stats(frame)})
            outputs[window.name] = {"frame": frame, "path": destination, "cache_hit": False, "window": window}
    return outputs


def candidate_ids(prepared, policy, window=None):
    """Uses schedules alone; no viewing table or validation labels are accepted."""
    if policy["kind"] == "file":
        catalogues = json.loads((prepared / "candidates.json").read_text(encoding="utf-8"))["by_file"]
        if policy["file"] not in catalogues:
            raise ValueError(f"Unknown catalogue file: {policy['file']}")
        ids = catalogues[policy["file"]]
    elif policy["kind"] == "window_overlap":
        if window is None:
            raise ValueError("window_overlap requires an explicit temporal window")
        ids = set()
        for batch in pq.ParquetFile(prepared / "broadcast_metadata.parquet").iter_batches(columns=["tv_show_id", "start_time", "end_time"]):
            frame = batch.to_pandas()
            start, end = epoch_column(frame.start_time), epoch_column(frame.end_time)
            valid = (end > start) & (end > window.start) & (start < window.end)
            ids.update(int(x) for x in frame.loc[valid, "tv_show_id"])
    else:
        raise ValueError(f"Unknown candidate policy: {policy}")
    return sorted({int(item) for item in ids if int(item) > 0})


def viewing_activity(prepared, windows, cohort, batch_rows):
    """Separate no activity from activity with no identifiable programme label."""
    users = {window.name: set() for window in windows}
    seconds = {window.name: 0 for window in windows}
    cohort = set(cohort)
    for batch in pq.ParquetFile(prepared / "normalized_viewing.parquet").iter_batches(batch_size=batch_rows, columns=["user_id", "start_time", "end_time"]):
        frame = batch.to_pandas()
        frame = frame.loc[frame.user_id.isin(cohort)]
        if frame.empty:
            continue
        start, end = epoch_column(frame.start_time), epoch_column(frame.end_time)
        for window in windows:
            overlap = np.minimum(end, window.end) - np.maximum(start, window.start)
            valid = overlap > 0
            users[window.name].update(frame.loc[valid, "user_id"])
            seconds[window.name] += int(overlap[valid].sum())
    return {name: {"users": sorted(value), "users_with_viewing": len(value), "user_channel_union_seconds": seconds[name]} for name, value in users.items()}


def historical_metadata(prepared, cutoff):
    """An as-of metadata interface for displays and later historical feature fitting."""
    frames = []
    for batch in pq.ParquetFile(prepared / "broadcast_metadata.parquet").iter_batches(columns=["tv_show_id", "start_time", "duration", "tv_show_title", "tv_show_category"]):
        frame = batch.to_pandas()
        frame = frame.loc[(epoch_column(frame.start_time) < cutoff) & (frame.duration > 0)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["tv_show_id", "tv_show_title", "tv_show_category"])
    frame = pd.concat(frames, ignore_index=True).sort_values(["start_time", "tv_show_id", "tv_show_title"], kind="mergesort")
    return frame.drop_duplicates("tv_show_id")[["tv_show_id", "tv_show_title", "tv_show_category"]].reset_index(drop=True)
