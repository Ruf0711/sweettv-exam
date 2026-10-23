"""Uninterrupted channel sessions and exact broadcast-instance overlap evidence."""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.cache import CacheBuild, complete_manifest, verify_cache
from tvrec.interactions import epoch_column
from tvrec.intervals import normalize_viewing
from tvrec.io import LOG_COLUMNS, SCHEDULE_COLUMNS, UTC_TYPE, arrow_utc, atomic_json, csv_chunks, file_hash, integer_column, stable_hash, validate_user_ids
from tvrec.resources import progress
from tvrec.time_utils import LOG_FORMAT, SCHEDULE_FORMAT, local_epoch, parse_local_times, zone_fingerprint
from pathlib import Path


def completion_code_hash():
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), package / "io.py", package / "time_utils.py", package / "intervals.py", package / "interactions.py"]
    return stable_hash({str(p.relative_to(package)): file_hash(p) for p in paths})

EVENT_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("tv_show_id", pa.int64()), ("channel_id", pa.int32()),
    ("session_id", pa.int64()), ("broadcast_id", pa.int64()),
    ("start_time", UTC_TYPE), ("end_time", UTC_TYPE),
    ("broadcast_start", UTC_TYPE), ("broadcast_end", UTC_TYPE), ("broadcast_seconds", pa.int64()),
])
SESSION_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("channel_id", pa.int32()), ("session_id", pa.int64()),
    ("start_time", UTC_TYPE), ("end_time", UTC_TYPE),
])


def continuous_sessions(users, channels, starts, ends):
    """A later different-channel start terminates a run; old channels never resume.

    At an exact same-user/start tie, differing channel IDs create an unknown
    break. Device IDs are absent, so cross-channel concurrency is ambiguous.
    """
    u, c, s, e = (np.asarray(a) for a in [users, channels, starts, ends])
    valid = e > s
    u, c, s, e = (a[valid] for a in [u, c, s, e])
    if not len(u):
        return {"user_index": u.astype(np.int32), "channel_id": c.astype(np.int32), "start": s.astype(np.int64), "end": e.astype(np.int64)}, {"continuous_sessions": 0}
    order = np.lexsort((c, e, s, u))
    u, c, s, e = (a[order] for a in [u, c, s, e])
    first = np.r_[True, (u[1:] != u[:-1]) | (s[1:] != s[:-1])]
    boundaries = np.flatnonzero(first)
    minimum_channel = np.minimum.reduceat(c, boundaries)
    maximum_channel = np.maximum.reduceat(c, boundaries)
    ties = minimum_channel != maximum_channel
    u, s = u[boundaries], s[boundaries]
    c = np.where(ties, -1, minimum_channel).astype(np.int32)
    e = np.maximum.reduceat(e, boundaries)
    run_first = np.r_[True, (u[1:] != u[:-1]) | (c[1:] != c[:-1])]
    run_at = np.flatnonzero(run_first)
    run_ids = np.cumsum(run_first, dtype=np.int64) - 1
    limits = np.full(len(run_at), int(e.max()), dtype=np.int64)
    same_user_next = u[run_at[:-1]] == u[run_at[1:]]
    limits[:-1][same_user_next] = s[run_at[1:]][same_user_next]
    clipped = np.minimum(e, limits[run_ids])
    truncations = int((clipped < e).sum())
    keep = (c >= 0) & (clipped > s)
    views, _ = normalize_viewing(u[keep], run_ids[keep].astype(np.int32), s[keep], clipped[keep], clipped[keep] - s[keep])
    views["channel_id"] = c[run_at][views["channel_id"]].astype(np.int32)
    return views, {"continuous_sessions": len(views["start"]), "same_timestamp_conflicting_channel_groups": int(ties.sum()), "events_truncated_by_channel_change": truncations, "continuous_user_seconds": int((views["end"] - views["start"]).sum())}


def blackout_intervals(starts, ends):
    """Return disjoint intervals with more than one source broadcast active."""
    if len(starts) < 2 or np.all(starts[1:] >= np.maximum.accumulate(ends)[:-1]):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    times = np.r_[starts, ends]
    delta = np.r_[np.ones(len(starts), dtype=np.int64), -np.ones(len(ends), dtype=np.int64)]
    order = np.argsort(times, kind="stable")
    times, delta = times[order], delta[order]
    index = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    boundaries = times[index]
    active = np.cumsum(np.add.reduceat(delta, index))
    mask = active[:-1] > 1
    return boundaries[:-1][mask], boundaries[1:][mask]


def interval_covered(starts, ends, low, high):
    if not len(starts):
        return np.zeros(len(low), dtype=np.int64)
    prefix = np.r_[0, np.cumsum(ends - starts)]
    def integral(times):
        index = np.searchsorted(starts, times, side="right") - 1
        result = np.zeros(len(times), dtype=np.int64)
        valid = index >= 0
        j = index[valid]
        result[valid] = prefix[j] + np.minimum(times[valid] - starts[j], ends[j] - starts[j])
        return result
    return integral(high) - integral(low)


def match_broadcasts(starts, ends, broadcast):
    bs, be = broadcast["start"], broadcast["end"]
    low = np.searchsorted(np.maximum.accumulate(be), starts, side="right")
    high = np.searchsorted(bs, ends, side="left")
    size = np.maximum(0, high - low)
    rows = np.repeat(np.arange(len(starts)), size)
    if not len(rows):
        return (np.empty(0, dtype=np.int64),) * 4
    index = np.repeat(low, size) + np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
    left, right = np.maximum(starts[rows], bs[index]), np.minimum(ends[rows], be[index])
    valid = right > left
    return rows[valid], index[valid], left[valid], right[valid]


def build_completion_cache(settings, options, fingerprints, prepared, resources):
    identity = {
        "version": 1, "inputs": {k: v["sha256"] for k, v in fingerprints.items()},
        "stage1_preparation": prepared.name, "policies": options["completion"],
        "timezone": zone_fingerprint(settings.zone), "code": completion_code_hash(),
    }
    destination = settings.artifacts_dir / "completion_cache" / stable_hash(identity)
    if destination.exists():
        verify_cache(destination, identity)
        return destination
    with CacheBuild(destination) as directory:
        with resources.phase("stage2_read_source_broadcast_instances"):
            parts = []
            for name in settings.values["sources"]["schedules"]:
                for chunk in csv_chunks(settings.source(name), SCHEDULE_COLUMNS, 250000):
                    channel = integer_column(chunk.channel_id, "channel_id", bits=32)
                    item = integer_column(chunk.tv_show_id, "tv_show_id", missing_zero=True)
                    length = integer_column(chunk.duration, "duration", signed=True)
                    start = parse_local_times(chunk.start_time, SCHEDULE_FORMAT, settings.zone)
                    valid = length > 0
                    parts.append(pd.DataFrame({"channel_id": channel[valid], "tv_show_id": item[valid], "start": start[valid], "end": start[valid] + length[valid]}))
            frame = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(["channel_id", "start", "end", "tv_show_id"], kind="mergesort").reset_index(drop=True)
            frame["broadcast_id"] = np.arange(len(frame), dtype=np.int64)
            broadcasts, blackouts = {}, {}
            for channel, group in frame.groupby("channel_id", sort=True):
                bs, be = blackout_intervals(group.start.to_numpy(), group.end.to_numpy())
                if len(bs):
                    blackouts[str(int(channel))] = {"start": bs.tolist(), "end": be.tolist()}
                positive = group.loc[group.tv_show_id > 0]
                if len(positive):
                    broadcasts[int(channel)] = {name: positive[name].to_numpy() for name in ["start", "end", "tv_show_id", "broadcast_id"]}
            del frame, parts
            atomic_json(directory / "blackouts.json", blackouts)
        with resources.phase("stage2_reconstruct_uninterrupted_channel_sessions"):
            users = json.loads((prepared / "users.json").read_text(encoding="utf-8"))["user_ids"]
            lookup = {user: i for i, user in enumerate(users)}
            arrays = {name: [] for name in ["user_index", "channel_id", "start", "end"]}
            stop = local_epoch(settings.values["preparation"]["viewing_end_exclusive_local"], settings.zone)
            count = 0
            excluded = 0
            for chunk in csv_chunks(settings.source(settings.values["sources"]["viewing"]), LOG_COLUMNS, 250000):
                validate_user_ids(chunk.user_id)
                channel = integer_column(chunk.vsetv_id, "vsetv_id", bits=32)
                s = parse_local_times(chunk.start_time, LOG_FORMAT, settings.zone)
                e = parse_local_times(chunk.stop_time, LOG_FORMAT, settings.zone)
                duration = integer_column(chunk.duraton, "duraton", signed=True)
                if np.any((e > s) & (e - s != duration)):
                    raise ValueError("Stage 2 UTC duration mismatch")
                valid = (e > s) & (s < stop)
                excluded += int((s >= stop).sum())
                ids = chunk.loc[valid, "user_id"].map(lookup)
                if ids.isna().any():
                    raise ValueError("A viewing user is absent from the verified Stage 1 user registry")
                for name, value in [("user_index", ids.to_numpy(dtype=np.int32)), ("channel_id", channel[valid]), ("start", s[valid]), ("end", np.minimum(e[valid], stop))]:
                    arrays[name].append(value)
                count += len(chunk)
                if count % 500000 < len(chunk):
                    progress("Stage 2 raw sessions read", count)
            arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
            sessions, diagnostics = continuous_sessions(arrays["user_index"], arrays["channel_id"], arrays["start"], arrays["end"])
            diagnostics["source_rows"] = count
            diagnostics["excluded_future_rows"] = excluded
            del arrays
        with resources.phase("stage2_indexed_broadcast_instance_join"):
            names = np.asarray(users, dtype=object)
            event_count = 0
            chunk_rows = options["completion"]["event_chunk_rows"]
            with pq.ParquetWriter(directory / "events.parquet", EVENT_SCHEMA, compression="zstd") as writer, pq.ParquetWriter(directory / "continuous_sessions.parquet", SESSION_SCHEMA, compression="zstd") as session_writer:
                total = len(sessions["start"])
                for low in range(0, total, chunk_rows):
                    high = min(total, low + chunk_rows)
                    s, e, c, u = (sessions[name][low:high] for name in ["start", "end", "channel_id", "user_index"])
                    session_writer.write_table(pa.Table.from_arrays([pa.array(names[u]), pa.array(c, type=pa.int32()), pa.array(np.arange(low, high)), arrow_utc(s), arrow_utc(e)], schema=SESSION_SCHEMA))
                    for channel, ix in pd.Series(c).groupby(pd.Series(c), sort=False).indices.items():
                        if int(channel) not in broadcasts:
                            continue
                        broadcast = broadcasts[int(channel)]
                        row, j, left, right = match_broadcasts(s[ix], e[ix], broadcast)
                        if not len(row):
                            continue
                        table = pa.Table.from_arrays([
                            pa.array(names[u[ix[row]]], type=pa.string()), pa.array(broadcast["tv_show_id"][j], type=pa.int64()),
                            pa.array(np.full(len(row), int(channel)), type=pa.int32()), pa.array(low + ix[row], type=pa.int64()),
                            pa.array(broadcast["broadcast_id"][j], type=pa.int64()), arrow_utc(left), arrow_utc(right),
                            arrow_utc(broadcast["start"][j]), arrow_utc(broadcast["end"][j]),
                            pa.array(broadcast["end"][j] - broadcast["start"][j], type=pa.int64()),
                        ], schema=EVENT_SCHEMA)
                        writer.write_table(table)
                        event_count += len(row)
                    if high % 500000 < high - low or high == total:
                        progress("Continuous sessions joined", high, total)
            diagnostics["positive_broadcast_session_pairs"] = event_count
            diagnostics["blackout_channel_count"] = len(blackouts)
            atomic_json(directory / "diagnostics.json", diagnostics)
        complete_manifest(directory, identity, ["events.parquet", "continuous_sessions.parquet", "blackouts.json", "diagnostics.json"])
    return destination


def window_event_portion(frame, window, blackouts):
    low = np.maximum(epoch_column(frame.start_time), window.start)
    high = np.minimum(epoch_column(frame.end_time), window.end)
    keep = high > low
    frame = frame.loc[keep].copy()
    low, high = low[keep], high[keep]
    seconds = high - low
    for channel, ix in frame.reset_index(drop=True).groupby("channel_id", sort=False).indices.items():
        blackout = blackouts.get(str(int(channel)))
        if blackout:
            seconds[ix] -= interval_covered(np.asarray(blackout["start"]), np.asarray(blackout["end"]), low[ix], high[ix])
    frame["observed_seconds"] = seconds
    frame["last_observed_epoch"] = high
    frame["ge50"] = 2 * seconds >= frame.broadcast_seconds.to_numpy()
    frame["gt80"] = 5 * seconds > 4 * frame.broadcast_seconds.to_numpy()
    return frame


def completion_windows(settings, options, completion_path, windows):
    identity = {"completion_key": completion_path.name, "windows": {w.name: w.identity() for w in windows}, "version": 1, "code": completion_code_hash()}
    destination = settings.artifacts_dir / "completion_windows" / stable_hash(identity)
    if destination.exists():
        verify_cache(destination, identity)
        return {w.name: pq.read_table(destination / (w.name + ".parquet")).to_pandas() for w in windows}, destination
    blackouts = json.loads((completion_path / "blackouts.json").read_text(encoding="utf-8"))
    parts = {w.name: [] for w in windows}
    for batch in pq.ParquetFile(completion_path / "events.parquet").iter_batches(batch_size=131072):
        frame = batch.to_pandas()
        for window in windows:
            clipped = window_event_portion(frame, window, blackouts)
            clipped = clipped.loc[clipped.ge50 | clipped.gt80]
            if len(clipped):
                grouped = clipped.groupby(["user_id", "tv_show_id", "broadcast_id"], sort=False).agg(
                    ge50=("ge50", "max"), gt80=("gt80", "max"),
                    seconds=("observed_seconds", "max"), last_epoch=("last_observed_epoch", "max"),
                ).reset_index()
                parts[window.name].append(grouped)
    outputs = {}
    with CacheBuild(destination) as staging:
        for window in windows:
            if parts[window.name]:
                per_instance = pd.concat(parts[window.name], ignore_index=True).groupby(["user_id", "tv_show_id", "broadcast_id"], as_index=False).agg(ge50=("ge50", "max"), gt80=("gt80", "max"), seconds=("seconds", "max"), last_epoch=("last_epoch", "max"))
                result = per_instance.groupby(["user_id", "tv_show_id"], as_index=False).agg(count50=("ge50", "sum"), count80=("gt80", "sum"), seconds50=("seconds", "sum"), last_epoch=("last_epoch", "max"))
            else:
                result = pd.DataFrame({"user_id": pd.Series(dtype=str), "tv_show_id": pd.Series(dtype="int64"), "count50": pd.Series(dtype="int64"), "count80": pd.Series(dtype="int64"), "seconds50": pd.Series(dtype="float64"), "last_epoch": pd.Series(dtype="int64")})
            result["count50"] = result.count50.astype("int64")
            result["count80"] = result.count80.astype("int64")
            outputs[window.name] = result
            pq.write_table(pa.Table.from_pandas(result, preserve_index=False), staging / (window.name + ".parquet"), compression="zstd")
        complete_manifest(staging, identity, [w.name + ".parquet" for w in windows])
    return outputs, destination
