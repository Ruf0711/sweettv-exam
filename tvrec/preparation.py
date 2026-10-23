"""One full join; immutable, timestamped positive evidence for later experiments."""

import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cache import CacheBuild, complete_manifest, preparation_identity, verify_cache
from .intervals import canonical_timeline, normalize_viewing
from .io import LOG_COLUMNS, SCHEDULE_COLUMNS, UTC_TYPE, arrow_utc, atomic_json, csv_chunks, integer_column, stable_hash, validate_user_ids
from .resources import LOGGER, progress
from .time_utils import LOG_FORMAT, SCHEDULE_FORMAT, local_epoch, parse_local_times, utc_iso

EVIDENCE_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("tv_show_id", pa.int64()), ("channel_id", pa.int32()),
    ("view_id", pa.int64()), ("segment_id", pa.int64()),
    ("start_time", UTC_TYPE), ("end_time", UTC_TYPE),
    ("allocation_denominator", pa.int32()), ("overlap_seconds", pa.int64()), ("credited_seconds", pa.float64()),
])
VIEW_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("channel_id", pa.int32()), ("view_id", pa.int64()),
    ("start_time", UTC_TYPE), ("end_time", UTC_TYPE), ("source_rows", pa.int64()),
])
_META_TEXT = [c for c in SCHEDULE_COLUMNS if c not in {"channel_id", "start_time", "duration", "tv_show_id"}]
BROADCAST_SCHEMA = pa.schema([
    ("tv_show_id", pa.int64()), ("channel_id", pa.int32()), ("start_time", UTC_TYPE),
    ("end_time", UTC_TYPE), ("duration", pa.int64()), ("source_file", pa.string()),
    *[(name, pa.string()) for name in _META_TEXT],
])
PROGRAMME_SCHEMA = pa.schema([
    ("tv_show_id", pa.int64()), ("tv_show_title", pa.string()), ("tv_show_category", pa.string()),
    ("title_variants", pa.list_(pa.string())), ("category_values", pa.list_(pa.string())),
    *[(name + "_values", pa.list_(pa.string())) for name in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]],
    ("channel_ids", pa.list_(pa.int32())), ("first_start_time", UTC_TYPE), ("last_end_time", UTC_TYPE),
])


def _broadcast_table(chunk, channels, ids, starts, ends, duration, filename, positive):
    arrays = {
        "tv_show_id": pa.array(ids[positive], type=pa.int64()), "channel_id": pa.array(channels[positive], type=pa.int32()),
        "start_time": arrow_utc(starts[positive]), "end_time": arrow_utc(ends[positive]),
        "duration": pa.array(duration[positive], type=pa.int64()),
        "source_file": pa.array([filename] * int(positive.sum()), type=pa.string()),
    }
    for name in _META_TEXT:
        arrays[name] = pa.array(chunk.loc[positive, name].tolist(), type=pa.string())
    return pa.Table.from_pydict(arrays, schema=BROADCAST_SCHEMA)


def load_schedules(settings, directory):
    parts, per_file, by_file = [], [], {}
    prep = settings.values["preparation"]
    with pq.ParquetWriter(directory / "broadcast_metadata.parquet", BROADCAST_SCHEMA, compression="zstd") as writer:
        for filename in settings.values["sources"]["schedules"]:
            stats = Counter()
            ids_in_file = set()
            for chunk in csv_chunks(settings.source(filename), SCHEDULE_COLUMNS, prep["csv_chunk_rows"]):
                channels = integer_column(chunk.channel_id, "channel_id", bits=32)
                ids = integer_column(chunk.tv_show_id, "tv_show_id", missing_zero=True)
                duration = integer_column(chunk.duration, "duration", signed=True)
                starts = parse_local_times(chunk.start_time, SCHEDULE_FORMAT, settings.zone)
                ends = starts + duration
                stats["rows"] += len(chunk)
                stats["missing_programme_ids"] += int((chunk.tv_show_id == "").sum())
                stats["zero_programme_rows"] += int((ids == 0).sum())
                stats["zero_duration_rows"] += int((duration == 0).sum())
                stats["negative_duration_rows"] += int((duration < 0).sum())
                stats["positive_programme_rows"] += int((ids > 0).sum())
                ids_in_file.update(int(x) for x in ids[ids > 0])
                writer.write_table(_broadcast_table(chunk, channels, ids, starts, ends, duration, filename, ids > 0))
                valid = duration > 0
                parts.append(pd.DataFrame({"channel_id": channels[valid], "start": starts[valid], "end": ends[valid], "tv_show_id": ids[valid]}))
                if stats["rows"] % prep["progress_every_rows"] < len(chunk):
                    progress(filename, stats["rows"])
            stats["distinct_positive_ids"] = len(ids_in_file)
            per_file.append({"file": filename, **dict(stats)})
            by_file[filename] = sorted(ids_in_file)
    frame = pd.concat(parts, ignore_index=True)
    del parts
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().sort_values(["channel_id", "start", "end", "tv_show_id"], kind="mergesort")
    timelines = {}
    base = 0
    competing_segments = 0
    ambiguous_seconds = 0
    for channel, group in frame.groupby("channel_id", sort=True):
        timeline = canonical_timeline(group.start.to_numpy(), group.end.to_numpy(), group.tv_show_id.to_numpy(), segment_base=base)
        timelines[int(channel)] = timeline
        base += len(timeline.starts)
        sizes = np.diff(timeline.label_offsets)
        ambiguous = sizes > 1
        competing_segments += int(ambiguous.sum())
        ambiguous_seconds += int((timeline.ends[ambiguous] - timeline.starts[ambiguous]).sum())
    all_positive = set().union(*(set(value) for value in by_file.values()))
    stats = {
        "files": per_file, "distinct_channels": len(timelines), "distinct_positive_programmes": len(all_positive),
        "exact_duplicate_broadcast_rows": duplicate_count, "canonical_segments": base,
        "segments_with_competing_distinct_labels": competing_segments,
        "ambiguous_distinct_label_seconds": ambiguous_seconds,
        "allocation_policy": prep["competing_broadcast_policy"],
        "zero_label_share": "Unknown label 0 participates in equal sharing but is never emitted as an item",
    }
    atomic_json(directory / "candidates.json", {"by_file": by_file})
    _write_programmes(directory)
    return timelines, stats


def _write_programmes(directory):
    frame = pq.read_table(directory / "broadcast_metadata.parquet").to_pandas()
    frame = frame.sort_values(["start_time", "tv_show_id", "channel_id", "source_file"], kind="mergesort")
    records = []
    text_fields = ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]
    for item, group in frame.groupby("tv_show_id", sort=True):
        titles = [x for x in dict.fromkeys(group.tv_show_title) if x]
        categories = [x for x in dict.fromkeys(group.tv_show_category) if x]
        record = {
            "tv_show_id": int(item), "tv_show_title": titles[0] if titles else "",
            "tv_show_category": categories[0] if categories else "", "title_variants": titles,
            "category_values": categories, "channel_ids": sorted(int(x) for x in group.channel_id.unique()),
            "first_start_time": group.start_time.min(), "last_end_time": group.end_time.max(),
        }
        for name in text_fields:
            record[name + "_values"] = [x for x in dict.fromkeys(group[name]) if x]
        records.append(record)
    pq.write_table(pa.Table.from_pylist(records, schema=PROGRAMME_SCHEMA), directory / "programme_metadata.parquet", compression="zstd")


def load_viewing(settings, timelines):
    prep = settings.values["preparation"]
    allowed_end = local_epoch(prep["viewing_end_exclusive_local"], settings.zone)
    arrays = {name: [] for name in ["user_index", "channel_id", "start", "end", "duration"]}
    users, user_index, source_users, source_channels = [], {}, set(), set()
    stats, missing_channels = Counter(), Counter()
    earliest, latest = None, None
    for chunk in csv_chunks(settings.source(settings.values["sources"]["viewing"]), LOG_COLUMNS, prep["csv_chunk_rows"]):
        validate_user_ids(chunk.user_id)
        channels = integer_column(chunk.vsetv_id, "vsetv_id", bits=32)
        duration = integer_column(chunk.duraton, "duraton", signed=True)
        starts = parse_local_times(chunk.start_time, LOG_FORMAT, settings.zone)
        ends = parse_local_times(chunk.stop_time, LOG_FORMAT, settings.zone)
        elapsed = ends - starts
        stats["source_rows"] += len(chunk)
        source_users.update(chunk.user_id.unique())
        source_channels.update(int(x) for x in np.unique(channels))
        valid = (elapsed > 0) & (duration > 0)
        stats["excluded_nonpositive_rows"] += int((~valid).sum())
        mismatch = valid & (elapsed != duration)
        stats["utc_duration_mismatches"] += int(mismatch.sum())
        if mismatch.any() and prep["require_elapsed_duration_agreement"]:
            raise ValueError("UTC elapsed duration disagrees with duraton; review the timezone assumption")
        future = starts >= allowed_end
        stats["excluded_future_rows"] += int((valid & future).sum())
        valid &= ~future
        stats["clipped_at_viewing_end_rows"] += int((valid & (ends > allowed_end)).sum())
        ends = np.minimum(ends, allowed_end)
        valid &= ends > starts
        frame = chunk.loc[valid]
        for user in frame.user_id.unique():
            if user not in user_index:
                user_index[user] = len(users)
                users.append(user)
        ui = frame.user_id.map(user_index).to_numpy(dtype=np.int32)
        c, s, e, d = channels[valid], starts[valid], ends[valid], duration[valid]
        arrays["user_index"].append(ui)
        arrays["channel_id"].append(c)
        arrays["start"].append(s)
        arrays["end"].append(e)
        arrays["duration"].append(d)
        stats["allowed_raw_rows"] += len(s)
        stats["allowed_elapsed_seconds"] += int((e - s).sum())
        if len(s):
            earliest = int(s.min()) if earliest is None else min(earliest, int(s.min()))
            latest = int(e.max()) if latest is None else max(latest, int(e.max()))
        for channel, ix in pd.Series(c).groupby(pd.Series(c), sort=False).indices.items():
            channel = int(channel)
            if channel not in timelines:
                missing_channels[channel] += len(ix)
                continue
            stats["rows_with_channel_in_schedules"] += len(ix)
            stats["seconds_on_channels_in_schedules"] += int((e[ix] - s[ix]).sum())
            timeline = timelines[channel]
            any_seconds = timeline.covered_seconds(s[ix], e[ix])
            positive_seconds = timeline.covered_seconds(s[ix], e[ix], positive=True)
            stats["raw_any_broadcast_match_rows"] += int((any_seconds > 0).sum())
            stats["raw_positive_programme_match_rows"] += int((positive_seconds > 0).sum())
            stats["raw_covered_union_seconds"] += int(any_seconds.sum())
            stats["raw_positive_union_seconds"] += int(positive_seconds.sum())
        if stats["source_rows"] % prep["progress_every_rows"] < len(chunk):
            progress("Viewing CSV parsed", stats["source_rows"])
    if not users:
        raise ValueError("No allowed positive-duration viewing intervals")
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
    elapsed = arrays["end"] - arrays["start"]
    stats = dict(stats)
    stats.update({
        "distinct_source_users": len(source_users), "distinct_allowed_users": len(users),
        "distinct_source_channels": len(source_channels),
        "direct_channel_id_intersection": len(source_channels & set(timelines)),
        "missing_channel_ids": sorted(source_channels - set(timelines)),
        "missing_channel_row_counts": {str(k): int(v) for k, v in sorted(missing_channels.items())},
        "observed_start_utc": utc_iso(earliest), "observed_start_epoch": earliest,
        "observed_end_utc": utc_iso(latest), "allowed_end_exclusive_utc": utc_iso(allowed_end),
        "duration_quantiles_seconds": {str(q): float(np.quantile(elapsed, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]},
        "duration_threshold_counts": {str(t): int((elapsed > t).sum()) for t in [21600, 43200, 86400]},
    })
    stats["raw_any_match_fraction"] = stats.get("raw_any_broadcast_match_rows", 0) / stats["allowed_raw_rows"]
    stats["raw_positive_match_fraction"] = stats.get("raw_positive_programme_match_rows", 0) / stats["allowed_raw_rows"]
    return arrays, users, stats


def _view_table(views, usernames, low, high):
    sl = slice(low, high)
    return pa.Table.from_arrays([
        pa.array(usernames[views["user_index"][sl]], type=pa.string()), pa.array(views["channel_id"][sl], type=pa.int32()),
        pa.array(np.arange(low, high, dtype=np.int64)), arrow_utc(views["start"][sl]), arrow_utc(views["end"][sl]),
        pa.array(views["source_rows"][sl], type=pa.int64()),
    ], schema=VIEW_SCHEMA)


def join_views(settings, directory, timelines, views, users):
    prep = settings.values["preparation"]
    usernames = np.asarray(users, dtype=object)
    total = len(views["start"])
    stats = Counter()
    known_users, known_items, known_channels = set(), set(), set()
    with pq.ParquetWriter(directory / "normalized_viewing.parquet", VIEW_SCHEMA, compression="zstd") as view_writer, pq.ParquetWriter(directory / "evidence.parquet", EVIDENCE_SCHEMA, compression="zstd") as evidence_writer:
        for low in range(0, total, prep["join_chunk_rows"]):
            high = min(total, low + prep["join_chunk_rows"])
            view_writer.write_table(_view_table(views, usernames, low, high))
            s, e, c = (views[name][low:high] for name in ["start", "end", "channel_id"])
            covered = np.zeros(len(s), dtype=np.int64)
            positive_credit = np.zeros(len(s), dtype=np.float64)
            unknown_credit = np.zeros(len(s), dtype=np.float64)
            evidence_parts = []
            for channel, ix in pd.Series(c).groupby(pd.Series(c), sort=False).indices.items():
                if int(channel) not in timelines:
                    continue
                timeline = timelines[int(channel)]
                row, segment, left, right = timeline.overlaps(s[ix], e[ix])
                if not len(row):
                    continue
                physical = right - left
                covered[ix] = np.bincount(row, weights=physical, minlength=len(ix)).astype(np.int64)
                sizes = timeline.label_offsets[segment + 1] - timeline.label_offsets[segment]
                which = np.repeat(np.arange(len(row)), sizes)
                count = int(sizes.sum())
                label_ix = np.repeat(timeline.label_offsets[segment], sizes) + np.arange(count) - np.repeat(np.cumsum(sizes) - sizes, sizes)
                label = timeline.labels[label_ix]
                credits = physical[which] / sizes[which]
                expanded_rows = row[which]
                positive = label > 0
                positive_credit[ix] = np.bincount(expanded_rows[positive], weights=credits[positive], minlength=len(ix))
                unknown_credit[ix] = np.bincount(expanded_rows[~positive], weights=credits[~positive], minlength=len(ix))
                if positive.any():
                    p = which[positive]
                    view_id = low + ix[row[p]]
                    evidence_parts.append({
                        "user_index": views["user_index"][view_id], "tv_show_id": label[positive],
                        "channel_id": np.full(len(p), int(channel), dtype=np.int32), "view_id": view_id,
                        "segment_id": timeline.segment_base + segment[p], "start": left[p], "end": right[p],
                        "allocation_denominator": sizes[p].astype(np.int32), "overlap_seconds": physical[p], "credited_seconds": credits[positive],
                    })
                    known_users.update(int(x) for x in np.unique(views["user_index"][view_id]))
                    known_items.update(int(x) for x in np.unique(label[positive]))
                    known_channels.add(int(channel))
            if not np.allclose(positive_credit + unknown_credit, covered, rtol=0, atol=1e-7) or np.any(covered > e - s):
                raise AssertionError("Duration conservation failed in the interval join")
            stats["normalized_intervals"] += len(s)
            stats["any_match_intervals"] += int((covered > 0).sum())
            stats["positive_match_intervals"] += int((positive_credit > 0).sum())
            stats["viewing_union_seconds"] += int((e - s).sum())
            stats["covered_union_seconds"] += int(covered.sum())
            stats["positive_credited_seconds"] += float(positive_credit.sum())
            stats["unknown_credited_seconds"] += float(unknown_credit.sum())
            if evidence_parts:
                keys = list(evidence_parts[0])
                combined = {key: np.concatenate([part[key] for part in evidence_parts]) for key in keys}
                order = np.lexsort((combined["tv_show_id"], combined["segment_id"], combined["view_id"]))
                combined = {key: value[order] for key, value in combined.items()}
                table = pa.Table.from_arrays([
                    pa.array(usernames[combined["user_index"]], type=pa.string()), pa.array(combined["tv_show_id"], type=pa.int64()),
                    pa.array(combined["channel_id"], type=pa.int32()), pa.array(combined["view_id"], type=pa.int64()),
                    pa.array(combined["segment_id"], type=pa.int64()), arrow_utc(combined["start"]), arrow_utc(combined["end"]),
                    pa.array(combined["allocation_denominator"], type=pa.int32()), pa.array(combined["overlap_seconds"], type=pa.int64()),
                    pa.array(combined["credited_seconds"], type=pa.float64()),
                ], schema=EVIDENCE_SCHEMA)
                evidence_writer.write_table(table)
                stats["positive_evidence_fragments"] += table.num_rows
            if high % prep["progress_every_rows"] < high - low or high == total:
                progress("Normalized intervals joined", high, total)
    result = dict(stats)
    result.update({"users_with_identified_history": len(known_users), "programmes_with_identified_history": len(known_items), "channels_with_identified_history": len(known_channels)})
    return result


def prepare(settings, fingerprints, resources):
    identity = preparation_identity(settings, fingerprints)
    key = stable_hash(identity)
    destination = settings.artifacts_dir / "cache" / key
    if destination.exists():
        LOGGER.info("Reusing verified preparation cache %s", key)
        verify_cache(destination, identity)
        return destination, json.loads((destination / "diagnostics.json").read_text(encoding="utf-8")), True
    with CacheBuild(destination) as directory:
        with resources.phase("load_schedules_and_build_channel_indexes"):
            timelines, schedule_stats = load_schedules(settings, directory)
        with resources.phase("parse_viewing_and_measure_raw_coverage"):
            arrays, users, raw_stats = load_viewing(settings, timelines)
        with resources.phase("deduplicate_and_union_user_channel_viewing"):
            views, normalized_stats = normalize_viewing(arrays["user_index"], arrays["channel_id"], arrays["start"], arrays["end"], arrays["duration"])
            del arrays
            gc.collect()
        with resources.phase("indexed_join_and_write_timestamped_evidence"):
            join_stats = join_views(settings, directory, timelines, views, users)
        atomic_json(directory / "users.json", {"user_ids": users, "external_id_type": "exact_string"})
        diagnostics = {
            "preparation_key": key, "timezone": settings.values["timezone"], "timezone_assumption": settings.values["timezone_assumption"],
            "raw": raw_stats, "schedule": schedule_stats, "normalized": normalized_stats, "joined": join_stats,
            "metadata_policy": "programme_metadata is an all-schedule catalogue reference; use timestamped broadcast_metadata for historical features",
            "count_policy": "Evidence fragments are canonical segment/normalized-view/positive-label intersections; view counts refer to distinct normalized view IDs",
        }
        atomic_json(directory / "diagnostics.json", diagnostics)
        filenames = ["broadcast_metadata.parquet", "programme_metadata.parquet", "normalized_viewing.parquet", "evidence.parquet", "candidates.json", "users.json", "diagnostics.json"]
        complete_manifest(directory, identity, filenames)
    return destination, diagnostics, False
