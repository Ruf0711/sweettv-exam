"""Strict broadcast-completion counts from the existing session/broadcast join."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.cache import CacheBuild, complete_manifest, verify_cache
from tvrec.io import file_hash, stable_hash
from tvrec.stage2.completion import window_event_portion


def instance_portion(frame, window, blackouts):
    frame = frame.loc[frame.tv_show_id > 0]
    clipped = window_event_portion(frame, window, blackouts)
    seconds = clipped.observed_seconds.to_numpy()
    duration = clipped.broadcast_seconds.to_numpy()
    keep = 10 * seconds > 3 * duration
    clipped = clipped.loc[keep].copy()
    seconds, duration = seconds[keep], duration[keep]
    clipped["last50"] = np.where(2 * seconds > duration, clipped.last_observed_epoch, 0)
    clipped["last_ge50"] = np.where(2 * seconds >= duration, clipped.last_observed_epoch, 0)
    return clipped[["user_id", "tv_show_id", "broadcast_id", "broadcast_seconds", "observed_seconds", "last50", "last_ge50"]]


def finish_instances(parts):
    if not parts:
        return pd.DataFrame(columns=["user_id", "tv_show_id", "count30", "count50", "count80", "exact50_only", "seconds50", "last50", "ge50seconds", "last_ge50"])
    # Maximum SINGLE sitting, never sum separate sittings. A user/broadcast is
    # counted once even if duplicated across source chunks or watched twice.
    instances = pd.concat(parts, ignore_index=True).groupby(["user_id", "tv_show_id", "broadcast_id"], as_index=False).agg(
        broadcast_seconds=("broadcast_seconds", "first"), observed_seconds=("observed_seconds", "max"),
        last50=("last50", "max"), last_ge50=("last_ge50", "max"))
    seconds, duration = instances.observed_seconds.to_numpy(), instances.broadcast_seconds.to_numpy()
    instances["count30"] = (10 * seconds > 3 * duration).astype(np.int64)
    instances["count50"] = (2 * seconds > duration).astype(np.int64)
    instances["count80"] = (5 * seconds > 4 * duration).astype(np.int64)
    instances["exact50_only"] = (2 * seconds == duration).astype(np.int64)
    instances["seconds50"] = np.where(2 * seconds > duration, seconds, 0)
    instances["ge50seconds"] = np.where(2 * seconds >= duration, seconds, 0)
    return instances.groupby(["user_id", "tv_show_id"], as_index=False).agg(
        count30=("count30", "sum"), count50=("count50", "sum"), count80=("count80", "sum"),
        exact50_only=("exact50_only", "sum"), seconds50=("seconds50", "sum"), last50=("last50", "max"),
        ge50seconds=("ge50seconds", "sum"), last_ge50=("last_ge50", "max"))


def build_features(destination_root, completion, windows, cohort):
    identity = {"completion_cache": completion.name, "windows": {name: window.identity() for name, window in windows.items()},
                "cohort_hash": stable_hash({"users": cohort}), "feature_code_sha256": file_hash(Path(__file__)),
                "semantics": "strict thresholds; distinct broadcasts; maximum one-sitting unambiguous seconds; cutoff-clipped numerator"}
    destination = destination_root / "features" / stable_hash(identity)
    if destination.exists():
        verify_cache(destination, identity)
        return {name: pq.read_table(destination / (name + ".parquet")).to_pandas() for name in windows}, destination
    blackouts = json.loads((completion / "blackouts.json").read_text(encoding="utf-8"))
    parts = {name: [] for name in windows}
    for batch in pq.ParquetFile(completion / "events.parquet").iter_batches(batch_size=131072):
        frame = batch.to_pandas()
        frame = frame.loc[frame.user_id.isin(cohort)]
        if frame.empty:
            continue
        for name, window in windows.items():
            part = instance_portion(frame, window, blackouts)
            if len(part):
                parts[name].append(part.groupby(["user_id", "tv_show_id", "broadcast_id"], as_index=False).agg(
                    broadcast_seconds=("broadcast_seconds", "first"), observed_seconds=("observed_seconds", "max"),
                    last50=("last50", "max"), last_ge50=("last_ge50", "max")))
    result = {}
    with CacheBuild(destination) as staging:
        for name in windows:
            frame = finish_instances(parts.pop(name))
            if len(frame) and ((frame.last50 > windows[name].end).any() or (frame.last_ge50 > windows[name].end).any()):
                raise ValueError("Post-cutoff activity reached count features")
            result[name] = frame
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), staging / (name + ".parquet"), compression="zstd")
        complete_manifest(staging, identity, [name + ".parquet" for name in windows])
    return result, destination


def verify_against_existing(frame, original, cohort):
    """Exact equality at >=50 and >80 anchors this extension to verified data."""
    actual = frame.loc[frame.count50 + frame.exact50_only > 0, ["user_id", "tv_show_id", "count50", "count80", "exact50_only", "ge50seconds", "last_ge50"]].copy()
    actual["count50"] = actual.count50 + actual.exact50_only
    actual = actual.drop(columns="exact50_only").rename(columns={"ge50seconds": "seconds50", "last_ge50": "last_epoch"})
    expected = original.loc[original.user_id.isin(cohort), actual.columns]
    sort = lambda data: data.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sort(actual), sort(expected), check_dtype=False)
    return {"matched_original_inclusive50_and_strict80_pairs": len(actual), "exactly_half_broadcasts_excluded_by_strict50": int(frame.exact50_only.sum())}


def score_counts(frame, cohort, candidates, popularity, cutoff, spec, airing=None):
    """Small deterministic rankers; personal counts first, popularity fallback."""
    if frame.duplicated(["user_id", "tv_show_id"]).any():
        raise ValueError("Count features must be unique user/programme rows")
    if (frame.last50 > cutoff).any() or (frame.last_ge50 > cutoff).any():
        raise ValueError("Post-cutoff features reached the ranker")
    kind = spec["kind"]
    if kind == "mixture":
        value = frame[["count30", "count50", "count80"]].to_numpy(dtype=np.float64) @ np.asarray(spec["weights"])
    else:
        value = frame["count" + str(spec.get("threshold", 50))].to_numpy(dtype=np.float64)
    if kind == "signals":
        age = np.where(value > 0, np.maximum(0, cutoff - frame.last50.to_numpy()) / 86400., 0.)
        value *= (1 + spec["duration_coefficient"] * np.log1p(frame.seconds50.to_numpy() / 3600.)) * np.exp2(-age / spec["half_life_days"])
    if kind == "airing":
        value *= (1 + np.array([airing.get(int(item), 0) for item in frame.tv_show_id])) ** spec["exponent"]
    if np.any(value < 0) or not np.isfinite(value).all():
        raise ValueError("Invalid count ranking weights")
    users, items = {user: i for i, user in enumerate(cohort)}, {int(item): i for i, item in enumerate(candidates)}
    scores = np.zeros((len(cohort), len(candidates)), dtype=np.float64)
    valid = frame.user_id.isin(users).to_numpy() & frame.tv_show_id.isin(items).to_numpy() & (value > 0)
    rows = frame.loc[valid, "user_id"].map(users).to_numpy(dtype=np.int64)
    columns = frame.loc[valid, "tv_show_id"].map(items).to_numpy(dtype=np.int64)
    scores[rows, columns] = value[valid]
    supported = (scores > 0).sum(axis=1)
    warm_users = set(frame.loc[value > 0, "user_id"])
    cold = np.asarray([user not in warm_users for user in cohort])
    # Float64 keeps the original 1e-7 popularity prior as a meaningful tie-break
    # for integer counts. It cannot reverse a one-count primary difference.
    popularity = np.asarray(popularity, dtype=np.float64)
    scores += (popularity / max(1., float(popularity.max()))) * 1e-7
    scores[cold] = popularity
    return scores, {"candidate_count": len(candidates), "cold_users": int(cold.sum()),
                    "users_without_personal_candidate_support": int((supported == 0).sum()),
                    "users_with_fewer_than_five_personal_candidates": int((supported < 5).sum()),
                    "candidate_ids_with_personal_support": int(np.unique(columns).size),
                    "fallback_positions": int(np.maximum(0, 5 - supported).sum()), "output_user_coverage": len(cohort)}
