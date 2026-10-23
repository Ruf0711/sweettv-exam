"""Independent full-data completion/metric checks, without production label code.

Run with ``python -B -m tests.stage2_independent`` after the comparison CLI.
Reference overlap uses direct endpoint arithmetic; AP uses exact Fractions.
"""

import json
import math
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tests.test_stage2_integrity import slow_sessions
from tvrec.io import atomic_json, file_hash


def epochs(series):
    return series.to_numpy(dtype="datetime64[s]").astype(np.int64)


def rebuild_completion(run, manifest):
    cache = Path(manifest["completion_cache"])
    reference = pq.read_table(cache / "continuous_sessions.parquet", columns=["channel_id", "session_id", "start_time", "end_time"]).to_pandas()
    session_start, session_end = epochs(reference.start_time), epochs(reference.end_time)
    session_channel = reference.channel_id.to_numpy()
    np.testing.assert_array_equal(reference.session_id.to_numpy(), np.arange(len(reference)))
    assert (session_end > session_start).all()
    del reference
    blackouts = json.loads((cache / "blackouts.json").read_text(encoding="utf-8"))
    windows = {}
    for fold in ["tuning", "main", "recent_b"]:
        context = json.loads((run / fold / "context.json").read_text(encoding="utf-8"))
        for kind in ["train", "validation"]:
            bounds = context[kind + "_window"]
            windows[(fold, kind)] = {"start": bounds["start_epoch"], "end": bounds["end_exclusive_epoch"], "path": Path(context["completion_window_path"]) / (kind + ".parquet"), "instances": {}}
    event_count, source_broadcasts = 0, {}
    for batch in pq.ParquetFile(cache / "events.parquet").iter_batches(batch_size=131072):
        frame = batch.to_pandas()
        start, end = epochs(frame.start_time), epochs(frame.end_time)
        bs, be = epochs(frame.broadcast_start), epochs(frame.broadcast_end)
        duration = frame.broadcast_seconds.to_numpy()
        sid, channel = frame.session_id.to_numpy(), frame.channel_id.to_numpy()
        assert (frame.tv_show_id.to_numpy() > 0).all()
        assert (sid >= 0).all() and (sid < len(session_start)).all()
        np.testing.assert_array_equal(start, np.maximum(session_start[sid], bs))
        np.testing.assert_array_equal(end, np.minimum(session_end[sid], be))
        np.testing.assert_array_equal(channel, session_channel[sid])
        np.testing.assert_array_equal(duration, be - bs)
        assert (end > start).all() and (duration > 0).all()
        for row in frame.drop_duplicates("broadcast_id").itertuples():
            value = (int(row.channel_id), int(row.tv_show_id), int(row.broadcast_start.timestamp()), int(row.broadcast_end.timestamp()))
            old = source_broadcasts.setdefault(int(row.broadcast_id), value)
            assert old == value
        for window in windows.values():
            left, right = np.maximum(start, window["start"]), np.minimum(end, window["end"])
            seconds = np.maximum(0, right - left)
            for c, blocked in blackouts.items():
                ix = np.flatnonzero((channel == int(c)) & (seconds > 0))
                if len(ix):
                    for low, high in zip(blocked["start"], blocked["end"]):
                        seconds[ix] -= np.maximum(0, np.minimum(right[ix], high) - np.maximum(left[ix], low))
            assert (seconds >= 0).all() and (seconds <= duration).all()
            keep = (seconds > 0) & (2 * seconds >= duration)
            if not keep.any():
                continue
            part = frame.loc[keep, ["user_id", "tv_show_id", "broadcast_id"]].copy()
            part["seconds"] = seconds[keep]
            part["last_epoch"] = right[keep]
            part["gt80"] = 5 * seconds[keep] > 4 * duration[keep]
            part = part.groupby(["user_id", "tv_show_id", "broadcast_id"], as_index=False).max()
            for user, item, broadcast, watched, last, gt80 in part.itertuples(index=False, name=None):
                key = (user, int(item), int(broadcast))
                prior = window["instances"].get(key)
                value = (int(watched), int(last), bool(gt80))
                window["instances"][key] = value if prior is None else (max(prior[0], value[0]), max(prior[1], value[1]), prior[2] or value[2])
        event_count += len(frame)
    del session_start, session_end, session_channel
    metadata = pq.read_table(Path(manifest["stage1_preparation"]) / "broadcast_metadata.parquet", columns=["channel_id", "tv_show_id", "start_time", "end_time"]).to_pandas()
    starts, ends = epochs(metadata.start_time), epochs(metadata.end_time)
    source_keys = set(zip(metadata.channel_id.to_numpy().tolist(), metadata.tv_show_id.to_numpy().tolist(), starts.tolist(), ends.tolist()))
    assert all(value in source_keys for value in source_broadcasts.values())
    del metadata, source_keys
    rebuilt, evidence = {}, {}
    for key, window in windows.items():
        pairs = {}
        for (user, item, _), (seconds, last, gt80) in window.pop("instances").items():
            value = pairs.setdefault((user, item), [0, 0, 0, 0])
            value[0] += 1
            value[1] += int(gt80)
            value[2] += seconds
            value[3] = max(value[3], last)
        expected = pd.DataFrame([(u, item, *value) for (u, item), value in pairs.items()], columns=["user_id", "tv_show_id", "count50", "count80", "seconds50", "last_epoch"]).sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
        actual = pq.read_table(window["path"]).to_pandas().sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
        assert (actual.last_epoch <= window["end"]).all()
        rebuilt[key] = pairs
        evidence["/".join(key)] = {"user_programme_pairs_checked": len(pairs), "completed_broadcasts_ge50_checked": int(actual.count50.sum()), "completed_broadcasts_gt80_checked": int(actual.count80.sum())}
    return rebuilt, {"broadcast_session_events_checked": event_count, "distinct_broadcast_instances_checked_against_schedule": len(source_broadcasts), "windows": evidence}


def raw_session_reference(root, run, manifest, options, base):
    users = options["example_user_ids"]
    records = {user: [] for user in users}
    zone = ZoneInfo(base["timezone"])
    stop = int(datetime.strptime(base["preparation"]["viewing_end_exclusive_local"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).timestamp())
    for chunk in pd.read_csv(root / base["data_dir"] / base["sources"]["viewing"], dtype=str, keep_default_na=False, chunksize=250000):
        for row in chunk.loc[chunk.user_id.isin(users)].itertuples(index=False):
            start = int(datetime.strptime(row.start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).timestamp())
            end = int(datetime.strptime(row.stop_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).timestamp())
            assert end - start == int(row.duraton)
            if end > start and start < stop:
                records[row.user_id].append((row.user_id, int(row.vsetv_id), start, min(end, stop)))
    actual = pq.read_table(Path(manifest["completion_cache"]) / "continuous_sessions.parquet", filters=[("user_id", "in", users)]).to_pandas()
    output = {}
    for user in users:
        raw = records[user]
        expected = slow_sessions(*zip(*raw)) if raw else []
        subset = actual.loc[actual.user_id == user]
        observed = sorted(zip(subset.user_id, subset.channel_id, epochs(subset.start_time), epochs(subset.end_time)))
        assert expected == observed
        output[user] = {"raw_rows": len(raw), "continuous_sessions_checked": len(expected)}
    return output


def rational_metrics(root, run, options, base, rebuilt):
    cohort = pd.read_csv(root / base["data_dir"] / base["sources"]["template"], dtype=str, keep_default_na=False).user_id.tolist()
    assert len(cohort) == 1260 and len(set(cohort)) == len(cohort)
    all_scores, models_checked = {}, 0
    for fold in ["tuning", "main", "recent_b"]:
        context = json.loads((run / fold / "context.json").read_text(encoding="utf-8"))
        candidates = set(context["candidate_ids"])
        truth = {rule: {} for rule in options["relevance_rules"]}
        cumulative = pq.read_table(Path(context["cumulative_paths"]["validation"]) / "interactions.parquet", columns=["user_id", "tv_show_id", "seconds"]).to_pandas()
        for row in cumulative.loc[cumulative.seconds >= 60].itertuples(index=False):
            if row.tv_show_id in candidates:
                truth["cumulative_ge60"].setdefault(row.user_id, set()).add(int(row.tv_show_id))
        for (user, item), value in rebuilt[(fold, "validation")].items():
            if item in candidates:
                truth["completion_ge50"].setdefault(user, set()).add(item)
                if value[1] > 0:
                    truth["completion_gt80"].setdefault(user, set()).add(item)
        all_scores[fold] = {}
        for path in sorted((run / fold).glob("*/recommendations.parquet")):
            predictions = pq.read_table(path).to_pylist()
            assert [r["user_id"] for r in predictions] == cohort
            recorded = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
            per_user = pq.read_table(path.parent / "per_user_metrics.parquet").to_pandas()
            all_scores[fold][path.parent.name] = {}
            for rule, labels in truth.items():
                total, eligible = Fraction(), 0
                details = per_user.loc[per_user.relevance_rule == rule].set_index("user_id")
                for row in predictions:
                    user, items = row["user_id"], row["programme_ids"]
                    assert isinstance(user, str) and len(items) == 5 and len(set(items)) == 5
                    assert all(item > 0 and item in candidates for item in items)
                    relevant = labels.get(user, set())
                    hits, ap = 0, Fraction()
                    if relevant:
                        eligible += 1
                        for rank, item in enumerate(items, 1):
                            if item in relevant:
                                hits += 1
                                ap += Fraction(hits, rank)
                        ap /= min(5, len(relevant))
                    assert math.isclose(float(ap), details.loc[user, "ap_at_5"], rel_tol=0, abs_tol=2e-16)
                    assert len(relevant) == details.loc[user, "relevant_count"]
                    total += ap
                fixed = total / len(cohort)
                evaluable = total / eligible if eligible else Fraction()
                assert math.isclose(float(fixed), recorded[rule]["map_at_5_fixed"], rel_tol=0, abs_tol=2e-16)
                assert math.isclose(float(evaluable), recorded[rule]["map_at_5_evaluable"], rel_tol=0, abs_tol=2e-16)
                assert eligible == recorded[rule]["evaluable_users"]
                all_scores[fold][path.parent.name][rule] = {"fixed_map_fraction": str(fixed), "fixed_map": float(fixed), "evaluable_map_fraction": str(evaluable)}
            models_checked += 1
    return {"model_fold_outputs_checked": models_checked, "metric_values_checked": models_checked * 3, "per_user_ap_values_checked": models_checked * 3 * len(cohort), "scores": all_scores}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    stage2 = root / "artifacts/stage2"
    latest = json.loads((stage2 / "latest_comparison.json").read_text(encoding="utf-8"))
    run = Path(latest["run_dir"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    options = json.loads((run / "config.json").read_text(encoding="utf-8"))
    base = json.loads((run / "effective_base_config.json").read_text(encoding="utf-8"))
    tests = json.loads((stage2 / "tests/unit_tests.json").read_text(encoding="utf-8"))
    assert tests["successful"] and tests["stage2_source_code_hash"] == manifest["source_code_hash"]
    rebuilt, completion = rebuild_completion(run, manifest)
    metrics = rational_metrics(root, run, options, base, rebuilt)
    del rebuilt
    sessions = raw_session_reference(root, run, manifest, options, base)
    result = {"status": "passed", "run_dir": str(run), "script_sha256": file_hash(Path(__file__)), "application_source_hash_matches_executed_tests": True, "completion": completion, "raw_session_reconstruction": sessions, "independent_rational_metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
    atomic_json(stage2 / "checks/independent.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "independent_rational_metrics"}, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in metrics.items() if key != "scores"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
