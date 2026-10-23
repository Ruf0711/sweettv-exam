"""Eight cheap inner trials, frozen outer comparison, conditional final export."""

import json
import math
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.cache import verify_cache
from tvrec.interactions import Window, epoch_column
from tvrec.io import atomic_json, atomic_parquet, file_hash, read_template, stable_hash
from tvrec.metrics import evaluate
from tvrec.pipeline import PREDICTION_SCHEMA
from tvrec.recommendations import Recommendation, format_submission
from tvrec.resources import LOGGER, Resources
from tvrec.stage2.rankers import PersonalHistory, global_scores, recommendations_from_scores
from tvrec_final.common import execution_environment, load_context, verify_provenance, verify_recorded_outputs
from tvrec_final.verification import read_future_catalogue, verify_csv, verify_regenerated
from tvrec_final.workflow import write_submission

from .features import build_features, score_counts, verify_against_existing


def code_hash():
    return stable_hash({path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))})


def load_options(path):
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8"))
    context = load_context(path.parent / options["final_v1_config"])
    if options["history_parameters"] != context.options["history"]:
        raise ValueError("Duration/recency signals must retain the validated v1 parameters")
    if options["submission_file"] != "submission_history_v2.csv":
        raise ValueError("The second candidate must not overwrite v1")
    return context, options


def preservation(root, destination):
    snapshot = json.loads((destination / "preservation_before.json").read_text(encoding="utf-8"))
    for name, expected in snapshot["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or path.stat().st_mtime_ns != expected["mtime_ns"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Pre-existing project evidence changed: {name}")
    previous = {name for name in snapshot["files"] if name.startswith("artifacts/")}
    current = {p.relative_to(root).as_posix() for p in (root / "artifacts").rglob("*") if p.is_file() and not p.is_relative_to(destination)}
    if previous != current:
        raise ValueError("Prior artifact inventory changed")
    return {"status": "passed", "files_checked": len(snapshot["files"]), "sha256_sizes_mtimes_and_prior_artifact_inventory_unchanged": True,
            "v1_sha256": file_hash(root / "submission_history_v1.csv"),
            "ai_session_backup_sha256": file_hash(root / "exam_session_backup.json") if (root / "exam_session_backup.json").exists() else None}


def specifications(options):
    specs = [{"id": f"count{threshold}", "family": f"count{threshold}", "kind": "count", "threshold": threshold} for threshold in [30, 50, 80]]
    specs += [{"id": "mixture_" + str(index), "family": "mixture", "kind": "mixture", "weights": weights} for index, weights in enumerate(options["mixture_weights"])]
    specs += [{"id": "count50_signals", "family": "signals", "kind": "signals", "threshold": 50, **options["history_parameters"]}]
    specs += [{"id": "count50_airing_" + str(index), "family": "airing", "kind": "airing", "threshold": 50, "exponent": exponent} for index, exponent in enumerate(options["airing_exponents"])]
    return specs


def schedule_frame(prepared):
    frame = pq.read_table(prepared / "broadcast_metadata.parquet", columns=["tv_show_id", "channel_id", "start_time", "end_time", "source_file"]).to_pandas()
    frame["start"] = epoch_column(frame.start_time)
    frame["end"] = epoch_column(frame.end_time)
    return frame.loc[(frame.tv_show_id > 0) & (frame.end > frame.start)].drop_duplicates(["channel_id", "tv_show_id", "start", "end"])


def airing_frequency(metadata, start, end, source=None):
    keep = (metadata.start < end) & (metadata.end > start)
    if source is not None:
        keep &= metadata.source_file == source
    return {int(item): int(count) for item, count in metadata.loc[keep].groupby("tv_show_id").size().items()}


def load_fold(context, fold, cohort, features, metadata):
    recorded = json.loads((context.stage2_run / fold / "context.json").read_text(encoding="utf-8"))
    paths = [Path(recorded["completion_window_path"]), Path(recorded["cumulative_paths"]["train"])]
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        verify_cache(path, manifest["identity"])
    feedback = pq.read_table(paths[0] / "train.parquet").to_pandas()
    cumulative = pq.read_table(paths[1] / "interactions.parquet").to_pandas()
    validation = pq.read_table(paths[0] / "validation.parquet").to_pandas()
    candidates = recorded["candidate_ids"]
    truth = validation.loc[(validation.count80 > 0) & validation.tv_show_id.isin(candidates)].groupby("user_id").tv_show_id.agg(set).to_dict()
    start, end = recorded["validation_window"]["start_epoch"], recorded["validation_window"]["end_exclusive_epoch"]
    return {"fold": fold, "cohort": cohort, "candidates": candidates, "truth": truth, "feedback": feedback,
            "counts": features[fold], "cutoff": recorded["train_window"]["end_exclusive_epoch"],
            "popularity": global_scores(cumulative, candidates), "airing": airing_frequency(metadata, start, end),
            "feature_anchor": verify_against_existing(features[fold], feedback, cohort), "original_context": recorded}


def evaluate_spec(spec, data, history_parameters):
    started = time.perf_counter()
    if spec["id"] == "HistoryV1":
        model = PersonalHistory().fit(data["feedback"], data["cutoff"], data["candidates"], data["popularity"], history_parameters)
        scores = model.score(data["cohort"])
        personal = data["feedback"].loc[data["feedback"].tv_show_id.isin(data["candidates"]) & data["feedback"].user_id.isin(data["cohort"])]
        support = personal.groupby("user_id").tv_show_id.nunique().to_dict()
        supported = np.array([support.get(user, 0) for user in data["cohort"]])
        warm = set(data["feedback"].user_id)
        coverage = {"candidate_count": len(data["candidates"]), "output_user_coverage": len(data["cohort"]),
                    "cold_users": sum(user not in warm for user in data["cohort"]),
                    "users_without_personal_candidate_support": int((supported == 0).sum()),
                    "users_with_fewer_than_five_personal_candidates": int((supported < 5).sum()),
                    "candidate_ids_with_personal_support": int(personal.tv_show_id.nunique()),
                    "fallback_positions": int(np.maximum(0, 5 - supported).sum())}
    else:
        scores, coverage = score_counts(data["counts"], data["cohort"], data["candidates"], data["popularity"], data["cutoff"], spec, data["airing"])
    rows = recommendations_from_scores(data["cohort"], data["candidates"], scores)
    metrics, per_user = evaluate(data["cohort"], rows, data["truth"], data["candidates"])
    return {"specification": spec, "metrics": metrics, "per_user": per_user, "rows": rows,
            "runtime_seconds": time.perf_counter() - started, "coverage": coverage}


def save_evaluation(directory, result):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "result.json", {key: value for key, value in result.items() if key not in ["rows", "per_user"]})
    atomic_parquet(directory / "per_user_metrics.parquet", pa.Table.from_pandas(result["per_user"], preserve_index=False))
    atomic_parquet(directory / "recommendations.parquet", pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in result["rows"]], schema=PREDICTION_SCHEMA))


def compact(result, baseline):
    return {"map_at_5": result["metrics"]["map_at_5_fixed"], "difference_vs_v1": result["metrics"]["map_at_5_fixed"] - baseline,
            "runtime_seconds": result["runtime_seconds"], "parameters": result["specification"], "coverage": result["coverage"]}


def paired_interval(challenger, baseline, options):
    difference = challenger.ap_at_5.to_numpy() - baseline.ap_at_5.to_numpy()
    rng = np.random.default_rng(options["seed"])
    samples = rng.integers(0, len(difference), size=(options["replicates"], len(difference)))
    means = difference[samples].mean(axis=1)
    low = options["lower_quantile"]
    return {"mean_difference": float(difference.mean()), "paired_bootstrap_95_percent_interval": [float(np.quantile(means, low)), float(np.quantile(means, 1 - low))],
            "replicates": options["replicates"], "seed": options["seed"], "users": len(difference),
            "note": "Descriptive paired user bootstrap; the overlapping temporal folds are not independent."}


def generate_v2(context, options, run, spec, features_path, metadata, v1):
    started = time.perf_counter()
    settings = context.settings
    cohort = read_template(settings.source(settings.values["sources"]["template"])).user_id.tolist()
    actual_future = settings.source(context.options["candidate_schedule"])
    display = read_future_catalogue(actual_future)
    candidates = sorted(display)
    counts = pq.read_table(features_path / "full.parquet").to_pandas()
    cumulative = pq.read_table(Path(v1["training_evidence"]["cumulative_window"]) / "interactions.parquet").to_pandas()
    popularity = global_scores(cumulative, candidates)
    future_end = int(metadata.loc[metadata.source_file == context.options["candidate_schedule"], "end"].max())
    airing = airing_frequency(metadata, context.cutoff, future_end, context.options["candidate_schedule"])
    scores, coverage = score_counts(counts, cohort, candidates, popularity, context.cutoff, spec, airing)
    rows = recommendations_from_scores(cohort, candidates, scores)
    payload = format_submission(cohort, rows, candidates).encode("utf-8")
    submission = settings.root / options["submission_file"]
    write_submission(submission, payload)
    directory = run / "final"
    directory.mkdir(exist_ok=False)
    np.savez_compressed(directory / "candidate_scores.npz", user_ids=np.asarray(cohort, dtype=str), candidate_ids=np.asarray(candidates, dtype=np.int64), scores=scores)
    atomic_parquet(directory / "recommendations.parquet", pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in rows], schema=PREDICTION_SCHEMA))
    csv_check = verify_csv(submission, settings.source(settings.values["sources"]["template"]), actual_future, directory / "candidate_scores.npz", directory / "recommendations.parquet")
    fresh_counts = pq.read_table(features_path / "full.parquet").to_pandas()
    fresh_scores, _ = score_counts(fresh_counts, cohort, candidates, popularity, context.cutoff, spec, airing)
    fresh_rows = recommendations_from_scores(cohort, candidates, fresh_scores)
    regenerated = format_submission(cohort, fresh_rows, candidates).encode("utf-8")
    reproducibility = verify_regenerated(submission, regenerated, scores, fresh_scores)
    write_submission(directory / "regenerated_submission.csv", regenerated)
    selected_users = set(context.options["example_user_ids"])
    atomic_json(directory / "examples_final.json", [{"user_id": row.user_id, "stage": "final_v2", "recommendations": [{"tv_show_id": item, "score": score, **display[item]} for item, score in zip(row.programme_ids, row.scores)]} for row in rows if row.user_id in selected_users])
    result = {"submission_path": str(submission), "sha256": file_hash(submission), "bytes": submission.stat().st_size,
              "specification": spec, "coverage": coverage, "cutoff_exclusive_utc": v1["training_window"]["end_exclusive_utc"],
              "csv_verification": csv_check, "reproducibility": reproducibility, "runtime_seconds": time.perf_counter() - started}
    atomic_json(directory / "manifest.json", result)
    return result


def run_experiment(config):
    context, options = load_options(config)
    destination = context.settings.root / options["artifacts_dir"]
    with Resources() as resources:
        with resources.phase("verify_preserved_sources_and_cached_evidence"):
            preservation(context.settings.root, destination)
            inputs = verify_provenance(context)
            tests = json.loads((destination / "tests/unit_tests.json").read_text(encoding="utf-8"))
            if not tests["successful"] or tests["optimization_code_hash"] != code_hash():
                raise ValueError("Passing tests for this count experiment are required")
            v1 = json.loads((context.settings.artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
            if file_hash(context.settings.root / "submission_history_v1.csv") != v1["submission"]["sha256"]:
                raise ValueError("V1 submission changed")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = destination / "runs" / ("counts-" + stamp)
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(run / "config.json", options)
        atomic_json(run / "external_reference.json", options["external_reference"])
        cohort = read_template(context.settings.source(context.settings.values["sources"]["template"])).user_id.tolist()
        records = {fold: json.loads((context.stage2_run / fold / "context.json").read_text(encoding="utf-8")) for fold in ["tuning", "main", "recent_b"]}
        windows = {fold: Window(fold, record["train_window"]["start_epoch"], record["train_window"]["end_exclusive_epoch"]) for fold, record in records.items()}
        windows["full"] = Window("full", v1["training_window"]["start_epoch"], context.cutoff)
        with resources.phase("single_cached_event_scan_strict_completion_features"):
            features, feature_path = build_features(destination, context.completion, windows, cohort)
            original_full = pq.read_table(Path(v1["training_evidence"]["completion_window"]) / "train_full.parquet").to_pandas()
            full_anchor = verify_against_existing(features["full"], original_full, cohort)
            metadata = schedule_frame(context.prepared)
        specs = specifications(options)
        baseline_spec = {"id": "HistoryV1", "family": "baseline", "kind": "unchanged_validated_history", **context.options["history"]}
        with resources.phase("inner_only_small_variant_selection"):
            tuning = load_fold(context, "tuning", cohort, features, metadata)
            inner = {}
            for spec in [baseline_spec, *specs]:
                result = evaluate_spec(spec, tuning, context.options["history"])
                inner[spec["id"]] = result
                save_evaluation(run / "tuning" / spec["id"], result)
            best_by_family = {}
            for spec in specs:
                family = spec["family"]
                old = best_by_family.get(family)
                if old is None or inner[spec["id"]]["metrics"]["map_at_5_fixed"] > inner[old["id"]]["metrics"]["map_at_5_fixed"]:
                    best_by_family[family] = spec
            selected = list(best_by_family.values())
            winner = max([baseline_spec, *selected], key=lambda spec: inner[spec["id"]]["metrics"]["map_at_5_fixed"])
            selection = {"parameters_selected_on": "existing inner chronological split only", "locked_before_outer_metrics": True,
                         "selected_variants": selected, "inner_winner": winner,
                         "inner_scores": {name: result["metrics"]["map_at_5_fixed"] for name, result in inner.items()},
                         "acceptance_rule": {"minimum_gain_both_outer_folds": options["minimum_outer_gain_each"], "both_paired_bootstrap_lower_bounds_above_zero": True}}
            atomic_json(run / "selection.json", selection)
        comparison, details, anchors = {}, {}, {"full": full_anchor, "tuning": tuning["feature_anchor"]}
        for fold in ["main", "recent_b"]:
            with resources.phase(f"{fold}_frozen_outer_comparison"):
                data = load_fold(context, fold, cohort, features, metadata)
                anchors[fold] = data["feature_anchor"]
                atomic_json(run / fold / "context.json", data["original_context"])
                comparison[fold], details[fold] = {}, {}
                base = None
                for spec in [baseline_spec, *selected]:
                    result = evaluate_spec(spec, data, context.options["history"])
                    if spec["id"] == "HistoryV1":
                        base = result["metrics"]["map_at_5_fixed"]
                        original = pq.read_table(context.stage2_run / fold / "PersonalHistory/recommendations.parquet").to_pylist()
                        if [{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in result["rows"]] != original:
                            raise ValueError("Existing History v1 was not exactly reproduced")
                    save_evaluation(run / fold / spec["id"], result)
                    comparison[fold][spec["id"]] = compact(result, base)
                    details[fold][spec["id"]] = result["per_user"]
                    LOGGER.info("%s %s MAP@5=%.8f delta=%+.8f", fold, spec["id"], result["metrics"]["map_at_5_fixed"], comparison[fold][spec["id"]]["difference_vs_v1"])
        intervals = {fold: paired_interval(details[fold][winner["id"]], details[fold]["HistoryV1"], options["bootstrap"]) for fold in details}
        accepted = winner["id"] != "HistoryV1" and all(
            intervals[fold]["mean_difference"] >= options["minimum_outer_gain_each"] and intervals[fold]["paired_bootstrap_95_percent_interval"][0] > 0 for fold in intervals)
        final = generate_v2(context, options, run, winner, feature_path, metadata, v1) if accepted else None
        atomic_json(run / "comparison.json", comparison)
        atomic_json(run / "acceptance.json", {"accepted": accepted, "winner": winner, "paired_intervals": intervals,
                    "recommendation": "v2 candidate" if accepted else "retain v1; no consistent independently validated gain from the frozen inner winner"})
        atomic_json(run / "feature_anchors.json", anchors)
        preserved = preservation(context.settings.root, destination)
        manifest = {"status": "complete", "run_dir": str(run), "optimization_code_hash": code_hash(), "config_sha256": file_hash(Path(config)),
                    "environment": execution_environment(), "inputs": inputs, "feature_path": str(feature_path),
                    "selected_variants": selected, "inner_winner": winner, "v2_generated": accepted, "final": final,
                    "comparison": comparison, "paired_intervals": intervals, "preservation": preserved,
                    "resources": resources.report(), "tests": {"path": str(destination / "tests/unit_tests.json"), "passed": tests["passed"]},
                    "methodological_reference": options["external_reference"], "kaggle_scores_used_for_selection": False,
                    "uploads_performed": False, "pdf_generated": False,
                    "outputs": {str(path.relative_to(run)): {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(run.rglob("*")) if path.is_file()}}
        atomic_json(run / "manifest.json", manifest)
        atomic_json(destination / "manifest.json", manifest)
    return {"status": "complete", "run_dir": str(run), "inner_winner": winner, "comparison": comparison,
            "paired_intervals": intervals, "v2_generated": accepted, "final": final, "resources": manifest["resources"], "preservation": preserved}


def verify_experiment(config):
    context, options = load_options(config)
    destination = context.settings.root / options["artifacts_dir"]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest["optimization_code_hash"] != code_hash() or manifest["config_sha256"] != file_hash(Path(config)):
        raise ValueError("Experiment code/configuration changed")
    run = Path(manifest["run_dir"])
    verify_recorded_outputs(run, manifest["outputs"])
    path = Path(manifest["feature_path"])
    cached = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    verify_cache(path, cached["identity"])
    cohort = read_template(context.settings.source(context.settings.values["sources"]["template"])).user_id.tolist()
    checked, ap_checked = 0, 0
    for fold in ["tuning", "main", "recent_b"]:
        source = json.loads((context.stage2_run / fold / "context.json").read_text(encoding="utf-8"))
        labels = pq.read_table(Path(source["completion_window_path"]) / "validation.parquet").to_pandas()
        candidates = set(source["candidate_ids"])
        truths = labels.loc[(labels.count80 > 0) & labels.tv_show_id.isin(candidates)].groupby("user_id").tv_show_id.agg(set).to_dict()
        for predictions in (run / fold).glob("*/recommendations.parquet"):
            rows = pq.read_table(predictions).to_pylist()
            if [row["user_id"] for row in rows] != cohort:
                raise ValueError("Template users/order changed")
            per_user = pq.read_table(predictions.parent / "per_user_metrics.parquet").to_pandas().set_index("user_id")
            result = json.loads((predictions.parent / "result.json").read_text(encoding="utf-8"))
            total = Fraction()
            for row in rows:
                items, user = row["programme_ids"], row["user_id"]
                if len(items) != 5 or len(set(items)) != 5 or any(item <= 0 or item not in candidates for item in items) or not all(math.isfinite(score) for score in row["scores"]):
                    raise ValueError("Invalid recommendation")
                truth = truths.get(user, set())
                ap, hits = Fraction(), 0
                if truth:
                    for rank, item in enumerate(items, 1):
                        if item in truth:
                            hits += 1
                            ap += Fraction(hits, rank)
                    ap /= min(5, len(truth))
                if not math.isclose(float(ap), per_user.loc[user, "ap_at_5"], rel_tol=0, abs_tol=2e-16):
                    raise ValueError("Independent rational AP differs")
                total += ap
                ap_checked += 1
            if not math.isclose(float(total / len(cohort)), result["metrics"]["map_at_5_fixed"], rel_tol=0, abs_tol=2e-16):
                raise ValueError("Independent rational MAP differs")
            checked += 1
    final_check = None
    if manifest["v2_generated"]:
        v1 = json.loads((context.settings.artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
        counts = pq.read_table(path / "full.parquet").to_pandas()
        cumulative = pq.read_table(Path(v1["training_evidence"]["cumulative_window"]) / "interactions.parquet").to_pandas()
        metadata = schedule_frame(context.prepared)
        future = context.settings.source(context.options["candidate_schedule"])
        candidates = sorted(read_future_catalogue(future))
        end = int(metadata.loc[metadata.source_file == context.options["candidate_schedule"], "end"].max())
        airing = airing_frequency(metadata, context.cutoff, end, context.options["candidate_schedule"])
        scores, _ = score_counts(counts, cohort, candidates, global_scores(cumulative, candidates), context.cutoff, manifest["inner_winner"], airing)
        payload = format_submission(cohort, recommendations_from_scores(cohort, candidates, scores), candidates).encode("utf-8")
        submission = context.settings.root / options["submission_file"]
        with np.load(run / "final/candidate_scores.npz", allow_pickle=False) as saved:
            regeneration = verify_regenerated(submission, payload, saved["scores"], scores)
        final_check = verify_csv(submission, context.settings.source(context.settings.values["sources"]["template"]), future,
                                 run / "final/candidate_scores.npz", run / "final/recommendations.parquet")
        final_check["fresh_process_regeneration"] = regeneration
    result = {"status": "passed", "model_fold_outputs_checked": checked, "independent_rational_per_user_ap_values": ap_checked,
              "final_v2": final_check, "preservation": preservation(context.settings.root, destination)}
    atomic_json(destination / "verification.json", result)
    return result
