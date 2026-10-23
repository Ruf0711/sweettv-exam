"""Reproducible Stage 1 execution, diagnostics, and independently checkable results."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cache import code_hash, environment
from .interactions import Window, aggregate_windows, candidate_ids, historical_metadata, interaction_stats, viewing_activity
from .io import atomic_json, atomic_parquet, file_hash, fingerprint_files, read_template, stable_hash
from .metrics import evaluate, relevance_sets
from .popularity import Popularity
from .preparation import prepare
from .recommendations import Recommendation
from .resources import LOGGER, Resources
from .time_utils import local_epoch, utc_iso

PREDICTION_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("programme_ids", pa.list_(pa.int64())), ("scores", pa.list_(pa.float64())),
])
AUDIT_LOG_HASH = "8c9e5f058157041e5c33f13e64dc1625a5177cc6d5e434ba18cd8a69afafb53b"
AUDIT_REFERENCE = {
    "source_rows": 5821987, "distinct_source_users": 4914, "distinct_source_channels": 202,
    "direct_channel_id_intersection": 197, "raw_any_broadcast_match_rows": 5408690,
    "raw_positive_programme_match_rows": 2964378, "raw_covered_union_seconds": 8107698218,
    "raw_positive_union_seconds": 4575229505,
}


def source_paths(settings):
    source = settings.values["sources"]
    names = [source["viewing"], *source["schedules"], source["template"]]
    if source.get("archive"):
        names.append(source["archive"])
    paths = [settings.source(name) for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing original source file: {path}")
    return paths


def fold_windows(settings, fold_name, diagnostics):
    fold = settings.values["folds"][fold_name]
    start = local_epoch(fold["train_start_local"], settings.zone) if fold.get("train_start_local") else diagnostics["raw"]["observed_start_epoch"]
    cut = local_epoch(fold["cutoff_local"], settings.zone)
    end = local_epoch(fold["validation_end_local"], settings.zone)
    permitted_end = local_epoch(settings.values["preparation"]["viewing_end_exclusive_local"], settings.zone)
    if end > permitted_end:
        raise ValueError("Validation extends into the prohibited future viewing period")
    train, validation = Window("train", start, cut), Window("validation", cut, end)
    lookback = settings.values["popularity"]["lookback_days"]
    if lookback is None:
        support = train
    else:
        wall = datetime.strptime(fold["cutoff_local"], "%Y-%m-%d %H:%M:%S") - timedelta(days=lookback)
        support = Window("popularity_support", max(start, local_epoch(wall.strftime("%Y-%m-%d %H:%M:%S"), settings.zone)), cut)
    return train, validation, support


def cohort_diagnostics(cohort, train, validation, candidates, activity):
    cohort_set, candidate_set = set(cohort), set(candidates)
    train_cohort = train.loc[train.user_id.isin(cohort_set)]
    val_cohort = validation.loc[validation.user_id.isin(cohort_set)]
    threshold = val_cohort.loc[val_cohort.relevant]
    included = threshold.loc[threshold.tv_show_id.isin(candidate_set)]
    excluded = threshold.loc[~threshold.tv_show_id.isin(candidate_set)]
    validation_viewers = set(activity["validation"]["users"])
    identified_viewers = set(val_cohort.user_id)
    return {
        "fixed_cohort_users": len(cohort),
        "validation_users_with_any_viewing": len(validation_viewers),
        "validation_users_without_any_viewing": len(cohort_set - validation_viewers),
        "validation_users_with_identified_seconds": len(identified_viewers),
        "validation_active_users_without_identified_labels": len(validation_viewers - identified_viewers),
        "validation_users_without_identified_labels": len(cohort_set - identified_viewers),
        "validation_users_with_threshold_relevance_before_catalogue_filter": int(threshold.user_id.nunique()),
        "validation_users_with_eligible_relevance": int(included.user_id.nunique()),
        "zero_eligible_positive_users": len(cohort) - int(included.user_id.nunique()),
        "training_users_without_any_identified_support": len(cohort_set - set(train_cohort.user_id)),
        "training_users_without_threshold_support": len(cohort_set - set(train_cohort.loc[train_cohort.relevant, "user_id"])),
        "validation_pairs_below_relevance_threshold": int((~val_cohort.relevant).sum()),
        "validation_relevant_pairs_before_catalogue_filter": len(threshold),
        "validation_relevant_pairs_after_catalogue_filter": len(included),
        "validation_relevant_pairs_excluded_by_catalogue": len(excluded),
        "validation_distinct_relevant_programmes_excluded_by_catalogue": int(excluded.tv_show_id.nunique()),
        "all_user_validation_relevant_pairs_excluded_by_catalogue": int((validation.relevant & ~validation.tv_show_id.isin(candidate_set)).sum()),
        "candidate_count": len(candidate_set),
        "candidate_count_with_positive_training_support": len(candidate_set & set(train.tv_show_id)),
        "candidate_count_with_threshold_training_support": len(candidate_set & set(train.loc[train.relevant, "tv_show_id"])),
    }


def validation_examples(settings, prepared, cohort, train, validation, recommendations, cutoff):
    # IDs were explicitly preselected in configuration before evaluation.
    chosen = settings.values["evaluation"]["example_user_ids"]
    lookup = {row.user_id: row for row in recommendations}
    metadata = historical_metadata(prepared, cutoff).set_index("tv_show_id").to_dict("index")
    examples = []
    for user in chosen:
        if user not in set(cohort):
            raise ValueError(f"Configured example user is absent from the template: {user}")
        history = train.loc[train.user_id == user]
        heldout = validation.loc[validation.user_id == user]
        prediction = lookup[user]
        items = []
        for item, score in zip(prediction.programme_ids, prediction.scores):
            record = metadata.get(item, {})
            items.append({"tv_show_id": item, "distinct_training_viewers": int(score), "title": record.get("tv_show_title"), "category": record.get("tv_show_category")})
        examples.append({
            "user_id": user, "training_programme_count": len(history), "training_identified_seconds": float(history.seconds.sum()),
            "training_threshold_programme_count": int(history.relevant.sum()),
            "validation_threshold_programme_count_before_catalogue_filter": int(heldout.relevant.sum()),
            "recommendations": items,
        })
    return {"selection": "Explicit IDs preselected in configuration; no selection using validation scores", "metadata_as_of_exclusive_utc": utc_iso(cutoff), "global_popularity_is_identical_for_all_users": True, "users": examples}


def compare_audit(fingerprints, settings, raw):
    applicable = fingerprints[settings.values["sources"]["viewing"]]["sha256"] == AUDIT_LOG_HASH
    differences = {key: raw.get(key, 0) - value for key, value in AUDIT_REFERENCE.items()} if applicable else {}
    return {
        "applicable_to_original_viewing_file": applicable, "reference": AUDIT_REFERENCE,
        "measured_minus_reference": differences,
        "all_compared_raw_counts_agree": all(value == 0 for value in differences.values()) if applicable else None,
        "normalization_explanation": "Raw coverage retains original source rows for audit comparison. Model evidence removes exact duplicates, unions overlapping/touching same-user/channel intervals, and shares competing seconds equally across distinct active labels including unknown 0. Its denominators and credited seconds therefore differ.",
    }


def run_pipeline(settings, fold_name="main", command="run"):
    np.random.seed(settings.values["seed"])
    with Resources() as resources:
        with resources.phase("fingerprint_original_inputs"):
            paths = source_paths(settings)
            before = fingerprint_files(paths)
            template = read_template(settings.source(settings.values["sources"]["template"]))
            cohort = template.user_id.tolist()
            expected = settings.values["evaluation"].get("expected_cohort_size", len(cohort))
            if len(cohort) != expected:
                raise ValueError(f"Expected {expected} template users, found {len(cohort)}")
        with resources.phase("preparation_total"):
            prepared, diagnostics, cache_hit = prepare(settings, before, resources)
        if command == "prepare":
            after = fingerprint_files(paths)
            if before != after:
                raise RuntimeError("An original source changed during preparation")
            summary = {"preparation_path": str(prepared), "preparation_cache_hit": cache_hit, "original_inputs_unchanged": True, "resources": resources.report()}
            atomic_json(settings.artifacts_dir / "latest_preparation.json", summary)
            return summary
        train_window, val_window, support_window = fold_windows(settings, fold_name, diagnostics)
        windows = [train_window, val_window] + ([support_window] if support_window.name != "train" else [])
        with resources.phase("aggregate_temporal_interactions"):
            aggregated = aggregate_windows(settings, prepared, diagnostics["preparation_key"], windows)
        train, validation, support = (aggregated[name]["frame"] for name in ["train", "validation", support_window.name])
        with resources.phase("independent_candidate_catalogues_and_activity"):
            policy = settings.values["folds"][fold_name]["candidate_policy"]
            candidates = candidate_ids(prepared, policy, val_window)
            final_candidates = candidate_ids(prepared, settings.values["final_candidate_policy"])
            activity = viewing_activity(prepared, [train_window, val_window], cohort, settings.values["preparation"]["parquet_batch_rows"])
        with resources.phase("fit_popularity_and_evaluate"):
            model = Popularity(settings.values["popularity"]["support_min_seconds"]).fit(support, candidates)
            recommendations = model.recommend(cohort, k=5)
            truth = relevance_sets(validation, candidates)
            scores, per_user = evaluate(cohort, recommendations, truth, candidates)
        with resources.phase("validation_examples_as_of_training_cutoff"):
            examples = validation_examples(settings, prepared, cohort, train, validation, recommendations, train_window.end)
        identity = {"config": settings.values, "preparation_key": diagnostics["preparation_key"], "code_hash": code_hash(), "fold": fold_name, "model": "popularity"}
        run_key = stable_hash(identity)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = settings.artifacts_dir / "runs" / f"stage1-{fold_name}-{run_key[:12]}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        cohort_report = cohort_diagnostics(cohort, train, validation, candidates, activity)
        metrics = {
            "model": "popularity", "fold": fold_name, **scores,
            "training": interaction_stats(train), "validation": interaction_stats(validation),
            "training_cohort": interaction_stats(train, cohort), "validation_cohort": interaction_stats(validation, cohort),
            "cohort_diagnostics": cohort_report,
            "candidate_policy": policy, "candidate_count": len(candidates), "final_candidate_count": len(final_candidates),
            "relevance_seconds": settings.values["interactions"]["relevance_seconds"],
            "training_window": train_window.identity(), "validation_window": val_window.identity(),
            "popularity_support_window": support_window.identity(),
            "popularity_support_rule": "Number of distinct users with >0 credited seconds; if support_min_seconds >0, additionally require accumulated seconds >= that parameter",
            "popularity_options": settings.values["popularity"],
            "repeat_policy": "Previously viewed programmes are allowed; IDs within a recommendation must be distinct",
            "label_status": "60-second threshold is an explicit configurable validation proxy; official hidden-label construction is unverified",
        }
        with resources.phase("write_and_verify_run_artifacts"):
            atomic_json(run_dir / "config.json", settings.values)
            atomic_json(run_dir / "input_fingerprints.json", before)
            atomic_json(run_dir / "metrics.json", metrics)
            atomic_json(run_dir / "diagnostics.json", {**diagnostics, "cohort": cohort_report, "activity": activity, "audit_comparison": compare_audit(before, settings, diagnostics["raw"])})
            atomic_json(run_dir / "candidate_ids.json", {"validation": candidates, "final_catalogue_only_no_recommendations": final_candidates})
            atomic_json(run_dir / "validation_examples.json", examples)
            prediction_table = pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in recommendations], schema=PREDICTION_SCHEMA)
            atomic_parquet(run_dir / "validation_recommendations.parquet", prediction_table)
            atomic_parquet(run_dir / "validation_user_metrics.parquet", pa.Table.from_pandas(per_user, preserve_index=False))
            after = fingerprint_files(paths)
            if before != after:
                raise RuntimeError("An original data file changed during execution")
            checks = {**scores["recommendation_contract"], "original_input_hashes_and_mtimes_unchanged": True, "future_viewing_used": False, "training_end_exclusive_utc": utc_iso(train_window.end), "kaggle_submission_written": False}
            atomic_json(run_dir / "checks.json", checks)
        test_path = settings.artifacts_dir / "tests" / "unit_tests.json"
        test_evidence = None
        if test_path.is_file():
            result = json.loads(test_path.read_text(encoding="utf-8"))
            test_evidence = {"path": str(test_path), "sha256": file_hash(test_path), "code_hash_matches": result.get("application_code_hash") == code_hash(), "tests_run": result.get("tests_run"), "failures": result.get("failures"), "errors": result.get("errors")}
        manifest = {
            "status": "complete", "run_key": run_key, "identity": identity, "created_utc": stamp,
            "preparation_path": str(prepared), "preparation_cache_hit": cache_hit,
            "windows": {name: {"path": str(value["path"]), "cache_hit": value["cache_hit"]} for name, value in aggregated.items()},
            "environment": environment(), "resources": resources.report(), "test_evidence": test_evidence,
            "outputs": {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in sorted(run_dir.iterdir()) if path.is_file()},
        }
        atomic_json(run_dir / "run_manifest.json", manifest)
        summary = {"run_dir": str(run_dir), "preparation_path": str(prepared), "preparation_cache_hit": cache_hit, "cohort_users": len(cohort), "evaluable_users": scores["evaluable_users"], "map_at_5_fixed": scores["map_at_5_fixed"], "map_at_5_evaluable": scores["map_at_5_evaluable"], "runtime_seconds": manifest["resources"]["elapsed_seconds"], "os_peak_working_set_bytes": manifest["resources"]["os_peak_working_set_bytes"]}
        atomic_json(settings.artifacts_dir / "latest_run.json", summary)
        LOGGER.info("MAP@5 fixed=%.12f; evaluable=%.12f (%d/%d users)", scores["map_at_5_fixed"], scores["map_at_5_evaluable"], scores["evaluable_users"], len(cohort))
        return summary


def verify_run(settings, run_dir):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Run is incomplete")
    for name, expected in manifest["outputs"].items():
        path = run_dir / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Run output integrity failure: {name}")
    original = json.loads((run_dir / "input_fingerprints.json").read_text(encoding="utf-8"))
    if fingerprint_files(source_paths(settings)) != original:
        raise ValueError("Original inputs changed after this run")
    cohort = read_template(settings.source(settings.values["sources"]["template"])).user_id.tolist()
    predictions = pq.read_table(run_dir / "validation_recommendations.parquet")
    if predictions.schema.field("user_id").type != pa.string():
        raise ValueError("Prediction user IDs are not stored as strings")
    rows = [Recommendation(row["user_id"], tuple(row["programme_ids"]), tuple(row["scores"])) for row in predictions.to_pylist()]
    candidates = json.loads((run_dir / "candidate_ids.json").read_text(encoding="utf-8"))["validation"]
    validation = pq.read_table(Path(manifest["windows"]["validation"]["path"]) / "interactions.parquet").to_pandas()
    recomputed, per_user = evaluate(cohort, rows, relevance_sets(validation, candidates), candidates)
    recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in ["cohort_users", "evaluable_users", "zero_eligible_positive_users", "map_at_5_fixed", "map_at_5_evaluable", "ap_sum"]:
        if recomputed[key] != recorded[key]:
            raise ValueError(f"Recomputed metric differs: {key}")
    stored = pq.read_table(run_dir / "validation_user_metrics.parquet").to_pandas()
    pd.testing.assert_frame_equal(stored, per_user)
    result = {"status": "passed", **recomputed["recommendation_contract"], "metric_recomputation_exact": True, "source_integrity_verified": True, "output_integrity_verified": True}
    atomic_json(run_dir / "verification.json", result)
    return result
