"""Locked inner selection, outer ablations and conditional deployment refit."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.io import atomic_json, atomic_parquet, file_hash, read_template
from tvrec.metrics import evaluate
from tvrec.pipeline import PREDICTION_SCHEMA
from tvrec.recommendations import format_submission
from tvrec.resources import LOGGER, Resources
from tvrec.stage2.rankers import recommendations_from_scores
from tvrec_final.common import execution_environment, verify_provenance
from tvrec_final.verification import verify_csv, verify_regenerated
from tvrec_final.workflow import write_submission
from .common import admissible_training, code_hash, forecasts, lightgbm, load_options, preservation
from .datasets import load_dataset, prepare, prepare_one, truth_at
from .evidence import EvidenceStore
from .models import fit_model, rerank_scores


def read_archive(path):
    with np.load(Path(path) / "retrieval_scores.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def paired_bootstrap(challenger, baseline, settings):
    if challenger.user_id.tolist() != baseline.user_id.tolist():
        raise ValueError("Paired bootstrap user order differs")
    values = challenger.ap_at_5.to_numpy() - baseline.ap_at_5.to_numpy()
    rng = np.random.default_rng(settings["seed"])
    samples = rng.integers(0, len(values), size=(settings["replicates"], len(values)))
    means = values[samples].mean(axis=1)
    return {"difference": float(values.mean()), "interval95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "replicates": settings["replicates"], "seed": settings["seed"], "note": "Descriptive paired-user bootstrap. The two temporal outer windows overlap."}


def save_evaluation(directory, scores, data, cohort, candidates):
    directory.mkdir(parents=True, exist_ok=False)
    rows = recommendations_from_scores(cohort, candidates, scores)
    result, per_user = evaluate(cohort, rows, truth_at(data["path"]), candidates)
    atomic_json(directory / "metrics.json", result)
    atomic_parquet(directory / "per_user_metrics.parquet", pa.Table.from_pandas(per_user, preserve_index=False))
    atomic_parquet(directory / "recommendations.parquet", pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in rows], schema=PREDICTION_SCHEMA))
    return result, per_user


def assess_model(model, report, data, archive, cohort, blend, directory):
    started = time.perf_counter()
    raw = model.predict(data["features"][report["feature_names"]].to_numpy(dtype=np.float32), num_threads=1)
    scores = rerank_scores(data["features"], raw, archive, blend)
    metrics, per_user = save_evaluation(directory, scores, data, cohort, archive["candidate_ids"])
    np.save(directory / "raw_predictions.npy", raw)
    return metrics, per_user, time.perf_counter() - started


def final_export(context, options, run, pseudo_specs, paths, selected, cohort, destination):
    started = time.perf_counter()
    store = EvidenceStore(context, cohort)
    actual_future = context.settings.source(context.options["candidate_schedule"])
    end = int(store.metadata.loc[store.metadata.source_file == context.options["candidate_schedule"], "end"].max())
    spec = {"name": "deployment", "cutoff": context.cutoff, "end": end,
            "candidate_policy": {"kind": "file", "file": context.options["candidate_schedule"]}}
    dataset_path = prepare_one(context, options, spec, store, cohort, destination)
    del store
    data = load_dataset(dataset_path)
    archive = read_archive(dataset_path)
    model, report = fit_model(selected["model"], pseudo_specs, paths, context.cutoff, options, run / "final/model")
    raw = model.predict(data["features"][report["feature_names"]].to_numpy(dtype=np.float32), num_threads=1)
    scores = rerank_scores(data["features"], raw, archive, selected["history_blend"])
    rows = recommendations_from_scores(cohort, archive["candidate_ids"], scores)
    payload = format_submission(cohort, rows, archive["candidate_ids"]).encode("utf-8")
    submission = context.settings.root / options["submission_file"]
    write_submission(submission, payload)
    final_dir = run / "final"
    np.savez_compressed(final_dir / "candidate_scores.npz", user_ids=np.asarray(cohort, dtype=str), candidate_ids=archive["candidate_ids"], scores=scores)
    atomic_parquet(final_dir / "recommendations.parquet", pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in rows], schema=PREDICTION_SCHEMA))
    checks = verify_csv(submission, context.settings.source(context.settings.values["sources"]["template"]), actual_future, final_dir / "candidate_scores.npz", final_dir / "recommendations.parquet")
    result = {"path": str(submission), "sha256": file_hash(submission), "bytes": submission.stat().st_size,
              "csv_verification": checks, "training": report, "deployment_dataset": str(dataset_path), "selected": selected,
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json(final_dir / "manifest.json", result)
    return result


def run(config):
    context, options = load_options(config)
    destination = context.settings.root / options["artifacts_dir"]
    tests = json.loads((destination / "tests/unit_tests.json").read_text(encoding="utf-8"))
    if not tests["successful"] or tests["stage4_code_hash"] != code_hash():
        raise ValueError("Stage 4 needs passing tests for its current source")
    with Resources() as resources:
        with resources.phase("verify_sources_and_preservation"):
            preservation(destination)
            provenance = verify_provenance(context)
        specifications = forecasts(context, options)
        pseudo_names = {item["name"] for item in options["pseudo_forecasts"]}
        pseudo_specs = [item for item in specifications if item["name"] in pseudo_names]
        inner_spec = next(item for item in specifications if item["name"] == "inner")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = destination / "runs" / ("ltr-" + stamp)
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / "config.json", options)
        atomic_json(run_dir / "external_reference.json", options["external_reference"])
        with resources.phase("chronological_forecast_features_and_retrieval"):
            path_strings = prepare(context, options, specifications, destination)
        paths = {name: Path(path) for name, path in path_strings.items()}
        cohort = read_template(context.settings.source(context.settings.values["sources"]["template"])).user_id.tolist()
        inner = load_dataset(paths["inner"])
        inner_archive = read_archive(paths["inner"])
        inner_trials, selected = [], None
        with resources.phase("inner_only_model_and_blend_selection"):
            for name in options["models"]:
                model, report = fit_model(name, pseudo_specs, paths, inner_spec["cutoff"], options, run_dir / "inner" / name / "model")
                for blend in options["history_rank_blends"]:
                    metrics, _, elapsed = assess_model(model, report, inner, inner_archive, cohort, blend, run_dir / "inner" / name / ("blend-" + str(blend)))
                    trial = {"model": name, "history_blend": blend, "map_at_5": metrics["map_at_5_fixed"], "train_seconds": report["train_seconds"], "predict_and_evaluate_seconds": elapsed}
                    inner_trials.append(trial)
                    if selected is None or trial["map_at_5"] > selected["map_at_5"]:
                        selected = trial
                    LOGGER.info("Inner %s blend=%.2f MAP@5=%.8f", name, blend, trial["map_at_5"])
        selection = {"locked_before_outer_evaluation": True, "selected_using": "Only chronological April 20-May 18 inner labels; training labels end by April 20.",
                     "selected": selected, "inner_trials": inner_trials, "outer_ablation_blend": selected["history_blend"],
                     "acceptance": {"minimum_gain_each_outer": options["minimum_gain_each_outer"], "both_paired_bootstrap_lower_bounds_positive": True}}
        atomic_json(run_dir / "selection.json", selection)
        del inner, inner_archive
        comparison, intervals = {}, {}
        for fold in ["main", "recent_b"]:
            with resources.phase(fold + "_frozen_outer_ablations"):
                data = load_dataset(paths[fold])
                archive = read_archive(paths[fold])
                cutoff = data["meta"]["specification"]["cutoff"]
                base, base_users = save_evaluation(run_dir / fold / "HistoryV1", archive["history"], data, cohort, archive["candidate_ids"])
                comparison[fold] = {"HistoryV1": {"map_at_5": base["map_at_5_fixed"], "difference_vs_v1": 0.}, "candidate_metrics": data["meta"]["candidate_metrics"]}
                intervals[fold] = {}
                for name in options["models"]:
                    model, report = fit_model(name, pseudo_specs, paths, cutoff, options, run_dir / fold / name / "model")
                    metrics, per_user, elapsed = assess_model(model, report, data, archive, cohort, selected["history_blend"], run_dir / fold / name / "evaluation")
                    interval = paired_bootstrap(per_user, base_users, options["bootstrap"])
                    intervals[fold][name] = interval
                    comparison[fold][name] = {"map_at_5": metrics["map_at_5_fixed"], "difference_vs_v1": interval["difference"], "bootstrap95": interval["interval95"],
                                               "train_seconds": report["train_seconds"], "fit_total_seconds": report["elapsed_seconds"], "predict_and_evaluate_seconds": elapsed,
                                               "feature_count": report["feature_count"], "training_rows": report["training_rows"], "training_queries": report["query_groups"],
                                               "latest_training_label_end_exclusive": report["latest_label_end_exclusive"], "history_blend": selected["history_blend"]}
                    LOGGER.info("Outer %s %s MAP@5=%.8f delta=%+.8f", fold, name, metrics["map_at_5_fixed"], interval["difference"])
                del data, archive
        accepted = all(intervals[fold][selected["model"]]["difference"] >= options["minimum_gain_each_outer"] and intervals[fold][selected["model"]]["interval95"][0] > 0 for fold in intervals)
        with resources.phase("conditional_final_refit_and_csv"):
            final = final_export(context, options, run_dir, pseudo_specs, paths, selected, cohort, destination) if accepted else None
        atomic_json(run_dir / "comparison.json", comparison)
        atomic_json(run_dir / "acceptance.json", {"accepted": accepted, "selected": selected, "recommendation": "ranker_v2" if accepted else "History v1; frozen inner winner did not robustly improve both folds", "paired_intervals": intervals})
        preserved = preservation(destination)
        manifest = {"status": "complete", "run_dir": str(run_dir), "stage4_code_hash": code_hash(), "config_sha256": file_hash(Path(config)),
                    "environment": {**execution_environment(), "lightgbm": lightgbm().__version__}, "original_sources": provenance,
                    "datasets": path_strings, "selection": selection, "comparison": comparison, "accepted": accepted, "final": final,
                    "tests": tests, "preservation": preserved, "resources": resources.report(), "kaggle_scores_used_for_selection": False,
                    "external_reference": options["external_reference"], "uploads_performed": False, "report_generated": False,
                    "outputs": {str(path.relative_to(run_dir)): {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in run_dir.rglob("*") if path.is_file()}}
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(destination / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "selected": selected, "comparison": comparison, "accepted": accepted,
            "final_csv": final["path"] if final else None, "resources": manifest["resources"], "preservation": preserved}
