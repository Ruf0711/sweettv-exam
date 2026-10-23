"""Fit the unchanged validated History model and export final predictions."""

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from tvrec.cache import verify_cache
from tvrec.interactions import Window, aggregate_windows, candidate_ids, epoch_column, historical_metadata
from tvrec.io import atomic_json, atomic_parquet, file_hash, read_template
from tvrec.pipeline import PREDICTION_SCHEMA
from tvrec.recommendations import format_submission
from tvrec.resources import Resources
from tvrec.stage2.common import source_code_hash
from tvrec.stage2.completion import completion_windows
from tvrec.stage2.rankers import PersonalHistory, global_scores, recommendations_from_scores
from tvrec.time_utils import utc_iso, zone_fingerprint

from .common import ensure_preservation_snapshot, execution_environment, final_code_hash, verify_preservation, verify_provenance, verify_recorded_outputs
from .verification import read_future_catalogue, verify_csv, verify_regenerated


def write_submission(path, content):
    """Never replace an unfamiliar or different pre-existing root submission."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing submission differs; inspect it before replacing: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def fit_selected(feedback, cumulative, candidates, cohort, cutoff, parameters):
    started = time.perf_counter()
    popularity = global_scores(cumulative, candidates)
    popularity_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = PersonalHistory().fit(feedback, cutoff, candidates, popularity, parameters)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = model.score(cohort)
    rows = recommendations_from_scores(cohort, candidates, scores)
    payload = format_submission(cohort, rows, candidates).encode("utf-8")
    return model, scores, rows, payload, {"global_popularity_fit_seconds": popularity_seconds,
                                        "history_fit_seconds": fit_seconds,
                                        "score_sort_format_seconds": time.perf_counter() - started}


def validate_training(feedback, cumulative, window):
    if feedback.empty or cumulative.empty:
        raise ValueError("The full permitted historical training window is empty")
    if (feedback.tv_show_id <= 0).any() or (feedback.count50 <= 0).any() or (feedback.count80 > feedback.count50).any():
        raise ValueError("Invalid completed-broadcast training features")
    if (feedback.last_epoch > window.end).any() or (feedback.last_epoch <= window.start).any():
        raise ValueError("Training features cross the exclusive permitted cutoff")
    if not np.isfinite(feedback.seconds50).all() or not np.isfinite(cumulative.seconds).all():
        raise ValueError("Non-finite historical training features")
    if feedback.duplicated(["user_id", "tv_show_id"]).any() or cumulative.duplicated(["user_id", "tv_show_id"]).any():
        raise ValueError("Duplicate historical user/programme features")


def git_ignore_checks(root):
    """Use a disposable Git directory; leave this workspace's repository state alone."""
    local = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Temp/opencode"
    cases = {
        "submission_history_v1.csv": False, "README.md": False,
        "data/submission.csv": True, "artifacts/final/manifest.json": True,
        ".kaggle/kaggle.json": True, "kaggle.json": True, ".env": True,
        "cache/example.parquet": True, "large_model.npz": True,
    }
    with tempfile.TemporaryDirectory(dir=local if local.is_dir() else None) as temporary:
        repo = Path(temporary) / "ignore-check"
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True, capture_output=True, text=True)
        results = {}
        for name, expected in cases.items():
            result = subprocess.run(["git", "-c", "core.excludesFile=NUL", "--git-dir=" + str(repo / ".git"),
                                     "--work-tree=" + str(root), "check-ignore", "--no-index", "--stdin", "-z"],
                                    input=name.encode("utf-8") + b"\0", capture_output=True, cwd=root)
            if result.returncode not in [0, 1] or (result.returncode == 0) != expected:
                raise ValueError(f"Unexpected Git ignore policy for {name}: {result.stdout} {result.stderr}")
            results[name] = {"ignored": expected, "git_exit_code": result.returncode}
    return {"status": "passed", "root_submission_not_ignored": True, "cases": results}


def save_examples(context, directory, feedback, cumulative, model, rows, future_metadata):
    users = context.options["example_user_ids"]
    prior = historical_metadata(context.prepared, context.cutoff).set_index("tv_show_id").to_dict("index")
    viewing = pq.read_table(context.completion / "continuous_sessions.parquet", filters=[("user_id", "in", users)]).to_pandas()
    if (epoch_column(viewing.start_time) >= context.cutoff).any() or (epoch_column(viewing.end_time) > context.cutoff).any():
        raise ValueError("Example viewing history crosses the final cutoff")
    atomic_parquet(directory / "example_viewing_sessions.parquet", pa.Table.from_pandas(viewing, preserve_index=False))
    predictions = {row.user_id: row for row in rows}
    examples, histories = [], []
    candidates = set(future_metadata)
    for user in users:
        observations = feedback.loc[feedback.user_id == user]
        completed = {int(row.tv_show_id): row for row in observations.itertuples()}
        weights = {}
        if user in model.user_map:
            personal = model.matrix[model.user_map[user]]
            weights = dict(zip(model.items[personal.indices].tolist(), personal.data.tolist()))
        activity = viewing.loc[viewing.user_id == user]
        summary = {"continuous_sessions": len(activity), "same_channel_seconds": int((epoch_column(activity.end_time) - epoch_column(activity.start_time)).sum()),
                   "programmes_with_completion_ge50": len(observations), "completed_broadcasts_ge50": int(observations.count50.sum()),
                   "completed_broadcasts_gt80": int(observations.count80.sum())}
        history = []
        for row in cumulative.loc[cumulative.user_id == user].itertuples():
            item = int(row.tv_show_id)
            item_metadata = prior.get(item, {})
            observed = completed.get(item)
            history.append({"tv_show_id": item, "title": item_metadata.get("tv_show_title"), "category": item_metadata.get("tv_show_category"),
                            "metadata_origin": "pre_cutoff_broadcast_metadata", "cumulative_credited_seconds": float(row.seconds),
                            "completed_broadcasts_ge50": int(observed.count50) if observed is not None else 0,
                            "completed_broadcasts_gt80": int(observed.count80) if observed is not None else 0,
                            "completion_training_seconds": float(observed.seconds50) if observed is not None else 0.,
                            "last_completed_viewing_utc": utc_iso(int(observed.last_epoch)) if observed is not None else None,
                            "personal_history_weight": float(weights.get(item, 0.)), "available_in_future_catalogue": item in candidates})
        history.sort(key=lambda item: (-item["personal_history_weight"], item["tv_show_id"]))
        prediction = predictions[user]
        examples.append({"prediction_stage": "final_competition_weeks_31_42", "held_out_validation_example": False,
                         "user_id": user, "training_cutoff_exclusive_utc": utc_iso(context.cutoff), "training_activity": summary,
                         "recommendations": [{"tv_show_id": item, "score": float(score), **future_metadata[item],
                                              "metadata_origin": context.options["candidate_schedule"],
                                              "ranking_source": "personal_history" if weights.get(item, 0.) > 0 else "global_popularity_fallback"}
                                             for item, score in zip(prediction.programme_ids, prediction.scores)]})
        histories.append({"user_id": user, "training_cutoff_exclusive_utc": utc_iso(context.cutoff), "training_activity": summary,
                          "history_order": "personal_weight_descending_then_programme_id_ascending", "programme_history": history})
    atomic_json(directory / "examples_final.json", examples)
    atomic_json(directory / "example_histories.json", histories)


def refit_from_disk(context, completion_path, cumulative_path, cohort, candidates):
    feedback = pq.read_table(completion_path / "train_full.parquet").to_pandas()
    cumulative = pq.read_table(cumulative_path / "interactions.parquet").to_pandas()
    return fit_selected(feedback, cumulative, candidates, cohort, context.cutoff, context.options["history"])


def generate(context):
    settings = context.settings
    with Resources() as resources:
        with resources.phase("verify_prior_experiments_and_original_sources"):
            inputs = verify_provenance(context)
            snapshot = ensure_preservation_snapshot(settings.root)
            verify_preservation(settings.root, snapshot)
            test_path = settings.artifacts_dir / "tests/unit_tests.json"
            tests = json.loads(test_path.read_text(encoding="utf-8"))
            if not tests["successful"] or tests["final_source_code_hash"] != final_code_hash():
                raise ValueError("Passing tests for the current final-stage code are required")
            future = settings.source(context.options["candidate_schedule"])
            future_metadata = read_future_catalogue(future)
            candidates = candidate_ids(context.prepared, {"kind": "file", "file": context.options["candidate_schedule"]})
            if candidates != sorted(future_metadata):
                raise ValueError("Cached candidate catalogue differs from the actual future CSV")
            template = settings.source(settings.values["sources"]["template"])
            cohort = read_template(template).user_id.tolist()
            if len(cohort) != context.options["expected_users"]:
                raise ValueError("The final cohort must contain exactly 1260 users")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = settings.artifacts_dir / "runs" / ("history-v1-" + stamp)
        run.mkdir(parents=True, exist_ok=False)
        diagnostics = json.loads((context.prepared / "diagnostics.json").read_text(encoding="utf-8"))
        window = Window("train_full", diagnostics["raw"]["observed_start_epoch"], context.cutoff)
        with resources.phase("aggregate_all_permitted_historical_evidence"):
            completed, completion_path = completion_windows(settings, context.stage2_options, context.completion, [window])
            aggregated = aggregate_windows(settings, context.prepared, context.prepared.name, [window])["train_full"]
            feedback, cumulative = completed["train_full"], aggregated["frame"]
            cumulative_path = aggregated["path"]
            validate_training(feedback, cumulative, window)
        with resources.phase("fit_frozen_history_and_score_future_catalogue"):
            model, scores, rows, payload, timings = fit_selected(feedback, cumulative, candidates, cohort, context.cutoff, context.options["history"])
        with resources.phase("save_root_csv_model_scores_and_final_examples"):
            submission = settings.root / context.options["submission_file"]
            write_submission(submission, payload)
            atomic_json(run / "frozen_config.json", context.options)
            atomic_json(run / "effective_base_config.json", settings.values)
            atomic_json(run / "test_evidence.json", tests)
            atomic_parquet(run / "recommendations.parquet", pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in rows], schema=PREDICTION_SCHEMA))
            np.savez_compressed(run / "candidate_scores.npz", user_ids=np.asarray(cohort, dtype=str), candidate_ids=np.asarray(candidates, dtype=np.int64), scores=scores)
            sparse.save_npz(run / "history_weights.npz", model.matrix)
            atomic_json(run / "model_registry.json", {"user_ids": model.users, "programme_ids": model.items.tolist(), "candidate_ids": candidates})
            save_examples(context, run, feedback, cumulative, model, rows, future_metadata)
        with resources.phase("independent_csv_readback_and_fresh_model_regeneration"):
            verification = verify_csv(submission, template, future, run / "candidate_scores.npz", run / "recommendations.parquet")
            _, regenerated_scores, _, regenerated, regeneration_times = refit_from_disk(context, completion_path, cumulative_path, cohort, candidates)
            reproducibility = verify_regenerated(submission, regenerated, scores, regenerated_scores)
            write_submission(run / "regenerated_submission.csv", regenerated)
            atomic_json(run / "csv_verification.json", verification)
            atomic_json(run / "reproducibility.json", {**reproducibility, "refit_timings": regeneration_times})
            git_policy = git_ignore_checks(settings.root)
            atomic_json(run / "git_ignore_verification.json", git_policy)
        with resources.phase("verify_source_and_prior_evidence_preservation"):
            preservation = verify_preservation(settings.root, snapshot)
            atomic_json(run / "preservation.json", preservation)
        warm = set(feedback.user_id)
        supported = feedback.loc[feedback.tv_show_id.isin(candidates) & feedback.user_id.isin(cohort)].groupby("user_id").tv_show_id.nunique().to_dict()
        cold = [user for user in cohort if user not in warm]
        supported_pairs = set(zip(feedback.user_id, feedback.tv_show_id))
        fallback_positions = sum((row.user_id, item) not in supported_pairs for row in rows for item in row.programme_ids)
        manifest = {
            "status": "complete_and_verified", "stage": 3, "designated_first_submission": True,
            "run_dir": str(run), "created_utc": datetime.now(timezone.utc).isoformat(),
            "submission": {"path": str(submission), "sha256": file_hash(submission), "bytes": submission.stat().st_size,
                           "data_rows": len(cohort), "columns": ["user_id", "tv_show_id"]},
            "training_window": window.identity(), "training_cutoff_local_exclusive": context.options["training_cutoff_local_exclusive"],
            "timezone": zone_fingerprint(settings.zone), "weeks_31_42_viewing_used": False,
            "future_schedule_usage": "Known candidate availability and display metadata only; never fitted viewing evidence.",
            "method": "tvrec.stage2.rankers.PersonalHistory", "configuration": context.options,
            "training_feedback": context.options["training_feedback"],
            "score_policy": "Unchanged validated float32 history weights plus original popularity prior; cold users receive global distinct-viewer counts; equal final scores use ascending programme ID.",
            "candidate_count": len(candidates), "candidate_source": str(future),
            "training": {"completion50_user_programme_pairs": len(feedback), "completed_broadcasts_ge50": int(feedback.count50.sum()),
                         "completed_broadcasts_gt80": int(feedback.count80.sum()), "training_users_with_completion50": len(warm),
                         "cumulative_positive_user_programme_pairs": len(cumulative), "last_completed_observation_utc": utc_iso(int(feedback.last_epoch.max())),
                         "cohort_cold_users": len(cold), "cohort_cold_user_ids": cold,
                         "cohort_users_without_future_history_support": sum(supported.get(user, 0) == 0 for user in cohort),
                         "cohort_users_with_fewer_than_five_future_history_items": sum(supported.get(user, 0) < 5 for user in cohort),
                         "popularity_fallback_recommendation_positions": fallback_positions},
            "inputs": inputs, "stage2_run": str(context.stage2_run),
            "stage2_manifest_sha256": file_hash(context.stage2_run / "manifest.json"),
            "validated_stage1_stage2_code_hash": source_code_hash(), "final_source_code_hash": final_code_hash(),
            "environment": execution_environment(), "config_sha256": file_hash(context.config_path),
            "training_evidence": {"completion_cache": str(context.completion), "stage1_preparation": str(context.prepared),
                                  "completion_window": str(completion_path), "cumulative_window": str(cumulative_path)},
            "tests": {"path": str(test_path), "sha256": file_hash(test_path), "tests_run": tests["tests_run"], "passed": tests["passed"]},
            "verification": verification, "reproducibility": reproducibility, "preservation": preservation,
            "git_ignore_verification": git_policy, "fit_and_prediction_timings": timings, "resources": resources.report(),
            "documentation": {name: file_hash(settings.root / name) for name in ["README.md", ".gitignore"]},
            "uploads_performed": False, "kaggle_score_claimed": False, "final_pdf_generated": False,
            "outputs": {str(path.relative_to(run)): {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(run.rglob("*")) if path.is_file()},
        }
        atomic_json(run / "manifest.json", manifest)
        atomic_json(settings.artifacts_dir / "manifest.json", manifest)
    return {"status": manifest["status"], "submission": manifest["submission"], "candidate_count": len(candidates),
            "training": manifest["training"], "manifest": str(settings.artifacts_dir / "manifest.json"),
            "resources": manifest["resources"], "verification": verification, "reproducibility": reproducibility,
            "preservation": preservation}


def verify_final(context):
    with Resources() as resources:
        manifest = json.loads((context.settings.artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["final_source_code_hash"] != final_code_hash() or manifest["config_sha256"] != file_hash(context.config_path):
            raise ValueError("Final code or configuration differs from the saved execution")
        verify_provenance(context)
        run = Path(manifest["run_dir"])
        if json.loads((run / "manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Root and archived final manifests disagree")
        verify_recorded_outputs(run, manifest["outputs"])
        for name, expected in manifest["documentation"].items():
            if file_hash(context.settings.root / name) != expected:
                raise ValueError("Final documentation changed since generation")
        evidence = manifest["training_evidence"]
        completion_path, cumulative_path = Path(evidence["completion_window"]), Path(evidence["cumulative_window"])
        for path in [completion_path, cumulative_path]:
            cached = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            verify_cache(path, cached["identity"])
        completion_identity = json.loads((completion_path / "manifest.json").read_text(encoding="utf-8"))["identity"]
        cumulative_identity = json.loads((cumulative_path / "manifest.json").read_text(encoding="utf-8"))["identity"]
        if (completion_identity["windows"]["train_full"] != manifest["training_window"] or
                cumulative_identity["window"] != manifest["training_window"] or
                manifest["training_window"]["end_exclusive_epoch"] != context.cutoff):
            raise ValueError("Saved training windows do not match the authorized final cutoff")
        feedback = pq.read_table(completion_path / "train_full.parquet").to_pandas()
        cumulative = pq.read_table(cumulative_path / "interactions.parquet").to_pandas()
        window = Window("train_full", manifest["training_window"]["start_epoch"], context.cutoff)
        validate_training(feedback, cumulative, window)
        settings = context.settings
        submission = settings.root / context.options["submission_file"]
        template = settings.source(settings.values["sources"]["template"])
        future = settings.source(context.options["candidate_schedule"])
        verification = verify_csv(submission, template, future, run / "candidate_scores.npz", run / "recommendations.parquet")
        if verification["submission_sha256"] != manifest["submission"]["sha256"]:
            raise ValueError("Root submission SHA-256 differs from its final manifest")
        cohort = read_template(template).user_id.tolist()
        candidates = sorted(read_future_catalogue(future))
        _, scores, _, payload, timings = fit_selected(feedback, cumulative, candidates, cohort, context.cutoff, context.options["history"])
        with np.load(run / "candidate_scores.npz", allow_pickle=False) as original:
            reproducibility = verify_regenerated(submission, payload, original["scores"], scores)
        if (run / "regenerated_submission.csv").read_bytes() != payload:
            raise ValueError("Archived regenerated CSV differs from fresh regeneration")
        preservation = verify_preservation(settings.root, settings.artifacts_dir / "preservation_before.json")
        result = {"status": "passed", "independent_disk_csv_checks": verification, "fresh_process_regeneration": reproducibility,
                  "refit_timings": timings, "prior_experiments_and_all_recorded_outputs_integrity": "passed",
                  "preservation": preservation, "git_ignore_verification": git_ignore_checks(settings.root),
                  "resources": resources.report(), "manifest_sha256": file_hash(settings.artifacts_dir / "manifest.json")}
        atomic_json(settings.artifacts_dir / "verification.json", result)
    return result
