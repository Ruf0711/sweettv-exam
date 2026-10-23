"""Disk-first rational metric checks, temporal audits and fresh-model replay."""

import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tvrec.io import atomic_json, file_hash, read_template
from tvrec.recommendations import format_submission
from tvrec.stage2.rankers import recommendations_from_scores
from tvrec_final.common import verify_recorded_outputs
from tvrec_final.verification import verify_csv, verify_regenerated
from tvrec_final.workflow import write_submission
from .common import admissible_training, assert_training_boundary, code_hash, forecasts, lightgbm, load_options, preservation
from .datasets import load_dataset, truth_at
from .models import fit_model, rerank_scores
from .retrieval import candidate_metrics, retrieve
from .workflow import read_archive


def rational_ap(items, relevant):
    hits, value = 0, Fraction()
    for rank, item in enumerate(items, 1):
        if item in relevant:
            hits += 1
            value += Fraction(hits, rank)
    return value / min(5, len(relevant)) if relevant else Fraction()


def verify(config):
    started = time.perf_counter()
    context, options = load_options(config)
    destination = context.settings.root / options["artifacts_dir"]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest["stage4_code_hash"] != code_hash() or manifest["config_sha256"] != file_hash(Path(config)):
        raise ValueError("Stage 4 code or configuration changed")
    run = Path(manifest["run_dir"])
    verify_recorded_outputs(run, manifest["outputs"])
    cohort = read_template(context.settings.source(context.settings.values["sources"]["template"])).user_id.tolist()
    paths = {name: Path(path) for name, path in manifest["datasets"].items()}
    pseudo_names = {spec["name"] for spec in options["pseudo_forecasts"]}
    pseudo_specs = [spec for spec in forecasts(context, options) if spec["name"] in pseudo_names]
    dataset_checks, ap_count, outputs_checked, temporal_models = [], 0, 0, 0
    for name, path in paths.items():
        data = load_dataset(path)
        archive = read_archive(path)
        meta, frame, labels = data["meta"], data["features"], data["labels"]
        spec = meta["specification"]
        if meta["maximum_clipped_past_end_epoch"] > spec["cutoff"] or archive["user_ids"].tolist() != cohort:
            raise ValueError("Feature timestamps or exact string users violate the contract")
        if not frame[["user_id", "tv_show_id"]].equals(labels[["user_id", "tv_show_id"]]) or frame.duplicated(["user_id", "tv_show_id"]).any():
            raise ValueError("Invalid or mismatched candidate/label keys")
        recovered = retrieve(cohort, archive["candidate_ids"], {key: archive[key] for key in options["retrieval_quotas"]}, options["retrieval_quotas"])
        if not frame[recovered.columns].equals(recovered.astype({key: frame[key].dtype for key in recovered.columns})):
            raise ValueError("Candidate features are not the label-blind retriever output")
        truth = truth_at(path)
        expected = np.fromiter((int(item in truth.get(user, set())) for user, item in zip(frame.user_id, frame.tv_show_id)), dtype=np.int8, count=len(frame))
        if not np.array_equal(labels.label.to_numpy(), expected):
            raise ValueError("Training labels were injected or disagree with strict-80 truth")
        recovered_metrics = candidate_metrics(recovered, cohort, truth)
        if recovered_metrics != meta["candidate_metrics"]:
            raise ValueError("Candidate recall/oracle differs on disk recomputation")
        dataset_checks.append({"forecast": name, "candidate_rows": len(frame), "positive_labels": int(expected.sum()), "features_max_end_no_later_than_cutoff": True})
    for training_path in run.rglob("training.json"):
        report = json.loads(training_path.read_text(encoding="utf-8"))
        for spec in report["training_forecasts"]:
            assert_training_boundary(spec, report["deployment_cutoff"])
        expected_names = [spec["name"] for spec in admissible_training(pseudo_specs, report["deployment_cutoff"])]
        if [spec["name"] for spec in report["training_forecasts"]] != expected_names:
            raise ValueError("Model training datasets differ from chronological admission")
        if any(name in report["feature_names"] for name in ["label", "tv_show_id", "user_id", "user_index", "label_end"]):
            raise ValueError("Labels, raw identities or label times reached model features")
        temporal_models += 1
    for fold in ["inner", "main", "recent_b"]:
        data = load_dataset(paths[fold], verify=False)
        archive = read_archive(paths[fold])
        truth = truth_at(paths[fold])
        allowed = set(archive["candidate_ids"].tolist())
        for prediction_path in (run / fold).rglob("recommendations.parquet"):
            rows = pq.read_table(prediction_path).to_pylist()
            if [row["user_id"] for row in rows] != cohort:
                raise ValueError("Prediction users/order differ from template")
            saved = pq.read_table(prediction_path.parent / "per_user_metrics.parquet").to_pandas().set_index("user_id")
            metrics = json.loads((prediction_path.parent / "metrics.json").read_text(encoding="utf-8"))
            total = Fraction()
            for row in rows:
                items = row["programme_ids"]
                if len(items) != 5 or len(set(items)) != 5 or not set(items).issubset(allowed) or any(item <= 0 for item in items) or not all(math.isfinite(score) for score in row["scores"]):
                    raise ValueError("Invalid Top-5 prediction")
                value = rational_ap(items, truth.get(row["user_id"], set()))
                if not math.isclose(float(value), saved.loc[row["user_id"], "ap_at_5"], rel_tol=0, abs_tol=2e-16):
                    raise ValueError("Independent rational AP differs")
                total += value
                ap_count += 1
            if not math.isclose(float(total / len(cohort)), metrics["map_at_5_fixed"], rel_tol=0, abs_tol=2e-16):
                raise ValueError("Independent rational MAP differs")
            if (prediction_path.parent / "raw_predictions.npy").exists():
                model_root = prediction_path.parent.parent / "model"
                report = json.loads((model_root / "training.json").read_text(encoding="utf-8"))
                model = lightgbm().Booster(model_file=str(model_root / "model.txt"))
                raw = model.predict(data["features"][report["feature_names"]].to_numpy(dtype=np.float32), num_threads=1)
                if not np.array_equal(raw, np.load(prediction_path.parent / "raw_predictions.npy", allow_pickle=False)):
                    raise ValueError("Fresh-process model prediction differs")
                blend = float(prediction_path.parent.name.removeprefix("blend-")) if fold == "inner" else manifest["selection"]["selected"]["history_blend"]
                scores = rerank_scores(data["features"], raw, archive, blend)
                regenerated = recommendations_from_scores(cohort, archive["candidate_ids"], scores)
                if [{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in regenerated] != rows:
                    raise ValueError("Fresh-process deterministic ranking differs")
            outputs_checked += 1
    final_check = None
    if manifest["accepted"]:
        final = manifest["final"]
        data = load_dataset(Path(final["deployment_dataset"]))
        archive = read_archive(final["deployment_dataset"])
        selected = manifest["selection"]["selected"]
        model, report = fit_model(selected["model"], pseudo_specs, paths, context.cutoff, options, destination / "verification/fresh_final_model")
        raw = model.predict(data["features"][report["feature_names"]].to_numpy(dtype=np.float32), num_threads=1)
        scores = rerank_scores(data["features"], raw, archive, selected["history_blend"])
        rows = recommendations_from_scores(cohort, archive["candidate_ids"], scores)
        payload = format_submission(cohort, rows, archive["candidate_ids"]).encode("utf-8")
        submission = Path(final["path"])
        with np.load(run / "final/candidate_scores.npz", allow_pickle=False) as saved:
            repeated = verify_regenerated(submission, payload, saved["scores"], scores)
        write_submission(destination / "verification/regenerated_submission.csv", payload)
        final_check = verify_csv(submission, context.settings.source(context.settings.values["sources"]["template"]), context.settings.source(context.options["candidate_schedule"]), run / "final/candidate_scores.npz", run / "final/recommendations.parquet")
        final_check["fresh_process_refit"] = repeated
    result = {"status": "passed", "datasets_checked": dataset_checks, "chronological_model_training_audits": temporal_models,
              "model_fold_predictions_replayed": outputs_checked, "independent_rational_ap_values": ap_count,
              "all_retrievals_reconstructed_without_labels": True, "no_evaluation_positive_injection": True,
              "all_training_label_end_exclusive_epochs_no_later_than_model_cutoffs": True,
              "final_csv": final_check, "preservation": preservation(destination), "elapsed_seconds": time.perf_counter() - started}
    atomic_json(destination / "verification.json", result)
    return result
