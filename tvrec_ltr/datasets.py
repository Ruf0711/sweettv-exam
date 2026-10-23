"""Immutable forecast datasets: retrieve/features first, labels afterwards."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.interactions import Window, candidate_ids
from tvrec.io import atomic_json, atomic_parquet, file_hash, read_template, stable_hash
from tvrec.pipeline import PREDICTION_SCHEMA
from tvrec.resources import LOGGER
from tvrec.stage2.rankers import ContentBased, PersonalHistory, SparseALS, global_scores, recommendations_from_scores
from tvrec_final.common import verify_recorded_outputs
from .evidence import EvidenceStore
from .features import build_features
from .retrieval import candidate_metrics, retrieve


def dataset_identity(context, options, spec):
    paths = [Path(__file__).with_name(name + ".py") for name in ["datasets", "evidence", "features", "retrieval"]]
    return {"forecast": spec, "feature_code": {path.name: file_hash(path) for path in paths},
            "completion_cache": context.completion.name, "prepared": context.prepared.name,
            "cohort_source_sha256": file_hash(context.settings.source(context.settings.values["sources"]["template"])),
            "retrieval_quotas": options["retrieval_quotas"], "seed": options["seed"], "retrieval_settings": context.stage2_options}


def load_dataset(directory, verify=True):
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if verify:
        verify_recorded_outputs(directory, meta["outputs"])
    return {"path": directory, "meta": meta, "features": pq.read_table(directory / "features.parquet").to_pandas(),
            "labels": pq.read_table(directory / "labels.parquet").to_pandas()}


def truth_at(directory):
    directory = Path(directory)
    truth = pq.read_table(directory / "truth.parquet").to_pandas()
    return truth.groupby("user_id").tv_show_id.agg(set).to_dict()


def components(context, cohort, candidates, history, prior, upcoming, cutoff):
    options = context.stage2_options
    params = context.stage2_manifest["selection"]["als_parameters"]
    popularity = global_scores(history["cumulative"], candidates)
    result, timing = {}, {}
    for name in ["history", "popularity", "als", "cbf"]:
        started = time.perf_counter()
        if name == "history":
            model = PersonalHistory().fit(history["feedback"], cutoff, candidates, popularity, options["history"])
        elif name == "als":
            model = SparseALS().fit(history["feedback"], cutoff, candidates, popularity, options["history"], params, options["seed"], 1)
        elif name == "cbf":
            model = ContentBased().fit(history["feedback"], cutoff, candidates, popularity, options["history"], options["content"], prior, upcoming)
        else:
            model = None
        fit = time.perf_counter() - started
        result[name] = np.broadcast_to(popularity, (len(cohort), len(candidates))).copy() if model is None else model.score(cohort)
        timing[name] = {"fit_seconds": fit, "fit_and_score_seconds": time.perf_counter() - started}
    return result, timing


def prepare_one(context, options, spec, store, cohort, destination):
    directory = destination / "datasets" / spec["name"]
    identity = dataset_identity(context, options, spec)
    if (directory / "manifest.json").exists():
        meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if meta["identity"] != identity:
            raise ValueError(f"Dataset inputs/code differ from its immutable identity: {directory}")
        verify_recorded_outputs(directory, meta["outputs"])
        LOGGER.info("Reusing Stage 4 dataset %s", spec["name"])
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cutoff, end = spec["cutoff"], spec["end"]
    history = store.history(cutoff)
    candidates = candidate_ids(context.prepared, spec["candidate_policy"], Window("forecast", cutoff, end))
    if "recorded_candidates" in spec and candidates != spec["recorded_candidates"]:
        raise ValueError("The existing outer candidate universe changed")
    prior, upcoming = store.metadata_at(cutoff, end)
    scores, timing = components(context, cohort, candidates, history, prior, upcoming, cutoff)
    if spec["name"] in ["main", "recent_b"]:
        old = pq.read_table(context.stage2_run / spec["name"] / "PersonalHistory/recommendations.parquet").to_pylist()
        rows = recommendations_from_scores(cohort, candidates, scores["history"])
        if [{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in rows] != old:
            raise ValueError("Unchanged History v1 no longer reproduces archived outer predictions")
    retrieval_started = time.perf_counter()
    retrieved = retrieve(cohort, candidates, scores, options["retrieval_quotas"])
    retrieval_seconds = time.perf_counter() - retrieval_started
    feature_started = time.perf_counter()
    frame = build_features(retrieved, cohort, history, prior, upcoming, cutoff, end, store.start, context.settings.zone)
    feature_seconds = time.perf_counter() - feature_started
    atomic_parquet(directory / "features.parquet", pa.Table.from_pandas(frame, preserve_index=False))
    warm = set(history["feedback"].user_id)
    np.savez_compressed(directory / "retrieval_scores.npz", user_ids=np.asarray(cohort, dtype=str), candidate_ids=np.asarray(candidates, dtype=np.int64),
                        cold=np.array([user not in warm for user in cohort]), **scores)
    # Labels are only read after the label-blind candidate table has been built.
    truth = store.labels(cutoff, end, candidates) if end <= context.cutoff else {}
    label = frame[ ["user_id", "tv_show_id"] ].copy()
    label["label"] = np.fromiter((int(item in truth.get(user, set())) for user, item in zip(frame.user_id, frame.tv_show_id)), dtype=np.int8, count=len(frame))
    atomic_parquet(directory / "labels.parquet", pa.Table.from_pandas(label, preserve_index=False))
    truth_rows = [(user, int(item)) for user in cohort for item in sorted(truth.get(user, set()))]
    truth_frame = pd.DataFrame(truth_rows, columns=["user_id", "tv_show_id"])
    atomic_parquet(directory / "truth.parquet", pa.Table.from_pandas(truth_frame, preserve_index=False))
    if spec["name"] in ["main", "recent_b", "inner"]:
        original_name = "tuning" if spec["name"] == "inner" else spec["name"]
        old_context = json.loads((context.stage2_run / original_name / "context.json").read_text(encoding="utf-8"))
        validation = pq.read_table(Path(old_context["completion_window_path"]) / "validation.parquet").to_pandas()
        old_truth = validation.loc[(validation.count80 > 0) & validation.user_id.isin(cohort) & validation.tv_show_id.isin(candidates)].groupby("user_id").tv_show_id.agg(set).to_dict()
        if old_truth != truth:
            raise ValueError("The unchanged outer/inner strict-80 labels were not reproduced")
    meta = {"identity": identity, "specification": spec, "cohort_users": len(cohort), "rows": len(frame),
            "candidate_universe_size": len(candidates), "candidate_metrics": candidate_metrics(retrieved, cohort, truth),
            "history_anchor": store.anchor(spec["name"], history), "all_features_use_past_evidence_before": cutoff,
            "maximum_clipped_past_end_epoch": history["observed_max_clipped_epoch"],
            "label_window": {"start_inclusive": cutoff, "end_exclusive": end},
            "labels_constructed_after_retrieval": True, "evaluation_positives_injected": False,
            "component_runtime": timing, "retrieval_seconds": retrieval_seconds, "feature_seconds": feature_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in directory.iterdir() if path.is_file()}}
    atomic_json(directory / "manifest.json", meta)
    LOGGER.info("Prepared %s: %d rows; %.2fs", spec["name"], len(frame), meta["elapsed_seconds"])
    return directory


def prepare(context, options, specs, destination):
    cohort = read_template(context.settings.source(context.settings.values["sources"]["template"])).user_id.tolist()
    store = EvidenceStore(context, cohort)
    outputs = {}
    for spec in sorted(specs, key=lambda row: (row["cutoff"], row["end"], row["name"])):
        outputs[spec["name"]] = str(prepare_one(context, options, spec, store, cohort, destination))
    atomic_json(destination / "datasets.json", outputs)
    return outputs
