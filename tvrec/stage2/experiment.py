"""Inner temporal tuning, frozen outer comparisons, and complete run evidence."""

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.interactions import Window, aggregate_windows, candidate_ids
from tvrec.cache import verify_cache
from tvrec.io import atomic_json, atomic_parquet, file_hash, read_template
from tvrec.metrics import evaluate
from tvrec.pipeline import PREDICTION_SCHEMA
from tvrec.resources import LOGGER, Resources
from tvrec.time_utils import local_epoch
from .common import source_code_hash, versions
from .completion import completion_windows
from .rankers import ContentBased, PersonalHistory, SparseALS, global_scores, metadata_frames, recommendations_from_scores, reciprocal_rank_fusion


def preserve_check(settings):
    snapshot = json.loads((settings.artifacts_dir / "preservation_before.json").read_text(encoding="utf-8"))
    for name, expected in snapshot.items():
        path = settings.root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Preserved Stage 1 file changed: {name}")
    return {"status": "passed", "original_files_checked": len(snapshot), "stage1_code_tests_data_and_artifacts_unchanged": True}


def truths_and_coverage(cumulative, completion, cohort, candidates):
    before = {
        "cumulative_ge60": cumulative.loc[cumulative.seconds >= 60, ["user_id", "tv_show_id"]],
        "completion_ge50": completion.loc[completion.count50 > 0, ["user_id", "tv_show_id"]],
        "completion_gt80": completion.loc[completion.count80 > 0, ["user_id", "tv_show_id"]],
    }
    truths, coverage = {}, {}
    for rule, frame in before.items():
        target = frame.loc[frame.user_id.isin(cohort)]
        included = target.loc[target.tv_show_id.isin(candidates)]
        truths[rule] = included.groupby("user_id").tv_show_id.agg(lambda values: set(int(x) for x in values)).to_dict()
        sizes = [len(truths[rule].get(user, set())) for user in cohort]
        coverage[rule] = {
            "fixed_cohort_users": len(cohort), "evaluable_users": sum(n > 0 for n in sizes),
            "zero_positive_users": sum(n == 0 for n in sizes),
            "all_user_pairs_before_catalogue": len(frame), "cohort_pairs_before_catalogue": len(target),
            "cohort_pairs_after_catalogue": len(included), "cohort_excluded_pairs": len(target) - len(included),
            "candidate_count": len(candidates), "median_relevant_count_including_empty": float(np.median(sizes)),
        }
    return truths, coverage


def build_fold_data(settings, options, old, prepared, completion_path, fold_name, cohort, resources):
    spec = options["folds"][fold_name]
    diagnostics = json.loads((prepared / "diagnostics.json").read_text(encoding="utf-8"))
    cut = local_epoch(spec["cutoff_local"], settings.zone)
    end = local_epoch(spec["validation_end_local"], settings.zone)
    train_window = Window("train", diagnostics["raw"]["observed_start_epoch"], cut)
    val_window = Window("validation", cut, end)
    if end > local_epoch(settings.values["preparation"]["viewing_end_exclusive_local"], settings.zone):
        raise ValueError("A fold crosses into test viewing")
    with resources.phase(f"{fold_name}_window_features_and_labels"):
        completed, completion_window_path = completion_windows(settings, options, completion_path, [train_window, val_window])
        if fold_name == "main":
            for window in [train_window, val_window]:
                path = Path(old["windows"][window.name]["path"])
                cached = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                if cached["identity"]["window"] != window.identity():
                    raise ValueError("Preserved cumulative window boundaries do not match this fold")
                verify_cache(path, cached["identity"])
            cumulative = {name: pq.read_table(Path(old["windows"][name]["path"]) / "interactions.parquet").to_pandas() for name in ["train", "validation"]}
            cumulative_paths = {name: old["windows"][name]["path"] for name in cumulative}
        else:
            aggregates = aggregate_windows(settings, prepared, prepared.name, [train_window, val_window])
            cumulative = {name: value["frame"] for name, value in aggregates.items()}
            cumulative_paths = {name: str(value["path"]) for name, value in aggregates.items()}
        candidates = candidate_ids(prepared, spec["candidate_policy"], val_window)
        truths, coverage = truths_and_coverage(cumulative["validation"], completed["validation"], cohort, candidates)
        prior, upcoming = metadata_frames(prepared, cut, end)
    return {
        "name": fold_name, "train_window": train_window, "validation_window": val_window,
        "candidates": np.asarray(candidates, dtype=np.int64), "truths": truths, "coverage": coverage,
        "feedback": completed["train"], "cumulative": cumulative,
        "popularity": global_scores(cumulative["train"], candidates), "prior": prior, "upcoming": upcoming,
        "completion_window_path": str(completion_window_path), "cumulative_paths": cumulative_paths,
        "candidate_policy": spec["candidate_policy"],
    }


def evaluate_scores(cohort, data, scores):
    rows = recommendations_from_scores(cohort, data["candidates"], scores)
    results, frames = {}, []
    for rule, truth in data["truths"].items():
        result, per_user = evaluate(cohort, rows, truth, data["candidates"])
        results[rule] = result
        per_user["relevance_rule"] = rule
        frames.append(per_user)
    return rows, results, pd.concat(frames, ignore_index=True)


def fit_component(name, cohort, data, options, params=None):
    with Resources() as resources:
        started = time.perf_counter()
        model = None
        if name == "GlobalPopularity":
            fitted_popularity = global_scores(data["cumulative"]["train"], data["candidates"])
        elif name == "PersonalHistory":
            model = PersonalHistory().fit(data["feedback"], data["train_window"].end, data["candidates"], data["popularity"], options["history"])
        elif name == "ALS":
            model = SparseALS().fit(data["feedback"], data["train_window"].end, data["candidates"], data["popularity"], options["history"], params, options["seed"], options["threads"])
        elif name == "ContentBased":
            model = ContentBased().fit(data["feedback"], data["train_window"].end, data["candidates"], data["popularity"], options["history"], options["content"], data["prior"], data["upcoming"])
        else:
            raise ValueError(name)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = np.broadcast_to(fitted_popularity, (len(cohort), len(data["candidates"]))).copy() if model is None else model.score(cohort)
        score_seconds = time.perf_counter() - started
        rows, metrics, per_user = evaluate_scores(cohort, data, scores)
        report = resources.report()
        report.update({"fit_seconds": fit_seconds, "score_seconds": score_seconds, "model": name, "parameters": params or (options["content"] if name == "ContentBased" else options["history"] if name == "PersonalHistory" else {"distinct_viewers": "any_positive_seconds", "lookback": "all_history"})})
        report["warm_training_users"] = int(data["feedback"].user_id.nunique()) if model is not None else int(data["cumulative"]["train"].user_id.nunique())
        report["candidate_items_without_completion_training_support"] = int((~np.isin(data["candidates"], data["feedback"].tv_show_id)).sum())
        if name == "ALS":
            report["training_nnz"] = model.training_nnz
            report["iteration_losses"] = model.losses
        if name == "ContentBased":
            report["sparse_profile_nnz"] = int(model.profiles.nnz)
            report["cold_candidates_with_content"] = model.cold_candidates_with_content
    return {"name": name, "model": model, "scores": scores, "rows": rows, "metrics": metrics, "per_user": per_user, "resources": report}


def display_metadata(data):
    records = {}
    for source, frame in [("training_as_of", data["prior"]), ("known_upcoming_schedule", data["upcoming"])]:
        for row in frame.drop_duplicates("tv_show_id").itertuples():
            records.setdefault(int(row.tv_show_id), {"title": row.tv_show_title, "category": row.tv_show_category, "metadata_origin": source})
    return records


def save_component(directory, component, data, cohort, options):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "metrics.json", component["metrics"])
    atomic_json(directory / "resources.json", component["resources"])
    atomic_parquet(directory / "recommendations.parquet", pa.Table.from_pylist([{"user_id": row.user_id, "programme_ids": list(row.programme_ids), "scores": list(row.scores)} for row in component["rows"]], schema=PREDICTION_SCHEMA))
    atomic_parquet(directory / "per_user_metrics.parquet", pa.Table.from_pandas(component["per_user"], preserve_index=False))
    metadata = display_metadata(data)
    predictions = {row.user_id: row for row in component["rows"]}
    examples = []
    for user in options["example_user_ids"]:
        row = predictions[user]
        history = data["feedback"].loc[data["feedback"].user_id == user]
        examples.append({"user_id": user, "training_programmes_ge50": len(history), "training_completed_broadcasts_ge50": int(history.count50.sum()), "recommendations": [{"tv_show_id": item, "score": score, **metadata.get(item, {"title": None, "category": None, "metadata_origin": "unavailable_in_window"})} for item, score in zip(row.programme_ids, row.scores)]})
    atomic_json(directory / "examples.json", examples)
    if component["name"] == "ALS":
        component["model"].save(directory / "factors.npz")
    if component["name"] == "ContentBased":
        component["model"].save(directory)


def save_fold_context(directory, data):
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "label_diagnostics.json", data["coverage"])
    atomic_json(directory / "context.json", {
        "train_window": data["train_window"].identity(), "validation_window": data["validation_window"].identity(),
        "candidate_ids": data["candidates"].tolist(), "candidate_policy": data["candidate_policy"],
        "completion_window_path": data["completion_window_path"], "cumulative_paths": data["cumulative_paths"],
        "training_completion50_user_programme_pairs": len(data["feedback"]),
        "training_completion50_users": int(data["feedback"].user_id.nunique()),
        "training_completion50_broadcasts": int(data["feedback"].count50.sum()),
    })


def hybrid_trials(cohort, data, options, components):
    primary = options["primary_relevance"]
    history = components["PersonalHistory"]
    trials = []
    for second in ["ALS", "ContentBased"]:
        # Complementarity is evaluated only after the individual components.
        if components[second]["metrics"][primary]["map_at_5_fixed"] <= 0:
            continue
        for weight in options["hybrid"]["history_weights"]:
            params = {"selected_on": "inner_tuning_only", "second": second, "history_weight": weight, "rrf_constant": options["hybrid"]["rrf_constant"]}
            trials.append(fit_hybrid(f"HybridHistory{weight}-{second}", cohort, data, components, params))
    return trials


def fit_hybrid(name, cohort, data, components, params):
    history, second = components["PersonalHistory"], components[params["second"]]
    with Resources() as resources:
        start = time.perf_counter()
        scores = reciprocal_rank_fusion([history["scores"], second["scores"]], [params["history_weight"], 1 - params["history_weight"]], params["rrf_constant"])
        fusion_seconds = time.perf_counter() - start
        rows, metrics, per_user = evaluate_scores(cohort, data, scores)
        report = resources.report()
        report.update({
            "parameters": params, "fusion_seconds": fusion_seconds,
            "fit_seconds": history["resources"]["fit_seconds"] + second["resources"]["fit_seconds"],
            "score_seconds": history["resources"]["score_seconds"] + second["resources"]["score_seconds"] + fusion_seconds,
            "timing_note": "Fit/score sums include the measured upstream components; elapsed_seconds measures fusion and evaluation only. RSS includes the current shared experiment process.",
        })
    return {"name": name, "model": None, "scores": scores, "rows": rows, "metrics": metrics, "per_user": per_user, "resources": report}


def run_experiment(settings, options, old, prepared, completion_path, resources):
    np.random.seed(options["seed"])
    cohort = read_template(settings.source(settings.values["sources"]["template"])).user_id.tolist()
    if len(cohort) != 1260:
        raise ValueError("Stage 2 requires the fixed original 1260-user cohort")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = settings.artifacts_dir / "runs" / ("comparison-" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    atomic_json(run / "config.json", options)
    atomic_json(run / "effective_base_config.json", settings.values)
    atomic_json(run / "source_provenance.json", options["source_evidence"])
    primary = options["primary_relevance"]
    summary, all_resources = {}, {}
    # Parameter selection ends before the MAIN outer training cutoff. Neither
    # outer validation fold is consulted when selecting parameters or a hybrid.
    tuning = build_fold_data(settings, options, old, prepared, completion_path, "tuning", cohort, resources)
    save_fold_context(run / "tuning", tuning)
    components = {}
    for name in ["GlobalPopularity", "PersonalHistory", "ContentBased"]:
        LOGGER.info("Inner tuning component %s", name)
        component = fit_component(name, cohort, tuning, options)
        components[name] = component
        save_component(run / "tuning" / name, component, tuning, cohort, options)
    grid = options["als_grid"]
    best, grid_results = None, []
    for factors, regularization, scale in itertools.product(grid["factors"], grid["regularization"], grid["confidence_scale"]):
        params = {"factors": factors, "regularization": regularization, "confidence_scale": scale, "iterations": grid["iterations"]}
        component = fit_component("ALS", cohort, tuning, options, params)
        value = component["metrics"][primary]["map_at_5_fixed"]
        name = f"ALS-f{factors}-r{regularization}-c{scale}"
        save_component(run / "tuning" / name, component, tuning, cohort, options)
        grid_results.append({"parameters": params, "metrics": component["metrics"], "resources": component["resources"]})
        LOGGER.info("ALS inner parameters=%s MAP@5 %s=%.8f", params, primary, value)
        if best is None or value > best["metrics"][primary]["map_at_5_fixed"]:
            best = component
    components["ALS"] = best
    atomic_json(run / "als_grid.json", grid_results)
    trials = hybrid_trials(cohort, tuning, options, components)
    for trial in trials:
        save_component(run / "tuning" / trial["name"], trial, tuning, cohort, options)
    best_individual = max(components, key=lambda name: components[name]["metrics"][primary]["map_at_5_fixed"])
    best_individual_value = components[best_individual]["metrics"][primary]["map_at_5_fixed"]
    best_hybrid = max(trials, key=lambda t: t["metrics"][primary]["map_at_5_fixed"]) if trials else None
    hybrid_selected = bool(best_hybrid and best_hybrid["metrics"][primary]["map_at_5_fixed"] > best_individual_value + options["hybrid"]["minimum_tuning_gain"])
    selected = {
        "selected_before_outer_evaluation": True, "selection_fold": options["folds"]["tuning"],
        "als_parameters": best["resources"]["parameters"], "best_individual_on_tuning": best_individual,
        "best_individual_primary_map": best_individual_value, "hybrid_selected": hybrid_selected,
        "hybrid_parameters": best_hybrid["resources"]["parameters"] if best_hybrid else None,
        "best_hybrid_primary_map": best_hybrid["metrics"][primary]["map_at_5_fixed"] if best_hybrid else None,
    }
    atomic_json(run / "selection.json", selected)
    del components, tuning, best, trials, best_hybrid
    for fold_name in ["main", "recent_b"]:
        data = build_fold_data(settings, options, old, prepared, completion_path, fold_name, cohort, resources)
        save_fold_context(run / fold_name, data)
        components, summary[fold_name], all_resources[fold_name] = {}, {}, {}
        for name in ["GlobalPopularity", "PersonalHistory", "ALS", "ContentBased"]:
            LOGGER.info("Outer fold %s model %s", fold_name, name)
            component = fit_component(name, cohort, data, options, selected["als_parameters"] if name == "ALS" else None)
            components[name] = component
            save_component(run / fold_name / name, component, data, cohort, options)
            summary[fold_name][name] = component["metrics"]
            all_resources[fold_name][name] = component["resources"]
            LOGGER.info("%s %s primary MAP@5=%.8f", fold_name, name, component["metrics"][primary]["map_at_5_fixed"])
        if fold_name == "main":
            baseline = summary[fold_name]["GlobalPopularity"]["cumulative_ge60"]["map_at_5_fixed"]
            if baseline != 0.7950661375661375:
                raise ValueError("Preserved Stage 1 cumulative popularity score was not reproduced")
        if hybrid_selected:
            params = selected["hybrid_parameters"]
            component = fit_hybrid("Hybrid", cohort, data, components, params)
            save_component(run / fold_name / "Hybrid", component, data, cohort, options)
            summary[fold_name]["Hybrid"] = component["metrics"]
            all_resources[fold_name]["Hybrid"] = component["resources"]
        del components, data
    preservation = preserve_check(settings)
    atomic_json(run / "comparison.json", summary)
    atomic_json(run / "preservation.json", preservation)
    manifest = {
        "status": "complete", "stage": 2, "source_code_hash": source_code_hash(), "versions": versions(),
        "completion_cache": str(completion_path), "stage1_preparation": str(prepared), "selection": selected,
        "resources": resources.report(), "model_resources": all_resources,
        "outputs": {str(p.relative_to(run)): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(run.rglob("*")) if p.is_file()},
    }
    atomic_json(run / "manifest.json", manifest)
    result = {"run_dir": str(run), "comparison_path": str(run / "comparison.json"), "selection": selected, "resources": manifest["resources"], "preservation": preservation}
    atomic_json(settings.artifacts_dir / "latest_comparison.json", result)
    return result


def check_experiment(settings, options):
    latest = json.loads((settings.artifacts_dir / "latest_comparison.json").read_text(encoding="utf-8"))
    run = Path(latest["run_dir"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_code_hash"] != source_code_hash():
        raise ValueError("Application code differs from the recorded Stage 2 run")
    cache_paths = {Path(manifest["completion_cache"]), Path(manifest["stage1_preparation"])}
    for name, expected in manifest["outputs"].items():
        path = run / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Stage 2 output changed: {name}")
    cohort = read_template(settings.source(settings.values["sources"]["template"])).user_id.tolist()
    models_checked = 0
    for fold in ["tuning", "main", "recent_b"]:
        context = json.loads((run / fold / "context.json").read_text(encoding="utf-8"))
        cache_paths.add(Path(context["completion_window_path"]))
        cache_paths.update(Path(path) for path in context["cumulative_paths"].values())
        completion = pq.read_table(Path(context["completion_window_path"]) / "validation.parquet").to_pandas()
        cumulative = pq.read_table(Path(context["cumulative_paths"]["validation"]) / "interactions.parquet").to_pandas()
        truths, _ = truths_and_coverage(cumulative, completion, cohort, context["candidate_ids"])
        for prediction_path in (run / fold).glob("*/recommendations.parquet"):
            from tvrec.recommendations import Recommendation
            rows = [Recommendation(r["user_id"], tuple(r["programme_ids"]), tuple(r["scores"])) for r in pq.read_table(prediction_path).to_pylist()]
            recorded = json.loads((prediction_path.parent / "metrics.json").read_text(encoding="utf-8"))
            parts = []
            for rule, truth in truths.items():
                result, per_user = evaluate(cohort, rows, truth, context["candidate_ids"])
                if result != recorded[rule]:
                    raise ValueError(f"Metric recomputation differs: {prediction_path} {rule}")
                per_user["relevance_rule"] = rule
                parts.append(per_user)
            pd.testing.assert_frame_equal(pd.concat(parts, ignore_index=True), pq.read_table(prediction_path.parent / "per_user_metrics.parquet").to_pandas())
            models_checked += 1
    for path in sorted(cache_paths):
        cached = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        verify_cache(path, cached["identity"])
    result = {"status": "passed", "model_fold_outputs_checked": models_checked, "recommendations_checked": models_checked * len(cohort) * 5, "caches_integrity_checked": len(cache_paths), "all_three_metrics_recomputed_exactly": True, "preservation": preserve_check(settings)}
    atomic_json(settings.artifacts_dir / "verification.json", result)
    return result
