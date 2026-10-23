"""Compact fixed-budget models, chronological admission and deterministic ranks."""

import json
import time

import numpy as np
import pandas as pd

from tvrec.io import atomic_json
from .common import admissible_training, assert_training_boundary, lightgbm
from .datasets import load_dataset
from .features import feature_columns


def binary_weights(labels, groups):
    weights = np.empty(len(labels), dtype=np.float32)
    offset = 0
    for size in groups:
        part = labels[offset:offset + size]
        positives = int(part.sum())
        boost = min(20., (size - positives) / max(1, positives)) if positives else 1.
        raw = np.where(part > 0, boost, 1.)
        weights[offset:offset + size] = raw * size / raw.sum()
        offset += size
    if offset != len(labels):
        raise ValueError("User/query groups do not partition the training rows")
    return weights


def training_data(specs, paths, cutoff, family):
    chosen = admissible_training(specs, cutoff)
    frames, labels, groups = [], [], []
    columns = None
    for spec in chosen:
        assert_training_boundary(spec, cutoff)
        data = load_dataset(paths[spec["name"]])
        meta = data["meta"]
        if meta["specification"]["end"] != spec["end"] or meta["maximum_clipped_past_end_epoch"] > spec["cutoff"]:
            raise ValueError("Dataset provenance violates temporal boundaries")
        x, y = data["features"], data["labels"]
        if not x[["user_id", "tv_show_id"]].equals(y[["user_id", "tv_show_id"]]):
            raise ValueError("Feature and binary-label keys differ")
        names = feature_columns(x, family)
        if columns is not None and columns != names:
            raise ValueError("Forecast feature schemas differ")
        columns = names
        group = x.groupby("user_index", sort=False).size().to_numpy(dtype=np.int32)
        expected = np.repeat(np.arange(len(group)), group)
        if not np.array_equal(x.user_index.to_numpy(), expected):
            raise ValueError("LambdaRank requires contiguous complete user groups")
        groups.extend(group.tolist())
        frames.append(x[columns].to_numpy(dtype=np.float32))
        labels.append(y.label.to_numpy(dtype=np.int32))
    return np.concatenate(frames), np.concatenate(labels), np.asarray(groups, dtype=np.int32), columns, chosen


def fit_model(name, specs, paths, cutoff, options, directory):
    started = time.perf_counter()
    family = name.split("_", 1)[1]
    x, y, groups, columns, chosen = training_data(specs, paths, cutoff, family)
    preparation_seconds = time.perf_counter() - started
    lgb = lightgbm()
    ranking = name.startswith("lambda")
    params = {**options["lightgbm"], "objective": "lambdarank" if ranking else "binary",
              "metric": "map" if ranking else "binary_logloss", "verbosity": -1,
              "seed": options["seed"], "data_random_seed": options["seed"], "feature_fraction_seed": options["seed"],
              "bagging_seed": options["seed"], "num_threads": options["threads"],
              "deterministic": True, "force_col_wise": True}
    if ranking:
        params.update({"eval_at": [5], "lambdarank_truncation_level": 10, "lambdarank_norm": True, "label_gain": [0, 1]})
    weights = np.ones(len(y), dtype=np.float32) if ranking else binary_weights(y, groups)
    dataset = lgb.Dataset(x, label=y, weight=weights, group=groups if ranking else None, feature_name=columns,
                          categorical_feature=[column for column in ["category_id", "channel_id"] if column in columns], free_raw_data=True)
    training_started = time.perf_counter()
    model = lgb.train(params, dataset, num_boost_round=options["boosting_rounds"])
    train_seconds = time.perf_counter() - training_started
    directory.mkdir(parents=True, exist_ok=False)
    model.save_model(str(directory / "model.txt"))
    importance = sorted([{"feature": feature, "gain": float(gain), "splits": int(splits)} for feature, gain, splits in zip(columns, model.feature_importance("gain"), model.feature_importance("split"))], key=lambda row: (-row["gain"], row["feature"]))
    report = {"name": name, "feature_family": family, "feature_names": columns, "feature_count": len(columns),
              "training_rows": len(y), "positives": int(y.sum()), "query_groups": len(groups),
              "training_forecasts": chosen, "deployment_cutoff": cutoff, "latest_label_end_exclusive": max(spec["end"] for spec in chosen),
              "all_label_windows_end_no_later_than_cutoff": True, "no_random_row_split": True,
              "user_class_weighting": "Ranking: equal query rows and LightGBM query normalization. Binary: equal per-query total weight; positive multiplier capped at 20.",
              "parameters": params, "boosting_rounds": options["boosting_rounds"], "lightgbm_version": lgb.__version__,
              "data_loading_seconds": preparation_seconds, "train_seconds": train_seconds, "elapsed_seconds": time.perf_counter() - started,
              "feature_importance": importance}
    atomic_json(directory / "training.json", report)
    return model, report


def rerank_scores(frame, raw, archive, blend):
    candidates = archive["candidate_ids"]
    history = archive["history"]
    scores = np.full(history.shape, -1000., dtype=np.float64)
    if len(raw) != len(frame) or not np.isfinite(raw).all():
        raise ValueError("Nonfinite or mismatched reranker predictions")
    for user, part in frame.groupby("user_index", sort=False):
        ix = part.index.to_numpy()
        ids = part.tv_show_id.to_numpy(dtype=np.int64)
        history_rank = part.hist_rank.to_numpy()
        order = np.lexsort((ids, history_rank, -raw[ix]))
        ordinal = np.empty(len(part), dtype=np.float64)
        ordinal[order] = np.arange(len(part))
        historical = np.empty(len(part), dtype=np.float64)
        historical[np.argsort(history_rank, kind="stable")] = np.arange(len(part))
        # Rank blending is scale-free for both binary margins and LambdaRank.
        value = (1 - blend) * (1 - ordinal / (len(part) - 1)) + blend * (1 - historical / (len(part) - 1))
        value += (1 - historical / (len(part) - 1)) * 1e-10
        scores[int(user), np.searchsorted(candidates, ids)] = value
    scores[archive["cold"]] = history[archive["cold"]]
    return scores
