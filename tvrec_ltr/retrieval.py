"""Label-blind, deterministic sequential model-quota retrieval."""

import numpy as np
import pandas as pd


def score_ranks(scores):
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, scores.shape[1] + 1), scores.shape), axis=1)
    return order, ranks


def retrieve(cohort, candidates, scores, quotas):
    """No validation labels or positive-item override are accepted by this API."""
    candidates = np.asarray(candidates, dtype=np.int64)
    total = sum(quotas.values())
    if len(candidates) < total or not np.all(candidates > 0) or not np.all(candidates[1:] > candidates[:-1]):
        raise ValueError("Insufficient or unsorted distinct eligible candidate universe")
    orders, ranks = {}, {}
    for name in quotas:
        if scores[name].shape != (len(cohort), len(candidates)) or not np.isfinite(scores[name]).all():
            raise ValueError("Invalid retrieval scores")
        orders[name], ranks[name] = score_ranks(scores[name])
    rows, columns, sources = [], [], []
    for user_index in range(len(cohort)):
        chosen, origins = {}, {}
        for name, quota in quotas.items():
            count = 0
            for column in orders[name][user_index]:
                column = int(column)
                if column not in chosen:
                    chosen[column] = True
                    origins[column] = name
                    count += 1
                    if count == quota:
                        break
        selected = sorted(chosen)
        if len(selected) != total:
            raise ValueError("Retrieval quota coverage failed")
        rows.extend([user_index] * total)
        columns.extend(selected)
        sources.extend(origins[column] for column in selected)
    users, columns = np.asarray(rows), np.asarray(columns)
    frame = pd.DataFrame({"user_id": np.asarray(cohort)[users], "user_index": users.astype(np.int32), "tv_show_id": candidates[columns]})
    for name in quotas:
        prefix = "hist" if name == "history" else "pop" if name == "popularity" else name
        frame[prefix + "_score"] = scores[name][users, columns]
        frame[prefix + "_rank"] = ranks[name][users, columns]
        frame[prefix + "_reciprocal_rank"] = 1. / ranks[name][users, columns]
        frame[prefix + "_top_quota"] = (ranks[name][users, columns] <= quotas[name]).astype(np.float32)
        frame["source_" + name] = (np.asarray(sources) == name).astype(np.float32)
    return frame


def candidate_metrics(frame, cohort, truth):
    grouped = frame.groupby("user_id").tv_show_id.agg(set).to_dict()
    hits = total = nonempty = 0
    macro = oracle = 0.
    for user in cohort:
        relevant = truth.get(user, set())
        found = len(grouped[user] & relevant)
        hits += found
        total += len(relevant)
        if relevant:
            nonempty += 1
            macro += found / len(relevant)
            oracle += min(5, found) / min(5, len(relevant))
    return {"candidate_count_per_user": int(frame.groupby("user_id").size().min()),
            "users": len(cohort), "relevant_pairs": total, "retrieved_relevant_pairs": hits,
            "micro_candidate_recall": hits / total if total else 0.,
            "macro_candidate_recall_nonempty": macro / nonempty if nonempty else 0.,
            "candidate_oracle_map_at_5_fixed": oracle / len(cohort), "evaluable_users": nonempty}
