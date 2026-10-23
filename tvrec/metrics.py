"""Duplicate-safe AP@k and fixed-cohort macro evaluation for every model."""

import math

import pandas as pd

from .recommendations import validate_recommendations


def ap_at_k(predictions, relevant, k=5):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    seen, hits, terms = set(), 0, []
    for rank, item in enumerate(predictions[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            terms.append(hits / rank)
        seen.add(item)
    return math.fsum(terms) / min(k, len(relevant))


def relevance_sets(interactions, candidates):
    allowed = set(candidates)
    eligible = interactions.loc[interactions.relevant & interactions.tv_show_id.isin(allowed)]
    return eligible.groupby("user_id").tv_show_id.agg(lambda values: set(int(x) for x in values)).to_dict()


def evaluate(cohort, recommendations, truth, candidates, k=5):
    checks = validate_recommendations(cohort, recommendations, candidates, k)
    rows = []
    for prediction in recommendations:
        relevant = truth.get(prediction.user_id, set())
        rows.append({"user_id": prediction.user_id, "relevant_count": len(relevant), "ap_at_5": ap_at_k(prediction.programme_ids, relevant, k)})
    frame = pd.DataFrame(rows, columns=["user_id", "relevant_count", "ap_at_5"])
    total = math.fsum(frame.ap_at_5)
    eligible = int((frame.relevant_count > 0).sum())
    return {
        "k": k, "cohort_users": len(cohort), "evaluable_users": eligible,
        "zero_eligible_positive_users": len(cohort) - eligible,
        "ap_sum": total, "map_at_5_fixed": total / len(cohort) if cohort else 0.0,
        "map_at_5_evaluable": total / eligible if eligible else 0.0,
        "empty_relevance_ap": 0.0, "duplicate_hit_credit": False,
        "recommendation_contract": checks,
    }, frame
