"""A shared ranking/evaluation boundary, plus strict in-memory CSV formatting."""

import csv
import io
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Recommendation:
    user_id: str
    programme_ids: tuple[int, ...]
    scores: tuple[float, ...]


class Recommender(Protocol):
    def recommend(self, user_ids: Sequence[str], k: int = 5) -> list[Recommendation]: ...


def validate_recommendations(cohort, recommendations, candidates, k=5):
    if not all(isinstance(user, str) and user.isdigit() for user in cohort):
        raise ValueError("External user IDs must remain exact digit strings")
    if len(set(cohort)) != len(cohort):
        raise ValueError("The evaluation cohort must contain unique users")
    if [row.user_id for row in recommendations] != list(cohort):
        raise ValueError("Recommendations must preserve the exact template user order")
    permitted = set(candidates)
    for row in recommendations:
        ids = row.programme_ids
        if len(ids) != k or len(row.scores) != k or len(set(ids)) != k:
            raise ValueError("Each recommendation must contain exactly k distinct IDs and scores")
        if any(not isinstance(item, Integral) or isinstance(item, bool) or item <= 0 for item in ids):
            raise ValueError("Programme IDs must be positive integers")
        if not set(ids).issubset(permitted):
            raise ValueError("Recommendation outside the permitted candidate catalogue")
    return {"users_checked": len(cohort), "recommendations_checked": len(cohort) * k, "k": k, "all_unique_positive_permitted": True, "exact_template_order": True, "external_user_ids_are_strings": True}


def format_submission(cohort, recommendations, candidates):
    """Pure formatter tested in memory; Stage 1 never writes a Kaggle CSV."""
    validate_recommendations(cohort, recommendations, candidates, k=5)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["user_id", "tv_show_id"])
    for row in recommendations:
        writer.writerow([row.user_id, " ".join(str(item) for item in row.programme_ids)])
    return stream.getvalue()
