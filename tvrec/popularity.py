"""Distinct historical viewers, with deterministic candidate-only ranking."""

from .recommendations import Recommendation


class Popularity:
    def __init__(self, support_min_seconds=0.0):
        if support_min_seconds < 0:
            raise ValueError("support_min_seconds must be nonnegative")
        self.support_min_seconds = support_min_seconds
        self.ranking = None
        self.support = {}

    def fit(self, interactions, candidates):
        pool = sorted({int(item) for item in candidates if int(item) > 0})
        if len(pool) < 5:
            raise ValueError("At least five distinct positive candidates are required")
        supported = interactions.loc[interactions.seconds > 0]
        if self.support_min_seconds > 0:
            supported = supported.loc[supported.seconds >= self.support_min_seconds]
        counts = supported.groupby("tv_show_id").user_id.nunique().to_dict()
        self.support = {item: int(counts.get(item, 0)) for item in pool}
        self.ranking = tuple(sorted(pool, key=lambda item: (-self.support[item], item)))
        return self

    def recommend(self, user_ids, k=5):
        if self.ranking is None:
            raise RuntimeError("Fit the popularity baseline first")
        if k <= 0 or len(self.ranking) < k:
            raise ValueError("Not enough candidates for k recommendations")
        if not all(isinstance(user, str) and user.isdigit() for user in user_ids):
            raise ValueError("User IDs must be exact digit strings")
        ids = self.ranking[:k]
        scores = tuple(float(self.support[item]) for item in ids)
        # The global ranking is also the explicit cold-user fallback.
        return [Recommendation(user, ids, scores) for user in user_ids]
