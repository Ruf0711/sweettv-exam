"""History, genuine sparse implicit ALS, and cutoff-safe sparse content profiles."""

import json
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from tvrec.interactions import epoch_column
from tvrec.recommendations import Recommendation, validate_recommendations


def global_scores(interactions, candidates):
    counts = interactions.loc[interactions.seconds > 0].groupby("tv_show_id").user_id.nunique().to_dict()
    return np.asarray([counts.get(int(item), 0) for item in candidates], dtype=np.float32)


def history_weights(frame, cutoff, half_life_days=90., duration_coefficient=.25):
    if np.any(frame.last_epoch.to_numpy() > cutoff):
        raise ValueError("Post-cutoff activity reached model training features")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    age = np.maximum(0, cutoff - frame.last_epoch.to_numpy()) / 86400
    return (np.log1p(frame.count50.to_numpy()) *
            (1 + duration_coefficient * np.log1p(frame.seconds50.to_numpy() / 3600)) *
            np.exp2(-age / half_life_days)).astype(np.float32)


def feedback_matrix(frame, weights):
    users = sorted(frame.user_id.unique().tolist())
    items = sorted(int(item) for item in frame.tv_show_id.unique() if item > 0)
    user_map, item_map = {u: i for i, u in enumerate(users)}, {item: i for i, item in enumerate(items)}
    valid = (frame.tv_show_id > 0).to_numpy() & (weights > 0)
    matrix = sparse.coo_matrix((weights[valid], (frame.loc[valid, "user_id"].map(user_map), frame.loc[valid, "tv_show_id"].map(item_map))), shape=(len(users), len(items)), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, users, np.asarray(items, dtype=np.int64)


def fallback_scores(raw, warm_users, popularity):
    popularity = np.asarray(popularity, dtype=np.float32)
    prior = popularity / max(1., float(popularity.max(initial=0)))
    scores = np.asarray(raw, dtype=np.float32).copy()
    scores += prior[None, :] * 1e-7
    scores[~np.asarray(warm_users, dtype=bool)] = popularity
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite ranking scores")
    return scores


def recommendations_from_scores(cohort, candidates, scores):
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) < 5 or len(np.unique(candidates)) != len(candidates) or np.any(candidates <= 0):
        raise ValueError("At least five distinct positive candidates are required")
    if not np.all(candidates[1:] > candidates[:-1]):
        raise ValueError("Candidate columns must be sorted for deterministic tie-breaking")
    if scores.shape != (len(cohort), len(candidates)) or not np.isfinite(scores).all():
        raise ValueError("Invalid score matrix")
    top = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    rows = [Recommendation(user, tuple(int(item) for item in candidates[top[i]]), tuple(float(value) for value in scores[i, top[i]])) for i, user in enumerate(cohort)]
    validate_recommendations(cohort, rows, candidates)
    return rows


class PersonalHistory:
    def fit(self, feedback, cutoff, candidates, popularity, params):
        self.candidates = np.asarray(candidates)
        self.popularity = popularity
        weight = history_weights(feedback, cutoff, **params)
        self.matrix, self.users, self.items = feedback_matrix(feedback, weight)
        self.user_map = {user: i for i, user in enumerate(self.users)}
        self.item_map = {int(item): i for i, item in enumerate(self.items)}
        return self

    def score(self, cohort):
        raw = np.zeros((len(cohort), len(self.candidates)), dtype=np.float32)
        warm = np.array([user in self.user_map for user in cohort])
        item_dest = np.flatnonzero(np.isin(self.candidates, self.items))
        item_source = [self.item_map[int(self.candidates[j])] for j in item_dest]
        users = np.flatnonzero(warm)
        if len(users) and len(item_dest):
            raw[np.ix_(users, item_dest)] = self.matrix[[self.user_map[cohort[j]] for j in users]][:, item_source].toarray()
        return fallback_scores(raw, warm, self.popularity)


class SparseALS:
    """Hu-Koren-Volinsky implicit objective via compiled CSR ALS, not SVD."""
    def fit(self, feedback, cutoff, candidates, popularity, history_params, params, seed=42, threads=1):
        from implicit.cpu.als import AlternatingLeastSquares
        weight = history_weights(feedback, cutoff, **history_params)
        matrix, self.users, self.items = feedback_matrix(feedback, weight)
        if not matrix.nnz:
            raise ValueError("ALS needs nonempty positive implicit feedback")
        self.candidates, self.popularity = np.asarray(candidates), popularity
        self.user_map = {user: i for i, user in enumerate(self.users)}
        self.item_map = {int(item): i for i, item in enumerate(self.items)}
        confidence = matrix.copy()
        confidence.data = (1 + params["confidence_scale"] * np.log1p(confidence.data)).astype(np.float32)
        self.training_nnz = int(confidence.nnz)
        self.losses = []
        with threadpool_limits(limits=threads):
            self.model = AlternatingLeastSquares(
                factors=params["factors"], regularization=params["regularization"], alpha=1.,
                iterations=params["iterations"], num_threads=threads, random_state=seed,
                dtype=np.float32, use_cg=True, calculate_training_loss=True,
            )
        def callback(iteration, elapsed, loss):
            self.losses.append({"iteration": int(iteration), "seconds": float(elapsed), "loss": float(loss)})
        with threadpool_limits(limits=threads):
            self.model.fit(confidence, show_progress=False, callback=callback)
        return self

    def score(self, cohort):
        raw = np.zeros((len(cohort), len(self.candidates)), dtype=np.float32)
        warm = np.array([user in self.user_map for user in cohort])
        dest = np.flatnonzero(np.isin(self.candidates, self.items))
        users = np.flatnonzero(warm)
        if len(users) and len(dest):
            with threadpool_limits(limits=1):
                raw[np.ix_(users, dest)] = self.model.user_factors[[self.user_map[cohort[j]] for j in users]] @ self.model.item_factors[[self.item_map[int(self.candidates[j])] for j in dest]].T
        return fallback_scores(raw, warm, self.popularity)

    def save(self, path):
        np.savez_compressed(path, user_factors=self.model.user_factors, item_factors=self.model.item_factors, user_ids=np.asarray(self.users, dtype=str), programme_ids=self.items)


def _text(value):
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


def metadata_documents(frame):
    docs = {}
    fields = ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]
    for item, group in frame.groupby("tv_show_id", sort=True):
        titles = list(dict.fromkeys(_text(x) for x in group.tv_show_title if x))[:8]
        title = " ".join(re.sub(r"\b\d+\b", " ", value) for value in titles)
        attrs = {}
        for name in fields:
            values = sorted({_text(x) for x in group[name] if x})
            for value in values[:8]:
                if name == "actors":
                    for actor in value.split(",")[:15]:
                        attrs["actor:" + actor.strip()] = .25
                elif name == "year_of_production":
                    match = re.search(r"\d{4}", value)
                    if match:
                        attrs["decade:" + str(int(match.group()) // 10)] = .25
                else:
                    attrs[name + ":" + value] = 3. if name == "tv_show_category" else (0.5 if name == "director" else 2.)
        channels = sorted(int(x) for x in group.channel_id.unique())
        for channel in channels:
            attrs["channel:" + str(channel)] = 1. / max(1, len(channels))
        docs[int(item)] = {"title": title, "attributes": attrs}
    return docs


def metadata_frames(prepared, cutoff, end):
    frame = pq_read_metadata(prepared)
    start = epoch_column(frame.start_time)
    stop = epoch_column(frame.end_time)
    prior = frame.loc[(start < cutoff) & (stop > start)].sort_values(["start_time", "tv_show_id"], kind="mergesort")
    upcoming = frame.loc[(start < end) & (stop > cutoff) & (stop > start)].sort_values(["start_time", "tv_show_id"], kind="mergesort")
    return prior, upcoming


def pq_read_metadata(prepared):
    import pyarrow.parquet as pq
    return pq.read_table(prepared / "broadcast_metadata.parquet").to_pandas()


def _prune_rows(matrix, keep=1024):
    matrix = matrix.tocsr()
    data, indices, pointers = [], [], [0]
    for row in range(matrix.shape[0]):
        low, high = matrix.indptr[row:row + 2]
        values, columns = matrix.data[low:high], matrix.indices[low:high]
        if len(values) > keep:
            chosen = np.argpartition(values, -keep)[-keep:]
            values, columns = values[chosen], columns[chosen]
        data.extend(values)
        indices.extend(columns)
        pointers.append(len(data))
    result = sparse.csr_matrix((np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(pointers, dtype=np.int64)), shape=matrix.shape)
    result.sort_indices()
    return result


class ContentBased:
    def fit(self, feedback, cutoff, candidates, popularity, history_params, params, prior, upcoming):
        if len(prior) and np.any(epoch_column(prior.start_time) >= cutoff):
            raise ValueError("Future metadata reached the fitted content corpus")
        self.prior_docs = metadata_documents(prior)
        self.candidate_docs = dict(self.prior_docs)
        for item, doc in metadata_documents(upcoming).items():
            if item not in self.candidate_docs:
                self.candidate_docs[item] = doc
        self.params = params
        self.candidates, self.popularity = np.asarray(candidates), popularity
        corpus = [self.prior_docs[item] for item in sorted(self.prior_docs)]
        if not corpus:
            raise ValueError("Content model needs pre-cutoff metadata")
        self.title = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2 if len(corpus) > 5 else 1, max_features=params["title_max_features"], sublinear_tf=True, dtype=np.float32)
        try:
            self.title.fit([doc["title"] for doc in corpus])
        except ValueError:
            self.title = None
        self.attributes = DictVectorizer(sparse=True, dtype=np.float32).fit([doc["attributes"] for doc in corpus])
        weights = history_weights(feedback, cutoff, **history_params)
        matrix, self.users, self.items = feedback_matrix(feedback, weights)
        self.user_map = {user: i for i, user in enumerate(self.users)}
        training_features = self.transform(self.items, self.prior_docs)
        profiles = matrix @ training_features
        self.profiles = normalize(_prune_rows(profiles), norm="l2", copy=False)
        self.candidate_features = self.transform(self.candidates, self.candidate_docs)
        self.fitted_programme_count = len(self.prior_docs)
        self.cold_candidates_with_content = int(np.sum((~np.isin(self.candidates, self.items)) & (self.candidate_features.getnnz(axis=1) > 0)))
        return self

    def transform(self, items, documents):
        docs = [documents.get(int(item), {"title": "", "attributes": {}}) for item in items]
        features = [self.attributes.transform([doc["attributes"] for doc in docs]) * self.params["metadata_weight"]]
        if self.title is not None:
            features.append(self.title.transform([doc["title"] for doc in docs]) * self.params["title_weight"])
        return normalize(sparse.hstack(features, format="csr", dtype=np.float32), norm="l2", copy=False)

    def score(self, cohort):
        scores = np.zeros((len(cohort), len(self.candidates)), dtype=np.float32)
        warm = np.array([user in self.user_map and self.profiles[self.user_map[user]].nnz > 0 for user in cohort])
        users = np.flatnonzero(warm)
        for low in range(0, len(users), 128):
            part = users[low:low + 128]
            scores[part] = (self.profiles[[self.user_map[cohort[j]] for j in part]] @ self.candidate_features.T).toarray()
        return fallback_scores(scores, warm, self.popularity)

    def save(self, directory):
        from tvrec.io import atomic_json
        atomic_json(directory / "content_transform.json", {"params": self.params, "title_vocabulary": {k: int(v) for k, v in self.title.vocabulary_.items()} if self.title else {}, "metadata_vocabulary": {k: int(v) for k, v in self.attributes.vocabulary_.items()}, "fitted_programme_count": self.fitted_programme_count, "cold_candidates_with_content": self.cold_candidates_with_content, "user_ids": self.users, "candidate_ids": self.candidates.tolist()})
        if self.title is not None:
            np.save(directory / "content_title_idf.npy", self.title.idf_)
        sparse.save_npz(directory / "content_user_profiles.npz", self.profiles)
        sparse.save_npz(directory / "content_candidate_features.npz", self.candidate_features)


def reciprocal_rank_fusion(components, weights, constant=60.):
    scores = np.zeros_like(components[0], dtype=np.float32)
    rank_values = (1 / (constant + np.arange(1, scores.shape[1] + 1))).astype(np.float32)
    for matrix, weight in zip(components, weights):
        order = np.argsort(-matrix, axis=1, kind="stable")
        contribution = np.empty_like(matrix)
        np.put_along_axis(contribution, order, np.broadcast_to(rank_values, matrix.shape), axis=1)
        scores += weight * contribution
    return scores
