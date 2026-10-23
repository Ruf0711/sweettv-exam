import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from tests.common import USER_A, USER_B, temporary_directory
from tvrec.interactions import Window
from tvrec.stage2.completion import completion_windows
from tvrec.stage2.rankers import ContentBased, PersonalHistory, SparseALS, feedback_matrix, global_scores, history_weights, recommendations_from_scores, reciprocal_rank_fusion


def feedback():
    return pd.DataFrame({"user_id": [USER_A, USER_A, USER_B, USER_B, "3", "3"], "tv_show_id": [1, 2, 2, 3, 3, 4], "count50": [10, 5, 6, 3, 7, 4], "count80": [8, 4, 5, 2, 6, 3], "seconds50": [30000, 15000, 18000, 9000, 21000, 12000], "last_epoch": [90, 80, 85, 75, 92, 70]})


def metadata():
    rows = []
    for item in range(1, 7):
        rows.append({"tv_show_id": item, "channel_id": 3, "start_time": 10 if item <= 4 else 110, "end_time": 50 if item <= 4 else 160, "duration": 40 if item <= 4 else 50, "tv_show_title": "Alpha series" if item <= 4 else "qxzj future title", "tv_show_category": "Known category", "tv_show_genre_1": "Known genre" if item <= 4 else "never_seen_genre", "tv_show_genre_2": "", "tv_show_genre_3": "", "year_of_production": "2020", "director": "Director", "actors": "Actor"})
    frame = pd.DataFrame(rows)
    return frame.iloc[:4].copy(), frame.iloc[4:].copy()


HISTORY = {"half_life_days": 90., "duration_coefficient": .25}
CONTENT = {"title_max_features": 100, "title_weight": .4, "metadata_weight": 1.}
ALS = {"factors": 4, "regularization": .1, "confidence_scale": 5., "iterations": 10}


class Stage2ModelTests(unittest.TestCase):
    def test_history_frequency_duration_and_recency(self):
        f = feedback()
        w = history_weights(f, 100, **HISTORY)
        self.assertGreater(w[0], w[1])
        future = f.copy()
        future.loc[0, "last_epoch"] = 101
        with self.assertRaises(ValueError):
            history_weights(future, 100, **HISTORY)

    def test_sparse_feedback_preserves_exact_external_ids(self):
        f = feedback()
        matrix, users, items = feedback_matrix(f, history_weights(f, 100))
        self.assertEqual(matrix.nnz, 6)
        self.assertIn(USER_A, users)
        self.assertEqual(items.tolist(), [1, 2, 3, 4])

    def test_history_cold_user_fallback_and_five_unique(self):
        model = PersonalHistory().fit(feedback(), 100, range(1, 7), np.array([1, 2, 3, 4, 5, 6], dtype=np.float32), HISTORY)
        rows = recommendations_from_scores([USER_A, "999"], range(1, 7), model.score([USER_A, "999"]))
        self.assertEqual(rows[0].programme_ids[0], 1)
        self.assertEqual(rows[1].programme_ids, (6, 5, 4, 3, 2))

    def test_als_genuine_factors_sparse_shape_and_cold_starts(self):
        with threadpool_limits(limits=1):
            model = SparseALS().fit(feedback(), 100, range(1, 7), np.arange(1, 7, dtype=np.float32), HISTORY, ALS, seed=42, threads=1)
        self.assertEqual(model.model.user_factors.shape, (3, 4))
        self.assertEqual(model.model.item_factors.shape, (4, 4))
        self.assertEqual(model.training_nnz, 6)
        self.assertEqual(len(model.losses), 10)
        self.assertLess(model.losses[-1]["loss"], model.losses[0]["loss"])
        scores = model.score([USER_A, "999"])
        self.assertTrue(np.isfinite(scores).all())
        self.assertNotIn(6, model.item_map)
        rows = recommendations_from_scores([USER_A, "999"], range(1, 7), scores)
        self.assertEqual(rows[1].programme_ids, (6, 5, 4, 3, 2))

    def test_als_reproducible_seed(self):
        with threadpool_limits(limits=1):
            a = SparseALS().fit(feedback(), 100, range(1, 7), np.ones(6), HISTORY, ALS, seed=42, threads=1)
            b = SparseALS().fit(feedback(), 100, range(1, 7), np.ones(6), HISTORY, ALS, seed=42, threads=1)
        np.testing.assert_allclose(a.model.user_factors, b.model.user_factors, rtol=0, atol=0)
        np.testing.assert_allclose(a.score([USER_A]), b.score([USER_A]), rtol=0, atol=0)

    def test_content_fit_excludes_future_vocabulary_but_supports_cold_items(self):
        prior, upcoming = metadata()
        model = ContentBased().fit(feedback(), 100, range(1, 7), np.arange(1, 7, dtype=np.float32), HISTORY, CONTENT, prior, upcoming)
        self.assertNotIn("tv_show_genre_1:never_seen_genre", model.attributes.vocabulary_)
        self.assertNotIn("qxz", model.title.vocabulary_)
        self.assertEqual(model.cold_candidates_with_content, 2)
        self.assertGreater(model.candidate_features[4].nnz, 0)
        rows = recommendations_from_scores([USER_A, "999"], range(1, 7), model.score([USER_A, "999"]))
        self.assertEqual(rows[1].programme_ids, (6, 5, 4, 3, 2))
        with temporary_directory() as temporary:
            model.save(Path(temporary))
            self.assertTrue((Path(temporary) / "content_transform.json").is_file())

    def test_future_metadata_cannot_be_passed_as_fitted_corpus(self):
        prior, upcoming = metadata()
        with self.assertRaises(ValueError):
            ContentBased().fit(feedback(), 100, range(1, 7), np.ones(6), HISTORY, CONTENT, pd.concat([prior, upcoming]), upcoming)

    def test_changing_future_metadata_keeps_fitted_transform_and_profiles(self):
        prior, upcoming = metadata()
        a = ContentBased().fit(feedback(), 100, range(1, 7), np.ones(6), HISTORY, CONTENT, prior, upcoming)
        changed = upcoming.copy()
        changed["tv_show_title"] = "different future-only phrase"
        b = ContentBased().fit(feedback(), 100, range(1, 7), np.ones(6), HISTORY, CONTENT, prior, changed)
        self.assertEqual(a.title.vocabulary_, b.title.vocabulary_)
        self.assertEqual(a.attributes.vocabulary_, b.attributes.vocabulary_)
        np.testing.assert_allclose(a.profiles.toarray(), b.profiles.toarray(), rtol=0, atol=0)

    def test_rrf_deterministic_positive_top5_contract(self):
        left = np.array([[6, 5, 4, 3, 2, 1]], dtype=np.float32)
        right = left[:, ::-1]
        scores = reciprocal_rank_fusion([left, right], [.75, .25])
        rows = recommendations_from_scores([USER_A], range(1, 7), scores)
        self.assertEqual(len(set(rows[0].programme_ids)), 5)
        self.assertEqual(rows[0].programme_ids[0], 1)

    def test_zero_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            recommendations_from_scores([USER_A], [0, 1, 2, 3, 4], np.ones((1, 5)))
