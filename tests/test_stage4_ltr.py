import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tests.common import USER_A, USER_B
from tvrec_ltr.common import admissible_training, assert_training_boundary, forecasts, load_options
from tvrec_ltr.evidence import cumulative_pairs, hourly_histograms
from tvrec_ltr.features import feature_columns
from tvrec_ltr.models import binary_weights, rerank_scores, training_data
from tvrec_ltr.retrieval import candidate_metrics, retrieve
from tvrec_ltr.verify import rational_ap


class Stage4TemporalTests(unittest.TestCase):
    def test_label_end_exclusive_at_cutoff_is_safe_but_one_second_later_is_not(self):
        assert_training_boundary({"cutoff": 10, "end": 20}, 20)
        with self.assertRaises(ValueError):
            assert_training_boundary({"cutoff": 10, "end": 21}, 20)
        with self.assertRaises(ValueError):
            assert_training_boundary({"cutoff": 20, "end": 10}, 30)

    def test_actual_main_recent_final_training_admission(self):
        context, options = load_options(Path(__file__).resolve().parents[1] / "configs/stage4.json")
        specs = forecasts(context, options)
        pseudo_names = {spec["name"] for spec in options["pseudo_forecasts"]}
        pseudo = [spec for spec in specs if spec["name"] in pseudo_names]
        for name, size in [("inner", 2), ("main", 6), ("recent_b", 10)]:
            cutoff = next(spec["cutoff"] for spec in specs if spec["name"] == name)
            allowed = admissible_training(pseudo, cutoff)
            self.assertEqual(len(allowed), size)
            self.assertTrue(all(spec["end"] <= cutoff for spec in allowed))
        self.assertEqual(len(admissible_training(pseudo, context.cutoff)), 11)

    def test_future_forecast_is_never_opened_by_training_loader(self):
        safe = {"name": "past", "cutoff": 0, "end": 10}
        future = {"name": "future", "cutoff": 10, "end": 30}
        x = pd.DataFrame({"user_id": [USER_A] * 2, "user_index": [0, 0], "tv_show_id": [1, 2], "pair_count50": [1., 0.]})
        y = x[["user_id", "tv_show_id"]].copy()
        y["label"] = [1, 0]
        data = {"features": x, "labels": y, "meta": {"specification": safe, "maximum_clipped_past_end_epoch": 0}}
        with patch("tvrec_ltr.models.load_dataset", return_value=data) as load:
            training_data([safe, future], {"past": "allowed", "future": "forbidden"}, 20, "history")
            load.assert_called_once_with("allowed")

    def test_future_viewing_cannot_change_cumulative_features(self):
        evidence = pd.DataFrame({"user_id": [USER_A, USER_A], "tv_show_id": [1, 2], "start_time": [0, 50], "end_time": [100, 100], "allocation_denominator": [1, 1]})
        past = cumulative_pairs(evidence, 0, 50)
        self.assertEqual(past.tv_show_id.tolist(), [1])
        self.assertEqual(past.seconds.tolist(), [50])
        evidence.loc[1, "end_time"] = 10000
        pd.testing.assert_frame_equal(past, cumulative_pairs(evidence, 0, 50))

    def test_histogram_splits_hour_boundaries_and_excludes_postcutoff_viewing(self):
        sessions = pd.DataFrame({"user_id": [USER_A, USER_A], "channel_id": [3, 4], "start_time": [1800, 7200], "end_time": [9000, 20000]})
        habits, channels, names = hourly_histograms(sessions, [USER_A, USER_B], 5400, ZoneInfo("Europe/Kyiv"))
        self.assertEqual(habits[0].sum(), 3600)
        self.assertEqual(np.sort(habits[0][habits[0] > 0]).tolist(), [1800, 1800])
        self.assertEqual(habits[1].sum(), 0)
        self.assertEqual(names, [3])
        self.assertEqual(channels.tolist(), [[3600], [0]])

    def test_ablation_columns_exclude_labels_and_raw_identities(self):
        frame = pd.DataFrame(columns=["user_id", "user_index", "tv_show_id", "label", "label_end", "pair_count50", "hist_rank", "schedule_future_count", "affinity_channel_share", "category_id", "als_score", "cbf_rank", "source_als"])
        self.assertEqual(feature_columns(frame, "history"), ["pair_count50", "hist_rank"])
        self.assertNotIn("als_score", feature_columns(frame, "context"))
        self.assertIn("category_id", feature_columns(frame, "context"))
        self.assertIn("als_score", feature_columns(frame, "full"))
        self.assertNotIn("label_end", feature_columns(frame, "full"))


class Stage4RetrievalAndModelTests(unittest.TestCase):
    def example(self):
        items = np.arange(1, 401)
        scores = {name: np.broadcast_to(np.arange(400, 0, -1, dtype=np.float32), (2, 400)).copy() for name in ["history", "als", "cbf", "popularity"]}
        quotas = {"history": 100, "als": 75, "cbf": 50, "popularity": 25}
        return items, scores, quotas

    def test_250_unique_eligible_string_user_candidates_even_for_identical_models(self):
        items, scores, quotas = self.example()
        frame = retrieve([USER_A, USER_B], items, scores, quotas)
        self.assertEqual(frame.groupby("user_id").size().tolist(), [250, 250])
        self.assertFalse(frame.duplicated(["user_id", "tv_show_id"]).any())
        self.assertEqual(frame.user_id.drop_duplicates().tolist(), [USER_A, USER_B])
        self.assertEqual(frame.groupby("user_id").source_history.sum().tolist(), [100, 100])
        self.assertEqual(frame.loc[frame.user_id == USER_A, "tv_show_id"].tolist(), list(range(1, 251)))

    def test_candidate_positives_are_not_injected_and_oracle_is_relevance_normalized(self):
        items, scores, quotas = self.example()
        frame = retrieve([USER_A, USER_B], items, scores, quotas)
        metrics = candidate_metrics(frame, [USER_A, USER_B], {USER_A: {1, 300}, USER_B: {399}})
        self.assertAlmostEqual(metrics["micro_candidate_recall"], 1/3)
        self.assertAlmostEqual(metrics["candidate_oracle_map_at_5_fixed"], .25)
        self.assertNotIn(300, frame.tv_show_id.to_list())

    def test_unsorted_universe_nonfinite_and_insufficient_candidates_rejected(self):
        items, scores, quotas = self.example()
        with self.assertRaises(ValueError):
            retrieve([USER_A, USER_B], items[::-1], scores, quotas)
        with self.assertRaises(ValueError):
            retrieve([USER_A, USER_B], items[:100], scores, quotas)
        scores["als"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            retrieve([USER_A, USER_B], items, scores, quotas)

    def test_class_weights_have_equal_user_mass_and_handle_zero_positive_queries(self):
        labels = np.array([1, 0, 0, 0, 0, 0])
        weights = binary_weights(labels, [3, 3])
        self.assertAlmostEqual(weights[:3].sum(), 3)
        self.assertAlmostEqual(weights[3:].sum(), 3)
        self.assertGreater(weights[0], weights[1])
        with self.assertRaises(ValueError):
            binary_weights(labels, [3])

    def test_reranking_preserves_cold_fallback_and_ascending_id_ties(self):
        frame = pd.DataFrame({"user_index": [0, 0, 0, 1, 1, 1], "tv_show_id": [1, 2, 3, 1, 2, 3], "hist_rank": [1, 2, 3, 1, 2, 3]})
        archive = {"candidate_ids": np.array([1, 2, 3, 4]), "history": np.array([[3., 2., 1., 0.], [0., 1., 2., 3.]]), "cold": np.array([False, True])}
        raw = np.array([.5, .5, .9, .9, .8, .7])
        scores = rerank_scores(frame, raw, archive, 0.)
        self.assertEqual(np.argsort(-scores[0], kind="stable").tolist(), [2, 0, 1, 3])
        np.testing.assert_array_equal(scores[1], archive["history"][1])
        np.testing.assert_array_equal(scores, rerank_scores(frame, raw, archive, 0.))

    def test_independent_rational_ap_and_oracle_empty_users(self):
        self.assertAlmostEqual(float(rational_ap([1, 2, 3, 4, 5], {1, 3, 5})), (1 + 2/3 + 3/5)/3)
        self.assertEqual(rational_ap([1, 2, 3, 4, 5], set()), 0)
