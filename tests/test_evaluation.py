import unittest

import pandas as pd

from tvrec.interactions import Window, aggregate_portion, finish_aggregation
from tvrec.metrics import ap_at_k, evaluate, relevance_sets
from tvrec.popularity import Popularity
from tvrec.recommendations import Recommendation
from tests.common import USER_A, USER_B


class TemporalAggregationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"user_id": [USER_A], "tv_show_id": [1], "view_id": [0], "start_time": [90], "end_time": [110], "allocation_denominator": [2]})

    def test_cutoff_crossing_is_split_before_aggregation(self):
        train = aggregate_portion(self.frame, Window("train", 0, 100))
        validation = aggregate_portion(self.frame, Window("validation", 100, 200))
        self.assertEqual(train.seconds.tolist(), [5.])
        self.assertEqual(validation.seconds.tolist(), [5.])

    def test_future_viewing_cannot_change_training_features(self):
        future = pd.DataFrame({"user_id": [USER_B], "tv_show_id": [2], "view_id": [1], "start_time": [100], "end_time": [200], "allocation_denominator": [1]})
        left = finish_aggregation([aggregate_portion(self.frame, Window("train", 0, 100))], 5, None)
        right = finish_aggregation([aggregate_portion(pd.concat([self.frame, future]), Window("train", 0, 100))], 5, None)
        pd.testing.assert_frame_equal(left, right)

    def test_confidence_cap_preserves_evidence_and_relevance(self):
        output = finish_aggregation([aggregate_portion(self.frame, Window("all", 0, 200))], 8, 3)
        self.assertEqual(output.seconds.tolist(), [10.])
        self.assertEqual(output.confidence_seconds.tolist(), [3.])
        self.assertEqual(output.relevant.tolist(), [True])

    def test_exact_relevance_threshold(self):
        output = finish_aggregation([aggregate_portion(self.frame, Window("all", 0, 200))], 10, None)
        self.assertEqual(output.relevant.tolist(), [True])


class RankingAndMetricTests(unittest.TestCase):
    def test_hand_calculated_ap(self):
        self.assertAlmostEqual(ap_at_k([1, 9, 2, 8, 7], {1, 2}), 5 / 6)

    def test_repeated_hits_do_not_receive_credit(self):
        self.assertAlmostEqual(ap_at_k([1, 1, 2, 2, 9], {1, 2}), 5 / 6)

    def test_empty_relevance_contributes_zero(self):
        self.assertEqual(ap_at_k([1, 2, 3, 4, 5], set()), 0.)

    def test_ap_uses_minimum_of_five_and_relevance_count(self):
        self.assertEqual(ap_at_k([1, 2, 3, 4, 5], set(range(1, 20))), 1.)

    def test_fixed_and_evaluable_cohorts(self):
        predictions = [Recommendation(user, (1, 2, 3, 4, 5), (1.,) * 5) for user in [USER_A, USER_B]]
        result, rows = evaluate([USER_A, USER_B], predictions, {USER_A: {1}}, range(1, 7))
        self.assertEqual(result["map_at_5_fixed"], .5)
        self.assertEqual(result["map_at_5_evaluable"], 1.)
        self.assertEqual(rows.user_id.tolist(), [USER_A, USER_B])

    def test_distinct_viewers_not_seconds_or_duplicate_rows(self):
        frame = pd.DataFrame({"user_id": [USER_A, USER_A, USER_B, USER_B], "tv_show_id": [1, 1, 1, 2], "seconds": [1., 100000., 1., 200000.], "relevant": [False, True, False, True]})
        model = Popularity().fit(frame, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(model.support[1], 2)
        self.assertEqual(model.support[2], 1)
        self.assertEqual(model.recommend([USER_B])[0].programme_ids, (1, 2, 3, 4, 5))

    def test_deterministic_ties_and_cold_user_fallback(self):
        frame = pd.DataFrame({"user_id": [USER_A], "tv_show_id": [9], "seconds": [60.]})
        model = Popularity().fit(frame, [9, 8, 7, 6, 5, 4])
        rows = model.recommend([USER_A, USER_B])
        self.assertEqual(rows[0].programme_ids, (9, 4, 5, 6, 7))
        self.assertEqual(rows[0].programme_ids, rows[1].programme_ids)

    def test_popularity_optional_support_threshold(self):
        frame = pd.DataFrame({"user_id": [USER_A, USER_B], "tv_show_id": [1, 2], "seconds": [59., 60.]})
        model = Popularity(support_min_seconds=60).fit(frame, range(1, 7))
        self.assertEqual(model.recommend([USER_A])[0].programme_ids[0], 2)

    def test_fewer_than_five_candidates_fails(self):
        with self.assertRaises(ValueError):
            Popularity().fit(pd.DataFrame(), [1, 2, 3, 4])

    def test_truth_respects_catalogue_and_threshold(self):
        frame = pd.DataFrame({"user_id": [USER_A] * 3, "tv_show_id": [1, 2, 3], "relevant": [True, False, True]})
        self.assertEqual(relevance_sets(frame, [1, 2]), {USER_A: {1}})
