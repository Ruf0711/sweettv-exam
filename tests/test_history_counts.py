import unittest

import numpy as np
import pandas as pd

from tests.common import USER_A, USER_B
from tvrec.interactions import Window
from tvrec.stage2.rankers import recommendations_from_scores
from tvrec_counts.experiment import airing_frequency
from tvrec_counts.features import finish_instances, instance_portion, score_counts, verify_against_existing


def events(rows):
    return pd.DataFrame([{"user_id": user, "tv_show_id": item, "channel_id": 3, "broadcast_id": broadcast,
                          "start_time": start, "end_time": end, "broadcast_seconds": duration}
                         for user, item, broadcast, start, end, duration in rows])


def aggregate(frame, start=0, end=1000, blackouts=None):
    return finish_instances([instance_portion(frame, Window("test", start, end), blackouts or {})])


class HistoryCountTests(unittest.TestCase):
    def test_strict30_50_80_boundaries_and_repeated_broadcasts(self):
        frame = events([(USER_A, 1, 1, 0, 30, 100), (USER_A, 1, 2, 100, 150, 100),
                        (USER_A, 1, 3, 200, 280, 100), (USER_A, 1, 4, 300, 381, 100)])
        row = aggregate(frame).iloc[0]
        self.assertEqual((row.count30, row.count50, row.count80, row.exact50_only), (3, 2, 1, 1))
        self.assertEqual((row.seconds50, row.ge50seconds), (161, 211))

    def test_separate_sittings_never_accumulate_to_completion(self):
        row = aggregate(events([(USER_A, 1, 1, 0, 40, 100), (USER_A, 1, 1, 60, 100, 100)])).iloc[0]
        self.assertEqual((row.count30, row.count50, row.count80), (1, 0, 0))

    def test_duplicate_sittings_and_batch_boundaries_count_one_instance(self):
        frame = events([(USER_A, 1, 1, 0, 90, 100)])
        part = instance_portion(frame, Window("all", 0, 1000), {})
        row = finish_instances([part, part.copy()]).iloc[0]
        self.assertEqual((row.count30, row.count50, row.count80, row.seconds50), (1, 1, 1, 90))

    def test_recency_uses_last_qualifying_sitting(self):
        row = aggregate(events([(USER_A, 1, 1, 0, 90, 100), (USER_A, 1, 1, 60, 100, 100)])).iloc[0]
        self.assertEqual(row.last50, 90)

    def test_cutoff_clips_numerator_not_broadcast_denominator(self):
        frame = events([(USER_A, 1, 1, 0, 100, 100)])
        row = aggregate(frame, end=50).iloc[0]
        self.assertEqual((row.count30, row.count50, row.count80, row.exact50_only), (1, 0, 0, 1))
        self.assertEqual(row.last_ge50, 50)

    def test_competing_unknown_seconds_excluded_and_zero_items_not_emitted(self):
        frame = events([(USER_A, 1, 1, 0, 100, 100), (USER_A, 0, 2, 0, 100, 100)])
        result = aggregate(frame, blackouts={"3": {"start": [30], "end": [80]}})
        self.assertEqual(result.tv_show_id.tolist(), [1])
        self.assertEqual(result.count50.tolist(), [0])

    def test_inclusive50_anchor_exactly_reconstructs_old_semantics(self):
        result = aggregate(events([(USER_A, 1, 1, 0, 50, 100), (USER_A, 1, 2, 100, 190, 100)]))
        original = pd.DataFrame({"user_id": [USER_A], "tv_show_id": [1], "count50": [2], "count80": [1], "seconds50": [140], "last_epoch": [190]})
        verified = verify_against_existing(result, original, [USER_A])
        self.assertEqual(verified["exactly_half_broadcasts_excluded_by_strict50"], 1)

    def test_pure_count_order_ties_and_cold_shortlist_fallback(self):
        frame = aggregate(events([(USER_A, 2, 1, 0, 90, 100), (USER_A, 2, 2, 100, 190, 100), (USER_A, 3, 3, 200, 290, 100)]))
        scores, coverage = score_counts(frame, [USER_A, USER_B], list(range(1, 7)), np.array([3, 4, 5, 6, 7, 8]), 1000, {"kind": "count", "threshold": 50})
        rows = recommendations_from_scores([USER_A, USER_B], list(range(1, 7)), scores)
        self.assertEqual(rows[0].programme_ids, (2, 3, 6, 5, 4))
        self.assertEqual(rows[1].programme_ids, (6, 5, 4, 3, 2))
        self.assertEqual(coverage["cold_users"], 1)
        self.assertEqual(coverage["fallback_positions"], 8)

    def test_mixture_is_linear_combination_and_signals_are_cutoff_safe(self):
        frame = aggregate(events([(USER_A, 2, 1, 0, 90, 100), (USER_A, 3, 2, 100, 150, 100)]))
        scores, _ = score_counts(frame, [USER_A], list(range(1, 7)), np.zeros(6), 1000, {"kind": "mixture", "weights": [.25, .5, .25]})
        self.assertEqual((scores[0, 1], scores[0, 2]), (1., .25))
        with self.assertRaises(ValueError):
            score_counts(frame, [USER_A], list(range(1, 7)), np.zeros(6), 50, {"kind": "signals", "threshold": 50, "half_life_days": 90., "duration_coefficient": .25})

    def test_airing_feature_is_schedule_only_and_half_open(self):
        metadata = pd.DataFrame({"tv_show_id": [1, 1, 2, 2], "start": [0, 10, 20, 30], "end": [10, 20, 30, 40], "source_file": ["old", "new", "new", "later"]})
        self.assertEqual(airing_frequency(metadata, 10, 30), {1: 1, 2: 1})
        self.assertEqual(airing_frequency(metadata, 10, 30, "new"), {1: 1, 2: 1})

    def test_randomized_max_single_sitting_reference(self):
        rng = np.random.default_rng(54)
        for case in range(60):
            source = []
            expected = {}
            for _ in range(40):
                broadcast = int(rng.integers(1, 12))
                watched = int(rng.integers(1, 101))
                source.append((USER_A, 1, broadcast, broadcast * 100, broadcast * 100 + watched, 100))
                expected[broadcast] = max(expected.get(broadcast, 0), watched)
            actual = aggregate(events(source), end=2000)
            count = (0, 0, 0) if actual.empty else tuple(int(actual.iloc[0]["count" + str(q)]) for q in [30, 50, 80])
            self.assertEqual(count, tuple(sum(seconds > q for seconds in expected.values()) for q in [30, 50, 80]), f"case {case}")
