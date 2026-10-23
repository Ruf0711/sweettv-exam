import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tvrec.intervals import canonical_timeline, normalize_viewing
from tvrec.preparation import join_views
from tests.common import USER_A, temporary_directory


def allocated(timeline, start, end):
    rows, indices, left, right = timeline.overlaps(np.array([start]), np.array([end]))
    result = {}
    for segment, s, e in zip(indices, left, right):
        labels = timeline.labels[timeline.label_offsets[segment]:timeline.label_offsets[segment + 1]]
        for item in labels:
            result[int(item)] = result.get(int(item), 0.) + (e - s) / len(labels)
    return result


class IntervalTests(unittest.TestCase):
    def test_endpoint_contact_is_zero(self):
        timeline = canonical_timeline([0, 10], [10, 20], [1, 2])
        self.assertEqual(allocated(timeline, 10, 15), {2: 5.})
        self.assertEqual(allocated(timeline, 20, 25), {})

    def test_multiple_programmes_and_midnight_independent(self):
        timeline = canonical_timeline([86300, 86400, 86500], [86400, 86500, 86600], [1, 2, 3])
        self.assertEqual(allocated(timeline, 86350, 86550), {1: 50., 2: 100., 3: 50.})

    def test_long_session_enumerates_all_programmes(self):
        starts = np.arange(0, 21000, 100)
        timeline = canonical_timeline(starts, starts + 100, np.arange(1, 211))
        self.assertEqual(len(allocated(timeline, 0, 21000)), 210)

    def test_competing_broadcasts_conserve_duration(self):
        timeline = canonical_timeline([0, 5], [10, 15], [1, 2])
        self.assertEqual(allocated(timeline, 0, 15), {1: 7.5, 2: 7.5})
        self.assertEqual(int(timeline.covered_seconds(np.array([0]), np.array([15]))[0]), 15)

    def test_duplicate_broadcast_label_does_not_take_extra_share(self):
        timeline = canonical_timeline([0, 0, 0], [10, 10, 10], [1, 1, 2])
        self.assertEqual(allocated(timeline, 0, 10), {1: 5., 2: 5.})

    def test_unknown_label_keeps_its_share(self):
        timeline = canonical_timeline([0, 0], [10, 10], [1, 0])
        self.assertEqual(allocated(timeline, 0, 10), {0: 5., 1: 5.})

    def test_zero_and_negative_schedule_duration_excluded(self):
        timeline = canonical_timeline([0, 5, 9], [10, 5, 8], [1, 2, 3])
        self.assertEqual(allocated(timeline, 0, 10), {1: 10.})

    def test_viewing_duplicates_and_union(self):
        views, stats = normalize_viewing([0, 0, 0, 0], [3, 3, 3, 4], [0, 0, 5, 5], [10, 10, 15, 15], [10, 10, 10, 10])
        self.assertEqual(stats["exact_duplicate_rows"], 1)
        self.assertEqual(stats["overlapping_rows_after_deduplication"], 1)
        self.assertEqual(stats["union_seconds"], 25)
        self.assertEqual(views["source_rows"].tolist(), [2, 1])

    def test_nested_viewing_does_not_split_a_union(self):
        views, _ = normalize_viewing([0, 0, 0], [3, 3, 3], [0, 5, 12], [20, 10, 15], [20, 5, 3])
        self.assertEqual(views["start"].tolist(), [0])
        self.assertEqual(views["end"].tolist(), [20])

    def test_missing_channel_and_zero_labels_not_emitted(self):
        with temporary_directory() as temporary:
            settings = SimpleNamespace(values={"preparation": {"join_chunk_rows": 10, "progress_every_rows": 10}})
            views = {"user_index": np.array([0, 0, 0]), "channel_id": np.array([3, 4, 999]), "start": np.array([0, 0, 0]), "end": np.array([10, 10, 10]), "source_rows": np.array([1, 1, 1])}
            stats = join_views(settings, Path(temporary), {3: canonical_timeline([0], [10], [0]), 4: canonical_timeline([0, 0], [10, 10], [1, 0])}, views, [USER_A])
            evidence = pq.read_table(Path(temporary) / "evidence.parquet").to_pandas()
            self.assertEqual(evidence.tv_show_id.tolist(), [1])
            self.assertEqual(evidence.credited_seconds.tolist(), [5.])
            self.assertEqual(stats["positive_credited_seconds"], 5.)
            self.assertEqual(stats["unknown_credited_seconds"], 15.)

    def test_randomized_dense_reference_100_cases(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            starts = rng.integers(0, 30, size=10)
            ends = starts + rng.integers(0, 15, size=10)
            labels = rng.integers(0, 5, size=10)
            timeline = canonical_timeline(starts, ends, labels)
            lo, hi = sorted(rng.integers(0, 45, size=2))
            actual = allocated(timeline, int(lo), int(hi))
            expected = {}
            for second in range(int(lo), int(hi)):
                active = set(labels[(starts <= second) & (ends > second)].tolist())
                for item in active:
                    expected[item] = expected.get(item, 0.) + 1 / len(active)
            self.assertEqual(set(actual), set(expected))
            for item in actual:
                self.assertAlmostEqual(actual[item], expected[item])
