import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.common import USER_A, temporary_directory, write_evidence
from tvrec.interactions import Window
from tvrec.stage2.completion import blackout_intervals, continuous_sessions, window_event_portion


def event_frame(starts, ends, durations=None, items=None, broadcasts=None, channels=None):
    n = len(starts)
    return pd.DataFrame({"user_id": [USER_A] * n, "tv_show_id": items or [1] * n, "broadcast_id": broadcasts or list(range(n)), "channel_id": channels or [3] * n, "start_time": starts, "end_time": ends, "broadcast_seconds": durations or [100] * n})


class CompletionTests(unittest.TestCase):
    def test_strict_80_and_inclusive_50_boundaries(self):
        f = event_frame([0, 0, 0], [50, 80, 81])
        output = window_event_portion(f, Window("all", 0, 200), {})
        self.assertEqual(output.ge50.tolist(), [True, True, True])
        self.assertEqual(output.gt80.tolist(), [False, False, True])

    def test_cutoff_clips_numerator_not_full_broadcast_denominator(self):
        f = event_frame([0], [100])
        train = window_event_portion(f, Window("train", 0, 60), {})
        val = window_event_portion(f, Window("validation", 60, 200), {})
        self.assertEqual(train.observed_seconds.tolist(), [60])
        self.assertEqual(train.gt80.tolist(), [False])
        self.assertEqual(val.ge50.tolist(), [False])

    def test_separate_sittings_cannot_be_accumulated(self):
        f = event_frame([0, 60], [45, 100], broadcasts=[1, 1])
        output = window_event_portion(f, Window("all", 0, 200), {})
        self.assertEqual(output.gt80.tolist(), [False, False])
        self.assertEqual(output.ge50.tolist(), [False, False])

    def test_repeated_broadcasts_are_not_one_long_programme(self):
        f = event_frame([0, 200], [90, 240], broadcasts=[1, 2])
        output = window_event_portion(f, Window("all", 0, 400), {})
        self.assertEqual(output.gt80.tolist(), [True, False])

    def test_channel_switch_truncates_and_does_not_resume(self):
        out, _ = continuous_sessions([0, 0], [3, 4], [0, 40], [100, 60])
        self.assertEqual(list(zip(out["channel_id"], out["start"], out["end"])), [(3, 0, 40), (4, 40, 60)])

    def test_explicit_return_starts_new_sitting(self):
        out, _ = continuous_sessions([0, 0, 0], [3, 4, 3], [0, 40, 60], [100, 60, 100])
        self.assertEqual(out["start"].tolist(), [0, 40, 60])
        self.assertEqual(out["end"].tolist(), [40, 60, 100])

    def test_same_channel_duplicates_and_contacts_union(self):
        out, _ = continuous_sessions([0, 0, 0], [3, 3, 3], [0, 0, 50], [50, 50, 100])
        self.assertEqual(out["start"].tolist(), [0])
        self.assertEqual(out["end"].tolist(), [100])

    def test_simultaneous_different_channels_create_unknown_break(self):
        out, stats = continuous_sessions([0, 0, 0, 0], [3, 3, 4, 3], [0, 30, 30, 70], [100, 60, 60, 100])
        self.assertEqual(out["start"].tolist(), [0, 70])
        self.assertEqual(out["end"].tolist(), [30, 100])
        self.assertEqual(stats["same_timestamp_conflicting_channel_groups"], 1)

    def test_competing_broadcast_seconds_are_not_double_credited(self):
        bs, be = blackout_intervals(np.array([0, 40]), np.array([100, 140]))
        self.assertEqual(bs.tolist(), [40])
        f = event_frame([0, 40], [100, 140])
        out = window_event_portion(f, Window("all", 0, 200), {"3": {"start": bs.tolist(), "end": be.tolist()}})
        self.assertEqual(out.observed_seconds.tolist(), [40, 40])
        self.assertFalse(out.ge50.any())

    def test_post_cutoff_activity_does_not_change_earlier_continuity(self):
        a, _ = continuous_sessions([0], [3], [0], [100])
        b, _ = continuous_sessions([0, 0], [3, 4], [0, 100], [100, 150])
        self.assertEqual(a["start"].tolist(), b["start"][:1].tolist())
        self.assertEqual(a["end"].tolist(), b["end"][:1].tolist())
