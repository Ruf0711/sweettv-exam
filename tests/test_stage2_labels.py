import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tests.common import USER_A, USER_B, temporary_directory
from tvrec.interactions import Window
from tvrec.stage2.completion import EVENT_SCHEMA, completion_windows, window_event_portion
from tvrec.stage2.experiment import truths_and_coverage
from tvrec.time_utils import LOG_FORMAT, parse_local_times


class Stage2LabelTests(unittest.TestCase):
    def test_full_broadcast_counts_deduplicate_sittings_and_separate_repeats(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            cache = root / "events_fixture"
            cache.mkdir()
            (cache / "blackouts.json").write_text("{}", encoding="utf-8")
            records = []
            for broadcast, start, end, bs in [(1, 0, 85, 0), (1, 10, 95, 0), (2, 200, 245, 200), (2, 260, 300, 200), (3, 400, 450, 400)]:
                records.append({"user_id": USER_A, "tv_show_id": 1, "channel_id": 3, "session_id": len(records), "broadcast_id": broadcast, "start_time": pd.Timestamp(start, unit="s", tz="UTC"), "end_time": pd.Timestamp(end, unit="s", tz="UTC"), "broadcast_start": pd.Timestamp(bs, unit="s", tz="UTC"), "broadcast_end": pd.Timestamp(bs + 100, unit="s", tz="UTC"), "broadcast_seconds": 100})
            pq.write_table(pa.Table.from_pylist(records, schema=EVENT_SCHEMA), cache / "events.parquet")
            settings = SimpleNamespace(artifacts_dir=root / "out")
            result, path = completion_windows(settings, {}, cache, [Window("all", 0, 1000)])
            row = result["all"].iloc[0]
            self.assertEqual(row.count50, 2)
            self.assertEqual(row.count80, 1)
            self.assertEqual(row.seconds50, 135)
            repeated, same_path = completion_windows(settings, {}, cache, [Window("all", 0, 1000)])
            pd.testing.assert_frame_equal(result["all"], repeated["all"])
            self.assertEqual(path, same_path)

    def test_dst_broadcast_uses_elapsed_utc_duration(self):
        times = parse_local_times(pd.Series(["2020-03-29 02:30:00", "2020-03-29 04:30:00"]), LOG_FORMAT, ZoneInfo("Europe/Kyiv"))
        frame = pd.DataFrame({"user_id": [USER_A], "tv_show_id": [1], "channel_id": [3], "broadcast_id": [1], "start_time": [int(times[0])], "end_time": [int(times[1])], "broadcast_seconds": [3600]})
        result = window_event_portion(frame, Window("dst", int(times[0]), int(times[1])), {})
        self.assertEqual(result.observed_seconds.tolist(), [3600])
        self.assertEqual(result.gt80.tolist(), [True])

    def test_three_rules_keep_fixed_cohort_and_independent_catalogue(self):
        cumulative = pd.DataFrame({"user_id": [USER_A, USER_A], "tv_show_id": [1, 2], "seconds": [60., 100.]})
        complete = pd.DataFrame({"user_id": [USER_A, USER_A], "tv_show_id": [1, 2], "count50": [1, 1], "count80": [0, 1]})
        truths, coverage = truths_and_coverage(cumulative, complete, [USER_A, USER_B], [1, 3, 4, 5, 6])
        self.assertEqual(truths["cumulative_ge60"], {USER_A: {1}})
        self.assertEqual(truths["completion_ge50"], {USER_A: {1}})
        self.assertEqual(truths["completion_gt80"], {})
        self.assertEqual(coverage["completion_gt80"]["zero_positive_users"], 2)
        self.assertEqual(coverage["cumulative_ge60"]["cohort_excluded_pairs"], 1)
