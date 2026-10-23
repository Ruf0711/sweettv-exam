import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tests.common import ROOT, USER_A, USER_B, fixture_project, schedule_row, temporary_directory, write_csv
from tvrec.interactions import Window
from tvrec.io import LOG_COLUMNS, SCHEDULE_COLUMNS, fingerprint_files
from tvrec.pipeline import source_paths
from tvrec.resources import Resources
from tvrec.stage2.common import load_stage2, validate_experiment
from tvrec.stage2.completion import blackout_intervals, build_completion_cache, completion_windows, continuous_sessions, match_broadcasts, window_event_portion
from tvrec.time_utils import local_epoch


def slow_sessions(users, channels, starts, ends):
    output = []
    for user in sorted(set(users)):
        grouped = {}
        for u, c, s, e in zip(users, channels, starts, ends):
            if u == user and e > s:
                group = grouped.setdefault(s, {})
                group[c] = max(group.get(c, e), e)
        current = None
        for start, group in sorted(grouped.items()):
            channel = next(iter(group)) if len(group) == 1 else None
            end = max(group.values())
            if current is not None:
                c, s, e = current
                if channel == c and start <= e:
                    current = (c, s, max(e, end))
                    continue
                output.append((user, c, s, min(e, start)))
            current = (channel, start, end) if channel is not None else None
        if current is not None:
            c, s, e = current
            output.append((user, c, s, e))
    return sorted(row for row in output if row[3] > row[2])


class Stage2IntegrityTests(unittest.TestCase):
    def test_random_sessions_match_independent_state_machine(self):
        rng = np.random.default_rng(421)
        for case in range(80):
            users = rng.integers(0, 3, 45)
            channels = rng.integers(1, 5, 45)
            starts = rng.integers(0, 35, 45)
            ends = starts + rng.integers(1, 15, 45)
            actual, _ = continuous_sessions(users, channels, starts, ends)
            rows = sorted(zip(actual["user_index"], actual["channel_id"], actual["start"], actual["end"]))
            self.assertEqual(rows, slow_sessions(users, channels, starts, ends), f"case {case}")

    def test_random_broadcast_join_and_completion_match_dense_seconds(self):
        rng = np.random.default_rng(422)
        for case in range(80):
            start = rng.integers(0, 40, 14)
            end = start + rng.integers(1, 14, 14)
            items = rng.integers(0, 5, 14)
            unique = sorted(set(zip(start.tolist(), end.tolist(), items.tolist())))
            start, end, items = (np.asarray(values) for values in zip(*unique))
            positive = items > 0
            broadcast = {"start": start[positive], "end": end[positive]}
            view_s = rng.integers(0, 40, 12)
            view_e = view_s + rng.integers(1, 20, 12)
            row, index, left, right = match_broadcasts(view_s, view_e, broadcast)
            expected = [(i, j, max(s, bs), min(e, be)) for i, (s, e) in enumerate(zip(view_s, view_e)) for j, (bs, be) in enumerate(zip(broadcast["start"], broadcast["end"])) if max(s, bs) < min(e, be)]
            self.assertEqual(sorted(zip(row, index, left, right)), sorted(expected))
            blackout_s, blackout_e = blackout_intervals(start, end)
            frame = pd.DataFrame({"user_id": [USER_A] * len(row), "tv_show_id": items[positive][index], "channel_id": [3] * len(row), "start_time": left, "end_time": right, "broadcast_seconds": broadcast["end"][index] - broadcast["start"][index]})
            result = window_event_portion(frame, Window("window", 5, 45), {"3": {"start": blackout_s.tolist(), "end": blackout_e.tolist()}})
            for event in result.itertuples():
                seconds = sum(int(((start <= t) & (end > t)).sum()) == 1 for t in range(max(event.start_time, 5), min(event.end_time, 45)))
                self.assertEqual(event.observed_seconds, seconds)
                self.assertEqual(event.ge50, seconds * 2 >= event.broadcast_seconds)
                self.assertEqual(event.gt80, seconds * 5 > 4 * event.broadcast_seconds)

    def test_completion_cache_excludes_unknowns_duplicates_and_future_viewing(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            settings, _ = fixture_project(root)
            schedules = settings.values["sources"]["schedules"]
            rows = [schedule_row(3, "18.05.2020 00:00:00", 100, 1), schedule_row(3, "18.05.2020 00:00:00", 100, 1), schedule_row(3, "18.05.2020 00:00:40", 20, ""), schedule_row(3, "18.05.2020 00:03:20", 100, 1)]
            write_csv(settings.source(schedules[0]), SCHEDULE_COLUMNS, rows)
            for name in schedules[1:]:
                write_csv(settings.source(name), SCHEDULE_COLUMNS, [])
            logs = [[USER_A, 3, "2020-05-18 00:00:00", "2020-05-18 00:01:40", 100], [USER_A, 3, "2020-05-18 00:00:00", "2020-05-18 00:01:40", 100], [USER_A, 3, "2020-05-18 00:03:20", "2020-05-18 00:04:50", 90], [USER_B, 3, "2020-05-18 00:00:00", "2020-05-18 00:01:40", 100], [USER_B, 4, "2020-05-18 00:00:40", "2020-05-18 00:01:00", 20], [USER_A, 3, "2020-07-27 00:00:00", "2020-07-27 00:01:40", 100]]
            write_csv(settings.source(settings.values["sources"]["viewing"]), LOG_COLUMNS, logs)
            prepared = root / "fixture_preparation"
            prepared.mkdir()
            (prepared / "users.json").write_text(json.dumps({"user_ids": [USER_A, USER_B]}), encoding="utf-8")
            options = json.loads((ROOT / "configs/stage2.json").read_text(encoding="utf-8"))
            with Resources() as resources:
                cache = build_completion_cache(settings, options, fingerprint_files(source_paths(settings)), prepared, resources)
            events = pq.read_table(cache / "events.parquet").to_pandas()
            self.assertEqual(len(events), 3)
            self.assertEqual(events.tv_show_id.unique().tolist(), [1])
            self.assertEqual(json.loads((cache / "diagnostics.json").read_text())["excluded_future_rows"], 1)
            start = local_epoch("2020-05-18 00:00:00", settings.zone)
            completed, _ = completion_windows(settings, options, cache, [Window("val", start, start + 3600)])
            self.assertEqual(completed["val"].user_id.tolist(), [USER_A])
            self.assertEqual(completed["val"].count50.tolist(), [2])
            self.assertEqual(completed["val"].count80.tolist(), [1])
            self.assertEqual(completed["val"].seconds50.tolist(), [170])

    def test_main_cutoff_cannot_silently_reuse_other_window(self):
        settings, options, stage1, _ = load_stage2(ROOT / "configs/stage2.json")
        options = copy.deepcopy(options)
        options["folds"]["main"]["cutoff_local"] = "2020-05-19 00:00:00"
        with self.assertRaises(ValueError):
            validate_experiment(settings, options, stage1)

    def test_inner_tuning_cannot_cross_outer_cutoff(self):
        settings, options, stage1, _ = load_stage2(ROOT / "configs/stage2.json")
        options = copy.deepcopy(options)
        options["folds"]["tuning"]["validation_end_local"] = "2020-05-19 00:00:00"
        with self.assertRaises(ValueError):
            validate_experiment(settings, options, stage1)

    def test_recent_fold_cannot_use_test_behaviour(self):
        settings, options, stage1, _ = load_stage2(ROOT / "configs/stage2.json")
        options = copy.deepcopy(options)
        options["folds"]["recent_b"]["validation_end_local"] = "2020-07-28 00:00:00"
        with self.assertRaises(ValueError):
            validate_experiment(settings, options, stage1)
