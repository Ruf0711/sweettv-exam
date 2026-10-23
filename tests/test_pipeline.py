import copy
import json
import unittest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tvrec.cache import complete_manifest, preparation_identity, verify_cache
from tvrec.config import Settings, load_settings
from tvrec.interactions import Window, aggregate_portion, aggregate_windows, evidence_batches, finish_aggregation, historical_metadata
from tvrec.io import LOG_COLUMNS, SCHEDULE_COLUMNS, fingerprint_files
from tvrec.pipeline import fold_windows, run_pipeline, source_paths, verify_run
from tvrec.time_utils import local_epoch
from tests.common import USER_A, USER_B, fixture_project, schedule_row, temporary_directory, write_csv, write_evidence


class CacheTests(unittest.TestCase):
    def test_changed_source_hash_invalidates_preparation(self):
        with temporary_directory() as temporary:
            settings, _ = fixture_project(Path(temporary))
            fingerprints = fingerprint_files(source_paths(settings))
            first = preparation_identity(settings, fingerprints)
            changed = copy.deepcopy(fingerprints)
            changed[settings.values["sources"]["viewing"]]["sha256"] = "different"
            self.assertNotEqual(first, preparation_identity(settings, changed))

    def test_threshold_changes_reuse_join_but_invalidate_window_cache(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            settings, _ = fixture_project(root)
            fingerprints = fingerprint_files(source_paths(settings))
            first_identity = preparation_identity(settings, fingerprints)
            prepared = root / "prepared"
            prepared.mkdir()
            write_evidence(prepared / "evidence.parquet", [(USER_A, 1, 0, 0, 100, 1)])
            first = aggregate_windows(settings, prepared, "fixture", [Window("train", 0, 100)])["train"]
            values = copy.deepcopy(settings.values)
            values["interactions"]["relevance_seconds"] = 300
            changed = Settings(values, settings.root, settings.config_path)
            self.assertEqual(first_identity, preparation_identity(changed, fingerprints))
            second = aggregate_windows(changed, prepared, "fixture", [Window("train", 0, 100)])["train"]
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual(first["frame"].relevant.tolist(), [True])
            self.assertEqual(second["frame"].relevant.tolist(), [False])

    def test_cache_content_corruption_is_rejected(self):
        with temporary_directory() as temporary:
            directory = Path(temporary)
            path = directory / "test.txt"
            path.write_text("original", encoding="utf-8")
            complete_manifest(directory, {"key": 1}, ["test.txt"])
            verify_cache(directory, {"key": 1})
            path.write_text("modified", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_cache(directory, {"key": 1})

    def test_incomplete_cache_is_rejected(self):
        with temporary_directory() as temporary:
            with self.assertRaises(ValueError):
                verify_cache(Path(temporary), {"key": 1})

    def test_view_count_survives_parquet_batch_boundary(self):
        with temporary_directory() as temporary:
            path = Path(temporary) / "evidence.parquet"
            write_evidence(path, [(USER_A, 1, 0, 0, 10, 1), (USER_A, 1, 0, 10, 20, 1), (USER_A, 1, 1, 30, 40, 1)])
            window = Window("all", 0, 100)
            parts = [aggregate_portion(batch, window) for batch in evidence_batches(path, 1)]
            output = finish_aggregation(parts, 60, None)
            self.assertEqual(output.seconds.tolist(), [30.])
            self.assertEqual(output.fragment_count.tolist(), [3])
            self.assertEqual(output.view_count.tolist(), [2])


class PipelineTests(unittest.TestCase):
    def test_complete_toy_pipeline_metrics_integrity_and_cache_reuse(self):
        with temporary_directory() as temporary:
            settings, _ = fixture_project(Path(temporary))
            first = run_pipeline(settings)
            self.assertFalse(first["preparation_cache_hit"])
            self.assertAlmostEqual(first["map_at_5_fixed"], 2 / 3)
            self.assertEqual(first["evaluable_users"], 2)
            result = verify_run(settings, Path(first["run_dir"]))
            self.assertEqual(result["status"], "passed")
            second = run_pipeline(settings)
            self.assertTrue(second["preparation_cache_hit"])
            self.assertEqual(first["map_at_5_fixed"], second["map_at_5_fixed"])
            manifest = json.loads((Path(second["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(value["cache_hit"] for value in manifest["windows"].values()))
            evidence = pq.read_table(Path(first["preparation_path"]) / "evidence.parquet")
            self.assertEqual(evidence.schema.field("start_time").type.tz, "UTC")
            self.assertNotIn(0, evidence.column("tv_show_id").to_pylist())
            self.assertFalse(list(settings.artifacts_dir.rglob("submission.csv")))

    def test_pipeline_leakage_regression_and_future_metadata_guard(self):
        with temporary_directory() as temporary:
            settings, logs = fixture_project(Path(temporary))
            first = run_pipeline(settings)
            original_manifest = json.loads((Path(first["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
            train_before = pq.read_table(Path(original_manifest["windows"]["train"]["path"]) / "interactions.parquet").to_pandas()
            # This joins an ongoing session after the cutoff and adds an entire
            # prohibited test-period session; neither can alter training support.
            logs += [[USER_A, 3, "2020-05-18 00:08:00", "2020-05-18 00:10:00", 120], [USER_B, 5, "2020-07-27 00:00:00", "2020-07-27 00:02:00", 120]]
            write_csv(settings.source(settings.values["sources"]["viewing"]), LOG_COLUMNS, logs)
            future_file = settings.values["sources"]["schedules"][2]
            future_rows = [schedule_row(i + 2, "27.07.2020 00:00:00", 3600, 100 + i) for i in range(1, 7)]
            future_rows.append(schedule_row(3, "27.07.2020 01:00:00", 3600, 1, "Future-only replacement title"))
            write_csv(settings.source(future_file), SCHEDULE_COLUMNS, future_rows)
            second = run_pipeline(settings)
            second_manifest = json.loads((Path(second["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
            train_after = pq.read_table(Path(second_manifest["windows"]["train"]["path"]) / "interactions.parquet").to_pandas()
            pd.testing.assert_frame_equal(train_before, train_after)
            before_predictions = pq.read_table(Path(first["run_dir"]) / "validation_recommendations.parquet").to_pylist()
            after_predictions = pq.read_table(Path(second["run_dir"]) / "validation_recommendations.parquet").to_pylist()
            self.assertEqual(before_predictions, after_predictions)
            before_candidates = json.loads((Path(first["run_dir"]) / "candidate_ids.json").read_text(encoding="utf-8"))["validation"]
            after_candidates = json.loads((Path(second["run_dir"]) / "candidate_ids.json").read_text(encoding="utf-8"))["validation"]
            self.assertEqual(before_candidates, after_candidates)
            diagnostics = json.loads((Path(second["preparation_path"]) / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["raw"]["excluded_future_rows"], 1)
            metadata = historical_metadata(Path(second["preparation_path"]), local_epoch("2020-05-18 00:00:00", settings.zone))
            self.assertNotIn("Future-only replacement title", metadata.tv_show_title.tolist())

    def test_recent_fold_and_recency_interfaces(self):
        with temporary_directory() as temporary:
            settings, _ = fixture_project(Path(temporary))
            diagnostics = {"raw": {"observed_start_epoch": local_epoch("2020-03-09 00:00:00", settings.zone)}}
            for name in ["recent_a", "recent_b", "twelve_week_stress"]:
                train, validation, support = fold_windows(settings, name, diagnostics)
                self.assertEqual(train.end, validation.start)
                self.assertEqual(support, train)
            values = copy.deepcopy(settings.values)
            values["popularity"]["lookback_days"] = 28
            changed = Settings(values, settings.root, settings.config_path)
            train, validation, support = fold_windows(changed, "main", diagnostics)
            self.assertEqual(support.start, local_epoch("2020-04-20 00:00:00", changed.zone))
            self.assertEqual(support.end, train.end)

    def test_artifacts_cannot_be_written_inside_original_data(self):
        with temporary_directory() as temporary:
            settings, _ = fixture_project(Path(temporary))
            values = copy.deepcopy(settings.values)
            values["artifacts_dir"] = "data/generated"
            settings.config_path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(settings.config_path)
