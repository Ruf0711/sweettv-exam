import copy
import csv
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tests.common import ROOT, USER_A, USER_B, temporary_directory, write_csv
from tvrec_final.common import ensure_preservation_snapshot, validate_frozen, verify_preservation
from tvrec_final.verification import verify_csv, verify_regenerated
from tvrec_final.workflow import fit_selected, git_ignore_checks, validate_training, write_submission
from tvrec.interactions import Window


class FinalCSVTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_directory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.template = self.root / "template.csv"
        self.future = self.root / "future.csv"
        self.submission = self.root / "submission.csv"
        self.archive = self.root / "scores.npz"
        self.sidecar = self.root / "recommendations.parquet"
        write_csv(self.template, ["user_id", "tv_show_id"], [[USER_A, "0 0 0 0 0"], [USER_B, "0 0 0 0 0"]])
        write_csv(self.future, ["tv_show_id", "tv_show_title", "tv_show_category"], [[item, f"Title {item}", "Category"] for item in [0, 1, 2, 3, 4, 5, 6, 6]])
        self.rows = [[USER_A, "1 2 3 4 5"], [USER_B, "1 2 3 4 5"]]
        self.scores = np.array([[5, 4, 3, 2, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
        self.write_rows(self.rows)
        self.write_scores(self.scores)
        pq.write_table(pa.Table.from_pylist([{"user_id": user, "programme_ids": [1, 2, 3, 4, 5], "scores": list(map(float, score[:5]))} for user, score in zip([USER_A, USER_B], self.scores)]), self.sidecar)

    def write_rows(self, rows, header=None):
        write_csv(self.submission, header or ["user_id", "tv_show_id"], rows)

    def write_scores(self, scores, users=None, candidates=None):
        np.savez_compressed(self.archive, user_ids=np.array(users or [USER_A, USER_B]), candidate_ids=np.array(candidates or [1, 2, 3, 4, 5, 6]), scores=scores)

    def verify(self, sidecar=True):
        return verify_csv(self.submission, self.template, self.future, self.archive, self.sidecar if sidecar else None, expected_users=2)

    def test_disk_readback_exact_large_user_ids_and_ties(self):
        result = self.verify()
        self.assertEqual(result["data_rows"], 2)
        self.assertEqual(result["candidate_count_from_actual_future_csv"], 6)
        self.assertEqual(result["selected_adjacent_score_ties"], 4)

    def test_extra_or_inexact_header_is_rejected(self):
        for header in [["user_id", "tv_show_id", "index"], ["User_id", "tv_show_id"]]:
            with self.subTest(header=header):
                self.write_rows(self.rows, header)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_extra_column_or_missing_row_is_rejected(self):
        for rows in [[self.rows[0]], [self.rows[0] + ["extra"], self.rows[1]]]:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_user_reordering_duplicates_and_numeric_corruption_are_rejected(self):
        for rows in [self.rows[::-1], [self.rows[0], self.rows[0]], [["1.8440717181607e19", self.rows[0][1]], self.rows[1]]]:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_invalid_zero_duplicate_noninteger_empty_and_spacing_cells(self):
        for cell in ["0 2 3 4 5", "1 1 3 4 5", "1.0 2 3 4 5", "1 -2 3 4 5", "", "1  2 3 4 5", "1 2 3 4", "1 2 3 4 5 6"]:
            with self.subTest(cell=cell):
                self.write_rows([[USER_A, cell], self.rows[1]])
                with self.assertRaises(ValueError):
                    self.verify()

    def test_candidate_membership_is_read_from_actual_future_csv(self):
        write_csv(self.future, ["tv_show_id", "tv_show_title", "tv_show_category"], [[item, "Title", "Category"] for item in [1, 2, 3, 4, 6]])
        with self.assertRaises(ValueError):
            self.verify()

    def test_nonfinite_scores_even_outside_top5_are_rejected(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            with self.subTest(invalid=invalid):
                scores = self.scores.copy()
                scores[0, 5] = invalid
                self.write_scores(scores)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_descending_score_and_tie_order_are_checked_independently(self):
        for rows in [[[USER_A, "2 1 3 4 5"], self.rows[1]], [self.rows[0], [USER_B, "2 1 3 4 5"]]]:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    self.verify(False)

    def test_score_archive_user_order_must_match(self):
        self.write_scores(self.scores, users=[USER_B, USER_A])
        with self.assertRaises(ValueError):
            self.verify()

    def test_score_sidecar_must_agree_with_csv(self):
        pq.write_table(pa.Table.from_pylist([{"user_id": user, "programme_ids": [1, 2, 3, 4, 6], "scores": [5., 4., 3., 2., 1.]} for user in [USER_A, USER_B]]), self.sidecar)
        with self.assertRaises(ValueError):
            self.verify()

    def test_refit_requires_exact_scores_and_byte_identical_csv(self):
        content = self.submission.read_bytes()
        self.assertTrue(verify_regenerated(self.submission, content, self.scores, self.scores.copy())["submission_regeneration_byte_identical"])
        with self.assertRaises(ValueError):
            verify_regenerated(self.submission, content + b"\n", self.scores, self.scores)
        with self.assertRaises(ValueError):
            verify_regenerated(self.submission, content, self.scores, self.scores + 1)

    def test_different_existing_submission_is_not_overwritten(self):
        old = self.submission.read_bytes()
        old_mtime = self.submission.stat().st_mtime_ns
        write_submission(self.submission, old)
        self.assertEqual(old_mtime, self.submission.stat().st_mtime_ns)
        with self.assertRaises(FileExistsError):
            write_submission(self.submission, b"different user work")
        self.assertEqual(self.submission.read_bytes(), old)


class FinalTrainingTests(unittest.TestCase):
    def test_git_ignore_checks_use_literal_paths_without_windows_crlf(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
            result = git_ignore_checks(root)
            self.assertFalse(result["cases"]["submission_history_v1.csv"]["ignored"])
            self.assertTrue(result["cases"]["kaggle.json"]["ignored"])
            self.assertTrue(result["cases"][".env"]["ignored"])
            self.assertTrue(result["cases"]["large_model.npz"]["ignored"])

    def test_frozen_parameters_and_cutoff_reject_changes(self):
        options = json.loads((ROOT / "configs/final_history_v1.json").read_text(encoding="utf-8"))
        stage2 = json.loads((ROOT / "configs/stage2.json").read_text(encoding="utf-8"))
        base = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
        validate_frozen(options, stage2, stage2["history"], base)
        for key, value in [("history", {"half_life_days": 30., "duration_coefficient": .25}), ("training_cutoff_local_exclusive", "2020-07-28 00:00:00"), ("candidate_schedule", "export_arh_21-30-final.csv")]:
            with self.subTest(key=key):
                changed = copy.deepcopy(options)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_frozen(changed, stage2, stage2["history"], base)

    def test_selected_history_refit_short_list_and_cold_user_fallback(self):
        feedback = pd.DataFrame({"user_id": [USER_A], "tv_show_id": [2], "count50": [3], "count80": [2], "seconds50": [9000], "last_epoch": [90]})
        cumulative = pd.DataFrame({"user_id": [USER_A, USER_B, "3"], "tv_show_id": [2, 6, 1], "seconds": [9000., 60., 60.]})
        _, scores, rows, payload, _ = fit_selected(feedback, cumulative, list(range(1, 7)), [USER_A, USER_B], 100, {"half_life_days": 90., "duration_coefficient": .25})
        self.assertEqual(rows[0].programme_ids, (2, 1, 6, 3, 4))
        self.assertEqual(rows[1].programme_ids, (1, 2, 6, 3, 4))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(payload.startswith(b"user_id,tv_show_id\n"))
        validate_training(feedback, cumulative, Window("all", 0, 100))
        feedback.loc[0, "last_epoch"] = 101
        with self.assertRaises(ValueError):
            validate_training(feedback, cumulative, Window("all", 0, 100))

    def test_preservation_detects_original_data_and_prior_artifact_changes(self):
        with temporary_directory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "data/original.csv").write_bytes(b"original\n")
            (root / "artifacts/stage2").mkdir(parents=True)
            (root / "artifacts/stage2/result.json").write_bytes(b"{}\n")
            snapshot = ensure_preservation_snapshot(root)
            self.assertEqual(ensure_preservation_snapshot(root), snapshot)
            self.assertEqual(verify_preservation(root, snapshot)["status"], "passed")
            (root / "data/original.csv").write_bytes(b"modified\n")
            with self.assertRaises(ValueError):
                verify_preservation(root, snapshot)
