import csv
import io
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tvrec.io import TEMPLATE_COLUMNS, csv_chunks, integer_column, read_template
from tvrec.recommendations import Recommendation, format_submission, validate_recommendations
from tvrec.time_utils import LOG_FORMAT, SCHEDULE_FORMAT, parse_local_times
from tests.common import USER_A, USER_B, temporary_directory, write_csv


class TimeAndInputTests(unittest.TestCase):
    def test_dst_utc_elapsed_duration(self):
        values = pd.Series(["2020-03-29 02:30:00", "2020-03-29 04:30:00"])
        times = parse_local_times(values, LOG_FORMAT, ZoneInfo("Europe/Kyiv"))
        self.assertEqual(int(times[1] - times[0]), 3600)

    def test_schedule_and_log_formats_agree(self):
        zone = ZoneInfo("Europe/Kyiv")
        left = parse_local_times(pd.Series(["2020-05-18 00:00:00"]), LOG_FORMAT, zone)
        right = parse_local_times(pd.Series(["18.05.2020 00:00:00"]), SCHEDULE_FORMAT, zone)
        self.assertEqual(int(left[0]), int(right[0]))

    def test_nonexistent_local_time_rejected(self):
        with self.assertRaises(ValueError):
            parse_local_times(pd.Series(["2020-03-29 03:30:00"]), LOG_FORMAT, ZoneInfo("Europe/Kyiv"))

    def test_ambiguous_local_time_rejected(self):
        with self.assertRaises(ValueError):
            parse_local_times(pd.Series(["2020-10-25 03:30:00"]), LOG_FORMAT, ZoneInfo("Europe/Kyiv"))

    def test_inexact_timestamp_format_rejected(self):
        with self.assertRaises(ValueError):
            parse_local_times(pd.Series(["2020-5-18 00:00:00"]), LOG_FORMAT, ZoneInfo("Europe/Kyiv"))

    def test_large_user_ids_and_template_order(self):
        with temporary_directory() as temporary:
            path = Path(temporary) / "template.csv"
            write_csv(path, TEMPLATE_COLUMNS, [[USER_A, "0 0 0 0 0"], [USER_B, "0 0 0 0 0"]])
            frame = read_template(path)
            self.assertEqual(frame.user_id.tolist(), [USER_A, USER_B])
            self.assertTrue(all(isinstance(value, str) for value in frame.user_id))
            self.assertGreater(int(frame.user_id.iloc[0]), 2**63 - 1)

    def test_exact_header_required(self):
        with temporary_directory() as temporary:
            path = Path(temporary) / "wrong.csv"
            write_csv(path, ["user_id", "programme"], [[USER_A, "0 0 0 0 0"]])
            with self.assertRaises(ValueError):
                list(csv_chunks(path, TEMPLATE_COLUMNS, 10))

    def test_duplicate_template_users_rejected(self):
        with temporary_directory() as temporary:
            path = Path(temporary) / "template.csv"
            write_csv(path, TEMPLATE_COLUMNS, [[USER_A, "0 0 0 0 0"], [USER_A, "0 0 0 0 0"]])
            with self.assertRaises(ValueError):
                read_template(path)

    def test_missing_programme_id_is_unknown_integer(self):
        self.assertEqual(integer_column(pd.Series(["", "0", "17"]), "tv_show_id", missing_zero=True).tolist(), [0, 0, 17])

    def test_float_programme_id_rejected(self):
        with self.assertRaises(ValueError):
            integer_column(pd.Series(["17.0"]), "tv_show_id")


class FormattingTests(unittest.TestCase):
    def setUp(self):
        self.cohort = [USER_A, USER_B]
        self.rows = [Recommendation(user, (1, 2, 3, 4, 5), (5., 4., 3., 2., 1.)) for user in self.cohort]

    def test_five_positive_unique_ids_and_exact_csv_contract(self):
        text = format_submission(self.cohort, self.rows, range(1, 7))
        parsed = list(csv.reader(io.StringIO(text)))
        self.assertEqual(parsed[0], ["user_id", "tv_show_id"])
        self.assertEqual(parsed[1:], [[USER_A, "1 2 3 4 5"], [USER_B, "1 2 3 4 5"]])

    def test_duplicate_prediction_rejected(self):
        self.rows[0] = Recommendation(USER_A, (1, 1, 3, 4, 5), (1.,) * 5)
        with self.assertRaises(ValueError):
            validate_recommendations(self.cohort, self.rows, range(1, 7))

    def test_zero_prediction_rejected(self):
        self.rows[0] = Recommendation(USER_A, (0, 2, 3, 4, 5), (1.,) * 5)
        with self.assertRaises(ValueError):
            format_submission(self.cohort, self.rows, range(0, 7))

    def test_short_prediction_rejected(self):
        self.rows[0] = Recommendation(USER_A, (1, 2, 3, 4), (1.,) * 4)
        with self.assertRaises(ValueError):
            format_submission(self.cohort, self.rows, range(1, 7))

    def test_outside_catalogue_rejected(self):
        with self.assertRaises(ValueError):
            format_submission(self.cohort, self.rows, [1, 2, 3, 4, 9])

    def test_reordered_users_rejected(self):
        with self.assertRaises(ValueError):
            format_submission(self.cohort, self.rows[::-1], range(1, 7))
