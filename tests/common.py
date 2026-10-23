import copy
import csv
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tvrec.config import load_settings
from tvrec.io import LOG_COLUMNS, SCHEDULE_COLUMNS, TEMPLATE_COLUMNS
from tvrec.preparation import EVIDENCE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
USER_A = "18440717181607000261"
USER_B = "8377619604347126107"


def temporary_directory():
    local = os.environ.get("LOCALAPPDATA")
    preferred = Path(local) / "Temp" / "opencode" if local else None
    return tempfile.TemporaryDirectory(dir=preferred if preferred and preferred.is_dir() else None)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def schedule_row(channel, start, duration, item, title=None):
    return [channel, f"Channel {channel}", start, duration, title or f"Programme {item}", item, "Category", "Genre", "", "", "2020", "", ""]


def fixture_project(root):
    values = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    values["project_root"] = str(root)
    values["sources"]["archive"] = None
    values["preparation"].update({"csv_chunk_rows": 3, "join_chunk_rows": 2, "parquet_batch_rows": 2, "progress_every_rows": 100})
    values["evaluation"].update({"expected_cohort_size": 2, "example_user_ids": [USER_A, USER_B]})
    data = root / "data"
    schedules = values["sources"]["schedules"]
    write_csv(data / schedules[0], SCHEDULE_COLUMNS, [schedule_row(i + 2, "17.05.2020 23:50:00", 1200, i) for i in range(1, 7)])
    rows = [schedule_row(i + 2, "18.05.2020 00:10:00", 1200, i) for i in range(1, 7)]
    rows += [schedule_row(99, "18.05.2020 00:00:00", 1200, 0), schedule_row(98, "18.05.2020 00:00:00", 0, "")]
    write_csv(data / schedules[1], SCHEDULE_COLUMNS, rows)
    write_csv(data / schedules[2], SCHEDULE_COLUMNS, [schedule_row(i + 2, "27.07.2020 00:00:00", 3600, 100 + i) for i in range(1, 7)])
    logs = [
        [USER_A, 3, "2020-05-17 23:55:00", "2020-05-18 00:05:00", 600],
        [USER_A, 3, "2020-05-17 23:55:00", "2020-05-18 00:05:00", 600],
        [USER_A, 3, "2020-05-17 23:58:00", "2020-05-18 00:08:00", 600],
        [USER_B, 4, "2020-05-17 23:56:00", "2020-05-17 23:59:00", 180],
        [USER_A, 4, "2020-05-18 00:11:00", "2020-05-18 00:16:00", 300],
        [USER_B, 777, "2020-05-18 00:00:00", "2020-05-18 00:02:00", 120],
        [USER_B, 99, "2020-05-18 00:00:00", "2020-05-18 00:01:00", 60],
        [USER_B, 5, "2020-05-18 00:11:00", "2020-05-18 00:13:00", 120],
    ]
    write_csv(data / values["sources"]["viewing"], LOG_COLUMNS, logs)
    write_csv(data / values["sources"]["template"], TEMPLATE_COLUMNS, [[USER_A, "0 0 0 0 0"], [USER_B, "0 0 0 0 0"]])
    config = root / "config.json"
    config.write_text(json.dumps(values), encoding="utf-8")
    return load_settings(config), logs


def write_evidence(path, rows):
    records = []
    for user, item, view, start, end, denominator in rows:
        records.append({
            "user_id": user, "tv_show_id": item, "channel_id": 3, "view_id": view, "segment_id": len(records),
            "start_time": pd.Timestamp(start, unit="s", tz="UTC"), "end_time": pd.Timestamp(end, unit="s", tz="UTC"),
            "allocation_denominator": denominator, "overlap_seconds": end - start, "credited_seconds": (end - start) / denominator,
        })
    pq.write_table(pa.Table.from_pylist(records, schema=EVIDENCE_SCHEMA), path)
