"""Exact source schemas, lossless identifiers, and atomic artifact writes."""

import csv
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOG_COLUMNS = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]
SCHEDULE_COLUMNS = [
    "channel_id", "channel_title", "start_time", "duration", "tv_show_title", "tv_show_id",
    "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
    "year_of_production", "director", "actors",
]
TEMPLATE_COLUMNS = ["user_id", "tv_show_id"]
UTC_TYPE = pa.timestamp("s", tz="UTC")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def fingerprint_files(paths: list[Path]) -> dict:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path), "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    }


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_chunks(path: Path, columns: list[str], chunk_rows: int):
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        actual = next(csv.reader(stream), None)
    if actual != columns:
        raise ValueError(f"Unexpected columns in {path.name}: {actual!r}; expected {columns!r}")
    yield from pd.read_csv(
        path, sep=",", quotechar='"', encoding="utf-8", encoding_errors="strict",
        dtype=str, keep_default_na=False, na_filter=False, chunksize=chunk_rows,
        on_bad_lines="error", low_memory=False,
    )


def integer_column(values: pd.Series, name: str, *, missing_zero: bool = False, signed: bool = False, bits: int = 64) -> np.ndarray:
    text = values.astype(str)
    if missing_zero:
        text = text.mask(text == "", "0")
    pattern = r"-?\d+" if signed else r"\d+"
    if not text.str.fullmatch(pattern).all():
        raise ValueError(f"Invalid integer in {name}")
    numbers = pd.to_numeric(text, errors="raise")
    limits = np.iinfo(f"int{bits}")
    if len(numbers) and (numbers.min() < limits.min or numbers.max() > limits.max):
        raise ValueError(f"{name} exceeds int{bits}")
    return numbers.to_numpy(dtype=f"int{bits}")


def validate_user_ids(values: pd.Series) -> None:
    if not values.map(lambda value: isinstance(value, str)).all() or not values.str.fullmatch(r"\d+").all():
        raise ValueError("User IDs must be nonempty, exact digit strings")


def read_template(path: Path) -> pd.DataFrame:
    frame = pd.concat(list(csv_chunks(path, TEMPLATE_COLUMNS, 10000)), ignore_index=True)
    validate_user_ids(frame.user_id)
    if frame.user_id.duplicated().any():
        raise ValueError("The template contains duplicate user IDs")
    if not frame.tv_show_id.str.fullmatch(r"\d+( \d+){4}").all():
        raise ValueError("Template predictions must contain exactly five single-space-separated integer tokens")
    return frame


def arrow_utc(values: np.ndarray) -> pa.Array:
    return pa.array(values, type=pa.int64()).cast(UTC_TYPE)
