"""Disk-first CSV verification independent of production ranking/formatting."""

import csv
import heapq
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tvrec.io import file_hash

HEADER = ["user_id", "tv_show_id"]


def read_template_users(path):
    with Path(path).open("r", encoding="utf-8", errors="strict", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows or rows[0] != HEADER or any(len(row) != 2 for row in rows[1:]):
        raise ValueError("Unexpected template header or row width")
    users = [row[0] for row in rows[1:]]
    if not all(re.fullmatch(r"[0-9]+", user) for user in users) or len(set(users)) != len(users):
        raise ValueError("Template users must be unique exact digit strings")
    return users


def read_future_catalogue(path):
    metadata = {}
    with Path(path).open("r", encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or not {"tv_show_id", "tv_show_title", "tv_show_category"}.issubset(reader.fieldnames):
            raise ValueError("The actual future schedule lacks programme columns")
        for row in reader:
            text = row["tv_show_id"]
            if text == "":
                continue
            if not re.fullmatch(r"[0-9]+", text):
                raise ValueError("Non-integer programme ID in the actual future schedule")
            item = int(text)
            if item > 0:
                metadata.setdefault(item, {"title": row["tv_show_title"], "category": row["tv_show_category"]})
    if len(metadata) < 5:
        raise ValueError("The future catalogue has fewer than five positive programmes")
    return metadata


def verify_csv(submission, template, future_schedule, score_archive, recommendations=None, expected_users=1260):
    """Read every row from disk and independently recover Top-5 from all scores."""
    cohort = read_template_users(template)
    if len(cohort) != expected_users:
        raise ValueError("Unexpected template cohort size")
    permitted = set(read_future_catalogue(future_schedule))
    with Path(submission).open("r", encoding="utf-8", errors="strict", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != HEADER:
            raise ValueError("Submission must have the exact two-column header")
        rows = list(reader)
    if len(rows) != expected_users or any(len(row) != 2 for row in rows):
        raise ValueError("Submission row count or column count is incorrect")
    users = [row[0] for row in rows]
    if users != cohort or len(set(users)) != expected_users:
        raise ValueError("Submission user strings/order differ from the original template")
    parsed = []
    for user, text in rows:
        if not re.fullmatch(r"[1-9][0-9]*( [1-9][0-9]*){4}", text):
            raise ValueError(f"Invalid five-positive-integer prediction cell: {user}")
        items = tuple(int(token) for token in text.split(" "))
        if len(set(items)) != 5 or not set(items).issubset(permitted):
            raise ValueError(f"Duplicate or out-of-catalogue prediction: {user}")
        parsed.append(items)
    with np.load(score_archive, allow_pickle=False) as saved:
        if set(saved.files) != {"user_ids", "candidate_ids", "scores"}:
            raise ValueError("Unexpected score archive fields")
        if saved["user_ids"].tolist() != cohort:
            raise ValueError("Score archive users/order differ from the template")
        candidates = saved["candidate_ids"]
        if candidates.tolist() != sorted(permitted):
            raise ValueError("Saved candidate columns differ from the actual future schedule")
        scores = saved["scores"]
        if scores.shape != (expected_users, len(permitted)) or not np.isfinite(scores).all():
            raise ValueError("Missing, malformed or non-finite candidate scores")
        sidecar = pq.read_table(recommendations).to_pylist() if recommendations is not None else None
        if sidecar is not None and [row["user_id"] for row in sidecar] != cohort:
            raise ValueError("Recommendation sidecar users/order differ")
        ties = 0
        for index, items in enumerate(parsed):
            top = heapq.nsmallest(5, range(len(candidates)), key=lambda j: (-float(scores[index, j]), int(candidates[j])))
            expected = tuple(int(candidates[j]) for j in top)
            if items != expected:
                raise ValueError("CSV Top-5 differs from descending scores / ascending-ID ties")
            selected_scores = [float(scores[index, j]) for j in top]
            ties += sum(selected_scores[j] == selected_scores[j + 1] for j in range(4))
            if sidecar is not None:
                if tuple(sidecar[index]["programme_ids"]) != items or sidecar[index]["scores"] != selected_scores:
                    raise ValueError("CSV, full candidate scores and Top-5 sidecar disagree")
    return {"status": "passed", "submission_sha256": file_hash(Path(submission)),
            "data_rows": len(rows), "unique_users": len(set(users)), "recommendation_positions": len(rows) * 5,
            "candidate_count_from_actual_future_csv": len(permitted), "exact_header": HEADER,
            "exact_template_user_strings_and_order": True, "all_five_distinct_positive_and_available": True,
            "all_candidate_and_top5_scores_finite": True, "descending_score_and_ascending_id_ties": True,
            "all_top5_recomputed_independently_from_scores": True, "selected_adjacent_score_ties": ties,
            "csv_contains_no_score_or_index_column": True}


def verify_regenerated(submission, regenerated_bytes, original_scores, regenerated_scores):
    if not np.array_equal(original_scores, regenerated_scores):
        raise ValueError("Refitted candidate scores are not exactly reproducible")
    if Path(submission).read_bytes() != regenerated_bytes:
        raise ValueError("Regenerated submission is not byte-identical")
    return {"fresh_model_refit": True, "all_candidate_scores_exactly_equal": True,
            "submission_regeneration_byte_identical": True, "sha256": file_hash(Path(submission))}
