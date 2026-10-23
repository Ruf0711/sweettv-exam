import json
import sys
from pathlib import Path

from tvrec.io import file_hash, stable_hash
from tvrec.time_utils import local_epoch
from tvrec_final.common import load_context


ROOT = Path(__file__).resolve().parents[1]


def code_hash():
    return stable_hash({path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))})


def lightgbm():
    vendor = ROOT / "artifacts/stage4/vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import lightgbm as lgb
    if lgb.__version__ != "4.6.0":
        raise ValueError("Stage 4 requires LightGBM 4.6.0")
    return lgb


def load_options(path):
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8"))
    context = load_context(path.parent / options["final_v1_config"])
    if options["submission_file"] != "submission_ranker_v2.csv" or options["artifacts_dir"] != "artifacts/stage4":
        raise ValueError("Stage 4 must remain isolated from existing deliverables")
    if not 200 <= sum(options["retrieval_quotas"].values()) <= 300:
        raise ValueError("Retrieval must produce 200-300 candidates")
    return context, options


def forecasts(context, options):
    result = []
    for spec in [*options["pseudo_forecasts"], options["inner"]]:
        result.append({"name": spec["name"], "cutoff": local_epoch(spec["cutoff"], context.settings.zone),
                       "end": local_epoch(spec["end"], context.settings.zone), "candidate_policy": {"kind": "window_overlap"}})
    for name in ["main", "recent_b"]:
        record = json.loads((context.stage2_run / name / "context.json").read_text(encoding="utf-8"))
        result.append({"name": name, "cutoff": record["train_window"]["end_exclusive_epoch"],
                       "end": record["validation_window"]["end_exclusive_epoch"], "candidate_policy": record["candidate_policy"],
                       "recorded_candidates": record["candidate_ids"]})
    return result


def admissible_training(specs, cutoff):
    chosen = [spec for spec in specs if spec["end"] <= cutoff]
    for spec in chosen:
        assert_training_boundary(spec, cutoff)
    if not chosen:
        raise ValueError("No strictly chronological supervised training forecasts")
    return chosen


def assert_training_boundary(spec, cutoff):
    if not spec["cutoff"] < spec["end"] <= cutoff:
        raise ValueError("Training-label window overlaps the deployment/validation cutoff")


def preservation(destination):
    snapshot = json.loads((destination / "preservation_before.json").read_text(encoding="utf-8"))
    for name, expected in snapshot["files"].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or path.stat().st_mtime_ns != expected["mtime_ns"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Preserved file changed: {name}")
    old = {name for name in snapshot["files"] if name.startswith("artifacts/")}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "artifacts").rglob("*") if p.is_file() and not p.is_relative_to(destination)}
    if old != actual:
        raise ValueError("Prior artifact inventory changed")
    return {"status": "passed", "files_checked": len(snapshot["files"]), "hash_size_mtime_and_inventory_unchanged": True,
            "history_v1_sha256": file_hash(ROOT / "submission_history_v1.csv"), "ai_session_backup_sha256": file_hash(ROOT / "exam_session_backup.json")}
