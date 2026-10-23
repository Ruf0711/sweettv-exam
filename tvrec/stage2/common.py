import copy
import importlib.metadata
import json
from pathlib import Path

from tvrec.config import Settings, load_settings
from tvrec.io import file_hash, stable_hash
from tvrec.time_utils import local_epoch


def source_code_hash():
    package = Path(__file__).resolve().parents[1]
    return stable_hash({str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))})


def load_stage2(path):
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "channel_switch_policy": "truncate_previous_run_at_next_different_channel_start_without_resuming",
        "simultaneous_different_channel_starts": "unknown_break",
        "broadcast_conflict_policy": "exclude_ambiguous_seconds",
        "denominator": "full_utc_broadcast_duration_even_at_window_boundaries",
        "repeated_broadcasts": "distinct_instances; any_single_sitting_may_qualify; count_each_user_broadcast_once",
    }
    if any(options["completion"].get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported completion policy")
    base = load_settings(path.parent / options["base_config"])
    values = copy.deepcopy(base.values)
    values["artifacts_dir"] = options["artifacts_subdir"]
    settings = Settings(values, base.root, base.config_path)
    old_manifest = path.parent / options["stage1_run_manifest"]
    manifest = json.loads(old_manifest.read_text(encoding="utf-8"))
    validate_experiment(settings, options, manifest)
    return settings, options, manifest, old_manifest.resolve()


def validate_experiment(settings, options, stage1):
    """Reject configurations that would silently reinterpret preserved caches."""
    if options["primary_relevance"] not in options["relevance_rules"]:
        raise ValueError("Primary relevance must be one of the evaluated rules")
    original = stage1["identity"]["config"]
    for key in ["sources", "timezone", "preparation", "interactions"]:
        if settings.values[key] != original[key]:
            raise ValueError(f"Stage 1 cache configuration changed: {key}")
    main = original["folds"]["main"]
    for key in ["cutoff_local", "validation_end_local", "candidate_policy"]:
        if options["folds"]["main"][key] != main[key]:
            raise ValueError(f"Main fold must preserve the Stage 1 {key}")
    boundaries = {}
    stop = local_epoch(settings.values["preparation"]["viewing_end_exclusive_local"], settings.zone)
    for name in ["tuning", "main", "recent_b"]:
        fold = options["folds"][name]
        start = local_epoch(fold["cutoff_local"], settings.zone)
        end = local_epoch(fold["validation_end_local"], settings.zone)
        if not start < end <= stop:
            raise ValueError(f"Invalid or test-crossing fold: {name}")
        boundaries[name] = (start, end)
    if boundaries["tuning"][1] > min(boundaries[name][0] for name in ["main", "recent_b"]):
        raise ValueError("Inner tuning must finish before both outer cutoffs")
    if options["threads"] != 1:
        raise ValueError("This reproducible Stage 2 configuration requires one thread")


def versions():
    return {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "implicit", "threadpoolctl", "tzdata"]}
