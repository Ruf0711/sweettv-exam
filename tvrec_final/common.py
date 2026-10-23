"""Frozen final-stage configuration and preservation of completed experiments."""

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tvrec.cache import environment, verify_cache
from tvrec.config import Settings
from tvrec.io import atomic_json, file_hash, fingerprint_files, stable_hash
from tvrec.pipeline import source_paths
from tvrec.stage2.common import load_stage2, source_code_hash, versions
from tvrec.stage2.completion import completion_code_hash
from tvrec.time_utils import local_epoch, zone_fingerprint


@dataclass(frozen=True)
class Context:
    settings: Settings
    options: dict
    stage2_options: dict
    stage2_manifest: dict
    stage2_run: Path
    stage1_manifest_path: Path
    config_path: Path
    prepared: Path
    completion: Path
    cutoff: int


def final_code_hash():
    package = Path(__file__).parent
    return stable_hash({path.name: file_hash(path) for path in sorted(package.glob("*.py"))})


def validate_frozen(options, stage2_options, validated_parameters, base):
    required = {
        "method": "PersonalHistory", "training_cutoff_local_exclusive": "2020-07-27 00:00:00",
        "timezone": "Europe/Kyiv", "candidate_schedule": "export_arh_31-42-final.csv",
        "submission_file": "submission_history_v1.csv", "artifacts_dir": "artifacts/final",
        "k": 5, "expected_users": 1260,
    }
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError("Final export contract or authorized training cutoff changed")
    if options["history"] != validated_parameters or options["history"] != stage2_options["history"]:
        raise ValueError("Personal History hyperparameters differ from the validated model")
    if options["training_feedback"] != stage2_options["training_feedback"]:
        raise ValueError("Final feedback semantics differ from Stage 2")
    if options["primary_evaluation_proxy"] != stage2_options["primary_relevance"]:
        raise ValueError("The documented evaluation proxy changed")
    if options["example_user_ids"] != stage2_options["example_user_ids"]:
        raise ValueError("Final examples must use the original two users")
    if options["training_cutoff_local_exclusive"] != base["preparation"]["viewing_end_exclusive_local"]:
        raise ValueError("Final cutoff does not match the permitted evidence boundary")


def load_context(path):
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding="utf-8"))
    settings, stage2_options, _, stage1_path = load_stage2(path.parent / options["stage2_config"])
    if (path.parent / options["base_config"]).resolve() != settings.config_path:
        raise ValueError("Final export must reuse the same base configuration")
    completed = json.loads((path.parent / options["stage2_completion"]).read_text(encoding="utf-8"))
    if completed["status"] != "complete_and_verified":
        raise ValueError("Stage 2 is not recorded as complete and verified")
    run = Path(completed["run_dir"])
    recorded_options = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if stage2_options != recorded_options:
        raise ValueError("Stage 2 configuration differs from the completed comparison")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    validated = json.loads((run / "main/PersonalHistory/resources.json").read_text(encoding="utf-8"))["parameters"]
    recent = json.loads((run / "recent_b/PersonalHistory/resources.json").read_text(encoding="utf-8"))["parameters"]
    if validated != recent:
        raise ValueError("Completed History models disagree on parameters")
    validate_frozen(options, stage2_options, validated, settings.values)
    values = copy.deepcopy(settings.values)
    values["artifacts_dir"] = options["artifacts_dir"]
    final_settings = Settings(values, settings.root, settings.config_path)
    return Context(final_settings, options, stage2_options, manifest, run, stage1_path, path,
                   Path(manifest["stage1_preparation"]), Path(manifest["completion_cache"]),
                   local_epoch(options["training_cutoff_local_exclusive"], settings.zone))


def verify_recorded_outputs(directory, outputs):
    for name, expected in outputs.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Recorded artifact changed: {path}")


def verify_provenance(context):
    manifest = context.stage2_manifest
    if source_code_hash() != manifest["source_code_hash"]:
        raise ValueError("The validated Stage 1/2 application code changed")
    if versions() != manifest["versions"]:
        raise ValueError("Dependency versions differ from the completed Stage 2 run")
    verify_recorded_outputs(context.stage2_run, manifest["outputs"])
    fingerprints = fingerprint_files(source_paths(context.settings))
    original = json.loads((context.stage1_manifest_path.parent / "input_fingerprints.json").read_text(encoding="utf-8"))
    if fingerprints != original:
        raise ValueError("Original source content, size or modification time differs from Stage 1")
    for path in [context.prepared, context.completion]:
        cached = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        verify_cache(path, cached["identity"])
    identity = json.loads((context.completion / "manifest.json").read_text(encoding="utf-8"))["identity"]
    if identity["code"] != completion_code_hash() or identity["timezone"] != zone_fingerprint(context.settings.zone):
        raise ValueError("Completion code or timezone data changed")
    if identity["inputs"] != {name: value["sha256"] for name, value in fingerprints.items()}:
        raise ValueError("Completion evidence source identity differs from original inputs")
    return fingerprints


def verify_preservation(root, snapshot_path):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    edits = []
    for name, expected in snapshot["files"].items():
        path = root / name
        if name in snapshot["authorized_documentation_edits"]:
            backup = Path(snapshot_path).parent / "preservation" / (name + ".before")
            if not backup.is_file() or file_hash(backup) != expected["sha256"]:
                raise ValueError(f"Original documentation backup is missing or changed: {name}")
            if not path.is_file():
                raise ValueError(f"Authorized documentation file was deleted: {name}")
            if file_hash(path) != expected["sha256"]:
                edits.append(name)
            continue
        if (not path.is_file() or path.stat().st_size != expected["bytes"] or
                path.stat().st_mtime_ns != expected["mtime_ns"] or file_hash(path) != expected["sha256"]):
            raise ValueError(f"Pre-existing file changed: {name}")
    prior_artifacts = {name for name in snapshot["files"] if name.startswith("artifacts/")}
    current_artifacts = {p.relative_to(root).as_posix() for p in (root / "artifacts").rglob("*") if p.is_file() and not p.is_relative_to(root / "artifacts/final")}
    if current_artifacts != prior_artifacts:
        raise ValueError("The pre-existing Stage 1/2 artifact inventory changed")
    return {"status": "passed", "pre_existing_files_checked": len(snapshot["files"]),
            "immutable_files_unchanged": len(snapshot["files"]) - len(snapshot["authorized_documentation_edits"]),
            "hashes_sizes_and_mtimes_checked": True, "prior_artifact_inventory_unchanged": True,
            "authorized_documentation_updates": edits, "original_documentation_backups_preserved": True,
            "ai_session_records": "No existing session record is rewritten or deleted by the final workflow."}


def ensure_preservation_snapshot(root):
    """Retain an existing snapshot; initialize one for a new reproduction workspace."""
    final = root / "artifacts/final"
    destination = final / "preservation_before.json"
    if destination.exists():
        return destination
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts and not path.is_relative_to(final):
            stat = path.stat()
            files[path.relative_to(root).as_posix()] = {"bytes": stat.st_size, "sha256": file_hash(path), "mtime_ns": stat.st_mtime_ns}
    authorized = [name for name in ["README.md", ".gitignore"] if name in files]
    for name in authorized:
        backup = final / "preservation" / (name + ".before")
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write((root / name).read_bytes())
    atomic_json(destination, {"root": str(root), "created_utc": datetime.now(timezone.utc).isoformat(),
                              "authorized_documentation_edits": authorized, "files": files})
    return destination


def execution_environment():
    result = environment()
    result["stage2_packages"] = versions()
    return result
