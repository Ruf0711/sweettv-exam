"""Content-addressed caches with atomic completion and verified output hashes."""

import importlib.metadata
import json
import platform
import shutil
import uuid
from pathlib import Path

from .io import atomic_json, file_hash, stable_hash
from .time_utils import zone_fingerprint

PREPARATION_SCHEMA_VERSION = 1
PREPARATION_MODULES = ["cache.py", "config.py", "io.py", "time_utils.py", "intervals.py", "preparation.py"]


def code_hash(names=None):
    package = Path(__file__).parent
    paths = sorted(package.glob("*.py")) if names is None else [package / name for name in names]
    return stable_hash({path.name: file_hash(path) for path in paths})


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "tzdata", "psutil"]},
    }


def preparation_identity(settings, fingerprints):
    source_names = [settings.values["sources"]["viewing"], *settings.values["sources"]["schedules"], settings.values["sources"]["template"]]
    computational = dict(settings.values["preparation"])
    computational.pop("progress_every_rows", None)
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "source_content": {name: {"sha256": fingerprints[name]["sha256"], "bytes": fingerprints[name]["bytes"]} for name in source_names},
        "timezone": zone_fingerprint(settings.zone),
        "timezone_assumption": settings.values["timezone_assumption"],
        "preparation": computational,
        "preparation_code_hash": code_hash(PREPARATION_MODULES),
        "environment": environment(),
    }


def complete_manifest(directory: Path, identity: dict, filenames: list[str], extra=None):
    manifest = {
        "status": "complete", "key": stable_hash(identity), "identity": identity,
        "outputs": {name: {"bytes": (directory / name).stat().st_size, "sha256": file_hash(directory / name)} for name in filenames},
    }
    if extra:
        manifest.update(extra)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def verify_cache(directory: Path, identity: dict):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete cache: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("key") != stable_hash(identity) or manifest.get("identity") != identity:
        raise ValueError(f"Cache identity mismatch: {directory}")
    for name, expected in manifest["outputs"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Corrupt cache artifact: {path}")
    return manifest


class CacheBuild:
    """Only a newly created staging directory is cleaned after a failed build."""

    def __init__(self, destination: Path):
        self.destination = destination
        self.staging = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex)

    def __enter__(self):
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    def __exit__(self, kind, value, traceback):
        if kind is None:
            if self.destination.exists():
                shutil.rmtree(self.staging)
                raise FileExistsError(f"Cache appeared during build: {self.destination}")
            self.staging.rename(self.destination)
        elif self.staging.exists():
            shutil.rmtree(self.staging)
