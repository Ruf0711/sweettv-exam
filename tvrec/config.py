"""Configuration validation and paths; original data is always read-only."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Settings:
    values: dict
    root: Path
    config_path: Path

    @property
    def data_dir(self) -> Path:
        return (self.root / self.values["data_dir"]).resolve()

    @property
    def artifacts_dir(self) -> Path:
        return (self.root / self.values["artifacts_dir"]).resolve()

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.values["timezone"])

    def source(self, filename: str) -> Path:
        path = (self.data_dir / filename).resolve()
        if not path.is_relative_to(self.data_dir):
            raise ValueError("Source paths must stay inside data_dir")
        return path


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).resolve()
    values = json.loads(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / values.get("project_root", "..")).resolve()
    settings = Settings(copy.deepcopy(values), root, config_path)
    ZoneInfo(values["timezone"])
    if not values.get("timezone_assumption"):
        raise ValueError("The evidence-based timezone assumption must be recorded")
    if settings.artifacts_dir.is_relative_to(settings.data_dir):
        raise ValueError("artifacts_dir must not be inside the original data directory")
    prep = values["preparation"]
    required_policies = {
        "competing_broadcast_policy": "equal_distinct_labels_including_unknown",
        "viewing_normalization": "deduplicate_then_union_same_user_channel_including_contact",
        "missing_programme_id_policy": "unknown_zero",
        "nonpositive_interval_policy": "exclude",
    }
    for key, expected in required_policies.items():
        if prep[key] != expected:
            raise ValueError(f"Unsupported {key}: {prep[key]!r}")
    for key in ["csv_chunk_rows", "join_chunk_rows", "parquet_batch_rows", "progress_every_rows"]:
        if int(prep[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if values["interactions"]["relevance_seconds"] <= 0:
        raise ValueError("relevance_seconds must be positive")
    cap = values["interactions"]["confidence_seconds_cap"]
    if cap is not None and cap <= 0:
        raise ValueError("confidence_seconds_cap must be positive or null")
    lookback = values["popularity"]["lookback_days"]
    if lookback is not None and (not isinstance(lookback, int) or lookback <= 0):
        raise ValueError("lookback_days must be a positive integer or null")
    if values["popularity"]["support_min_seconds"] < 0:
        raise ValueError("support_min_seconds must be nonnegative")
    if values["evaluation"]["k"] != 5:
        raise ValueError("Stage 1 requires exactly five recommendations")
    if values["evaluation"]["allow_repeats_from_history"] is not True:
        raise ValueError("Stage 1 permits programmes already viewed")
    if values["evaluation"]["empty_relevance_ap"] != 0:
        raise ValueError("Empty relevance must contribute zero")
    return settings
