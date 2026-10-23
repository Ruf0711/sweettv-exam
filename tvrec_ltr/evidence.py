"""Cutoff-clipped cached evidence; future labels are exposed separately."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tvrec.interactions import Window, epoch_column
from tvrec.stage2.completion import window_event_portion
from tvrec_counts.features import finish_instances, instance_portion, verify_against_existing


def cumulative_pairs(evidence, start, end):
    low = np.maximum(evidence.start_time.to_numpy(), start)
    high = np.minimum(evidence.end_time.to_numpy(), end)
    valid = (high > low) & (evidence.tv_show_id.to_numpy() > 0)
    part = evidence.loc[valid, ["user_id", "tv_show_id"]].copy()
    part["seconds"] = (high[valid] - low[valid]) / evidence.allocation_denominator.to_numpy()[valid]
    part["first_epoch"], part["last_epoch"] = low[valid], high[valid]
    return part.groupby(["user_id", "tv_show_id"], as_index=False).agg(seconds=("seconds", "sum"), first_epoch=("first_epoch", "min"), last_epoch=("last_epoch", "max"))


def inclusive_feedback(counts):
    frame = counts.loc[counts.count50 + counts.exact50_only > 0, ["user_id", "tv_show_id", "count50", "count80", "exact50_only", "ge50seconds", "last_ge50"]].copy()
    frame["count50"] += frame.exact50_only
    return frame.drop(columns="exact50_only").rename(columns={"ge50seconds": "seconds50", "last_ge50": "last_epoch"})


def hourly_histograms(sessions, cohort, cutoff, zone):
    """Split actual uninterrupted sessions at UTC hours, then localize bins."""
    subset = sessions.loc[sessions.user_id.isin(cohort) & (sessions.start_time < cutoff)].copy()
    start = subset.start_time.to_numpy(dtype=np.int64)
    end = np.minimum(subset.end_time.to_numpy(dtype=np.int64), cutoff)
    keep = end > start
    subset, start, end = subset.loc[keep], start[keep], end[keep]
    user_map = {user: i for i, user in enumerate(cohort)}
    user = subset.user_id.map(user_map).to_numpy(dtype=np.int64)
    size = (end - 1) // 3600 - start // 3600 + 1
    repeat = np.repeat(np.arange(len(start)), size)
    offset = np.arange(size.sum()) - np.repeat(np.cumsum(size) - size, size)
    hour_start = (start[repeat] // 3600 + offset) * 3600
    seconds = np.minimum(end[repeat], hour_start + 3600) - np.maximum(start[repeat], hour_start)
    local = pd.to_datetime(hour_start, unit="s", utc=True).tz_convert(zone)
    bins = local.dayofweek.to_numpy() * 24 + local.hour.to_numpy()
    habits = np.zeros((len(cohort), 168), dtype=np.float64)
    np.add.at(habits, (user[repeat], bins), seconds)
    channels = sorted(int(channel) for channel in subset.channel_id.unique())
    channel_map = {channel: i for i, channel in enumerate(channels)}
    channel_seconds = np.zeros((len(cohort), len(channels)), dtype=np.float64)
    np.add.at(channel_seconds, (user, subset.channel_id.map(channel_map).to_numpy(dtype=int)), end - start)
    return habits, channel_seconds, channels


class EvidenceStore:
    def __init__(self, context, cohort):
        self.context, self.cohort = context, cohort
        self.events = pq.read_table(context.completion / "events.parquet", columns=["user_id", "tv_show_id", "channel_id", "broadcast_id", "broadcast_seconds", "start_time", "end_time"]).to_pandas()
        self.cumulative = pq.read_table(context.prepared / "evidence.parquet", columns=["user_id", "tv_show_id", "start_time", "end_time", "allocation_denominator"]).to_pandas()
        self.sessions = pq.read_table(context.completion / "continuous_sessions.parquet").to_pandas()
        self.sessions = self.sessions.loc[self.sessions.user_id.isin(cohort)].copy()
        for frame in [self.events, self.cumulative, self.sessions]:
            for name in ["start_time", "end_time"]:
                frame[name] = epoch_column(frame[name])
        self.metadata = pq.read_table(context.prepared / "broadcast_metadata.parquet").to_pandas()
        self.metadata["start"] = epoch_column(self.metadata.start_time)
        self.metadata["end"] = epoch_column(self.metadata.end_time)
        self.metadata = self.metadata.loc[(self.metadata.tv_show_id > 0) & (self.metadata.end > self.metadata.start)].drop_duplicates(["channel_id", "tv_show_id", "start", "end"])
        self.blackouts = json.loads((context.completion / "blackouts.json").read_text(encoding="utf-8"))
        self.start = json.loads((context.prepared / "diagnostics.json").read_text(encoding="utf-8"))["raw"]["observed_start_epoch"]
        self._cutoff, self._history = None, None

    def completion(self, start, end, template_only=False):
        frame = self.events
        if template_only:
            frame = frame.loc[frame.user_id.isin(self.cohort)]
        return finish_instances([instance_portion(frame, Window("count", start, end), self.blackouts)])

    def history(self, cutoff):
        if self._cutoff == cutoff:
            return self._history
        counts = self.completion(self.start, cutoff)
        feedback = inclusive_feedback(counts)
        cumulative = cumulative_pairs(self.cumulative, self.start, cutoff)
        recent = {}
        for days in [7, 28]:
            low = max(self.start, cutoff - days * 86400)
            recent[days] = {"counts": self.completion(low, cutoff, True), "cumulative": cumulative_pairs(self.cumulative, low, cutoff)}
        habits, channel_seconds, channels = hourly_histograms(self.sessions, self.cohort, cutoff, self.context.settings.zone)
        self._history = {"counts": counts, "feedback": feedback, "cumulative": cumulative, "recent": recent,
                         "habits": habits, "channel_seconds": channel_seconds, "channels": channels,
                         "observed_max_clipped_epoch": int(cumulative.last_epoch.max())}
        self._cutoff = cutoff
        return self._history

    def labels(self, start, end, candidates):
        if end > self.context.cutoff:
            raise ValueError("Future competition viewing is not permitted as labels")
        frame = self.events.loc[self.events.user_id.isin(self.cohort)]
        observed = window_event_portion(frame, Window("labels", start, end), self.blackouts)
        eligible = observed.loc[observed.gt80 & observed.tv_show_id.isin(candidates), ["user_id", "tv_show_id"]].drop_duplicates()
        return eligible.groupby("user_id").tv_show_id.agg(set).to_dict()

    def metadata_at(self, cutoff, end):
        prior = self.metadata.loc[self.metadata.start < cutoff].sort_values(["start_time", "tv_show_id"], kind="mergesort")
        upcoming = self.metadata.loc[(self.metadata.start < end) & (self.metadata.end > cutoff)].sort_values(["start_time", "tv_show_id"], kind="mergesort")
        return prior, upcoming

    def anchor(self, name, history):
        if name not in ["main", "recent_b"]:
            return None
        record = json.loads((self.context.stage2_run / name / "context.json").read_text(encoding="utf-8"))
        old = pq.read_table(Path(record["completion_window_path"]) / "train.parquet").to_pandas()
        return verify_against_existing(history["counts"], old, old.user_id.unique().tolist())
