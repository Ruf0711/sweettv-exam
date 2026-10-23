"""Numeric history, user, item, channel/category and schedule features."""

import zlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


KEYS = ["user_id", "tv_show_id"]


def category_code(value):
    return int(zlib.crc32(str(value).casefold().encode("utf-8")) % 1000000)


def normalize_rows(array):
    return array / np.maximum(1., array.sum(axis=1, keepdims=True))


def indexed_take(frame, names, prefix):
    return frame[names].rename(columns={name: prefix + name for name in names if name not in KEYS})


def feature_columns(frame, family):
    excluded = {"user_id", "user_index", "tv_show_id", "label", "forecast", "label_end"}
    context = ("schedule_", "affinity_", "habit_")
    result = []
    for name in frame.columns:
        if name in excluded:
            continue
        is_context = name.startswith(context) or name in ["category_id", "channel_id"]
        is_full = name.startswith(("als_", "cbf_", "source_"))
        if family == "history" and (is_context or is_full):
            continue
        if family == "context" and is_full:
            continue
        result.append(name)
    return result


def build_features(retrieved, cohort, history, prior, upcoming, cutoff, end, history_start, zone):
    """Accepts past evidence and permitted schedules, never validation labels."""
    frame = retrieved.copy()
    counts = history["counts"].copy()
    counts["age50_days"] = np.where(counts.last50 > 0, (cutoff - counts.last50) / 86400., 999.)
    counts["age_ge50_days"] = np.where(counts.last_ge50 > 0, (cutoff - counts.last_ge50) / 86400., 999.)
    counts["inclusive50"] = counts.count50 + counts.exact50_only
    counts["completion80_ratio"] = counts.count80 / np.maximum(1, counts.count50)
    counts["mean_completed_seconds"] = counts.seconds50 / np.maximum(1, counts.count50)
    count_names = KEYS + ["count30", "count50", "count80", "inclusive50", "seconds50", "ge50seconds", "age50_days", "age_ge50_days", "completion80_ratio", "mean_completed_seconds"]
    frame = frame.merge(indexed_take(counts, count_names, "pair_"), on=KEYS, how="left", sort=False, validate="one_to_one")
    cumulative = history["cumulative"].copy()
    cumulative["age_any_days"] = (cutoff - cumulative.last_epoch) / 86400.
    cumulative["since_first_days"] = np.maximum(1., (cutoff - cumulative.first_epoch) / 86400.)
    frame = frame.merge(indexed_take(cumulative, KEYS + ["seconds", "age_any_days", "since_first_days"], "pair_"), on=KEYS, how="left", sort=False, validate="one_to_one")
    users = cumulative.groupby("user_id").agg(user_seconds=("seconds", "sum"), user_items=("tv_show_id", "nunique"), first=("first_epoch", "min"), last=("last_epoch", "max"))
    users["user_tenure_days"] = (cutoff - users["first"]) / 86400.
    users["user_age_days"] = (cutoff - users["last"]) / 86400.
    users = users.drop(columns=["first", "last"])
    user_counts = counts.groupby("user_id")[["count30", "count50", "count80"]].sum().add_prefix("user_")
    users = users.join(user_counts, how="left")
    frame = frame.join(users, on="user_id", how="left")
    items = cumulative.groupby("tv_show_id").agg(item_viewers=("user_id", "nunique"), item_seconds=("seconds", "sum"), last=("last_epoch", "max"))
    items["item_age_days"] = (cutoff - items["last"]) / 86400.
    items = items.drop(columns="last")
    items = items.join(counts.groupby("tv_show_id")[["count30", "count50", "count80"]].sum().add_prefix("item_"), how="left")
    frame = frame.join(items, on="tv_show_id", how="left")
    for days, data in history["recent"].items():
        name = str(days)
        recent_counts = data["counts"]
        recent_cumulative = data["cumulative"]
        rc = indexed_take(recent_counts, KEYS + ["count30", "count50", "count80"], "pair_recent" + name + "_")
        rc = rc.merge(indexed_take(recent_cumulative, KEYS + ["seconds"], "pair_recent" + name + "_"), on=KEYS, how="outer")
        frame = frame.merge(rc, on=KEYS, how="left", sort=False, validate="one_to_one")
        frame = frame.join(recent_cumulative.groupby("user_id").seconds.sum().rename("user_recent" + name + "_seconds"), on="user_id")
        frame = frame.join(recent_cumulative.groupby("tv_show_id").user_id.nunique().rename("item_recent" + name + "_viewers"), on="tv_show_id")

    # Every missing recency means no support, never a zero-day observation.
    for name in ["pair_age50_days", "pair_age_ge50_days", "pair_age_any_days", "pair_since_first_days", "user_age_days", "item_age_days"]:
        frame[name] = frame[name].fillna(999.)
    frame = frame.fillna(0)
    frame["pair_fraction_user_seconds"] = frame.pair_seconds / np.maximum(1, frame.user_seconds)
    frame["pair_fraction_user_completions"] = frame.pair_count50 / np.maximum(1, frame.user_count50)
    frame["pair_frequency_per_day"] = frame.pair_count50 / np.maximum(1, frame.pair_since_first_days)
    frame["pair_recent28_fraction"] = frame.pair_recent28_seconds / np.maximum(1, frame.pair_seconds)
    frame["pair_recent7_vs28_rate"] = (frame.pair_recent7_count50 + 1) / (frame.pair_recent28_count50 / 4 + 1)
    frame["user_recent28_fraction"] = frame.user_recent28_seconds / np.maximum(1, frame.user_seconds)

    # Metadata descriptors use prior metadata, supplemented by the known window.
    descriptors = pd.concat([prior, upcoming]).drop_duplicates("tv_show_id", keep="last").set_index("tv_show_id")
    categories = descriptors.tv_show_category.fillna("").astype(str)
    frame["category_id"] = frame.tv_show_id.map(categories.map(category_code)).fillna(0)
    history_categories = prior.drop_duplicates("tv_show_id", keep="last").set_index("tv_show_id").tv_show_category.fillna("").astype(str)
    category_seconds = cumulative.loc[cumulative.user_id.isin(cohort)].copy()
    category_seconds["category"] = category_seconds.tv_show_id.map(history_categories).fillna("")
    category_seconds = category_seconds.groupby(["user_id", "category"]).seconds.sum()
    pair_categories = frame.tv_show_id.map(categories).fillna("")
    category_index = pd.MultiIndex.from_arrays([frame.user_id, pair_categories])
    frame["affinity_category_share"] = category_seconds.reindex(category_index).fillna(0).to_numpy() / np.maximum(1, frame.user_seconds)

    item_ids = np.sort(frame.tv_show_id.unique())
    item_map = {int(item): index for index, item in enumerate(item_ids)}
    future = upcoming.loc[upcoming.tv_show_id.isin(item_ids)].copy()
    future["seconds"] = future.end - future.start
    local = pd.to_datetime(future.start, unit="s", utc=True).dt.tz_convert(zone)
    future["weekday"], future["hour"] = local.dt.dayofweek, local.dt.hour
    future["day"] = local.dt.strftime("%Y-%m-%d")
    future_stats = future.groupby("tv_show_id").agg(schedule_future_count=("start", "size"), schedule_future_days=("day", "nunique"), schedule_mean_duration=("seconds", "mean"), schedule_weekend_share=("weekday", lambda x: np.mean(x >= 5)))
    horizon_days = (end - cutoff) / 86400.
    future_stats["schedule_future_daily_rate"] = future_stats.schedule_future_count / horizon_days
    recent_schedule = prior.loc[prior.start >= cutoff - 28 * 86400].groupby("tv_show_id").size()
    future_stats["schedule_past28_count"] = recent_schedule.reindex(future_stats.index).fillna(0)
    future_stats["schedule_past_count"] = prior.groupby("tv_show_id").size().reindex(future_stats.index).fillna(0)
    future_stats["schedule_rate_change"] = (future_stats.schedule_future_daily_rate + .05) / (future_stats.schedule_past28_count / 28 + .05)
    future_stats["schedule_horizon_days"] = horizon_days
    frame = frame.join(future_stats, on="tv_show_id", how="left").fillna(0)
    frame["schedule_personal_exposure_rate"] = frame.pair_count50 / np.maximum(1, frame.schedule_past_count)
    frame["schedule_expected_completed"] = frame.schedule_personal_exposure_rate * frame.schedule_future_count

    primary_channel = future.groupby(["tv_show_id", "channel_id"]).size().reset_index(name="n").sort_values(["tv_show_id", "n", "channel_id"], ascending=[True, False, True]).drop_duplicates("tv_show_id").set_index("tv_show_id").channel_id
    frame["channel_id"] = frame.tv_show_id.map(primary_channel).fillna(0)
    channels = history["channels"]
    channel_map = {channel: index for index, channel in enumerate(channels)}
    future_channels = np.zeros((len(item_ids), len(channels)), dtype=np.float64)
    valid_channel = future.channel_id.isin(channel_map)
    np.add.at(future_channels, (future.loc[valid_channel].tv_show_id.map(item_map).to_numpy(int), future.loc[valid_channel].channel_id.map(channel_map).to_numpy(int)), 1)
    all_airings = future.groupby("tv_show_id").size().reindex(item_ids).fillna(0).to_numpy()
    future_channels /= np.maximum(1, all_airings[:, None])
    user_index = frame.user_index.to_numpy(int)
    item_index = frame.tv_show_id.map(item_map).to_numpy(int)
    with threadpool_limits(limits=1):
        affinity = normalize_rows(history["channel_seconds"]) @ future_channels.T
    frame["affinity_channel_share"] = affinity[user_index, item_index]
    future_habits = np.zeros((len(item_ids), 168), dtype=np.float64)
    np.add.at(future_habits, (future.tv_show_id.map(item_map).to_numpy(int), (future.weekday * 24 + future.hour).to_numpy(int)), 1)
    future_habits = normalize_rows(future_habits)
    habits = history["habits"]
    for name, a, b in [("weekhour", habits, future_habits), ("weekday", habits.reshape(-1, 7, 24).sum(axis=2), future_habits.reshape(-1, 7, 24).sum(axis=2)), ("hour", habits.reshape(-1, 7, 24).sum(axis=1), future_habits.reshape(-1, 7, 24).sum(axis=1))]:
        with threadpool_limits(limits=1):
            alignment = normalize_rows(a) @ b.T
        frame["habit_" + name + "_alignment"] = alignment[user_index, item_index]
    numeric = [name for name in frame if name not in ["user_id", "user_index", "tv_show_id"]]
    frame[numeric] = frame[numeric].astype(np.float32)
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Nonfinite cutoff-safe feature")
    if frame.duplicated(KEYS).any() or history["observed_max_clipped_epoch"] > cutoff:
        raise ValueError("Duplicate features or future historical evidence")
    return frame
