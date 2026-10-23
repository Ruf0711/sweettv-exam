"""Disjoint channel timelines, viewing union, and bounded indexed overlap lookup."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelTimeline:
    starts: np.ndarray
    ends: np.ndarray
    label_offsets: np.ndarray
    labels: np.ndarray
    segment_base: int = 0

    def __post_init__(self):
        if len(self.starts) != len(self.ends) or np.any(self.ends <= self.starts):
            raise ValueError("Timeline segments must have positive duration")
        if np.any(self.starts[1:] < self.ends[:-1]):
            raise ValueError("Canonical timeline segments must not overlap")
        sizes = np.diff(self.label_offsets)
        if len(sizes) != len(self.starts) or np.any(sizes <= 0):
            raise ValueError("Each segment needs at least one active label")
        if self.label_offsets[0] != 0 or self.label_offsets[-1] != len(self.labels):
            raise ValueError("Invalid label offsets")
        if np.any(self.labels < 0):
            raise ValueError("Negative programme labels are not permitted")
        self.has_positive = np.add.reduceat((self.labels > 0).astype(np.int64), self.label_offsets[:-1]) > 0
        lengths = self.ends - self.starts
        self._any_prefix = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
        self._positive_prefix = np.r_[0, np.cumsum(lengths * self.has_positive, dtype=np.int64)]

    def _integral(self, times: np.ndarray, positive: bool) -> np.ndarray:
        index = np.searchsorted(self.starts, times, side="right") - 1
        valid = index >= 0
        result = np.zeros(len(times), dtype=np.int64)
        j = index[valid]
        weights = self.has_positive[j] if positive else 1
        prefix = self._positive_prefix if positive else self._any_prefix
        result[valid] = prefix[j] + np.minimum(times[valid] - self.starts[j], self.ends[j] - self.starts[j]) * weights
        return result

    def covered_seconds(self, starts: np.ndarray, ends: np.ndarray, *, positive: bool = False) -> np.ndarray:
        if np.any(ends < starts):
            raise ValueError("Negative viewing interval")
        return self._integral(ends, positive) - self._integral(starts, positive)

    def overlaps(self, starts: np.ndarray, ends: np.ndarray):
        """Return viewing-row indices, segment indices, and exact overlap endpoints."""
        if np.any(ends < starts):
            raise ValueError("Negative viewing interval")
        low = np.searchsorted(self.ends, starts, side="right")
        high = np.searchsorted(self.starts, ends, side="left")
        sizes = np.maximum(high - low, 0)
        sizes[ends == starts] = 0
        rows = np.repeat(np.arange(len(starts)), sizes)
        count = int(sizes.sum())
        if count == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, empty
        indices = np.repeat(low, sizes) + np.arange(count) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        left = np.maximum(starts[rows], self.starts[indices])
        right = np.minimum(ends[rows], self.ends[indices])
        if np.any(right <= left):
            raise AssertionError("Indexed join emitted a nonpositive overlap")
        return rows, indices, left, right


def canonical_timeline(starts, ends, labels, *, segment_base=0) -> ChannelTimeline:
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    keep = ends > starts
    starts, ends, labels = starts[keep], ends[keep], labels[keep]
    if not len(starts):
        raise ValueError("Cannot construct an empty timeline")
    order = np.lexsort((labels, ends, starts))
    starts, ends, labels = starts[order], ends[order], labels[order]
    duplicate = np.r_[False, (starts[1:] == starts[:-1]) & (ends[1:] == ends[:-1]) & (labels[1:] == labels[:-1])]
    starts, ends, labels = starts[~duplicate], ends[~duplicate], labels[~duplicate]
    if np.all(starts[1:] >= ends[:-1]):
        return ChannelTimeline(starts, ends, np.arange(len(starts) + 1, dtype=np.int64), labels, segment_base)
    # Only channels with actual competition need a sweep. End events precede
    # starts at equal timestamps; emitted segments are half-open and disjoint.
    events = []
    for s, e, label in zip(starts, ends, labels):
        events.append((int(s), 1, int(label)))
        events.append((int(e), -1, int(label)))
    events.sort()
    active = Counter()
    out_s, out_e, out_labels, offsets = [], [], [], [0]
    previous = events[0][0]
    position = 0
    while position < len(events):
        stamp = events[position][0]
        if stamp > previous and active:
            keys = sorted(active)
            out_s.append(previous)
            out_e.append(stamp)
            out_labels.extend(keys)
            offsets.append(len(out_labels))
        while position < len(events) and events[position][0] == stamp:
            _, delta, label = events[position]
            active[label] += delta
            if active[label] == 0:
                del active[label]
            position += 1
        previous = stamp
    return ChannelTimeline(np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64), np.array(offsets, dtype=np.int64), np.array(out_labels, dtype=np.int64), segment_base)


def normalize_viewing(users, channels, starts, ends, durations):
    """Deduplicate source rows, then union overlapping or touching user/channel intervals."""
    users = np.asarray(users, dtype=np.int32)
    channels = np.asarray(channels, dtype=np.int32)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    durations = np.asarray(durations, dtype=np.int64)
    if not len(users):
        return {"user_index": users, "channel_id": channels, "start": starts, "end": ends, "source_rows": np.empty(0, dtype=np.int64)}, {"exact_duplicate_rows": 0, "overlapping_rows_after_deduplication": 0, "contact_rows_after_deduplication": 0, "normalized_intervals": 0, "deduplicated_seconds": 0, "union_seconds": 0}
    if np.any(ends <= starts):
        raise ValueError("normalize_viewing expects positive intervals")
    order = np.lexsort((durations, ends, starts, channels, users))
    u, c, s, e, d = (array[order] for array in [users, channels, starts, ends, durations])
    dup = np.r_[False, (u[1:] == u[:-1]) & (c[1:] == c[:-1]) & (s[1:] == s[:-1]) & (e[1:] == e[:-1]) & (d[1:] == d[:-1])]
    duplicates = int(dup.sum())
    u, c, s, e = (array[~dup] for array in [u, c, s, e])
    first = np.r_[True, (u[1:] != u[:-1]) | (c[1:] != c[:-1])]
    groups = np.cumsum(first, dtype=np.int64) - 1
    stride = int(e.max()) - int(e.min()) + 1
    if int(groups[-1]) * stride + int(e.max()) > np.iinfo(np.int64).max:
        raise ValueError("Time range too large for grouped prefix maximum")
    offset = groups * stride
    prefix = np.maximum.accumulate(e + offset) - offset
    previous_end = np.r_[s[0], prefix[:-1]]
    overlaps = (~first) & (s < previous_end)
    contacts = (~first) & (s == previous_end)
    new = first | (s > previous_end)
    boundaries = np.flatnonzero(new)
    normalized_end = np.maximum.reduceat(e, boundaries)
    source_rows = np.diff(np.r_[boundaries, len(s)])
    output = {"user_index": u[boundaries], "channel_id": c[boundaries], "start": s[boundaries], "end": normalized_end, "source_rows": source_rows}
    stats = {"exact_duplicate_rows": duplicates, "overlapping_rows_after_deduplication": int(overlaps.sum()), "contact_rows_after_deduplication": int(contacts.sum()), "normalized_intervals": len(boundaries), "deduplicated_seconds": int((e - s).sum()), "union_seconds": int((normalized_end - s[boundaries]).sum())}
    return output, stats
