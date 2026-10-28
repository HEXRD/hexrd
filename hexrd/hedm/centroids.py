"""Experimental streaming peak catalog; deliberately separate from fit-grains.

Detector-space centroids do not reproduce grain-dependent polar-window
measurements in general. This module supplies research data, never an implicit
replacement for pull_spots. Memory tracks thresholded pixels in adjacent frames
and active component statistics, rather than a dense rotation volume.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Centroid:
    row: float
    col: float
    omega: float  # degrees, intensity-weighted physical frame centers
    intensity: float
    maximum: float
    pixels: int
    bounds: tuple  # frame start/stop, row start/stop, column start/stop


def _pixels(frame, threshold, mask):
    if sparse.issparse(frame):
        frame = sparse.csr_array(frame, copy=True)
        frame.sum_duplicates()
        frame.sort_indices()
        rows = np.repeat(np.arange(frame.shape[0]), np.diff(frame.indptr))
        cols, values = frame.indices, frame.data
    else:
        frame = np.asarray(frame)
        rows, cols = np.nonzero(frame > threshold)
        values = frame[rows, cols]
    if len(frame.shape) != 2:
        raise ValueError('Centroid frames must be two-dimensional')
    keep = values > threshold
    if mask is not None:
        keep &= mask[rows, cols]
    return rows[keep], cols[keep], values[keep].astype(np.float64)


def _links(rows, cols, other_flat, width, height, shifts):
    """Pixel correspondences without a dense label image or row wraparound."""
    left, right = [], []
    for dr, dc in shifts:
        r, c = rows + dr, cols + dc
        flat = r * width + c
        positions = np.searchsorted(other_flat, flat)
        valid = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        valid &= positions < len(other_flat)
        ids = np.flatnonzero(valid)
        ids = ids[other_flat[positions[ids]] == flat[ids]]
        left.append(ids)
        right.append(positions[ids])
    return np.concatenate(left), np.concatenate(right)


def _groups(count, left, right):
    graph = sparse.coo_array(
        (np.ones(len(left), dtype=np.uint8), (left, right)), shape=(count, count)
    ).tocsr()
    return connected_components(graph, directed=False)


def _record(stats):
    mass, row, col, omega, maximum, count, *bounds = stats
    return Centroid(
        row / mass,
        col / mass,
        omega / mass,
        mass,
        maximum,
        int(count),
        tuple(map(int, bounds)),
    )


def iter_centroids(frames, omega, threshold, mask=None):
    """Yield finalized 26-connected components, consuming frames once.

    ``omega`` is an N×2 array in degrees; gaps and reversed scan direction
    split components. ``mask`` uses True for usable pixels. Threshold must be
    finite and nonnegative so implicit sparse zeros are background. Components
    touching the beginning/end of a periodic scan are kept separate. No peak
    splitting, background subtraction, or uncertainty model is applied.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('A finite nonnegative threshold is required')
    omega = np.asarray(omega, dtype=np.float64)
    if omega.ndim != 2 or omega.shape[1] != 2 or not np.all(np.isfinite(omega)):
        raise ValueError('omega must contain finite frame start/stop pairs')
    if np.any(omega[:, 0] == omega[:, 1]):
        raise ValueError('Omega frames must have nonzero width')
    previous_flat = previous_labels = np.empty(0, dtype=np.int64)
    active = np.empty((0, 12), dtype=np.float64)
    shape = None
    frame_count = 0
    for index, frame in enumerate(frames):
        if index >= len(omega):
            raise ValueError('More frames than omega intervals')
        frame_count += 1
        if shape is None:
            shape = frame.shape
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != shape:
                    raise ValueError('Mask shape differs from frame shape')
        elif frame.shape != shape:
            raise ValueError('All centroid frames must have the same shape')
        height, width = shape
        if index and (
            omega[index - 1, 1] != omega[index, 0]
            or np.sign(np.diff(omega[index - 1]))[0]
            != np.sign(np.diff(omega[index]))[0]
        ):
            for stats in active:
                yield _record(stats)
            active = np.empty((0, 12), dtype=np.float64)
            previous_flat = previous_labels = np.empty(0, dtype=np.int64)
        rows, cols, values = _pixels(frame, threshold, mask)
        if not np.all(np.isfinite(values)):
            raise ValueError('Nonfinite foreground intensities are unsupported')
        flat = rows * width + cols
        left, right = _links(
            rows, cols, flat, width, height, [(0, -1), (-1, -1), (-1, 0), (-1, 1)]
        )
        count, labels = _groups(len(flat), left, right)
        stats = np.zeros((count, 12), dtype=np.float64)
        for field, weights in enumerate(
            (values, rows * values, cols * values, values * np.mean(omega[index]))
        ):
            stats[:, field] = np.bincount(labels, weights, minlength=count)
        stats[:, 4] = -np.inf
        np.maximum.at(stats[:, 4], labels, values)
        stats[:, 5] = np.bincount(labels, minlength=count)
        stats[:, 6:8] = [index, index + 1]
        stats[:, [8, 10]] = np.inf
        np.minimum.at(stats[:, 8], labels, rows)
        np.maximum.at(stats[:, 9], labels, rows + 1)
        np.minimum.at(stats[:, 10], labels, cols)
        np.maximum.at(stats[:, 11], labels, cols + 1)
        left, right = _links(
            rows,
            cols,
            previous_flat,
            width,
            height,
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
        )
        old_count = len(active)
        merged_count, merged_labels = _groups(
            old_count + count, previous_labels[right], old_count + labels[left]
        )
        combined = np.concatenate((active, stats))
        merged = np.zeros((merged_count, 12), dtype=np.float64)
        for field in (0, 1, 2, 3, 5):
            merged[:, field] = np.bincount(
                merged_labels, combined[:, field], minlength=merged_count
            )
        for field in (4, 7, 9, 11):
            merged[:, field] = -np.inf
            np.maximum.at(merged[:, field], merged_labels, combined[:, field])
        for field in (6, 8, 10):
            merged[:, field] = np.inf
            np.minimum.at(merged[:, field], merged_labels, combined[:, field])
        live = np.zeros(merged_count, dtype=bool)
        live[merged_labels[old_count:]] = True
        for item in merged[~live]:
            yield _record(item)
        lookup = np.cumsum(live) - 1
        previous_labels = lookup[merged_labels[old_count:][labels]]
        previous_flat, active = flat, merged[live]
    if frame_count != len(omega):
        raise ValueError('Fewer frames than omega intervals')
    for stats in active:
        yield _record(stats)
