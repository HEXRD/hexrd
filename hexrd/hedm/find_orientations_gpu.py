"""CUDA port of the completeness scorer.

:func:`hexrd.hedm.find_orientations.score_orientations` imports this lazily
when a CUDA device is available; the numba CPU kernels there remain the
fallback and the reference implementation.  The device functions mirror the
CPU kernels' control flow in float64, comparison for comparison — including
numpy's remainder semantics for the angle wrapping — so both paths produce
bit-identical scores.
"""
import math

import numpy as np
from numba import cuda

TWO_PI = 2.0 * math.pi


def available() -> bool:
    """True when a usable CUDA device (and toolchain) is present."""
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


@cuda.jit(device=True)
def _wrap(value, lo):
    """np.mod(value - lo, 2*pi) + lo, with numpy remainder semantics."""
    r = math.fmod(value - lo, TWO_PI)
    if r != 0.0 and r < 0.0:
        r += TWO_PI
    return r + lo


@cuda.jit(device=True)
def _find_in_range(value, spans):
    """Index i such that spans[i-1] <= value < spans[i]; -2 if out of range."""
    if value < spans[0] or value >= spans[-1]:
        return -2
    li, ri = 0, len(spans)
    while li < ri:
        mi = (li + ri) // 2
        if value < spans[mi]:
            ri = mi
        else:
            li = mi + 1
    return li


@cuda.jit(device=True)
def _dilated_hit(eta, ome, dpix_eta, dpix_ome, ring_map, threshold):
    """1 if any map value above threshold in the tolerance window, -1 on NaN, else 0."""
    n_ome, n_eta = ring_map.shape
    for i in range(max(ome - dpix_ome, 0), min(ome + dpix_ome + 1, n_ome)):
        for j in range(max(eta - dpix_eta, 0), min(eta + dpix_eta + 1, n_eta)):
            if ring_map[i, j] > threshold:
                return 1
            if math.isnan(ring_map[i, j]):
                return -1
    return 0


@cuda.jit(cache=True)
def _count_hits_kernel(angs, ring_for_hkl, eta_edges, ome_edges,
                       valid_eta_spans, valid_ome_spans, ome_offset,
                       ring_maps, dpix_eta, dpix_ome, threshold, scores):
    """One thread per trial orientation: hits / total over both Bragg
    solutions of every symmetry-equivalent HKL (see the CPU _count_hits)."""
    q = cuda.grid(1)
    if q >= scores.shape[0]:
        return
    hits = 0
    total = 0
    for solution in range(2):
        for i in range(angs.shape[2]):
            if math.isnan(angs[solution, q, i, 0]):
                continue
            eta = _wrap(angs[solution, q, i, 1], -math.pi)
            if _find_in_range(eta, valid_eta_spans) & 1 == 0:
                continue
            ome = _wrap(angs[solution, q, i, 2], ome_offset)
            if _find_in_range(ome, valid_ome_spans) & 1 == 0:
                continue
            eta_idx = _find_in_range(eta, eta_edges) - 1
            ome_idx = _find_in_range(ome, ome_edges) - 1
            if eta_idx < 0 or ome_idx < 0:
                continue
            hit = _dilated_hit(eta_idx, ome_idx, dpix_eta, dpix_ome,
                               ring_maps[ring_for_hkl[i]], threshold)
            if hit >= 0:
                total += 1
                hits += hit
    scores[q] = hits / total if total > 0 else 0.0


def count_hits_all(angs, ring_for_hkl, eta_edges, ome_edges, valid_eta_spans,
                   valid_ome_spans, ome_offset, ring_maps, dpix_eta, dpix_ome,
                   threshold):
    """Per-trial completeness on the GPU; same contract as the CPU
    _count_hits_all, with the two Bragg solutions stacked as angs[0]/angs[1]."""
    n = angs.shape[1]
    scores = cuda.device_array(n, dtype=np.float64)
    threads = 128
    blocks = (n + threads - 1) // threads
    _count_hits_kernel[blocks, threads](
        cuda.to_device(np.ascontiguousarray(angs)),
        cuda.to_device(np.ascontiguousarray(ring_for_hkl, dtype=np.int64)),
        cuda.to_device(np.ascontiguousarray(eta_edges)),
        cuda.to_device(np.ascontiguousarray(ome_edges)),
        cuda.to_device(np.ascontiguousarray(valid_eta_spans)),
        cuda.to_device(np.ascontiguousarray(valid_ome_spans)),
        float(ome_offset),
        cuda.to_device(np.ascontiguousarray(ring_maps)),
        int(dpix_eta), int(dpix_ome), float(threshold), scores)
    return scores.copy_to_host()
