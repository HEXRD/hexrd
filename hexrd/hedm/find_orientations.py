"""
find-orientations: from a rotation series of detector images to grain orientations.

The whole workflow is `find_orientations` below. It reads as five steps, each
its own function:

    1. load_or_build_eta_omega_maps  images             -> per-ring (omega, eta) maps
    2. generate_orientation_fibers   seed-ring peaks    -> trial orientations
       (or a precomputed grid via `use_quaternion_grid`)
    3. score_orientations            trial orientations -> completeness per trial
    4. cluster_grains                scored trials      -> one orientation per grain
    5. write_results                 grains             -> grains.out, etc.

Inputs travel as two plain objects: an :class:`~hexrd.hedm.experiment.HedmExperiment`
(the parsed, typed config: instrument, image series, analysis parameters) and
a :class:`~hexrd.core.material.material_data.Material` (the crystal, exposing
a frozen PlaneData of arrays).  No stage reaches beyond those.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numba
import numpy as np

from hexrd.core import constants as const
from hexrd.core import distortion as distortion_pkg
from hexrd.core import matrixutil as mutil
from hexrd.core import rotations
from hexrd.hedm.experiment import (
    ClusteringAlgorithm, HedmExperiment, SeedSearchMethod,
)
from hexrd.core.extensions import transforms, transforms_c_api
from hexrd.core.material.material_data import Material, PlaneData
from hexrd.core.transforms import xfcapi

# scipy.ndimage and sklearn.cluster are only needed by the later stages
# (peak finding, clustering), so pull them in on a background thread while
# the earlier, numpy/IO-bound stages run.  The functions that use them do a
# local import, which by then is a dict lookup.
def _prefetch_heavy_imports():
    from scipy import ndimage  # noqa: F401
    from sklearn.cluster import dbscan  # noqa: F401


@dataclass
class EtaOmegaMaps:
    """Diffraction intensity binned over (omega, eta) for each active ring."""
    ring_maps: np.ndarray         # (n_rings, n_omega, n_eta) float64
    eta_edges: np.ndarray         # (n_eta + 1,)
    omega_edges: np.ndarray       # (n_omega + 1,)
    omegas: np.ndarray            # (n_omega,) bin centers
    omega_period: np.ndarray      # (2,) [start, start + 2*pi]
    two_theta_ranges: np.ndarray  # (n_rings, 2)
    ring_ids: np.ndarray          # (n_rings,) index among the non-excluded hkls
    eta_step: float               # radians


@dataclass
class FindOrientationsResult:
    """What find_orientations returns: the grains found, plus every scored trial.

    grain_orientations : (4, n_grains)  accepted grain orientations, as quaternions
    test_orientations  : (4, n_trials)  every trial orientation that was scored
    completeness       : (n_trials,)    completeness score per trial, in [0, 1]
    """
    grain_orientations: np.ndarray
    test_orientations: np.ndarray
    completeness: np.ndarray

    @property
    def num_grains(self) -> int:
        return self.grain_orientations.shape[1]


def _prefetch_skimage():
    from skimage.exposure import rescale_intensity  # noqa: F401
    from skimage.feature import blob_dog, blob_log  # noqa: F401


def find_orientations(experiment: HedmExperiment, material: Optional[Material] = None,
                      clean: bool = False) -> FindOrientationsResult:
    """Find every grain orientation in the rotation series.

    clean regenerates the eta-omega maps even when a cached file exists.
    """
    threading.Thread(target=_prefetch_heavy_imports, daemon=True).start()

    if material is None:
        material = experiment.get_active_material()
    plane_data = material.plane_data

    # scikit-image is only needed once the maps exist; import it while they build
    if experiment.find_orientations.seed_search.method is not SeedSearchMethod.LABEL:
        threading.Thread(target=_prefetch_skimage, daemon=True).start()

    maps = load_or_build_eta_omega_maps(experiment, plane_data, clean=clean)

    grid_file = experiment.quaternion_grid_file
    if grid_file is not None:
        # grid search: score a precomputed set of trial quaternions (4, n)
        fibers = np.load(grid_file)
        min_samples = 1
    else:
        fibers = generate_orientation_fibers(experiment, plane_data, maps)
        min_samples = estimate_min_samples(experiment, plane_data, maps)

    completeness = score_orientations(experiment, plane_data, maps, fibers)
    qbar = cluster_grains(experiment, plane_data, fibers, completeness,
                          min_samples=min_samples)

    return FindOrientationsResult(
        grain_orientations=qbar,
        test_orientations=fibers,
        completeness=completeness,
    )


# ---------------------------------------------------------------------------
# 1. eta-omega maps
# ---------------------------------------------------------------------------
def load_or_build_eta_omega_maps(experiment: HedmExperiment, plane_data: PlaneData,
                                 clean: bool = False) -> EtaOmegaMaps:
    """The eta-omega maps for this analysis, cached at experiment.eta_ome_maps_file.

    An existing cache is loaded unless `clean`; otherwise the maps are built
    and saved. The optional filter (orientation_maps: filter_maps) applies to
    loaded and freshly-built maps alike -- except under `clean`, which skips
    it (mirroring the long-standing behavior).
    """
    om = experiment.find_orientations.orientation_maps
    path = experiment.eta_ome_maps_file
    if clean or not os.path.exists(path):
        maps = build_eta_omega_maps(experiment, plane_data)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        _save_eta_omega_maps(maps, path)
        if clean:
            return maps
    else:
        maps = _load_eta_omega_maps(path, plane_data,
                                    experiment.active_material.two_theta_width)
    _filter_maps(maps, om.filter_maps, om.filter_fwhm)
    return maps


def _resolve_active_rings(active_hkls: Optional[np.ndarray],
                          plane_data: PlaneData) -> np.ndarray:
    """Ring indices (into the non-excluded hkl list) for the active hkls,
    given as [h, k, l] vectors or as master hkl IDs.

    None selects every non-excluded ring.
    """
    if active_hkls is None:
        return np.arange(len(plane_data.unexcluded_hkls))

    if active_hkls.ndim == 1:              # master hkl IDs
        bad = [i for i in active_hkls.tolist()
               if i >= len(plane_data.hkls) or plane_data.exclusions[i]]
        if bad:
            raise ValueError(
                f'active hkl IDs {bad} are not non-excluded hkls of the material')
        ring_of_hkl = np.cumsum(~plane_data.exclusions) - 1
        rings = ring_of_hkl[active_hkls]
    else:
        ring_of_vec = {tuple(h): i
                       for i, h in enumerate(plane_data.unexcluded_hkls.tolist())}
        missing = [h for h in active_hkls.tolist() if tuple(h) not in ring_of_vec]
        if missing:
            raise ValueError(
                f'active hkls {missing} are not non-excluded hkls of the material')
        rings = np.array([ring_of_vec[tuple(h)] for h in active_hkls.tolist()])
    if len(np.unique(rings)) != len(rings):
        raise ValueError('duplicate active_hkls specified')
    return rings


def build_eta_omega_maps(experiment: HedmExperiment,
                         plane_data: PlaneData) -> EtaOmegaMaps:
    """Accumulate every detector panel's intensity into per-ring (omega, eta) maps.

    Intensities are accumulated in float64: frame caches are often uint16,
    and a bin's summed intensity can exceed the native dtype's range.
    """
    fo = experiment.find_orientations

    ring_ids = _resolve_active_rings(fo.orientation_maps.active_hkls, plane_data)

    # a two-theta window around each active powder ring
    half = np.radians(experiment.active_material.two_theta_width) / 2
    ring_tths = plane_data.two_thetas[~plane_data.exclusions][ring_ids]
    two_theta_ranges = np.column_stack([ring_tths - half, ring_tths + half])

    eta_step = np.radians(fo.orientation_maps.eta_step)
    eta_edges = _eta_bin_edges(eta_step)
    n_eta = len(eta_edges) - 1
    row_of_frame, omegas, omega_edges, period = _omega_grid(
        experiment.image_series_list[0].omega)
    ring_maps = np.full((len(two_theta_ranges), len(omegas), n_eta), np.nan)

    # Pair each image series with its detector panel (by name) and bin every
    # frame's intensity into the shared map. Summing across panels is what makes
    # multi-panel instruments (e.g. Dexela) work.  Frames are sparse, so only
    # the stored pixels are binned, in a numba kernel parallel over frames
    # (frame intensities are integer-valued, and float64 sums of integers are
    # exact in any order).
    numba.set_num_threads(
        max(1, min(experiment.max_workers, numba.config.NUMBA_NUM_THREADS)))
    active = [ims for ims in experiment.image_series_list
              if ims.panel in experiment.detectors]

    # each panel's pixel angles and each series' frame decompression are
    # independent, and the C geometry routine releases the GIL: overlap them
    # all in one thread pool
    with ThreadPoolExecutor(max(1, min(experiment.max_workers, 16))) as pool:
        angle_futures = {panel: pool.submit(
            _panel_pixel_angles, experiment.detectors[panel], experiment.beam.vector)
                         for panel in {ims.panel for ims in active}}
        for ims in active:
            pool.submit(lambda series: series.images, ims)
        panel_angles = {panel: f.result() for panel, f in angle_futures.items()}

    for ims in active:
        ptth, peta = panel_angles[ims.panel]
        # np.histogram on an explicit bin array searches exactly like this,
        # with the last bin closed on the right
        eta_bin = np.minimum(
            np.searchsorted(eta_edges, peta, side='right') - 1, n_eta - 1)
        ring_pixels = [_ring_pixels(r, ptth, peta, eta_edges)
                       for r in two_theta_ranges]

        # a bin is NaN until a panel covers it; from there intensity adds up,
        # across overlapping panels too
        rows = row_of_frame[:len(ims.images)]
        for ring, rp in enumerate(ring_pixels):
            if rp is None:
                continue
            covered = np.ix_(rows, rp.bins)
            ring_maps[ring][covered] = np.nan_to_num(ring_maps[ring][covered])

        # pack ring membership into per-pixel bitmasks (63 rings per pass)
        threshold = float(fo.orientation_maps.threshold)
        for first in range(0, len(ring_pixels), 63):
            block = ring_pixels[first:first + 63]
            if all(rp is None for rp in block):
                continue
            ring_mask = np.zeros(ptth.size, dtype=np.int64)
            for bit, rp in enumerate(block):
                if rp is not None:
                    ring_mask[rp.in_ring] |= 1 << bit
            for ids, vals, offsets, chunk_rows in _pixel_chunks(
                    ims.images, rows):
                _bin_pixels(offsets, ids, vals, chunk_rows, ring_mask,
                            eta_bin, threshold,
                            ring_maps[first:first + 63])

    return EtaOmegaMaps(ring_maps, eta_edges, omega_edges, omegas, period,
                        two_theta_ranges, ring_ids, eta_step)


def _pixel_chunks(frames, row_of_frame, max_pixels: int = 8_000_000):
    """The stored pixels of consecutive frames, concatenated into flat arrays
    bounded by ``max_pixels``: (pixel_ids, values, frame offsets, map rows)."""
    start = 0
    while start < len(frames):
        stop, total = start, 0
        while stop < len(frames) and (
                stop == start or total + frames[stop].nnz <= max_pixels):
            total += frames[stop].nnz
            stop += 1
        chunk = [f.tocoo() for f in frames[start:stop]]
        if total > 0:
            ids = np.concatenate(
                [c.row.astype(np.int64) * c.shape[1] + c.col for c in chunk])
            vals = np.concatenate([c.data for c in chunk]).astype(np.float64)
            offsets = np.r_[0, np.cumsum([c.nnz for c in chunk])]
            yield ids, vals, offsets, np.asarray(row_of_frame[start:stop])
        start = stop


@numba.njit(nogil=True, cache=True, parallel=True)
def _bin_pixels(offsets, ids, vals, rows, ring_mask, eta_bin, threshold, out):
    """Add every stored pixel's intensity to its (ring, omega row, eta bin).

    Parallel over frames: within one image series every frame owns its own
    map row, so writes never collide.
    """
    for f in numba.prange(len(rows)):
        row = rows[f]
        for k in range(offsets[f], offsets[f + 1]):
            value = vals[k]
            if value < threshold:
                continue
            pixel = ids[k]
            mask = ring_mask[pixel]
            ring = 0
            while mask:
                if mask & 1:
                    out[ring, row, eta_bin[pixel]] += value
                mask >>= 1
                ring += 1


def _omega_grid(omega: np.ndarray):
    """Map rows and the omega axis for a scan's (n_frames, 2) omega spans.

    A contiguous scan gets one row per frame with the scan's own frame
    boundaries as bin edges.  A multi-wedge scan (gaps between frames) gets
    a uniform grid over the whole range: frames land on their grid row and
    the gap rows stay all-NaN, which excludes them from scoring the same
    way off-detector bins are excluded.

    Returns (row_of_frame, omegas, omega_edges, omega_period).
    """
    period = omega[0, 0] + np.r_[0.0, 2 * np.pi]
    span = period[1] - period[0]

    if np.allclose(omega[1:, 0], omega[:-1, 1]):     # contiguous scan
        omegas = np.mod(np.average(omega, axis=1) - period[0], span) + period[0]
        edges = np.mod(
            np.r_[omega[:, 0], omega[-1, 1]] - period[0], span) + period[0]
        # an exact full-circle scan wraps the final edge back onto the period
        # start; keep the edges monotonic for the scorer's binary search
        if edges[-1] <= edges[-2]:
            edges[-1] += span
        return np.arange(len(omega)), omegas, edges, period

    widths = omega[:, 1] - omega[:, 0]
    delta = float(widths[0])
    if not np.allclose(widths, delta):
        raise ValueError('a multi-wedge scan must use one omega step size')
    lo = period[0]
    rows_exact = (np.mod(omega[:, 0] - lo, span) + lo - lo) / delta
    rows = np.rint(rows_exact).astype(int)
    if not np.allclose(rows_exact, rows, atol=1e-3):
        raise ValueError('omega wedges must align to the frame-width grid')
    n_rows = rows[-1] + 1
    edges = lo + delta * np.arange(n_rows + 1)
    omegas = lo + delta * (np.arange(n_rows) + 0.5)
    return rows, omegas, edges, period


def _save_eta_omega_maps(maps: EtaOmegaMaps, path: str) -> None:
    np.savez_compressed(
        path,
        ring_maps=maps.ring_maps,
        eta_edges=maps.eta_edges, omega_edges=maps.omega_edges,
        omegas=maps.omegas, omega_period=maps.omega_period,
        two_theta_ranges=maps.two_theta_ranges, ring_ids=maps.ring_ids,
        eta_step=maps.eta_step)


def _load_eta_omega_maps(path: str, plane_data: PlaneData,
                         two_theta_width: float) -> EtaOmegaMaps:
    """Load a maps archive: the plain-array format written above, or a legacy
    hexrd eta-ome-maps npz."""
    with np.load(path) as d:
        if 'dataStore' in d:  # legacy hexrd archive
            ring_maps = np.array(d['dataStore'], dtype=np.float64)
            eta_edges, omega_edges = d['etaEdges'], d['omeEdges']
            omegas = d['omegas']
            # the archive's iHKLList holds master hkl IDs; convert to ring indices
            ring_of_hkl = np.cumsum(~plane_data.exclusions) - 1
            ring_ids = ring_of_hkl[np.asarray(d['iHKLList'])]
            half = np.radians(two_theta_width) / 2
            tths = plane_data.two_thetas[~plane_data.exclusions][ring_ids]
            two_theta_ranges = np.column_stack([tths - half, tths + half])
            period = omega_edges[0] + np.r_[0.0, 2 * np.pi]
            eta_step = float(d['etas'][1] - d['etas'][0])
        else:                 # plain-array archive
            ring_maps = d['ring_maps']
            eta_edges, omega_edges = d['eta_edges'], d['omega_edges']
            omegas, period = d['omegas'], d['omega_period']
            two_theta_ranges, ring_ids = d['two_theta_ranges'], d['ring_ids']
            eta_step = float(d['eta_step'])
    return EtaOmegaMaps(ring_maps, eta_edges, omega_edges, omegas, period,
                        two_theta_ranges, ring_ids, eta_step)


def _filter_maps(maps: EtaOmegaMaps, median: bool,
                 log_fwhm: Optional[float]) -> None:
    """In place: subtract each eta column's median (removes streak artifacts)
    and, when log_fwhm is given, apply a Laplacian-of-Gaussian filter of that
    FWHM in pixels."""
    if not median:
        return
    from scipy import ndimage

    for ring_map in maps.ring_maps:
        ring_map -= np.tile(np.nanmedian(ring_map, axis=0), (len(ring_map), 1))
        if log_fwhm is not None:
            ring_map[:] = -ndimage.gaussian_laplace(
                ring_map, const.fwhm_to_sigma * log_fwhm)


def _eta_bin_edges(step: float) -> np.ndarray:
    """Azimuthal bin edges over the full circle, branch-cut at +/- pi."""
    n = int(2 * np.pi / step)
    ang = np.nan_to_num(step * np.linspace(0.0, n, num=n + 1) - np.pi)
    edges = np.mod(ang + np.pi, 2 * np.pi) - np.pi
    edges[np.logical_and(edges == -np.pi, ang > -np.pi)] = np.pi
    return edges


def _panel_pixel_angles(detector, beam_vector):
    """Map a panel's pixels to (two-theta, eta) using its geometry and distortion."""
    rmat_b = np.eye(3)
    grid_i, grid_j = detector.pixel_coordinates
    xy = np.empty((grid_i.size, 2))
    xy[:, 0] = grid_j.ravel()
    xy[:, 1] = grid_i.ravel()
    xy = _apply_distortion(detector, xy)
    ac = np.ascontiguousarray
    (ptth, peta), _ = transforms_c_api.detectorXYToGvec(
        ac(xy),
        ac(detector.transform.rotation_matrix),
        ac(rmat_b),
        ac(detector.transform.translation).ravel(),
        np.zeros(3), np.zeros(3),
        ac(beam_vector), ac(rmat_b[:, 0]))
    return ptth, peta


def _apply_distortion(detector, xy: np.ndarray) -> np.ndarray:
    """Apply a panel's distortion model (from the distortion registry) to
    (col, row) coordinates; panels without one pass through unchanged."""
    spec = detector.distortion
    if not spec.function_name:
        return xy
    mapping = distortion_pkg.get_mapping(spec.function_name, spec.parameters)
    return mapping.apply(xy)


class _RingPixels(NamedTuple):
    """A panel's pixels on one ring: a membership mask over the panel's flat
    (row-major) pixel indices, and the eta bins the ring covers on it."""
    in_ring: np.ndarray
    bins: np.ndarray


def _ring_pixels(tth_range: np.ndarray, panel_tth: np.ndarray,
                 panel_eta: np.ndarray,
                 eta_edges: np.ndarray) -> Optional[_RingPixels]:
    """Which of a panel's pixels fall in a ring; None if the ring misses it."""
    in_ring = np.logical_and(panel_tth >= tth_range[0], panel_tth <= tth_range[1])
    if not np.any(in_ring):
        return None
    bins = np.where(np.histogram(panel_eta[in_ring], bins=eta_edges)[0])[0]
    return _RingPixels(in_ring, bins)


# ---------------------------------------------------------------------------
# 2. orientation fibers from seed peaks
# ---------------------------------------------------------------------------
def generate_orientation_fibers(experiment: HedmExperiment, plane_data: PlaneData,
                                maps: EtaOmegaMaps) -> np.ndarray:
    """Detect peaks in the seed-ring maps and expand each into a fiber of trial quats."""
    fo = experiment.find_orientations
    chi = experiment.oscillation_stage.chi
    csym = plane_data.laue_group
    fiber_ndiv = fo.seed_search.fiber_ndiv

    # hkl_seeds index into the active rings
    seed_ids = fo.seed_search.hkl_seeds
    if seed_ids.size == 0:
        raise ValueError('seed_search: hkl_seeds must be given for seeded search')
    seed_hkls = plane_data.unexcluded_hkls[maps.ring_ids[seed_ids]]
    seed_tths = np.average(maps.two_theta_ranges, axis=1)[seed_ids]
    d_omega = maps.omegas[1] - maps.omegas[0]

    # Find spots (peaks) in each seed-ring map (on a copy: scoring needs the
    # original, un-cleaned maps), then expand each spot into a discrete fiber
    # of candidate orientations.  Rings are independent and so are the spots,
    # so both passes run in a thread pool; order is preserved throughout.
    with ThreadPoolExecutor(max(1, min(experiment.max_workers, 8))) as pool:
        spots = list(pool.map(
            lambda i: _find_peaks(maps.ring_maps[i].copy(), fo.seed_search),
            seed_ids))

        centers = []
        for hkl, tth, (num_spots, coms) in zip(seed_hkls, seed_tths, spots):
            for ispot in range(num_spots):
                com = coms[ispot]
                if np.isnan(com[0]):
                    continue
                ome_c = maps.omega_edges[0] + (0.5 + com[0]) * d_omega
                eta_c = maps.eta_edges[0] + (0.5 + com[1]) * maps.eta_step
                centers.append((hkl, tth, eta_c, ome_c))
        q_fibers = list(pool.map(
            lambda c: _fiber(*c, chi, plane_data.B, fiber_ndiv, csym),
            centers))
    return np.hstack(q_fibers)


def _find_peaks(ring_map: np.ndarray, seed_search) -> tuple[int, np.ndarray]:
    """Number of spots and their (omega, eta) centers in one ring map,
    dispatching on the config's seed-search method."""
    _clean_map(ring_map)
    method, kwargs = seed_search.method, seed_search.method_kwargs

    if method is SeedSearchMethod.LABEL:
        from scipy import ndimage

        if kwargs.get('filter_radius'):
            stdev = const.fwhm_to_sigma * kwargs['filter_radius']
            ring_map = -ndimage.gaussian_laplace(ring_map, stdev)
        labels, num_spots = ndimage.label(
            ring_map > kwargs.get('threshold', 1),
            ndimage.generate_binary_structure(2, 1))
        coms = np.atleast_2d(ndimage.center_of_mass(
            ring_map, labels=labels, index=np.arange(1, np.amax(labels) + 1)))
        return num_spots, coms

    from skimage.exposure import rescale_intensity
    from skimage.feature import blob_dog, blob_log

    scl_map = rescale_intensity(ring_map, out_range=(-1, 1))
    detect = blob_log if method is SeedSearchMethod.BLOB_LOG else blob_dog
    blobs = np.atleast_2d(detect(scl_map, **kwargs))
    return len(blobs), blobs[:, :2]


def _clean_map(m: np.ndarray) -> None:
    """In place: fill NaN eta-gaps with the median, floor at it, and zero-base."""
    nan = np.isnan(m)
    med = np.median(m[~nan])
    m[nan] = med
    m[m <= med] = med
    m -= np.min(m)


def _fiber(hkl, tth, eta_c, ome_c, chi, b_matrix, fiber_ndiv, csym):
    """The discrete fiber of orientations consistent with one (tth, eta, omega)
    spot, reduced to unique quaternions in the fundamental region."""
    g_vec_s = xfcapi.angles_to_gvec(
        np.atleast_2d([tth, eta_c, ome_c]), chi=chi).T
    fiber = rotations.discreteFiber(
        hkl.reshape(3, 1), g_vec_s, B=b_matrix, ndiv=fiber_ndiv,
        invert=False, csym=csym)[0]
    return mutil.uniqueVectors(fiber)


# ---------------------------------------------------------------------------
# 3. completeness scoring
# ---------------------------------------------------------------------------
def score_orientations(experiment: HedmExperiment, plane_data: PlaneData,
                       maps: EtaOmegaMaps, q_fibers: np.ndarray) -> np.ndarray:
    """Completeness (fraction of expected reflections observed) for each trial
    orientation.

    This is the algorithm historically known as paintGrid, as one parallel
    numba pass shared by all trials.
    """
    fo = experiment.find_orientations

    # stack the active rings' symmetry-equivalent HKLs and remember which
    # map each belongs to
    symm_hkls_per_ring = [plane_data.symm_hkls[r] for r in maps.ring_ids]
    ring_for_hkl = np.repeat(np.arange(len(symm_hkls_per_ring)),
                             [s.shape[1] for s in symm_hkls_per_ring])
    symm_hkls = np.ascontiguousarray(
        np.vstack([s.T for s in symm_hkls_per_ring]), dtype=np.float64)

    eta_ranges = fo.eta.range
    ome_offset = maps.omega_period.min()
    valid_eta = normalize_ranges(eta_ranges[:, 0], eta_ranges[:, 1], -np.pi)
    valid_ome = normalize_ranges(
        np.array([maps.omega_edges.min()]),
        np.array([maps.omega_edges.max()]), ome_offset)
    dpix_ome = int(round(
        fo.omega.tolerance / abs(maps.omega_edges[1] - maps.omega_edges[0])))
    dpix_eta = int(round(
        fo.eta.tolerance / abs(maps.eta_edges[1] - maps.eta_edges[0])))

    ring_maps = np.ascontiguousarray(maps.ring_maps)
    bmat = np.ascontiguousarray(plane_data.B)

    beam_vector = experiment.beam.vector
    n = q_fibers.shape[1]
    angs = np.empty((2, n, len(symm_hkls), 3))
    rmats = rotations.rotMatOfQuat(q_fibers).reshape(n, 3, 3)
    for i, rmat in enumerate(rmats):
        angs[0, i], angs[1, i] = xfcapi.oscill_angles_of_hkls(
            symm_hkls, 0.0, rmat, bmat, plane_data.wavelength,
            beam_vec=beam_vector)

    gpu = _cuda_scorer(n)
    if gpu is not None:
        return gpu.count_hits_all(
            angs, ring_for_hkl, maps.eta_edges, maps.omega_edges,
            valid_eta, valid_ome, ome_offset, ring_maps,
            dpix_eta, dpix_ome, float(fo.threshold))
    numba.set_num_threads(
        max(1, min(experiment.max_workers, numba.config.NUMBA_NUM_THREADS)))
    return _count_hits_all(
        angs[0], angs[1], ring_for_hkl, maps.eta_edges, maps.omega_edges,
        valid_eta, valid_ome, ome_offset, ring_maps,
        dpix_eta, dpix_ome, float(fo.threshold))


# below this many trials, CUDA context setup outweighs the scoring itself
# in a fresh process, so seeded searches stay on the CPU kernels
_GPU_MIN_TRIALS = 100_000


def _cuda_scorer(n_trials: int):
    """The GPU scorer module when it's worth using, else None (CPU kernels).

    Both paths give bit-identical scores.  A usable CUDA device is picked up
    automatically for large searches (e.g. quaternion grids); set HEXRD_GPU=1
    to use it for any size, or HEXRD_DISABLE_GPU=1 to never use it.
    """
    def _set(name):
        return os.environ.get(name, '0').lower() not in ('0', '', 'false')

    if _set('HEXRD_DISABLE_GPU'):
        return None
    if n_trials < _GPU_MIN_TRIALS and not _set('HEXRD_GPU'):
        return None
    try:
        from hexrd.hedm import find_orientations_gpu
    except Exception:
        return None
    return find_orientations_gpu if find_orientations_gpu.available() else None


def normalize_ranges(starts: np.ndarray, stops: np.ndarray,
                     offset: float) -> np.ndarray:
    """Normalize (start, stop) angle ranges into ``[offset, offset + 2*pi)``.

    Returns a flat, sorted ``[start, stop, start, stop, ...]`` array; an
    angle mapped into the window lies inside a valid span exactly when its
    :func:`_find_in_range` insertion index is odd.  A range spanning the
    full circle collapses to ``[offset, offset + 2*pi]``.
    """
    if not np.all(starts < stops):
        raise ValueError('Invalid angle ranges')

    two_pi = 2 * np.pi
    if np.any((starts + two_pi) < stops + 1e-8):
        return np.array([offset, two_pi + offset])

    starts = np.mod(starts - offset, two_pi) + offset
    stops = np.mod(stops - offset, two_pi) + offset

    order = np.argsort(starts)
    result = np.hstack(
        (starts[order, np.newaxis], stops[order, np.newaxis])).ravel()
    # wrap-around in the last segment splits into a leading span
    if result[-1] < result[-2]:
        new_result = np.empty((len(result) + 2,), dtype=result.dtype)
        new_result[0] = offset
        new_result[1] = result[-1]
        new_result[2:-1] = result[0:-1]
        new_result[-1] = offset + two_pi
        result = new_result

    # any overlap between ranges shows up as an inversion in the
    # interleaved [start, stop, start, stop, ...] sequence
    if not np.all(np.diff(result) >= 0):
        raise ValueError('Angle ranges overlap')

    return result


@numba.njit(nogil=True, cache=True)
def _find_in_range(value, spans):
    """Index i such that spans[i-1] <= value < spans[i]; -2 if out of range.

    With spans as an interleaved, sorted [start, stop, start, stop, ...]
    array (see normalize_ranges), an odd result means value lies inside a
    valid span.
    """
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


@numba.njit(nogil=True, cache=True)
def _dilated_hit(eta, ome, dpix_eta, dpix_ome, ring_map, threshold):
    """1 if any map value above threshold in the tolerance window, -1 on NaN, else 0."""
    n_ome, n_eta = ring_map.shape
    for i in range(max(ome - dpix_ome, 0), min(ome + dpix_ome + 1, n_ome)):
        for j in range(max(eta - dpix_eta, 0), min(eta + dpix_eta + 1, n_eta)):
            if ring_map[i, j] > threshold:
                return 1
            if np.isnan(ring_map[i, j]):
                return -1
    return 0


@numba.njit(nogil=True, cache=True, parallel=True)
def _count_hits_all(angs_0, angs_1, ring_for_hkl, eta_edges, ome_edges,
                    valid_eta_spans, valid_ome_spans, ome_offset, ring_maps,
                    dpix_eta, dpix_ome, threshold):
    """Per-trial completeness: _count_hits for each trial, in parallel."""
    scores = np.empty(angs_0.shape[0])
    for q in numba.prange(angs_0.shape[0]):
        scores[q] = _count_hits(
            angs_0[q], angs_1[q], ring_for_hkl, eta_edges, ome_edges,
            valid_eta_spans, valid_ome_spans, ome_offset, ring_maps,
            dpix_eta, dpix_ome, threshold)
    return scores


@numba.njit(nogil=True, cache=True)
def _count_hits(angs_0, angs_1, ring_for_hkl, eta_edges, ome_edges,
                valid_eta_spans, valid_ome_spans, ome_offset, ring_maps,
                dpix_eta, dpix_ome, threshold):
    """hits / total over both Bragg solutions of every symmetry-equivalent HKL.

    A reflection counts toward the total if its predicted (eta, omega) lands
    in the valid spans and on the maps (NaN map bins don't count); it is a
    hit if the map exceeds the threshold anywhere within the
    (dpix_eta, dpix_ome) tolerance window.
    """
    hits = 0
    total = 0
    for solution in (angs_0, angs_1):
        for i in range(len(solution)):
            if np.isnan(solution[i, 0]):
                continue
            eta = np.mod(solution[i, 1] + np.pi, 2 * np.pi) - np.pi
            if _find_in_range(eta, valid_eta_spans) & 1 == 0:
                continue
            ome = np.mod(solution[i, 2] - ome_offset, 2 * np.pi) + ome_offset
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
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 4. clustering into grains
# ---------------------------------------------------------------------------
def estimate_min_samples(experiment: HedmExperiment, plane_data: PlaneData,
                         maps: EtaOmegaMaps, n_grains: int = 100) -> int:
    """dbscan's min_samples, from the seed reflections a typical grain produces.

    Simulates random grain orientations over the instrument, counts how many
    seed-ring reflections each would produce inside the eta/omega ranges and
    on a panel, and takes half the completeness threshold times the worst
    grain's count.  The random draw is seeded, so runs are reproducible.
    Panel distortion is ignored (sub-pixel effect on counts).
    """
    fo = experiment.find_orientations
    chi = experiment.oscillation_stage.chi
    bmat = np.ascontiguousarray(plane_data.B)

    # every non-excluded ring's symmetric equivalents
    symm_per_ring = plane_data.symm_hkls
    ring_for_hkl = np.repeat(np.arange(len(symm_per_ring)),
                             [s.shape[1] for s in symm_per_ring])
    symm_hkls = np.ascontiguousarray(
        np.vstack([s.T for s in symm_per_ring]), dtype=np.float64)
    seed_rings = set(maps.ring_ids[fo.seed_search.hkl_seeds].tolist())

    # valid spans as interleaved [start, stop, ...] arrays; a value is inside
    # a span iff its insertion index is odd (same machinery as the scorer)
    eta_ranges = fo.eta.range
    valid_eta = normalize_ranges(eta_ranges[:, 0], eta_ranges[:, 1], -np.pi)
    omega = experiment.image_series_list[0].omega
    ome_lo = omega[0, 0]
    # one span per omega wedge, so gaps in a multi-wedge scan don't count
    new_wedge = np.r_[True, ~np.isclose(omega[1:, 0], omega[:-1, 1])]
    valid_ome = normalize_ranges(omega[new_wedge, 0],
                                 omega[np.r_[new_wedge[1:], True], 1], ome_lo)

    def in_spans(values, spans):
        idx = np.searchsorted(spans, values, side='right')
        return (idx % 2 == 1) & (values >= spans[0]) & (values < spans[-1])

    rng = np.random.default_rng(0)
    quats = rng.normal(size=(4, n_grains))
    quats /= np.linalg.norm(quats, axis=0)

    ac = np.ascontiguousarray
    seed_refl_per_grain = np.empty(n_grains)
    for g in range(n_grains):
        rmat_c = ac(rotations.rotMatOfQuat(quats[:, g]))
        a0, a1 = xfcapi.oscill_angles_of_hkls(
            symm_hkls, chi, rmat_c, bmat, plane_data.wavelength,
            beam_vec=experiment.beam.vector)
        angs = np.vstack([a0, a1])
        rings = np.tile(ring_for_hkl, 2)
        ok = ~np.isnan(angs[:, 0])

        eta = np.mod(angs[:, 1] + np.pi, 2 * np.pi) - np.pi
        ok &= in_spans(eta, valid_eta)
        ome = np.mod(angs[:, 2] - ome_lo, 2 * np.pi) + ome_lo
        ok &= in_spans(ome, valid_ome)

        # project onto each panel; a reflection counts once per panel it hits
        count = 0
        angs, rings = angs[ok], rings[ok]
        gvecs = xfcapi.angles_to_gvec(angs, chi=chi, rmat_c=rmat_c)
        rmats_s = _sample_rmats(chi, angs[:, 2])
        for detector in experiment.detectors.values():
            xy = transforms_c_api.gvecToDetectorXYArray(
                ac(gvecs), ac(detector.transform.rotation_matrix), rmats_s,
                rmat_c, ac(detector.transform.translation).ravel(),
                np.zeros(3), np.zeros(3), experiment.beam.vector)
            on_panel = _clip_to_panel(detector, xy)
            count += np.isin(rings[on_panel], list(seed_rings)).sum()
        seed_refl_per_grain[g] = count

    return max(int(np.floor(
        0.5 * fo.clustering.completeness * seed_refl_per_grain.min())), 2)


def _sample_rmats(chi: float, omes: np.ndarray) -> np.ndarray:
    """Sample-frame rotation R_x(chi) @ R_y(omega) for each omega."""
    cx, sx = np.cos(chi), np.sin(chi)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    co, so = np.cos(omes), np.sin(omes)
    ry = np.zeros((len(omes), 3, 3))
    ry[:, 0, 0], ry[:, 0, 2] = co, so
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0], ry[:, 2, 2] = -so, co
    return np.ascontiguousarray(rx @ ry)


def _clip_to_panel(detector, xy: np.ndarray) -> np.ndarray:
    """Points inside the panel's extents, trimmed by its edge buffer (mm)."""
    half_x = 0.5 * detector.pixels.columns * detector.pixels.size[1]
    half_y = 0.5 * detector.pixels.rows * detector.pixels.size[0]
    buf_x, buf_y = detector.buffer if detector.buffer is not None else (0.0, 0.0)
    with np.errstate(invalid='ignore'):
        return ((np.abs(xy[:, 0]) <= half_x - buf_x)
                & (np.abs(xy[:, 1]) <= half_y - buf_y))


def cluster_grains(experiment: HedmExperiment, plane_data: PlaneData,
                   q_fibers: np.ndarray, completeness: np.ndarray,
                   min_samples: int = 1) -> np.ndarray:
    """Cluster the high-completeness trials and average each cluster into one grain.

    The algorithm comes from the config (dbscan, ort-dbscan, sph-dbscan, or
    fclusterdata); the quaternion-metric algorithms fall back to euclidean
    orthographic DBSCAN above 25000 candidates (they are O(n^2)), and the
    euclidean algorithms get a final duplicate-merging pass.
    """
    Algo = ClusteringAlgorithm

    fo = experiment.find_orientations
    radius = fo.clustering.radius
    qsym = np.ascontiguousarray(plane_data.q_sym)

    keep = q_fibers[:, np.asarray(completeness) > fo.clustering.completeness]
    n = keep.shape[1]
    if n == 0:
        return np.zeros((4, 0))
    if n == 1:
        return keep.copy()

    algorithm = fo.clustering.algorithm
    if n > 25000 and algorithm in (Algo.SPH_DBSCAN, Algo.FCLUSTERDATA):
        algorithm = Algo.ORT_DBSCAN

    def quat_distance(x, y):
        return xfcapi.quat_distance(np.ascontiguousarray(x),
                                    np.ascontiguousarray(y), qsym)

    if algorithm is Algo.FCLUSTERDATA:
        from scipy.cluster.hierarchy import fclusterdata
        labels = fclusterdata(keep.T, np.radians(radius),
                              criterion='distance', metric=quat_distance)
        labels = np.asarray(labels, dtype=int)  # already 1..N, no noise
    else:
        from sklearn.cluster import dbscan

        if algorithm is Algo.SPH_DBSCAN:
            from sklearn.metrics.pairwise import pairwise_distances
            pdist = pairwise_distances(keep.T, metric=quat_distance, n_jobs=1)
            _, labels = dbscan(pdist, eps=np.radians(radius),
                               min_samples=min_samples, metric='precomputed')
        else:
            if algorithm is Algo.ORT_DBSCAN:
                pts, eps = keep[1:, :].T, 0.25 * np.radians(radius)
            else:
                pts, eps = keep.T, 0.5 * np.radians(radius)
            _, labels = dbscan(pts, eps=eps, min_samples=min_samples,
                               metric='minkowski', p=2, n_jobs=1)

        # relabel: clusters 1..N, noise -1
        labels = np.asarray(labels, dtype=int)
        noise = labels == -1
        labels += 1
        labels[noise] = -1

    n_grains = len(np.unique(labels)) - (1 if -1 in labels else 0)
    qbar = np.zeros((4, n_grains))
    for i in range(n_grains):
        members = keep[:, labels == i + 1]
        qbar[:, i] = rotations.quatAverageCluster(members, qsym).flatten()

    # the euclidean algorithms can split one orientation across clusters;
    # merge any centroids closer than the clustering radius
    if algorithm in (Algo.DBSCAN, Algo.ORT_DBSCAN) and n_grains > 1:
        qbar = _merge_duplicates(qbar, qsym, quat_distance, radius)
    return qbar


def _merge_duplicates(qbar, qsym, quat_distance, radius) -> np.ndarray:
    from scipy.cluster.hierarchy import fclusterdata

    cl = fclusterdata(qbar.T, np.radians(radius),
                      criterion='distance', metric=quat_distance)
    n_merged = len(np.unique(cl))
    if n_merged == qbar.shape[1]:
        return qbar
    merged = np.zeros((4, n_merged))
    for i in range(n_merged):
        members = qbar[:, cl == i + 1].reshape(4, np.sum(cl == i + 1))
        merged[:, i] = rotations.quatAverageCluster(members, qsym).flatten()
    return merged


# ---------------------------------------------------------------------------
# 5. output
# ---------------------------------------------------------------------------
def write_results(results: FindOrientationsResult,
                  experiment: HedmExperiment) -> str:
    """Write find-orientations outputs in the standard file formats;
    returns the analysis directory."""
    qbar = np.atleast_2d(results.grain_orientations)
    analysis_dir = experiment.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)
    actmat = experiment.active_material.active

    np.savetxt(os.path.join(analysis_dir, f'accepted-orientations-{actmat}.dat'),
               qbar.T, fmt='%.18e', delimiter='\t')
    np.savez_compressed(
        os.path.join(analysis_dir, f'scored-orientations-{actmat}.npz'),
        test_quaternions=results.test_orientations, score=results.completeness)
    _write_grains_out(os.path.join(analysis_dir, 'grains.out'), qbar)
    return analysis_dir


def _write_grains_out(path: str, qbar: np.ndarray) -> None:
    """grains.out in the standard GrainDataWriter format.

    For find-orientations every grain is written with completeness 1.0, zero
    centroid, and identity strain (inv(V_s)=[1,1,1,0,0,0], ln(V_s)=0); only the
    orientation (exp_map_c) carries information.
    """
    header_items = (
        '# grain ID', 'completeness', 'chi^2',
        'exp_map_c[0]', 'exp_map_c[1]', 'exp_map_c[2]',
        't_vec_c[0]', 't_vec_c[1]', 't_vec_c[2]',
        'inv(V_s)[0,0]', 'inv(V_s)[1,1]', 'inv(V_s)[2,2]',
        'inv(V_s)[1,2]*sqrt(2)', 'inv(V_s)[0,2]*sqrt(2)', 'inv(V_s)[0,1]*sqrt(2)',
        'ln(V_s)[0,0]', 'ln(V_s)[1,1]', 'ln(V_s)[2,2]',
        'ln(V_s)[1,2]', 'ln(V_s)[0,2]', 'ln(V_s)[0,1]',
    )
    delim = '  '
    header = delim.join([
        delim.join(['{:<12}'] * 3).format(*header_items[:3]),
        delim.join(['{:<23}'] * (len(header_items) - 3)).format(*header_items[3:]),
    ])
    with open(path, 'w') as fid:
        fid.write(header + '\n')
        for gid, q in enumerate(qbar.T):
            phi = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
            v = q[1:]
            nrm = np.linalg.norm(v)
            n = v / nrm if nrm > 0 else np.zeros(3)
            grain_params = np.hstack(
                [phi * n, np.zeros(3), np.array([1., 1., 1., 0., 0., 0.])])
            res = [int(gid), 1.0, 0.0] + grain_params.tolist() + [0.0] * 6
            out = delim.join([
                delim.join(['{:<12d}', '{:<12f}', '{:<12e}']).format(*res[:3]),
                delim.join(['{:<23.16e}'] * (len(res) - 3)).format(*res[3:]),
            ])
            fid.write(out + '\n')
