"""Find grain orientations from detector images using independently callable stages."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from hexrd.core import constants as const
from hexrd.core import distortion as distortion_pkg
from hexrd.core import matrixutil as mutil
from hexrd.core import rotations
from hexrd.hedm.experiment import (
    ClusteringAlgorithm, HedmExperiment, SeedSearchMethod,
)
from hexrd.core.extensions import transforms_c_api, transforms as native
from hexrd.core.material.material_data import Material, PlaneData, load_plane_data_cached
from hexrd.core.transforms import xfcapi
from hexrd.core.utils.decorators import _lazy_njit
from hexrd.core.utils.npz import fingerprint, load_array_cache, save_array_cache

def _prefetch_heavy_imports(blob=False):
    """Import peak-finding and clustering dependencies in a worker thread."""
    from scipy import ndimage  # noqa: F401
    from scipy.spatial.transform import Rotation  # noqa: F401
    if blob:
        from skimage.exposure import rescale_intensity  # noqa: F401
        from skimage.feature import blob_dog, blob_log  # noqa: F401


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
        """Return the number of accepted grain orientations."""
        return self.grain_orientations.shape[1]


def find_orientations(experiment: HedmExperiment, material: Optional[Material] = None,
                      clean: bool = False) -> FindOrientationsResult:
    """Find grain orientations in a rotation series.

    Args:
        experiment: Experiment inputs and find-orientations parameters.
        material: Material to analyze. Uses the experiment's active material
            when omitted.
        clean: Whether to regenerate eta-omega maps instead of using a cached
            map file.

    Returns:
        The accepted grain orientations, trial orientations, and completeness
        scores.
    """
    grid_file = experiment.quaternion_grid_file
    fiber_cache = os.path.join(experiment.analysis_dir, '.hexrd', 'fibers.npz')
    if grid_file is None and not os.path.exists(fiber_cache):
        blob = experiment.find_orientations.seed_search.method is not SeedSearchMethod.LABEL
        threading.Thread(target=_prefetch_heavy_imports, args=(blob,), daemon=True).start()

    if material is None:
        from dataclasses import asdict

        def load_material():
            import numba

            # Set the limit before structure factors run, but leave the JIT
            # thread pool dormant when the numeric snapshot is already valid.
            numba.set_num_threads(max(1, min(
                experiment.max_workers, numba.config.NUMBA_NUM_THREADS)))
            return experiment.get_active_material()

        plane_data = load_plane_data_cached(
            experiment._materials_file,
            os.path.join(experiment.analysis_dir, '.hexrd', 'plane-data.npz'),
            dict(material=asdict(experiment.active_material),
                 beam_energy=experiment.beam.energy),
            load_material)
    else:
        plane_data = material.plane_data

    maps = load_or_build_eta_omega_maps(experiment, plane_data, clean=clean)

    if grid_file is not None:
        fibers = np.load(grid_file)
        min_samples = 1
    else:
        fibers = _load_or_generate_fibers(experiment, plane_data, maps, fiber_cache)
        min_samples = (estimate_min_samples(experiment, plane_data, maps)
                       if fibers.shape[1] else 1)

    completeness = score_orientations(experiment, plane_data, maps, fibers)
    qbar = cluster_grains(experiment, plane_data, fibers, completeness,
                          min_samples=min_samples)

    return FindOrientationsResult(
        grain_orientations=qbar,
        test_orientations=fibers,
        completeness=completeness,
    )


def load_or_build_eta_omega_maps(experiment: HedmExperiment, plane_data: PlaneData,
                                 clean: bool = False) -> EtaOmegaMaps:
    """Load cached eta-omega maps or build and cache new maps.

    The configured map filter is applied after loading or building unless
    ``clean`` is true.

    Args:
        experiment: Experiment inputs and orientation-map parameters.
        plane_data: Crystallographic data for the material being analyzed.
        clean: Whether to rebuild the maps even when a cache exists.

    Returns:
        The loaded or newly built eta-omega maps.
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
    """Resolve active HKLs to indices in the non-excluded HKL list.

    Args:
        active_hkls: HKLs as an ``(n, 3)`` array, master HKL IDs as an
            ``(n,)`` array, or ``None`` to select every non-excluded ring.
        plane_data: Crystallographic reflection data.

    Returns:
        Indices of the selected rings in ``plane_data.unexcluded_hkls``.

    Raises:
        ValueError: If an HKL is excluded, unknown, out of range, or repeated.
    """
    if active_hkls is None:
        return np.arange(len(plane_data.unexcluded_hkls))

    if active_hkls.ndim == 1:              # master hkl IDs
        bad = [i for i in active_hkls.tolist()
               if i < 0 or i >= len(plane_data.hkls) or plane_data.exclusions[i]]
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
    """Build per-ring eta-omega maps from detector image series.

    Intensities are accumulated in float64: frame caches are often uint16,
    and a bin's summed intensity can exceed the native dtype's range.

    Args:
        experiment: Experiment geometry, image series, and map parameters.
        plane_data: Crystallographic data for the material being analyzed.

    Returns:
        Intensity maps and their eta, omega, two-theta, and ring metadata.
    """
    import numba

    fo = experiment.find_orientations

    ring_ids = _resolve_active_rings(fo.orientation_maps.active_hkls, plane_data)

    half = np.radians(experiment.active_material.two_theta_width) / 2
    ring_tths = plane_data.two_thetas[~plane_data.exclusions][ring_ids]
    two_theta_ranges = np.column_stack([ring_tths - half, ring_tths + half])

    eta_step = np.radians(fo.orientation_maps.eta_step)
    eta_edges = _eta_bin_edges(eta_step)
    n_eta = len(eta_edges) - 1
    row_of_frame, omegas, omega_edges, period = _omega_grid(
        experiment.image_series_list[0].omega)
    ring_maps = np.full((len(two_theta_ranges), len(omegas), n_eta), np.nan)

    numba.set_num_threads(
        max(1, min(experiment.max_workers, numba.config.NUMBA_NUM_THREADS)))
    active = [ims for ims in experiment.image_series_list
              if ims.panel in experiment.detectors]

    for ims, (eta_bin, ring_masks, coverage) in _iter_panel_bins(
            experiment, active, two_theta_ranges, eta_edges):
        # Preserve NaN gaps; covered bins sum intensities across panels.
        rows = row_of_frame[:len(ims)]
        for ring, bins in enumerate(coverage):
            covered = np.ix_(rows, np.flatnonzero(bins))
            ring_maps[ring][covered] = np.nan_to_num(ring_maps[ring][covered])

        # Pack each frame chunk once, even for materials with over 63 rings.
        threshold = float(fo.orientation_maps.threshold)
        blocks = [(63 * block, mask) for block, mask in enumerate(ring_masks)
                  if np.any(coverage[63 * block:63 * (block + 1)])]
        if not blocks:
            continue
        chunks = (ims.pixel_chunks(rows) if hasattr(ims, 'pixel_chunks')
                  else _pixel_chunks(ims.images, rows))
        for ids, vals, offsets, chunk_rows in chunks:
            for first, ring_mask in blocks:
                _bin_pixels(offsets, ids, vals, chunk_rows, ring_mask,
                            eta_bin, threshold,
                            ring_maps[first:first + 63])

    return EtaOmegaMaps(ring_maps, eta_edges, omega_edges, omegas, period,
                        two_theta_ranges, ring_ids, eta_step)


def _iter_panel_bins(experiment, active, two_theta_ranges, eta_edges):
    """Keep only current/next panel geometry, overlapping IO with binning."""
    def load(series):
        return _cached_panel_bin_data(
            experiment, series.panel, two_theta_ranges, eta_edges)

    if experiment.max_workers == 1:
        for series in active:
            yield series, load(series)
        return
    if not active:
        return
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(load, active[0])
        for i, series in enumerate(active):
            data = future.result()
            future = pool.submit(load, active[i + 1]) if i + 1 < len(active) else None
            yield series, data


def _pixel_chunks(frames, row_of_frame, max_pixels: int = 8_000_000):
    """Yield stored pixels from consecutive sparse frames in bounded chunks.

    Args:
        frames: Sequence of CSR image frames.
        row_of_frame: Eta-omega map row for each frame.
        max_pixels: Preferred maximum number of stored pixels per chunk. A
            single frame may exceed this value.

    Yields:
        Tuples containing flat pixel IDs, float64 intensity values, per-frame
        offsets into those arrays, and the corresponding map rows.
    """

    from hexrd.core.experiment import _pixel_chunks as chunk_pixels

    def pixels():
        for frame in frames:
            ids = np.empty(frame.nnz, dtype=np.int64)
            _csr_pixel_ids(frame.indptr, frame.indices, frame.shape[1], ids)
            yield ids, frame.data

    yield from chunk_pixels(pixels(), row_of_frame, max_pixels)


@_lazy_njit
def _csr_pixel_ids(indptr, columns, n_columns, out):
    """Write flat pixel IDs from CSR indices without allocating COO rows.

    Args:
        indptr: CSR row offsets.
        columns: CSR column indices.
        n_columns: Number of columns in the image.
        out: Preallocated array receiving the row-major pixel IDs.
    """
    for row in range(len(indptr) - 1):
        for pixel in range(indptr[row], indptr[row + 1]):
            out[pixel] = row * n_columns + columns[pixel]

@_lazy_njit(parallel=True)
def _bin_pixels(offsets, ids, vals, rows, ring_mask, eta_bin, threshold, out):
    """Accumulate stored pixel intensities into ring, omega, and eta bins.

    Parallel over frames: within one image series every frame owns its own
    map row, so writes never collide.

    Args:
        offsets: Start offset for each frame in ``ids`` and ``vals``.
        ids: Row-major pixel IDs for all frames in the chunk.
        vals: Pixel intensities corresponding to ``ids``.
        rows: Eta-omega map row for each frame.
        ring_mask: Bit-packed ring membership for each detector pixel.
        eta_bin: Eta-bin index for each detector pixel.
        threshold: Minimum pixel intensity to accumulate.
        out: Ring maps updated in place.
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
    """Construct map rows and the omega axis for a rotation scan.

    A contiguous scan gets one row per frame with the scan's own frame
    boundaries as bin edges. A multi-wedge scan with gaps gets
    a uniform grid over the whole range: frames land on their grid row and
    the gap rows stay all-NaN, which excludes them from scoring the same
    way off-detector bins are excluded.

    Args:
        omega: Frame omega spans in radians with shape ``(n_frames, 2)``.

    Returns:
        A tuple containing each frame's map row, omega-bin centers, omega-bin
        edges, and the full omega period.

    Raises:
        ValueError: If multi-wedge frames have unequal widths or do not align
            with the frame-width grid.
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
    """Save eta-omega maps and their metadata to a compressed NPZ file.

    Args:
        maps: Maps and metadata to save.
        path: Destination NPZ path.
    """
    np.savez_compressed(
        path,
        ring_maps=maps.ring_maps,
        eta_edges=maps.eta_edges, omega_edges=maps.omega_edges,
        omegas=maps.omegas, omega_period=maps.omega_period,
        two_theta_ranges=maps.two_theta_ranges, ring_ids=maps.ring_ids,
        eta_step=maps.eta_step)


def _load_eta_omega_maps(path: str, plane_data: PlaneData,
                         two_theta_width: float) -> EtaOmegaMaps:
    """Load a current or legacy HEXRD eta-omega map archive.

    Args:
        path: Path to the NPZ map archive.
        plane_data: Crystallographic data used to interpret legacy ring IDs.
        two_theta_width: Full ring width in degrees for legacy archives.

    Returns:
        The maps and normalized metadata from the archive.
    """
    from hexrd.core.experiment import _NpzReader

    with _NpzReader(path) as d:
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
    """Filter eta-omega maps in place.

    Median subtraction removes streak artifacts. When requested, a
    Laplacian-of-Gaussian filter is then applied.

    Args:
        maps: Eta-omega maps to update.
        median: Whether to subtract each eta column's median.
        log_fwhm: Laplacian-of-Gaussian full width at half maximum in pixels,
            or ``None`` to skip that filter.
    """
    if not median:
        return
    from scipy import ndimage

    for ring_map in maps.ring_maps:
        ring_map -= np.nanmedian(ring_map, axis=0)
        if log_fwhm is not None:
            ring_map[:] = -ndimage.gaussian_laplace(
                ring_map, const.fwhm_to_sigma * log_fwhm)


def _eta_bin_edges(step: float) -> np.ndarray:
    """Construct eta-bin edges over a full circle.

    Args:
        step: Requested eta-bin width in radians.

    Returns:
        Eta-bin edges with the branch cut at ``-pi`` and ``pi``.
    """
    n = int(2 * np.pi / step)
    ang = np.nan_to_num(step * np.linspace(0.0, n, num=n + 1) - np.pi)
    edges = np.mod(ang + np.pi, 2 * np.pi) - np.pi
    edges[np.logical_and(edges == -np.pi, ang > -np.pi)] = np.pi
    return edges


def _panel_pixel_angles(detector, beam_vector):
    """Map detector pixels to two-theta and eta angles.

    Args:
        detector: Detector geometry and optional distortion specification.
        beam_vector: Incident beam direction in laboratory coordinates.

    Returns:
        A tuple of flat two-theta and eta arrays in radians.
    """
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
    """Apply a detector's registered distortion model to coordinates.

    Args:
        detector: Detector containing the distortion specification.
        xy: Detector coordinates with shape ``(n, 2)``.

    Returns:
        Distorted coordinates, or the input array when no model is configured.
    """
    spec = detector.distortion
    if not spec.function_name:
        return xy
    mapping = distortion_pkg.get_mapping(spec.function_name, spec.parameters)
    return mapping.apply(xy)


def _panel_bin_data(detector, beam_vector, two_theta_ranges, eta_edges):
    """Precompute eta bins, ring membership, and coverage for a panel.

    Args:
        detector: Detector geometry and distortion specification.
        beam_vector: Incident beam direction in laboratory coordinates.
        two_theta_ranges: Lower and upper two-theta bounds for each ring.
        eta_edges: Eta-bin edges in radians.

    Returns:
        A tuple containing per-pixel eta bins, bit-packed ring masks, and the
        eta-bin coverage of each ring.
    """

    ptth, peta = _panel_pixel_angles(detector, beam_vector)
    n_eta = len(eta_edges) - 1
    # Histogram bins are left-closed, with the final bin also right-closed.
    eta_bin = np.minimum(
        np.searchsorted(eta_edges, peta, side='right') - 1, n_eta - 1)
    eta_bin[(peta < eta_edges[0]) | (peta > eta_edges[-1]) | np.isnan(peta)] = -1
    masks, coverage = _ring_masks(ptth, eta_bin, two_theta_ranges, n_eta)
    return eta_bin, masks, coverage


@_lazy_njit
def _ring_masks(panel_tth, eta_bin, two_theta_ranges, n_eta):
    """Build bit-packed ring membership and eta-bin coverage arrays.

    Each mask block holds up to 63 rings.

    Args:
        panel_tth: Flat per-pixel two-theta angles in radians.
        eta_bin: Eta-bin index for each detector pixel, or ``-1`` if invalid.
        two_theta_ranges: Lower and upper two-theta bounds for each ring.
        n_eta: Number of eta bins.

    Returns:
        A tuple containing per-pixel ring masks and per-ring eta-bin coverage.
    """
    n_rings = len(two_theta_ranges)
    masks = np.zeros(((n_rings + 62) // 63, len(panel_tth)), dtype=np.int64)
    coverage = np.zeros((n_rings, n_eta), dtype=np.bool_)
    for ring in range(n_rings):
        lo, hi = two_theta_ranges[ring]
        block, bit = ring // 63, 1 << (ring % 63)
        for pixel in range(len(panel_tth)):
            if eta_bin[pixel] >= 0 and lo <= panel_tth[pixel] <= hi:
                masks[block, pixel] |= bit
                coverage[ring, eta_bin[pixel]] = True
    return masks, coverage


def _cached_panel_bin_data(experiment, panel, two_theta_ranges, eta_edges):
    """Reuse geometry across scans while still rebuilding maps from live pixels."""
    from dataclasses import asdict
    import hashlib
    from pathlib import Path
    from hexrd.core.extensions import transforms, inverse_distortion

    detector = experiment.detectors[panel]
    sources = [Path(__file__), Path(_lazy_njit.__code__.co_filename),
               Path(transforms_c_api.__file__), Path(transforms.__file__),
               Path(inverse_distortion.__file__)]
    sources.extend(sorted(Path(distortion_pkg.__file__).parent.glob('*.py')))
    if detector.distortion.function_name:
        import inspect

        mapping = distortion_pkg.get_mapping(
            detector.distortion.function_name, detector.distortion.parameters)
        try:
            sources.append(Path(inspect.getfile(type(mapping))))
        except TypeError:
            return _panel_bin_data(detector, experiment.beam.vector,
                                   two_theta_ranges, eta_edges)
    key = fingerprint(asdict(detector),
                      arrays=(experiment.beam.vector, two_theta_ranges, eta_edges),
                      files=sources, packages=('numpy', 'numba'))
    name = hashlib.sha256(panel.encode()).hexdigest()[:16]
    path = Path(experiment.analysis_dir) / '.hexrd' / f'panel-{name}.npz'
    saved = load_array_cache(path, key)
    if saved is not None and {'eta_bin', 'ring_masks', 'coverage'} <= saved.keys():
        return saved['eta_bin'], saved['ring_masks'], saved['coverage']

    eta_bin, masks, coverage = _panel_bin_data(
        detector, experiment.beam.vector, two_theta_ranges, eta_edges)
    eta_bin = eta_bin.astype(np.int32)
    save_array_cache(path, key, compressed=True, eta_bin=eta_bin,
                     ring_masks=masks, coverage=coverage)
    return eta_bin, masks, coverage


def _load_or_generate_fibers(experiment, plane_data, maps, cache_file):
    """Reuse seed search across scoring changes, validating the actual map data."""
    from dataclasses import asdict
    from pathlib import Path

    settings = dict(seed=asdict(experiment.find_orientations.seed_search),
                    chi=experiment.oscillation_stage.chi,
                    laue_group=plane_data.laue_group, eta_step=maps.eta_step)
    sources = [__file__, _lazy_njit.__code__.co_filename,
               rotations.__file__, mutil.__file__, xfcapi.__file__,
               xfcapi.cpp_transforms.__file__, transforms_c_api.__file__, const.__file__]
    key = fingerprint(settings, files=sources, packages=('numpy', 'scipy', 'scikit-image'),
                      arrays=(maps.ring_maps, maps.eta_edges, maps.omega_edges, maps.omegas,
                              maps.two_theta_ranges, maps.ring_ids,
                              plane_data.unexcluded_hkls, plane_data.B, plane_data.q_sym))
    cached = load_array_cache(cache_file, key)
    if cached is not None and 'fibers' in cached:
        fibers = cached['fibers']
        if fibers.ndim == 2 and fibers.shape[0] == 4 and fibers.dtype == np.float64:
            return fibers
    fibers = generate_orientation_fibers(experiment, plane_data, maps)
    save_array_cache(cache_file, key, fibers=fibers)
    return fibers


def generate_orientation_fibers(experiment: HedmExperiment, plane_data: PlaneData,
                                maps: EtaOmegaMaps) -> np.ndarray:
    """Generate trial orientations from peaks in the seed-ring maps.

    Args:
        experiment: Experiment geometry and seed-search parameters.
        plane_data: Crystallographic data for the material being analyzed.
        maps: Eta-omega maps containing the seed rings.

    Returns:
        Trial orientation quaternions with shape ``(4, n_trials)``.

    Raises:
        ValueError: If no seed-ring indices are configured.
    """
    fo = experiment.find_orientations
    chi = experiment.oscillation_stage.chi
    csym = plane_data.laue_group
    fiber_ndiv = fo.seed_search.fiber_ndiv

    seed_ids = fo.seed_search.hkl_seeds
    if seed_ids.size == 0:
        raise ValueError('seed_search: hkl_seeds must be given for seeded search')
    seed_hkls = plane_data.unexcluded_hkls[maps.ring_ids[seed_ids]]
    seed_tths = np.average(maps.two_theta_ranges, axis=1)[seed_ids]
    d_omega = maps.omegas[1] - maps.omegas[0]

    # Peak finding mutates maps; copies preserve the original scoring data.
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
    q_fibers = [
        _fiber(*center, chi, plane_data.B, fiber_ndiv, csym)
        for center in centers
    ]
    return np.hstack(q_fibers) if q_fibers else np.empty((4, 0))


def _find_peaks(ring_map: np.ndarray, seed_search) -> tuple[int, np.ndarray]:
    """Find peak centers in a seed-ring map.

    Args:
        ring_map: Two-dimensional omega-by-eta intensity map. The array is
            cleaned in place before peak detection.
        seed_search: Peak-detection method and its keyword arguments.

    Returns:
        A tuple containing the peak count and their omega-eta centers.
    """
    _clean_map(ring_map)
    method, kwargs = seed_search.method, seed_search.method_kwargs

    if method is SeedSearchMethod.LABEL:
        from scipy import ndimage

        if kwargs.get('filter_radius'):
            stdev = const.fwhm_to_sigma * kwargs['filter_radius']
            ring_map = -ndimage.gaussian_laplace(ring_map, stdev)
        labels, num_spots = ndimage.label(
            ring_map > kwargs.get('threshold', 1))
        if num_spots == 0:
            return 0, np.empty((0, 2))
        coms = np.atleast_2d(ndimage.center_of_mass(
            ring_map, labels=labels, index=np.arange(1, num_spots + 1)))
        return num_spots, coms

    from skimage.exposure import rescale_intensity
    from skimage.feature import blob_dog, blob_log

    scl_map = rescale_intensity(ring_map, out_range=(-1, 1))
    detect = blob_log if method is SeedSearchMethod.BLOB_LOG else blob_dog
    blobs = np.atleast_2d(detect(scl_map, **kwargs))
    return len(blobs), blobs[:, :2]


def _clean_map(m: np.ndarray) -> None:
    """Fill missing map values, floor at the median, and zero-base in place.

    Args:
        m: Eta-omega map to clean.
    """
    nan = np.isnan(m)
    if np.all(nan):
        m.fill(0.0)
        return
    med = np.median(m[~nan])
    m[nan] = med
    np.maximum(m, med, out=m)
    m -= med


def _fiber(hkl, tth, eta_c, ome_c, chi, b_matrix, fiber_ndiv, csym):
    """Generate the discrete orientation fiber for one diffraction spot.

    Args:
        hkl: Miller indices of the reflecting lattice plane.
        tth: Spot two-theta angle in radians.
        eta_c: Spot eta angle in radians.
        ome_c: Spot omega angle in radians.
        chi: Oscillation-stage chi angle in radians.
        b_matrix: Reciprocal-lattice B matrix.
        fiber_ndiv: Number of samples along the orientation fiber.
        csym: Crystal symmetry identifier.

    Returns:
        Unique orientation quaternions in the fundamental region.
    """
    g_vec_s = xfcapi.angles_to_gvec(
        np.atleast_2d([tth, eta_c, ome_c]), chi=chi).T
    fiber = rotations.discreteFiber(
        hkl.reshape(3, 1), g_vec_s, B=b_matrix, ndiv=fiber_ndiv,
        invert=False, csym=csym)[0]
    return mutil.uniqueVectors(fiber)


_hit_status_maps = native.orientationHitStatus


def score_orientations(experiment: HedmExperiment, plane_data: PlaneData,
                       maps: EtaOmegaMaps, q_fibers: np.ndarray) -> np.ndarray:
    """Score each trial orientation by reflection completeness.

    This is the algorithm historically known as paintGrid, scored in parallel
    over bounded batches of trials.

    Args:
        experiment: Experiment geometry and scoring parameters.
        plane_data: Crystallographic data for the material being analyzed.
        maps: Eta-omega intensity maps used to identify observed reflections.
        q_fibers: Trial orientation quaternions with shape ``(4, n_trials)``.

    Returns:
        Completeness score for each trial orientation.
    """
    fo = experiment.find_orientations
    if q_fibers.shape[1] == 0:
        return np.empty(0)

    symm_hkls_per_ring = [plane_data.symm_hkls[r] for r in maps.ring_ids]
    ring_for_hkl = np.repeat(np.arange(len(symm_hkls_per_ring)),
                             [s.shape[1] for s in symm_hkls_per_ring])
    symm_hkls = np.ascontiguousarray(
        np.vstack([s.T for s in symm_hkls_per_ring]), dtype=np.float64)

    eta_ranges = fo.eta.range
    ome_offset = fo.omega.period.min()
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
    beam_vector = np.ascontiguousarray(experiment.beam.vector, dtype=np.float64)
    n = q_fibers.shape[1]
    workers = max(1, min(experiment.max_workers, max(1, n // 2048)))
    hit_status = None
    window_area = (2 * dpix_eta + 1) * (2 * dpix_ome + 1)
    # Serial lookup construction must pay for each worker's share of scoring.
    if (n * len(symm_hkls) * window_area >= 8 * workers * ring_maps.size and window_area > 1
            and np.prod(ring_maps.shape[1:]) < np.iinfo(np.int32).max):
        hit_status = _hit_status_maps(ring_maps, dpix_eta, dpix_ome, fo.threshold)
    # The native scorer predicts and counts one trial at a time with private
    # bounded scratch, releasing the GIL for parallel chunks. No angle cube
    # or Python-to-C call per quaternion, and no scoring JIT warmup.
    bounds = np.linspace(0, n, workers + 1, dtype=int)

    def score_chunk(start, stop):
        rmats = rotations.rotMatOfQuat(q_fibers[:, start:stop]).reshape(-1, 3, 3)
        return native.scoreOrientations(
            rmats, symm_hkls, bmat, plane_data.wavelength, beam_vector,
            maps.eta_edges, maps.omega_edges, valid_eta, valid_ome, ome_offset,
            ring_maps, ring_for_hkl, dpix_eta, dpix_ome, float(fo.threshold), hit_status)

    if workers == 1:
        return score_chunk(0, n)
    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(score_chunk, bounds[:-1], bounds[1:])))

def normalize_ranges(starts: np.ndarray, stops: np.ndarray,
                     offset: float) -> np.ndarray:
    """Normalize angle ranges into one full-circle interval.

    The result is a flat, sorted ``[start, stop, start, stop, ...]`` array. An
    angle mapped into the window lies inside a valid span exactly when its
    ``_find_in_range`` insertion index is odd. A range spanning the
    full circle collapses to ``[offset, offset + 2*pi]``.

    Args:
        starts: Start angle of each range in radians.
        stops: Stop angle of each range in radians.
        offset: Lower bound of the normalized full-circle interval.

    Returns:
        Interleaved, sorted range boundaries in
        ``[offset, offset + 2*pi]``.

    Raises:
        ValueError: If a start is not less than its stop or normalized ranges
            overlap.
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

    if not np.all(np.diff(result) >= 0):
        raise ValueError('Angle ranges overlap')

    return result


def estimate_min_samples(experiment: HedmExperiment, plane_data: PlaneData,
                         maps: EtaOmegaMaps, n_grains: int = 100) -> int:
    """Estimate the DBSCAN minimum cluster size from seed reflections.

    Simulates random grain orientations over the instrument, counts how many
    seed-ring reflections each would produce inside the eta/omega ranges and
    on a panel, and takes half the completeness threshold times the worst
    grain's count. The random draw is seeded, so runs are reproducible.
    Panel distortion is ignored (sub-pixel effect on counts).

    Args:
        experiment: Experiment geometry and clustering parameters.
        plane_data: Crystallographic data for the material being analyzed.
        maps: Eta-omega maps identifying the active and seed rings.
        n_grains: Number of deterministic random orientations to simulate.

    Returns:
        The estimated ``min_samples`` value, with a lower bound of two.
    """
    fo = experiment.find_orientations
    chi = experiment.oscillation_stage.chi
    bmat = np.ascontiguousarray(plane_data.B)

    # Only seed reflections contribute to min_samples.
    seed_rings = np.unique(maps.ring_ids[fo.seed_search.hkl_seeds])
    symm_hkls = np.ascontiguousarray(
        np.vstack([plane_data.symm_hkls[r].T for r in seed_rings]),
        dtype=np.float64)

    eta_ranges = fo.eta.range
    valid_eta = normalize_ranges(eta_ranges[:, 0], eta_ranges[:, 1], -np.pi)
    omega = experiment.image_series_list[0].omega
    ome_lo = omega[0, 0]
    # one span per omega wedge, so gaps in a multi-wedge scan don't count
    new_wedge = np.r_[True, ~np.isclose(omega[1:, 0], omega[:-1, 1])]
    valid_ome = normalize_ranges(omega[new_wedge, 0],
                                 omega[np.r_[new_wedge[1:], True], 1], ome_lo)

    def in_spans(values, spans):
        """Return whether each value lies inside the interleaved spans.

        Args:
            values: Values to test.
            spans: Sorted, interleaved start and stop boundaries.

        Returns:
            A boolean mask with one entry per value.
        """
        idx = np.searchsorted(spans, values, side='right')
        return (idx % 2 == 1) & (values >= spans[0]) & (values < spans[-1])

    rng = np.random.default_rng(0)
    quats = rng.normal(size=(4, n_grains))
    quats /= np.linalg.norm(quats, axis=0)

    ac = np.ascontiguousarray
    panels = [(detector, ac(detector.transform.rotation_matrix),
               ac(detector.transform.translation).ravel())
              for detector in experiment.detectors.values()]
    min_reflections = np.inf
    for rmat_c in rotations.rotMatOfQuat(quats).reshape(-1, 3, 3):
        a0, a1 = xfcapi.oscill_angles_of_hkls(
            symm_hkls, chi, rmat_c, bmat, plane_data.wavelength,
            beam_vec=experiment.beam.vector)
        angs = np.vstack([a0, a1])
        ok = ~np.isnan(angs[:, 0])

        eta = np.mod(angs[:, 1] + np.pi, 2 * np.pi) - np.pi
        ok &= in_spans(eta, valid_eta)
        ome = np.mod(angs[:, 2] - ome_lo, 2 * np.pi) + ome_lo
        ok &= in_spans(ome, valid_ome)

        # Count a reflection once per panel it hits.
        count = 0
        angs = angs[ok]
        gvecs = xfcapi.angles_to_gvec(angs, chi=chi, rmat_c=rmat_c)
        rmats_s = _sample_rmats(chi, angs[:, 2])
        for detector, rmat_d, tvec_d in panels:
            xy = transforms_c_api.gvecToDetectorXYArray(
                ac(gvecs), rmat_d, rmats_s, rmat_c, tvec_d,
                np.zeros(3), np.zeros(3), experiment.beam.vector)
            count += np.count_nonzero(_clip_to_panel(detector, xy))
        min_reflections = min(min_reflections, count)
        if 0.5 * fo.clustering.completeness * min_reflections < 3:
            return 2

    return max(int(np.floor(
        0.5 * fo.clustering.completeness * min_reflections)), 2)


def _sample_rmats(chi: float, omes: np.ndarray) -> np.ndarray:
    """Construct sample-frame rotation matrices for omega angles.

    Args:
        chi: Oscillation-stage chi angle in radians.
        omes: Omega angles in radians.

    Returns:
        Rotation matrices ``R_x(chi) @ R_y(omega)`` with shape ``(n, 3, 3)``.
    """
    cx, sx = np.cos(chi), np.sin(chi)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    co, so = np.cos(omes), np.sin(omes)
    ry = np.zeros((len(omes), 3, 3))
    ry[:, 0, 0], ry[:, 0, 2] = co, so
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0], ry[:, 2, 2] = -so, co
    return np.ascontiguousarray(rx @ ry)


def _clip_to_panel(detector, xy: np.ndarray) -> np.ndarray:
    """Identify coordinates inside a detector's buffered active area.

    Args:
        detector: Detector dimensions, pixel sizes, and edge buffer.
        xy: Detector coordinates in millimeters with shape ``(n, 2)``.

    Returns:
        A boolean mask selecting coordinates inside the panel.
    """
    half_x = 0.5 * detector.pixels.columns * detector.pixels.size[1]
    half_y = 0.5 * detector.pixels.rows * detector.pixels.size[0]
    buf_x, buf_y = detector.buffer if detector.buffer is not None else (0.0, 0.0)
    with np.errstate(invalid='ignore'):
        return ((np.abs(xy[:, 0]) <= half_x - buf_x)
                & (np.abs(xy[:, 1]) <= half_y - buf_y))


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN for the small candidate sets produced by orientation scoring."""
    n = len(points)
    if n <= 1024:
        distance = np.zeros((n, n))
        for coordinate in points.T:
            distance += (coordinate[:, None] - coordinate[None, :]) ** 2
        neighbors = distance <= eps * eps
        core = neighbors.sum(axis=1) >= min_samples
        labels = np.full(n, -1, dtype=int)
        cluster = 0
        for i in range(n):
            if labels[i] >= 0 or not core[i]:
                continue
            pending = [i]
            while pending:
                j = pending.pop()
                if labels[j] >= 0:
                    continue
                labels[j] = cluster
                if core[j]:
                    pending.extend(np.flatnonzero(neighbors[j] & (labels < 0)))
            cluster += 1
        return labels

    from sklearn.cluster import dbscan

    return dbscan(points, eps=eps, min_samples=min_samples,
                  metric='minkowski', p=2, n_jobs=1)[1]


def cluster_grains(experiment: HedmExperiment, plane_data: PlaneData,
                   q_fibers: np.ndarray, completeness: np.ndarray,
                   min_samples: int = 1) -> np.ndarray:
    """Cluster high-completeness trials into grain orientations.

    Quaternion-metric algorithms fall back to orthographic DBSCAN above
    25000 candidates to avoid quadratic memory use. Euclidean algorithms
    merge duplicate grain centroids afterward.

    Args:
        experiment: Experiment clustering parameters.
        plane_data: Crystallographic symmetry data.
        q_fibers: Trial orientation quaternions with shape ``(4, n_trials)``.
        completeness: Completeness score for each trial orientation.
        min_samples: Minimum cluster size for DBSCAN algorithms.

    Returns:
        Averaged grain orientation quaternions with shape ``(4, n_grains)``.
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
        """Return the symmetry-aware angular distance between quaternions.

        Args:
            x: First quaternion.
            y: Second quaternion.

        Returns:
            Angular distance in radians.
        """
        return xfcapi.quat_distance(np.ascontiguousarray(x),
                                    np.ascontiguousarray(y), qsym)

    if algorithm is Algo.FCLUSTERDATA:
        from scipy.cluster.hierarchy import fclusterdata
        labels = fclusterdata(keep.T, np.radians(radius),
                              criterion='distance', metric=quat_distance)
        labels = np.asarray(labels, dtype=int)  # already 1..N, no noise
    else:
        if algorithm is Algo.SPH_DBSCAN:
            from sklearn.cluster import dbscan
            from sklearn.metrics.pairwise import pairwise_distances
            pdist = pairwise_distances(keep.T, metric=quat_distance, n_jobs=1)
            _, labels = dbscan(pdist, eps=np.radians(radius),
                               min_samples=min_samples, metric='precomputed')
        else:
            if algorithm is Algo.ORT_DBSCAN:
                pts, eps = keep[1:, :].T, 0.25 * np.radians(radius)
            else:
                pts, eps = keep.T, 0.5 * np.radians(radius)
            labels = _dbscan(pts, eps, min_samples)

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
    """Merge grain centroids within the clustering radius.

    Args:
        qbar: Grain orientation quaternions with shape ``(4, n_grains)``.
        qsym: Crystal-symmetry quaternions.
        quat_distance: Callable returning quaternion distance in radians.
        radius: Merge radius in degrees.

    Returns:
        Grain orientations after symmetry-aware duplicate merging.
    """
    # The usual few well-separated grains need no hierarchical clustering
    # (or SciPy import). Keep the established ordering for actual merges.
    if qbar.shape[1] <= 32:
        cutoff = np.radians(radius)
        if all(quat_distance(qbar[:, i], qbar[:, j]) > cutoff
               for i in range(qbar.shape[1]) for j in range(i)):
            return qbar

    from scipy.cluster.hierarchy import fclusterdata

    cl = fclusterdata(qbar.T, np.radians(radius),
                      criterion='distance', metric=quat_distance)
    n_merged = len(np.unique(cl))
    if n_merged == qbar.shape[1]:
        return qbar
    merged = np.zeros((4, n_merged))
    for i in range(n_merged):
        members = qbar[:, cl == i + 1]
        merged[:, i] = rotations.quatAverageCluster(members, qsym).flatten()
    return merged


def write_results(results: FindOrientationsResult,
                  experiment: HedmExperiment) -> str:
    """Write find-orientations results in the standard file formats.

    Args:
        results: Accepted grains, scored trials, and completeness values.
        experiment: Experiment output paths and analysis directory.

    Returns:
        The analysis directory containing the output files.
    """
    qbar = np.atleast_2d(results.grain_orientations)
    analysis_dir = experiment.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)
    np.savetxt(experiment.accepted_orientations_file,
               qbar.T, fmt='%.18e', delimiter='\t')
    np.savez_compressed(
        experiment.scored_orientations_file,
        test_quaternions=results.test_orientations, score=results.completeness)
    _write_grains_out(os.path.join(analysis_dir, 'grains.out'), qbar)
    return analysis_dir


def _write_grains_out(path: str, qbar: np.ndarray) -> None:
    """Write grain orientations in the standard ``grains.out`` format.

    For find-orientations every grain is written with completeness 1.0, zero
    centroid, and identity strain (inv(V_s)=[1,1,1,0,0,0], ln(V_s)=0); only the
    orientation (exp_map_c) carries information.

    Args:
        path: Destination ``grains.out`` path.
        qbar: Grain orientation quaternions with shape ``(4, n_grains)``.
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
