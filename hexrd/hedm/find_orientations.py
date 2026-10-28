"""Find grain orientations from detector images using independently callable stages."""
import os
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from contextvars import copy_context
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
from hexrd.core.utils.npz import fingerprint, load_array_cache, save_array_cache
from hexrd.hedm._progress import (
    activity, check_cancelled, diagnostic, progress_bar, report_hkls, report_peaks, stage,
)

logger = logging.getLogger(__name__)


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


class _LazyEtaOmegaMaps(EtaOmegaMaps):
    @property
    def ring_maps(self):
        if callable(self._ring_maps):
            self._ring_maps = self._ring_maps()
        return self._ring_maps

    @ring_maps.setter
    def ring_maps(self, value):
        self._ring_maps = value


def _cache_dir(experiment):
    """Keep derived caches outside the legacy analysis output tree."""
    from appdirs import user_cache_dir

    root = user_cache_dir('hexrd')
    key = fingerprint(dict(analysis_dir=os.path.abspath(experiment.analysis_dir)))
    return os.path.join(root, 'find-orientations', key)


def _map_key(maps):
    if isinstance(maps, _LazyEtaOmegaMaps) and callable(maps._ring_maps):
        return maps._content_key
    return fingerprint({}, arrays=(maps.ring_maps,))


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
    fo = experiment.find_orientations
    grid_file = experiment.quaternion_grid_file
    fiber_cache = os.path.join(_cache_dir(experiment), 'fibers.npz')

    with stage('Material'):
        if material is None and experiment._materials_file is not None:
            from dataclasses import asdict

            def load_material():
                import numba

                numba.set_num_threads(max(1, min(
                    experiment.max_workers, numba.config.NUMBA_NUM_THREADS)))
                return experiment.get_active_material()

            plane_data = load_plane_data_cached(
                experiment._materials_file,
                os.path.join(_cache_dir(experiment), 'plane-data.npz'),
                dict(material=asdict(experiment.active_material),
                     beam_energy=experiment.beam.energy),
                load_material)
        else:
            plane_data = (material or experiment.get_active_material()).plane_data

    with stage('Eta/omega maps'):
        maps = load_or_build_eta_omega_maps(experiment, plane_data, clean=clean)

    with stage('Fibers'):
        if grid_file is not None:
            activity('Loading quaternion grid')
            fibers = np.load(grid_file, mmap_mode='r')
        elif fo.direct_search is not None:
            spacing = fo.direct_search.average_angular_spacing
            eta_tol, ome_tol = np.degrees([fo.eta.tolerance, fo.omega.tolerance])
            if spacing > max(eta_tol, ome_tol):
                logger.warning(
                    'Direct-search grid spacing %g deg is coarser than the eta/omega '
                    'tolerances (%g/%g deg). Grains can be missed before clustering; '
                    'validate grid resolution or use seed_search.', spacing, eta_tol, ome_tol)
            activity('Generating uniform orientation grid')
            fibers = generate_orientation_grid(experiment, plane_data)
        else:
            report_hkls('Seed', plane_data.unexcluded_hkls[
                maps.ring_ids[fo.seed_search.hkl_seeds]])
            fibers = _load_or_generate_fibers(experiment, plane_data, maps, fiber_cache)
    with stage('Sampling'):
        min_samples = (_cached_min_samples(experiment, plane_data, maps)
                       if grid_file is None and fo.direct_search is None and fibers.shape[1]
                       else 1)

    with stage('Scoring'):
        report_hkls('Scoring', plane_data.unexcluded_hkls[maps.ring_ids])
        completeness = score_orientations(experiment, plane_data, maps, fibers)
    with stage('Clustering'):
        qbar = cluster_grains(experiment, plane_data, fibers, completeness,
                              min_samples=min_samples)
        diagnostic('Grains', f'{qbar.shape[1]:,}')

    return FindOrientationsResult(
        grain_orientations=qbar,
        test_orientations=fibers,
        completeness=completeness,
    )


def generate_orientation_grid(experiment: HedmExperiment,
                              plane_data: PlaneData) -> np.ndarray:
    """Return uniform scalar-first trial quaternions with shape ``(4, n)``.

    Configure ``find_orientations.direct_search.average_angular_spacing``
    in degrees. ``point_group_number`` optionally overrides the material.
    Generated trials use the same scoring, clustering, and output stages as
    seed-search trials; regular grids use a clustering minimum of one.
    """
    import numba
    from hexrd.hedm.sampleOrientations import sampleRFZ

    search = experiment.find_orientations.direct_search
    if search is None:
        raise ValueError('direct_search must be configured to generate an orientation grid')
    group = search.point_group_number or plane_data.point_group_number
    if not 1 <= group <= 32:
        raise ValueError('direct_search requires a material or explicit point_group_number (1-32)')
    numba.set_num_threads(max(1, min(experiment.max_workers, numba.config.NUMBA_NUM_THREADS)))

    def track(slabs):
        with progress_bar(total=len(slabs), desc='Generating orientation grid',
                          unit='batch', disable=None) as progress:
            for first in slabs:
                check_cancelled()
                yield first
                progress.update(1)

    grid = sampleRFZ(group, average_angular_spacing=search.average_angular_spacing,
                     progress=track).orientations.T
    logger.info('Generated %d trial orientations for point group %d at %g deg spacing',
                grid.shape[1], group, search.average_angular_spacing)
    return grid


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
    source_cache = os.path.join(_cache_dir(experiment), 'map-source.npz')
    if clean or not os.path.exists(path):
        activity('Building from detector frames')
        maps = build_eta_omega_maps(experiment, plane_data)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        activity('Saving eta/omega maps')
        _save_eta_omega_maps(maps, path, plane_data, np.radians(om.eta_step))
        save_array_cache(source_cache, fingerprint({}, files=(path,)), content_key=_map_key(maps))
        if clean:
            return maps
    else:
        activity('Loading saved maps')
        maps = _load_eta_omega_maps(path, plane_data,
                                    experiment.active_material.two_theta_width, source_cache)
        if np.any((maps.ring_ids < 0) | (maps.ring_ids >= len(plane_data.unexcluded_hkls))):
            maps = load_or_build_eta_omega_maps(experiment, plane_data, clean=True)
        else:
            report_hkls('Map', plane_data.unexcluded_hkls[maps.ring_ids])
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
        ring_of_hkl = {h: i for i, h in enumerate(plane_data.master_hkl_ids)}
        bad = [i for i in active_hkls.tolist() if i not in ring_of_hkl]
        if bad:
            raise ValueError(
                f'active hkl IDs {bad} are not non-excluded hkls of the material')
        rings = np.array([ring_of_hkl[h] for h in active_hkls], dtype=int)
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
    fo = experiment.find_orientations

    ring_ids = _resolve_active_rings(fo.orientation_maps.active_hkls, plane_data)
    report_hkls('Map', plane_data.unexcluded_hkls[ring_ids])

    half = np.radians(experiment.active_material.two_theta_width) / 2
    ring_tths = plane_data.two_thetas[~plane_data.exclusions][ring_ids]
    two_theta_ranges = np.column_stack([ring_tths - half, ring_tths + half])

    eta_step = np.radians(fo.orientation_maps.eta_step)
    eta_edges = _eta_bin_edges(fo.orientation_maps.eta_step)
    n_eta = len(eta_edges) - 1
    row_of_frame, omegas, omega_edges, period = _omega_grid(
        experiment.image_series_list[0])
    ring_maps = np.empty((len(two_theta_ranges), len(omegas), n_eta))
    for ring in ring_maps:
        for start in range(0, len(omegas), 256):
            check_cancelled()
            ring[start:start + 256].fill(np.nan)

    active = [ims for ims in experiment.image_series_list
              if ims.panel in experiment.detectors]

    with progress_bar(total=sum(map(len, active)), desc='Building eta/omega maps',
                      unit='frames', disable=None) as progress:
        for ims, (eta_bin, ring_masks, coverage) in _iter_panel_bins(
                experiment, active, two_theta_ranges, eta_edges):
            activity(f'Binning frames from {ims.panel}')
            processed = 0
            # Preserve NaN gaps; covered bins sum intensities across panels.
            rows = row_of_frame[:len(ims)]
            for ring, bins in enumerate(coverage):
                bins = np.flatnonzero(bins)
                for start in range(0, len(rows), 256):
                    check_cancelled()
                    covered = np.ix_(rows[start:start + 256], bins)
                    ring_maps[ring][covered] = np.nan_to_num(ring_maps[ring][covered], copy=False)

            # Pack each frame chunk once, even for materials with over 63 rings.
            threshold = float(fo.orientation_maps.threshold)
            blocks = [(63 * block, mask) for block, mask in enumerate(ring_masks)
                      if np.any(coverage[63 * block:63 * (block + 1)])]
            if not blocks:
                progress.update(len(ims))
                continue
            workers = max(1, min(experiment.max_workers, len(rows)))
            if len(np.unique(rows)) != len(rows):
                workers = 1
            chunk_size = 1 if workers > 1 else (262144 if not progress.disable else 8_000_000)
            chunks = (ims.pixel_chunks(rows, max_pixels=chunk_size, threshold=threshold) if hasattr(ims, 'pixel_chunks')
                      else _pixel_chunks(ims.images, rows, max_pixels=chunk_size))

            def bin_chunk(chunk):
                ids, vals, offsets, chunk_rows = chunk
                for start in range(0, len(ids), 262144):
                    stop = start + 262144
                    block_offsets = np.clip(offsets - start, 0, min(stop, len(ids)) - start)
                    for first, ring_mask in blocks:
                        check_cancelled()
                        _bin_pixels(block_offsets, ids[start:stop], vals[start:stop], chunk_rows,
                                    ring_mask, eta_bin, threshold, ring_maps[first:first + 63])
                progress.update(len(chunk_rows))

            # Exclusive omega rows preserve addition order without per-worker maps.
            with (ThreadPoolExecutor(workers) if workers > 1 else nullcontext()) as pool:
                pending, queued_pixels = {}, 0
                for chunk in chunks:
                    check_cancelled()
                    if pool is None:
                        bin_chunk(chunk)
                    else:
                        while pending and (len(pending) >= workers or
                                           queued_pixels + len(chunk[0]) > 64 * 1024**2):
                            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for future in completed:
                                future.result()
                                queued_pixels -= pending.pop(future)
                        pending[pool.submit(copy_context().run, bin_chunk, chunk)] = len(chunk[0])
                        queued_pixels += len(chunk[0])
                    processed += len(chunk[3])
                for future in pending:
                    future.result()

            progress.update(len(ims) - processed)

    eta_centers = eta_edges[:2] + 0.5 * eta_step
    eta_step = float(eta_centers[1] - eta_centers[0])
    return EtaOmegaMaps(ring_maps, eta_edges, omega_edges, omegas, period,
                        two_theta_ranges, ring_ids, eta_step)


def _iter_panel_bins(experiment, active, two_theta_ranges, eta_edges):
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


_csr_pixel_ids = native._csrPixelIds

_bin_pixels = native._binPixels


def _omega_grid(image_series):
    """Use the legacy wedge handling and arithmetic to preserve map-axis bytes."""
    from hexrd.core.imageseries.omega import OmegaImageSeries

    series = OmegaImageSeries(image_series)
    omega_degrees = series.omega
    period_degrees = omega_degrees[0, 0] + np.r_[0.0, 360.0]
    rows = np.arange(len(series))
    if series.nwedges > 1:
        wedges = series.omegawedges.wedges
        widths = [(w['ostop']-w['ostart'])/w['nsteps'] for w in wedges]
        if mutil.uniqueVectors(np.atleast_2d(widths), tol=1e-6).size != 1:
            raise ValueError('a multi-wedge scan must use one omega step size')
        lo = wedges[0]['ostart']
        hi = float(rotations.mapAngle(wedges[-1]['ostop'], period_degrees, units='degrees'))
        edges_degrees = np.linspace(lo, hi, int((hi-lo)/widths[0])+1)
        omega_degrees = np.column_stack((edges_degrees[:-1], edges_degrees[1:]))
        rows = np.flatnonzero([series.omega_to_frame(o)[0] != -1
                               for o in np.average(omega_degrees, axis=1)])
    period = np.radians(period_degrees)
    omegas = rotations.mapAngle(np.radians(np.average(omega_degrees, axis=1)), period)
    edges = rotations.mapAngle(np.radians(np.r_[omega_degrees[:, 0], omega_degrees[-1, 1]]), period)
    if abs(edges[0]-edges[-1]) <= const.sqrt_epsf:
        edges[-1] = edges[-2] + np.radians(omega_degrees[0, 1]-omega_degrees[0, 0])
    return rows, omegas, edges, period


def _save_eta_omega_maps(maps: EtaOmegaMaps, path: str,
                         plane_data: PlaneData, eta_step: float) -> None:
    """Write the legacy eta-omega NPZ schema, including crystallographic metadata."""
    import io
    import tempfile
    from hexrd.core.valunits import valWUnit

    class Writer(io.BufferedWriter):
        def write(self, data):
            check_cancelled()
            return super().write(data)

    args = np.array([list(plane_data.lattice_parameters), plane_data.laue_group,
                     valWUnit('wavelength', 'length', plane_data.wavelength, 'angstrom'),
                     float(plane_data.strain_magnitude)], dtype=object)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path) or '.') as directory:
        temporary = os.path.join(directory, 'maps.npz')
        with Writer(io.FileIO(temporary, 'w')) as stream:
            np.savez_compressed(
                stream, dataStore=maps.ring_maps, etas=maps.eta_edges[:-1] + 0.5 * eta_step,
                etaEdges=maps.eta_edges, iHKLList=plane_data.master_hkl_ids[maps.ring_ids],
                omegas=maps.omegas, omeEdges=maps.omega_edges, planeData_args=args,
                planeData_hkls=plane_data.hkls.T, planeData_excl=plane_data.exclusions)
        check_cancelled()
        os.replace(temporary, path)


def _load_eta_omega_maps(path: str, plane_data: PlaneData,
                         two_theta_width: float, source_cache=None) -> EtaOmegaMaps:
    """Load a current or legacy HEXRD eta-omega map archive.

    Args:
        path: Path to the NPZ map archive.
        plane_data: Crystallographic data used to interpret legacy ring IDs.
        two_theta_width: Full ring width in degrees for legacy archives.

    Returns:
        The maps and normalized metadata from the archive.
    """
    from hexrd.core.experiment import _NpzReader

    key = fingerprint({}, files=(path,)) if source_cache else None
    cached = load_array_cache(source_cache, key) if source_cache else None
    lazy = cached is not None and 'content_key' in cached and cached['content_key'].shape == ()
    def read_maps():
        with _NpzReader(path) as archive:
            return np.asarray(archive['dataStore'], dtype=np.float64) if legacy else archive['ring_maps']

    with _NpzReader(path) as d:
        legacy = 'dataStore' in d
        if legacy:  # legacy hexrd archive
            eta_edges, omega_edges = d['etaEdges'], d['omeEdges']
            omegas = d['omegas']
            # the archive's iHKLList holds master hkl IDs; convert to ring indices
            ring_of_hkl = {h: i for i, h in enumerate(plane_data.master_hkl_ids)}
            ring_ids = np.array([ring_of_hkl.get(h, len(plane_data.unexcluded_hkls))
                                 for h in d['iHKLList']], dtype=int)
            half = np.radians(two_theta_width) / 2
            tths = plane_data.two_thetas[~plane_data.exclusions][ring_ids]
            two_theta_ranges = np.column_stack([tths - half, tths + half])
            period = omega_edges[0] + np.r_[0.0, 2 * np.pi]
            eta_step = float(d['etas'][1] - d['etas'][0])
        else:                 # plain-array archive
            eta_edges, omega_edges = d['eta_edges'], d['omega_edges']
            omegas, period = d['omegas'], d['omega_period']
            two_theta_ranges, ring_ids = d['two_theta_ranges'], d['ring_ids']
            eta_step = float(d['eta_step'])
    maps = _LazyEtaOmegaMaps(read_maps, eta_edges, omega_edges, omegas, period,
                             two_theta_ranges, ring_ids, eta_step)
    if lazy:
        maps._content_key = str(cached['content_key'])
    else:
        content_key = fingerprint({}, arrays=(maps.ring_maps,))
        if source_cache:
            save_array_cache(source_cache, key, content_key=content_key)
    return maps


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
        step: Requested eta-bin width in degrees, as in make_powder_rings.

    Returns:
        Eta-bin edges with the branch cut at ``-pi`` and ``pi``.
    """
    n = int(360.0 / step)
    return rotations.mapAngle(np.radians(step * np.linspace(0.0, n, num=n + 1)) - np.pi)


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


_ring_masks = native._ringMasks


def _cached_panel_bin_data(experiment, panel, two_theta_ranges, eta_edges):
    """Reuse geometry across scans while still rebuilding maps from live pixels."""
    from dataclasses import asdict
    import hashlib
    from pathlib import Path
    from hexrd.core.extensions import transforms, inverse_distortion

    detector = experiment.detectors[panel]
    sources = [Path(__file__),
               Path(transforms_c_api.__file__), Path(transforms.__file__),
               Path(inverse_distortion.__file__)]
    sources.extend(sorted(Path(distortion_pkg.__file__).parent.glob('*.py')))
    name = detector.distortion.function_name
    mapping_type = distortion_pkg.Registry.distortion_registry.get(name)
    if name and (mapping_type is not None or
                 name.lower().removesuffix('_newton') not in distortion_pkg._builtin_modules):
        import inspect

        if mapping_type is None:
            mapping_type = type(distortion_pkg.get_mapping(name, detector.distortion.parameters))
        try:
            sources.append(Path(inspect.getfile(mapping_type)))
        except TypeError:
            return _panel_bin_data(detector, experiment.beam.vector,
                                   two_theta_ranges, eta_edges)
    key = fingerprint(asdict(detector),
                      arrays=(experiment.beam.vector, two_theta_ranges, eta_edges),
                      files=sources, packages=('numpy',))
    name = hashlib.sha256(panel.encode()).hexdigest()[:16]
    path = Path(_cache_dir(experiment)) / f'panel-{name}.npz'
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
                    laue_group=plane_data.laue_group, eta_step=maps.eta_step,
                    maps=_map_key(maps))
    sources = [__file__,
               rotations.__file__, mutil.__file__, xfcapi.__file__,
               xfcapi.cpp_transforms.__file__, transforms_c_api.__file__, const.__file__]
    key = fingerprint(settings, files=sources, packages=('numpy', 'scipy', 'scikit-image'),
                      arrays=(maps.eta_edges, maps.omega_edges, maps.omegas,
                              maps.two_theta_ranges, maps.ring_ids,
                              plane_data.unexcluded_hkls, plane_data.B, plane_data.q_sym))
    cached = load_array_cache(cache_file, key)
    if cached is not None and 'fibers' in cached:
        fibers = cached['fibers']
        if fibers.ndim == 2 and fibers.shape[0] == 4 and fibers.dtype == np.float64:
            activity('Loaded cached fibers')
            return fibers
    activity('Finding peaks and generating fibers')
    fibers = generate_orientation_fibers(experiment, plane_data, maps)
    save_array_cache(cache_file, key, fibers=fibers)
    return fibers


def generate_orientation_fibers(experiment: HedmExperiment, plane_data: PlaneData,
                                maps: EtaOmegaMaps) -> np.ndarray:
    """Generate trial orientations from peaks in the seed-ring maps.

    Use the configured worker count while preserving seed and peak order.

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
    csym = rotations.quatProductMatrix(
        rotations.quatOfLaueGroup(plane_data.laue_group), 'right')
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
        report_peaks(hkl, num_spots)
        if not num_spots:
            continue
        prepared = _prepare_fiber(hkl, plane_data.B, fiber_ndiv)
        for ispot in range(num_spots):
            com = coms[ispot]
            if np.isnan(com[0]):
                continue
            ome_c = maps.omega_edges[0] + (0.5 + com[0]) * d_omega
            eta_c = maps.eta_edges[0] + (0.5 + com[1]) * maps.eta_step
            centers.append((*prepared, tth, eta_c, ome_c))
    workers = max(1, min(experiment.max_workers, len(centers)))
    with progress_bar(total=len(centers), desc='Generating fibers', unit='fiber', disable=None) as progress:
        def generate(center):
            result = _fiber(*center, chi, csym)
            progress.update(1)
            return result

        if workers == 1:
            q_fibers = list(map(generate, centers))
        else:
            with ThreadPoolExecutor(workers) as pool:
                q_fibers = list(pool.map(generate, centers))
    return np.asfortranarray(np.hstack(q_fibers)) if q_fibers else np.empty((4, 0))


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


def _prepare_fiber(hkl, b_matrix, divisions):
    crystal = mutil.unitVector(np.dot(b_matrix, hkl.reshape(3, 1)))
    phi = np.arange(divisions) * (2*np.pi/float(divisions))
    quats = rotations.quatOfAngleAxis(phi, np.tile(crystal[:, 0], (divisions, 1)).T)
    return crystal, rotations.quatProductMatrix(quats, 'right')


def _fiber(crystal, fiber_rotations, tth, eta_c, ome_c, chi, csym):
    """Build one fiber using discreteFiber's arithmetic and uniqueVectors' groups."""
    sample = mutil.unitVector(xfcapi.angles_to_gvec(
        np.atleast_2d([tth, eta_c, ome_c]), chi=chi).T)
    axis = sample + crystal
    norm = mutil.columnNorm(axis).squeeze()
    axis = (axis / np.tile(norm, (3, 1)) if norm > const.sqrt_epsf
            else mutil.nullSpace(crystal)[:, :1])
    initial = np.vstack([np.zeros(1), axis])
    fiber = np.dot(fiber_rotations, initial).transpose(2, 1, 0).squeeze()
    reduced = native._reduceFiber(fiber, csym)
    return (reduced if reduced is not None else mutil.uniqueVectors(
        rotations.toFundamentalRegion(fiber, csym[:, :, 0].T)))


_hit_status_maps = native.orientationHitStatus


def _pack_hit_status(status):
    """Pack two-bit status codes into 8x8 tiles."""
    rings, rows, cols = status.shape
    packed = np.empty((rings, ((rows+7)//8) * ((cols+7)//8) * 16), dtype=np.int8)
    for ring, values in enumerate(status):
        padded = np.pad(values, ((0, -rows % 8), (0, -cols % 8)), constant_values=-1)
        tiles = padded.reshape((rows+7)//8, 8, (cols+7)//8, 8).transpose(0, 2, 1, 3)
        packed[ring] = np.packbits(
            np.stack((tiles >= 0, tiles > 0), axis=-1), bitorder='little').view(np.int8)
    return packed.ravel()


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

    bmat = np.ascontiguousarray(plane_data.B)
    beam_vector = np.ascontiguousarray(experiment.beam.vector, dtype=np.float64)
    n = q_fibers.shape[1]
    workers = max(1, min(experiment.max_workers, max(1, n // 2048)))
    shape = (len(maps.ring_ids), len(maps.omega_edges)-1, len(maps.eta_edges)-1)
    cache_dir = _cache_dir(experiment)
    sources = (__file__, native.__file__, rotations.__file__, const.__file__)
    cache_bins = (np.prod(shape) <= np.iinfo(np.int32).max and
                  n*len(symm_hkls)*8 <= 128*1024**2)
    bins_key = fingerprint(dict(wavelength=plane_data.wavelength, offset=ome_offset, shape=shape),
        files=sources, packages=('numpy',), arrays=(q_fibers, symm_hkls, bmat, beam_vector,
            maps.eta_edges, maps.omega_edges, valid_eta, valid_ome, ring_for_hkl)) if cache_bins else None
    bins_file = os.path.join(cache_dir, 'reflection-bins.npz')
    cached = load_array_cache(bins_file, bins_key, mmap=True)
    bins = cached.get('bins') if cached is not None else None
    if bins is not None and (bins.shape != (n, 2, len(symm_hkls)) or bins.dtype != np.int32):
        bins = None
    status_key = fingerprint(dict(maps=_map_key(maps), de=dpix_eta, dome=dpix_ome,
                                  threshold=fo.threshold), files=(native.__file__,))
    status_file = os.path.join(cache_dir, 'hit-status.npz')
    cached = load_array_cache(status_file, status_key, mmap=True)
    hit_status = cached.get('status') if cached is not None else None
    if hit_status is not None and (hit_status.shape != shape or hit_status.dtype != np.int8):
        hit_status = None
    # Only the shape is needed when hit status is cached.
    ring_maps = (np.ascontiguousarray(maps.ring_maps) if hit_status is None else
                 np.broadcast_to(np.zeros(()), shape))
    cache_bins = cache_bins and bins_key is not None
    window_area = (2 * dpix_eta + 1) * (2 * dpix_ome + 1)
    # Serial lookup construction must pay for each worker's share of scoring.
    if hit_status is None and (cache_bins or
            n * len(symm_hkls) * window_area >= 8 * workers * ring_maps.size and window_area > 1
            and np.prod(ring_maps.shape[1:]) < np.iinfo(np.int32).max):
        hit_status = _hit_status_maps(ring_maps, dpix_eta, dpix_ome, fo.threshold)
        save_array_cache(status_file, status_key, status=hit_status)
    if bins is not None and hit_status is not None:
        activity('Scoring cached reflection bins')
        return native._scoreOrientationBins(bins, hit_status)
    bins = np.empty((n, 2, len(symm_hkls)), dtype=np.int32) if cache_bins else None
    if hit_status is not None and hit_status.size >= 16*1024**2:
        hit_status = _pack_hit_status(hit_status)
    scores = np.empty(n)
    with progress_bar(total=n, desc='Scoring orientations', unit='orientation', disable=None) as progress:
        chunks = max(workers, (n + 262143) // 262144)
        bounds = np.linspace(0, n, chunks + 1, dtype=int)

        def score_chunk(start, stop):
            orientations = q_fibers[:, start:stop]
            if orientations.dtype != np.float64:
                orientations = rotations.rotMatOfQuat(orientations).reshape(-1, 3, 3)
            scores[start:stop] = native.scoreOrientations(
                orientations, symm_hkls, bmat, plane_data.wavelength, beam_vector,
                maps.eta_edges, maps.omega_edges, valid_eta, valid_ome, ome_offset,
                ring_maps, ring_for_hkl, dpix_eta, dpix_ome, float(fo.threshold), hit_status,
                bins[start:stop] if bins is not None else None)
            progress.update(stop - start)

        if chunks == 1:
            score_chunk(0, n)
        else:
            with ThreadPoolExecutor(workers) as pool:
                for _ in pool.map(score_chunk, bounds[:-1], bounds[1:]):
                    pass
    if bins is not None:
        save_array_cache(bins_file, bins_key, bins=bins)
    return scores

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


def _cached_min_samples(experiment, plane_data, maps):
    from dataclasses import asdict

    fo = experiment.find_orientations
    key = fingerprint(dict(chi=experiment.oscillation_stage.chi, wavelength=plane_data.wavelength,
        completeness=fo.clustering.completeness,
        detectors={name: asdict(panel) for name, panel in experiment.detectors.items()}),
        arrays=(plane_data.B, experiment.beam.vector, experiment.image_series_list[0].omega,
                maps.ring_ids, fo.seed_search.hkl_seeds, fo.eta.range, *plane_data.symm_hkls),
        files=(__file__, native.__file__, rotations.__file__, xfcapi.__file__, const.__file__),
        packages=('numpy',))
    path = os.path.join(_cache_dir(experiment), 'min-samples.npz')
    cached = load_array_cache(path, key)
    if cached is not None and 'value' in cached and cached['value'].shape == ():
        return int(cached['value'])
    value = estimate_min_samples(experiment, plane_data, maps)
    save_array_cache(path, key, value=value)
    return value


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
    diagnostic('Candidates',
               f'{n:,}/{q_fibers.shape[1]:,} (score > {fo.clustering.completeness:.1%})')
    diagnostic('Best trial', f'{np.max(completeness):.1%}' if len(completeness) else 'No trials')
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
    # Preserve centroid order when no merge is needed.
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
            phi = 2 * np.arccos(q[0])
            n = xfcapi.unit_vector(q[1:])
            grain_params = np.hstack(
                [phi * n, np.zeros(3), np.array([1., 1., 1., 0., 0., 0.])])
            res = [int(gid), 1.0, 0.0] + grain_params.tolist() + [0.0] * 6
            out = delim.join([
                delim.join(['{:<12d}', '{:<12f}', '{:<12e}']).format(*res[:3]),
                delim.join(['{:<23.16e}'] * (len(res) - 3)).format(*res[3:]),
            ])
            fid.write(out + '\n')
