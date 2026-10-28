"""The HEDM experiment: shared experiment inputs plus the parameters of the
HEDM analyses, as typed, frozen config sections.

:class:`HedmExperiment` extends :class:`hexrd.core.experiment.Experiment`
with the ``find_orientations`` section of the config.
"""
import enum
import os
import threading
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from pydantic import Field, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass

from hexrd.core.experiment import Beam, Experiment, ImageSeries, _pixel_chunks

_FCH5_PARALLEL_MIN_PIXELS = 128 * 1024**2


@lru_cache(maxsize=1)
def _fch5_file(filename):
    import h5py
    import hdf5plugin  # Register the frame cache compression filters.

    return h5py.File(filename, 'r')


def _fch5_pixels(group, start, stop, shape, dtype, threshold):
    from scipy.sparse import csr_array

    indices = group['indices'][int(start):int(stop)]
    data = group['data'][int(start):int(stop), 0].astype(dtype, copy=False)
    canonical = indices.dtype.kind in 'iu'
    if canonical:
        if np.any(indices < 0) or np.any(indices >= shape):
            raise ValueError('frame-cache pixel index is out of bounds')
        ids = np.multiply(indices[:, 0], shape[1], dtype=np.int64)
        ids += indices[:, 1]
        canonical = np.all(ids[1:] > ids[:-1])
    if not canonical:
        frame = csr_array((data, (indices[:, 0], indices[:, 1])), shape=shape, dtype=dtype).tocoo()
        ids, data = frame.row.astype(np.int64) * shape[1] + frame.col, frame.data
    if threshold is not None:
        data = data.astype(np.float64, copy=False)
        keep = ~(data < threshold)
        ids, data = ids[keep], data[keep]
    return ids, data


def _read_fch5_frame(args):
    index, filename, start, stop, shape, dtype, threshold = args
    return index, _fch5_pixels(_fch5_file(filename), start, stop, shape, dtype, threshold)


class _AdapterImageSeries(ImageSeries):
    """Expose registered image-series formats as sparse frames for map building."""

    def __init__(self, panel, filename, fmt, args, max_workers=1):
        from hexrd.core import imageseries

        self._initialize(panel, imageseries.open(filename, fmt, **args), max_workers)

    def _initialize(self, panel, series, max_workers):
        from hexrd.core.imageseries.omega import OmegaImageSeries

        while isinstance(series, OmegaImageSeries):
            series = series._adapter
        self._series = series
        self._max_workers = max_workers
        self.panel = panel if panel is not None else self._series.metadata.get('panel')
        self.omega = np.radians(self._series.metadata['omega'])
        self._n_frames = len(self._series)
        self._images = None
        self._lock = threading.Lock()

    def pixel_chunks(self, row_of_frame, max_pixels=8_000_000, threshold=None):
        from scipy.sparse import csr_array
        from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter
        from hexrd.core.extensions import transforms as native
        from hexrd.hedm._progress import activity, check_cancelled

        if isinstance(self._series, ImageSeries):
            yield from self._series.pixel_chunks(row_of_frame, max_pixels, threshold)
            return
        adapter = self._series._adapter
        if isinstance(adapter, FrameCacheImageSeriesAdapter) and adapter._style == 'fch5':
            import multiprocessing as mp
            import psutil

            with adapter._fch5_group_cm() as group:
                frames = group['frame_ids'][()].reshape(-1, 2)
                shape, dtype = tuple(adapter.shape), adapter.dtype
                largest = max((int(stop) - int(start) for start, stop in frames), default=0)
                budget = min(16 * 1024**3, psutil.virtual_memory().available // 4)
                # Reserve space for decoded coordinates, weights, and IPC copies.
                workers = min(self._max_workers, len(frames), max(1, budget // max(1, largest * 64)))
                large = sum(int(stop) - int(start) for start, stop in frames) >= _FCH5_PARALLEL_MIN_PIXELS
                if workers > 1 and large and isinstance(adapter._fname, (str, os.PathLike)):
                    activity(f'Reading {self.panel} with {workers} FCH5 decoder processes')
                    ordered = len(np.unique(row_of_frame)) != len(row_of_frame)
                    with mp.get_context('spawn').Pool(workers) as pool:
                        for first in range(0, len(frames), workers):
                            jobs = [(i, os.fspath(adapter._fname), *frames[i], shape, dtype, threshold)
                                    for i in range(first, min(first + workers, len(frames)))]
                            results = (pool.imap if ordered else pool.imap_unordered)(_read_fch5_frame, jobs)
                            for _ in jobs:
                                while True:
                                    check_cancelled()
                                    try:
                                        i, (ids, data) = results.next(timeout=0.1)
                                        break
                                    except mp.TimeoutError:
                                        pass
                                yield ids, data.astype(np.float64, copy=False), np.array([0, len(ids)]), row_of_frame[i:i+1]
                    return

                def pixels():
                    for start, stop in frames:
                        check_cancelled()
                        yield _fch5_pixels(group, start, stop, shape, dtype, threshold)

                yield from _pixel_chunks(pixels(), row_of_frame, max_pixels)
            return

        def pixels():
            for i in range(len(self)):
                check_cancelled()
                frame = csr_array(self._series[i])
                ids = np.empty(frame.nnz, dtype=np.int64)
                native._csrPixelIds(frame.indptr, frame.indices, frame.shape[1], ids)
                yield ids, frame.data

        yield from _pixel_chunks(pixels(), row_of_frame, max_pixels)

    @property
    def images(self):
        with self._lock:
            if self._images is None:
                from scipy.sparse import csr_array

                self._images = [csr_array(self._series[i]) for i in range(len(self))]
        return self._images

    @property
    def metadata(self):
        return self._series.metadata


class SeedSearchMethod(enum.StrEnum):
    LABEL = 'label'
    BLOB_LOG = 'blob_log'
    BLOB_DOG = 'blob_dog'


class ClusteringAlgorithm(enum.StrEnum):
    DBSCAN = 'dbscan'
    ORT_DBSCAN = 'ort-dbscan'
    SPH_DBSCAN = 'sph-dbscan'
    FCLUSTERDATA = 'fclusterdata'


def _hkl_array(value: Any, key: str) -> Optional[np.ndarray]:
    """Canonicalize an hkl selection: an (n, 3) array of [h, k, l] triples,
    an (n,) array of master hkl IDs, or None for all rings.

    Legacy scalar IDs and 'all' are normalized at the config boundary.
    """
    if value is None or (isinstance(value, str) and value == 'all'):
        return None
    hkls = np.atleast_1d(value) if isinstance(value, int) else np.asarray(value)
    if not np.issubdtype(hkls.dtype, np.integer) or hkls.ndim not in (1, 2) \
            or (hkls.ndim == 2 and hkls.shape[1] != 3):
        raise ValueError(
            f'{key} must be a list of [h, k, l] integer triples or of '
            f'integer hkl IDs (or null for all rings), got {value!r}')
    return hkls


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class OrientationMaps:
    threshold: float | None
    active_hkls: Optional[np.ndarray]   # (n, 3) hkl vectors or (n,) master
                                        # hkl IDs; None -> all rings
    eta_step: float                     # degrees
    file: Optional[str]                 # maps cache; None -> default path
    filter_maps: bool                   # subtract each eta column's median
    filter_fwhm: Optional[float]        # additionally LoG-filter, this FWHM in pixels

    @classmethod
    def from_dict(cls, d: dict) -> 'OrientationMaps':
        filter_maps = d.get('filter_maps', False)
        if filter_maps is None:
            filter_maps = False
        if not isinstance(filter_maps, (bool, int, float)):
            raise ValueError(
                'filter_maps must be a boolean or a numeric LoG width in pixels')
        filter_fwhm = d.get('filter_fwhm')
        if not isinstance(filter_maps, bool):
            if filter_maps and filter_fwhm is not None:
                raise ValueError('specify the LoG width only once')
            filter_fwhm = float(filter_maps) if filter_maps else None
            filter_maps = bool(filter_maps)
        if filter_fwhm is not None and not filter_maps:
            raise ValueError('filter_fwhm requires filter_maps: true')
        # a null threshold means "no thresholding", same as 0 for count data
        return cls(float(d.get('threshold') or 0),
                   _hkl_array(d.get('active_hkls'), 'active_hkls'),
                   float(d.get('eta_step', 0.25)), d.get('file'),
                   filter_maps,
                   None if filter_fwhm is None else float(filter_fwhm))


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class SeedSearch:
    hkl_seeds: np.ndarray               # indices into the active rings
    fiber_step: float                   # radians
    method: SeedSearchMethod
    method_kwargs: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict, omega_tolerance_deg: float) -> 'SeedSearch':
        # `method` is a one-entry mapping: {name: {kwargs...}}
        method_dict = d.get('method') or {'label': {}}
        (name, kwargs), = method_dict.items()
        seeds = np.atleast_1d(np.asarray(d.get('hkl_seeds', []), dtype=np.intp))
        if seeds.ndim != 1:
            raise ValueError(f'hkl_seeds must be a list of ring indices, '
                             f'got {d.get("hkl_seeds")!r}')
        return cls(seeds, np.radians(d.get('fiber_step', omega_tolerance_deg)),
                   SeedSearchMethod(name.lower()), kwargs or {})

    @property
    def fiber_ndiv(self) -> int:
        return int(round(2 * np.pi / self.fiber_step))


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class DirectSearch:
    """Uniform fundamental-zone sampling; angular spacing is in degrees.

    The point group defaults to the selected material's crystallographic
    point group (1-32), rather than its Laue group.

    Spacing must be explicit and chosen for the scoring tolerances. Master
    accepted externally sampled grids through ``use_quaternion_grid``;
    this section is a convenience for generating those trials in-process.

    Historical YAML, still supported::

        use_quaternion_grid: orientations.npy
    """

    average_angular_spacing: float = Field(gt=0, allow_inf_nan=False)
    point_group_number: int | None = Field(default=None, strict=True, ge=1, le=32)

    @field_validator('point_group_number', mode='before')
    @classmethod
    def normalize_point_group(cls, value):
        return int(value) if isinstance(value, np.integer) else value

    @classmethod
    def from_dict(cls, d: dict) -> 'DirectSearch':
        return TypeAdapter(cls).validate_python(d)


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class Omega:
    tolerance: float           # radians
    period: np.ndarray         # radians

    @classmethod
    def from_dict(cls, d: dict) -> 'Omega':
        period = np.radians(d.get('period', [-180.0, 180.0]))
        if not np.isclose(abs(period[1] - period[0]), 2 * np.pi,
                          rtol=0, atol=1e-8):
            raise ValueError('omega period range must be 360 degrees')
        return cls(np.radians(d.get('tolerance', 0.5)), period)


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class Eta:
    tolerance: float           # radians
    mask: Optional[float]      # radians

    @classmethod
    def from_dict(cls, d: dict) -> 'Eta':
        mask = d.get('mask', 5)
        return cls(np.radians(d.get('tolerance', 0.5)),
                   np.radians(mask) if mask is not None else None)

    @property
    def range(self) -> np.ndarray:
        """Valid eta spans, masking the region near the rotation axis.

        A null mask means no masking: the full circle is valid.
        """
        if self.mask is None:
            return np.array([[-np.pi, np.pi]])
        return np.array([[-np.pi / 2 + self.mask,     np.pi / 2 - self.mask],
                         [np.pi / 2 + self.mask, 3 * np.pi / 2 - self.mask]])


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class Clustering:
    radius: float
    completeness: float
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.DBSCAN

    @field_validator('algorithm', mode='before')
    @classmethod
    def normalize_algorithm(cls, value):
        return value.lower() if isinstance(value, str) else value

    @classmethod
    def from_dict(cls, d: dict) -> 'Clustering':
        return TypeAdapter(cls).validate_python(d)


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class FindOrientations:
    orientation_maps: OrientationMaps
    seed_search: SeedSearch
    omega: Omega
    eta: Eta
    clustering: Clustering
    threshold: float
    use_quaternion_grid: Optional[str]  # .npy of trial quats; bypasses generation
    direct_search: DirectSearch | None = None

    @classmethod
    def from_dict(cls, d: dict) -> 'FindOrientations':
        omega = Omega.from_dict(d.get('omega', {}))
        direct = d.get('direct_search')
        if direct is not None and (d.get('seed_search') or d.get('use_quaternion_grid')):
            raise ValueError('choose direct_search, seed_search, or use_quaternion_grid')
        return cls(OrientationMaps.from_dict(d.get('orientation_maps', {})),
                   SeedSearch.from_dict(d.get('seed_search', {}),
                                        np.degrees(omega.tolerance)),
                   omega,
                   Eta.from_dict(d.get('eta', {})),
                   Clustering.from_dict(d.get('clustering', {})),
                   float(d.get('threshold', 1)),
                   d.get('use_quaternion_grid'),
                   None if direct is None else DirectSearch.from_dict(direct))


def _pair(value):
    return [value, value] if isinstance(value, (int, float)) else value


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class FitGrainsTolerance:
    eta: list | None
    omega: list | None
    tth: list | None

    @classmethod
    def from_dict(cls, d: dict) -> 'FitGrainsTolerance':
        return cls(_pair(d.get('eta')), _pair(d.get('omega')),
                   _pair(d.get('tth')))


@dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class FitGrains:
    do_fit: bool
    estimate: str | None
    npdiv: int
    threshold: float | None
    tolerance: FitGrainsTolerance
    refit: list | None
    tth_max: bool | float
    reset_exclusions: bool
    exclusion_parameters: dict

    @classmethod
    def from_dict(cls, d: dict, path) -> 'FitGrains':
        estimate = d.get('estimate')
        if estimate is not None and not os.path.isabs(estimate):
            estimate = path(estimate)
        names = ('dmin', 'dmax', 'tthmin', 'tthmax', 'sfacmin', 'sfacmax',
                 'pintmin', 'pintmax')
        exclusions = {name: d.get(name) for name in names}
        exclusions['sfacmin'] = d.get('sfacmin', d.get('min_sfac_ratio'))
        return cls(d.get('do_fit', True), estimate, d.get('npdiv', 2),
                   d.get('threshold'),
                   FitGrainsTolerance.from_dict(d.get('tolerance', {})),
                   _pair(d.get('refit')), d.get('tth_max', True),
                   d.get('reset_exclusions', True), exclusions)


class HedmExperiment(Experiment):
    """An :class:`Experiment` plus the HEDM find-orientations parameters."""

    def __init__(self, filename: str, study: int | None = None, *,
                 working_dir: str | None = None, new_file_placement: bool = False):
        self._default_working_dir = working_dir
        self._default_file_placement = new_file_placement
        super().__init__(filename, study)
        self._configure_hedm()

    def _configure_hedm(self):
        if self.active_material.two_theta_width is None:
            self.active_material.two_theta_width = 0.25
        self.find_orientations = FindOrientations.from_dict(
            self.config['find_orientations'])
        self.fit_grains = FitGrains.from_dict(
            self.config.get('fit_grains', {}), self._path)

    @classmethod
    def from_objects(cls, instrument, image_series, plane_data, config):
        """Build from an HEDMInstrument, panel-to-series mapping, and PlaneData.

        ``config`` is a configuration dict. Relative paths use its working_dir
        (the current directory by default). Inputs are shared, not copied;
        series must include omega metadata in degrees. Material exclusions are
        preserved. Direct sampling defaults to the supplied Laue group's RFZ;
        use direct_search.point_group_number to specify a different point group.
        """
        from copy import deepcopy
        from types import SimpleNamespace
        from hexrd.core.imageseries.omega import OmegaImageSeries
        from hexrd.core.material.material_data import Material

        experiment = cls.__new__(cls)
        experiment.experiment_dir = os.getcwd()
        experiment.studies = []
        experiment._default_working_dir = os.getcwd()
        experiment._default_file_placement = False
        experiment._configure(deepcopy(config), instrument.write_config())
        experiment._instrument = instrument
        experiment.beam = Beam(instrument.beam_energy, instrument.beam_vector,
                               instrument.source_distance, instrument.energy_correction)
        experiment._image_series = {
            panel: series if isinstance(series, OmegaImageSeries) else OmegaImageSeries(series)
            for panel, series in image_series.items()}
        experiment.image_series_list = []
        for panel, series in image_series.items():
            if panel not in experiment.detectors:
                raise ValueError(f'unknown detector panel: {panel}')
            view = _AdapterImageSeries.__new__(_AdapterImageSeries)
            view._initialize(panel, series, experiment.max_workers)
            experiment.image_series_list.append(view)
        groups = {'ci': 2, 'c2h': 5, 'd2h': 8, 'c4h': 11, 'd4h': 15,
                  'c3i': 17, 's6': 17, 'd3d': 20, 'c6h': 23, 'd6h': 27,
                  'th': 29, 'oh': 32}
        material = Material.__new__(Material)
        material.name = experiment.active_material.active
        material.dmin = material.sfacmin = None
        material.reset_exclusions = False
        material._material = SimpleNamespace(
            planeData=plane_data, pgnum=groups[plane_data.laueGroup.lower()])
        material.plane_data = material._make_plane_data()
        experiment._active_material = material
        experiment._materials_file = None
        experiment._configure_hedm()
        return experiment

    def get_active_material(self):
        if hasattr(self, '_active_material'):
            return self._active_material
        return super().get_active_material()

    def get_materials(self):
        if self._materials_file is None:
            return [self.get_active_material()]
        return super().get_materials()

    def _load_image_series(self, spec):
        panel = spec.get('panel')
        fmt = self.config['image_series'].get('format', 'frame-cache')
        args = spec.get('args', {})
        if (panel is not None and fmt == 'frame-cache'
                and args.get('style', 'npz') == 'npz'
                and str(spec['file']).endswith('.npz')):
            panel, series = super()._load_image_series(spec)
        else:
            series = _AdapterImageSeries(panel, self._path(spec['file']), fmt, args, self.max_workers)
            if 'panel' not in spec:
                panel = series.panel
        if panel is None:
            panel = list(self.detectors)
        elif isinstance(panel, str) and panel not in self.detectors:
            panel = [name for name, det in self.detectors.items()
                     if det.group == panel] or panel
        return panel, series

    @property
    def eta_ome_maps_file(self) -> str:
        """Where the eta-omega maps cache lives (orientation_maps: file, or default)."""
        configured = self.find_orientations.orientation_maps.file
        if configured:
            return configured if os.path.isabs(configured) else self._path(configured)
        return self._output_file('eta-ome-maps-{material}.npz',
                                 '{analysis_id}_eta-ome_maps.npz')

    def _path(self, name: str) -> str:
        root = self.config.get('working_dir', self._default_working_dir)
        if root is None:
            return super()._path(name)
        if not os.path.isdir(root):
            raise IOError(f'"working_dir": {root} does not exist')
        return os.path.join(root, name)

    def _output_file(self, new_name, old_name):
        material = self.active_material.active.strip().replace(' ', '-')
        if self.config.get('new_file_placement', self._default_file_placement):
            return os.path.join(self.analysis_dir, new_name.format(material=material))
        analysis_id = f'{self.analysis_name.strip().replace(" ", "-")}_{material}'
        return self._path(old_name.format(analysis_id=analysis_id))

    @property
    def accepted_orientations_file(self) -> str:
        return self._output_file('accepted-orientations-{material}.dat',
                                 'accepted_orientations_{analysis_id}.dat')

    @property
    def scored_orientations_file(self) -> str:
        return self._output_file('scored-orientations-{material}.npz',
                                 'scored_orientations_{analysis_id}.npz')

    @property
    def find_orientations_logfile(self) -> str:
        return self._output_file('find-orientations-{material}.log',
                                 'find-orientations_{analysis_id}.log')

    @property
    def quaternion_grid_file(self) -> str | None:
        """Absolute path of the trial-quaternion grid, if configured."""
        grid = self.find_orientations.use_quaternion_grid
        if grid is None:
            return None
        return grid if os.path.isabs(grid) else self._path(grid)
