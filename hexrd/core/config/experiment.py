import enum
import io
import os
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import h5py
import yaml
from scipy.sparse import csr_array

from hexrd.core.material.material_data import Material
from hexrd.core.extensions import transforms


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
    """Canonicalize an hkl selection to an (n, 3) int array, or None for all.

    Exactly one form is accepted: a list of [h, k, l] triples. The looser
    forms hexrd tolerates (hkl IDs, single ints, 'all') are rejected so that
    one representation flows through the whole pipeline.
    """
    if value is None:
        return None
    hkls = np.asarray(value)
    if hkls.ndim != 2 or hkls.shape[1] != 3 or not np.issubdtype(hkls.dtype, np.integer):
        raise ValueError(
            f'{key} must be a list of [h, k, l] integer triples '
            f'(or null for all rings), got {value!r}')
    return hkls


# Numeric config defaults mirror hexrd.hedm.config.findorientations.
@dataclass(frozen=True)
class OrientationMaps:
    threshold: float
    active_hkls: Optional[np.ndarray]   # (n, 3) int hkl vectors; None -> all rings
    eta_step: float                     # degrees
    file: Optional[str]                 # maps cache; None -> default path
    filter_maps: bool                   # subtract each eta column's median
    filter_fwhm: Optional[float]        # additionally LoG-filter, this FWHM in pixels

    @classmethod
    def from_dict(cls, d: dict) -> 'OrientationMaps':
        filter_maps = d.get('filter_maps', False)
        if not isinstance(filter_maps, bool):
            raise ValueError(
                'filter_maps must be a boolean; give the LoG width (pixels) '
                'separately as filter_fwhm')
        filter_fwhm = d.get('filter_fwhm')
        if filter_fwhm is not None and not filter_maps:
            raise ValueError('filter_fwhm requires filter_maps: true')
        return cls(float(d.get('threshold', 0)),
                   _hkl_array(d.get('active_hkls'), 'active_hkls'),
                   float(d.get('eta_step', 0.25)), d.get('file'),
                   filter_maps,
                   None if filter_fwhm is None else float(filter_fwhm))


@dataclass(frozen=True)
class SeedSearch:
    hkl_seeds: np.ndarray               # indices into the active rings
    fiber_step: float                   # radians
    method: SeedSearchMethod
    method_kwargs: dict[str, float]

    @classmethod
    def from_dict(cls, d: dict, omega_tolerance_deg: float) -> 'SeedSearch':
        # `method` is a one-entry mapping: {name: {kwargs...}} (as in hexrd)
        method_dict = d.get('method') or {'label': {}}
        (name, kwargs), = method_dict.items()
        seeds = np.asarray(d.get('hkl_seeds', []), dtype=np.intp)
        if seeds.ndim != 1:
            raise ValueError(f'hkl_seeds must be a list of ring indices, '
                             f'got {d.get("hkl_seeds")!r}')
        return cls(seeds, np.radians(d.get('fiber_step', omega_tolerance_deg)),
                   SeedSearchMethod(name), kwargs or {})

    @property
    def fiber_ndiv(self) -> int:
        return int(round(2 * np.pi / self.fiber_step))


@dataclass(frozen=True)
class Omega:
    tolerance: float           # radians

    @classmethod
    def from_dict(cls, d: dict) -> 'Omega':
        return cls(np.radians(d.get('tolerance', 0.5)))


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Clustering:
    radius: float
    completeness: float
    algorithm: ClusteringAlgorithm

    @classmethod
    def from_dict(cls, d: dict) -> 'Clustering':
        missing = {'radius', 'completeness'} - d.keys()
        if missing:
            raise ValueError(f'clustering config requires {sorted(missing)}')
        return cls(float(d['radius']), float(d['completeness']),
                   ClusteringAlgorithm(d.get('algorithm', 'dbscan')))


@dataclass(frozen=True)
class FindOrientations:
    orientation_maps: OrientationMaps
    seed_search: SeedSearch
    omega: Omega
    eta: Eta
    clustering: Clustering
    threshold: float
    use_quaternion_grid: Optional[str]  # .npy of trial quats; replaces seed search

    @classmethod
    def from_dict(cls, d: dict) -> 'FindOrientations':
        omega = Omega.from_dict(d.get('omega', {}))
        return cls(OrientationMaps.from_dict(d.get('orientation_maps', {})),
                   SeedSearch.from_dict(d.get('seed_search', {}),
                                        np.degrees(omega.tolerance)),
                   omega,
                   Eta.from_dict(d.get('eta', {})),
                   Clustering.from_dict(d.get('clustering', {})),
                   float(d.get('threshold', 1)),
                   d.get('use_quaternion_grid'))


@dataclass
class ActiveMaterial:
    definitions: str
    active: str
    dmin: float                # angstrom
    two_theta_width: float     # degrees
    min_sfac_ratio: float

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('definitions'), d.get('active'), d.get('dmin'),
                   d.get('tth_width'), d.get('min_sfac_ratio'))


@dataclass
class Distortion:
    function_name: str
    parameters: np.ndarray

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('function_name', ''), np.array(d.get('parameters', [])))


@dataclass
class Transform:
    tilt: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d.get('tilt', [0.0, 0.0, 0.0])),
                   np.array(d.get('translation', [0.0, 0.0, 0.0])))

    @property
    def rotation_matrix(self):
        return transforms.make_rot_mat_of_exp_map(self.tilt)


@dataclass
class Pixels:
    columns: int
    rows: int
    size: list

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('columns', 0), d.get('rows', 0), d.get('size', [0.0, 0.0]))


@dataclass
class Detector:
    name: str
    distortion: Distortion
    pixels: Pixels
    transform: Transform
    buffer: Optional[np.ndarray]   # (2,) edge buffer in mm

    @classmethod
    def from_dict(cls, name, d):
        buffer = d.get('buffer')
        return cls(name, Distortion.from_dict(d.get('distortion', {})),
                   Pixels.from_dict(d.get('pixels', {})),
                   Transform.from_dict(d.get('transform', {})),
                   None if buffer is None else np.asarray(buffer, dtype=np.float64))

    @property
    def pixel_coordinates(self):
        """(row, col) lab coordinates of every pixel center, as an ij meshgrid."""
        rows, cols, (dy, dx) = self.pixels.rows, self.pixels.columns, self.pixels.size
        row = dy * (0.5 * (rows - 1) - np.arange(rows))
        col = dx * (np.arange(cols) - 0.5 * (cols - 1))
        return np.meshgrid(row, col, indexing='ij')


@dataclass
class OscillationStage:
    chi: float
    translation: np.ndarray

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('chi', 0.0), np.array(d.get('translation', [0.0, 0.0, 0.0])))


def _parse_multiprocessing(spec) -> int:
    """Worker count from the config's `multiprocessing` value (hexrd semantics):
    'all', 'half', a positive count, or a negative offset from the cpu count."""
    ncpus = os.cpu_count() or 1
    if spec == 'all':
        return ncpus
    if spec == 'half':
        return max(ncpus // 2, 1)
    if isinstance(spec, int):
        n = spec if spec > 0 else ncpus + spec
        return min(max(n, 1), ncpus)
    raise ValueError(f"multiprocessing must be 'all', 'half', or an integer, "
                     f"got {spec!r}")


def _frame_csr(data, row, col, shape, dtype) -> csr_array:
    """CSR frame from COO triplets, skipping the COO sort when already canonical.

    Frame caches are written row-major with unique pixels, so the flat indices
    are strictly increasing and the CSR arrays can be assembled directly; the
    result is identical to what the (data, (row, col)) constructor canonicalizes
    to. Falls back to that constructor otherwise.
    """
    flat = row.astype(np.int64) * shape[1] + col
    if flat.size == 0 or np.all(np.diff(flat) > 0):
        indptr = np.searchsorted(row, np.arange(shape[0] + 1))
        return csr_array((data, col, indptr), shape=shape, dtype=dtype)
    return csr_array((data, (row, col)), shape=shape, dtype=dtype)


class _NpzReader:
    """Random access to an .npz held in memory, bypassing zipfile's read path.

    np.load spends most of its time in per-member Python plumbing under the
    GIL; slicing the compressed bytes directly and handing them to
    zlib.decompress (which releases the GIL) yields the same bytes and lets
    member reads run in parallel.
    """

    def __init__(self, filename: str):
        with open(filename, 'rb') as f:
            self._buf = f.read()
        with zipfile.ZipFile(io.BytesIO(self._buf)) as zf:
            self._members = {i.filename: i for i in zf.infolist()}

    def __getitem__(self, name: str) -> np.ndarray:
        info = self._members[name + '.npy']
        # local file header: sizes of the variable name/extra fields at 26/28
        name_len, extra_len = struct.unpack_from('<HH', self._buf, info.header_offset + 26)
        start = info.header_offset + 30 + name_len + extra_len
        raw = self._buf[start:start + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(raw, wbits=-15)
        return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)


class ImageSeries:
    """A frame-cache (.npz) of sparse detector frames for one panel, with omegas.

    Construction only indexes the file; the frames themselves are decompressed
    on first access to ``images``, so runs that never look at the frames (e.g.
    with cached eta-omega maps) never pay for them.
    """

    def __init__(self, panel: str, filename: str, max_workers: int = 8):
        self.panel = panel
        self._arrs = _NpzReader(filename)
        self._shape = tuple(self._arrs['shape'])
        self._dtype = self._arrs['dtype'].tobytes().decode()
        self._n_frames = int(self._arrs['nframes'])
        self._max_workers = max_workers
        self.omega = np.radians(self._arrs['omega'])   # (n_frames, 2): [start, stop]
        self._images: list[csr_array] | None = None
        self._lock = threading.Lock()

    @property
    def images(self) -> list[csr_array]:
        with self._lock:
            if self._images is None:
                arrs, shape, dtype = self._arrs, self._shape, self._dtype

                def load_frames(lo: int, hi: int) -> list[csr_array]:
                    return [_frame_csr(arrs[f'{i}_data'], arrs[f'{i}_row'],
                                       arrs[f'{i}_col'], shape, dtype)
                            for i in range(lo, hi)]

                n_workers = min(self._max_workers, 8, max(self._n_frames, 1))
                bounds = np.linspace(0, self._n_frames, n_workers + 1).astype(int)
                with ThreadPoolExecutor(n_workers) as ex:
                    parts = ex.map(load_frames, bounds[:-1], bounds[1:])
                self._images = [im for part in parts for im in part]
        return self._images

    def __len__(self):
        return self._n_frames


def _merge_config(base: dict, overlay: dict) -> dict:
    """Recursively overlay one config document on another (hexrd multi-doc yml)."""
    merged = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], val)
        else:
            merged[key] = val
    return merged


class Experiment:
    """All inputs for one find-orientations analysis, loaded from a config file.

    The config may hold several yaml documents: a base followed by study
    overlays, each a sparse dict merged over the base (as in hexrd). ``study``
    selects one by index; the default runs the base document alone.
    """

    def __init__(self, filename: str, study: int | None = None):
        self.experiment_dir = os.path.dirname(filename)
        with open(filename) as f:
            config, *studies = yaml.safe_load_all(f)
        self.studies = list(studies)
        if study is not None:
            config = _merge_config(config, self.studies[study])
        self.config = config

        instrument = yaml.safe_load(open(self._path(config['instrument'])))
        self.detectors = [Detector.from_dict(name, d)
                          for name, d in instrument['detectors'].items()]
        self.beam_energy = instrument.get('beam', {}).get('energy')
        self.oscillation_stage = OscillationStage.from_dict(
            instrument.get('oscillation_stage', {}))

        self.active_material = ActiveMaterial.from_dict(config['material'])
        self.find_orientations = FindOrientations.from_dict(config['find_orientations'])
        self.max_workers = _parse_multiprocessing(config.get('multiprocessing', -1))
        self.image_series_list = [
            ImageSeries(d['panel'], self._path(d['file']), self.max_workers)
            for d in config['image_series']['data']
        ]

        self.analysis_name = config['analysis_name']
        self.analysis_dir = self._path(self.analysis_name)
        self._materials_file = self._path(config['material']['definitions'])

    def _path(self, name: str) -> str:
        return os.path.join(self.experiment_dir, name)

    @property
    def eta_ome_maps_file(self) -> str:
        """Where the eta-omega maps cache lives (orientation_maps: file, or default)."""
        configured = self.find_orientations.orientation_maps.file
        if configured:
            return configured if os.path.isabs(configured) else self._path(configured)
        actmat = self.active_material.active.strip().replace(' ', '-')
        return os.path.join(self.analysis_dir, f'eta-ome-maps-{actmat}.npz')

    @property
    def quaternion_grid_file(self) -> str | None:
        """Absolute path of the trial-quaternion grid, if configured."""
        grid = self.find_orientations.use_quaternion_grid
        if grid is None:
            return None
        return grid if os.path.isabs(grid) else self._path(grid)

    def get_materials(self) -> list[Material]:
        """Load every crystal material defined in the materials HDF5 file."""
        with h5py.File(self._materials_file, 'r') as f:
            return [self._material(name, f) for name in f]

    def get_active_material(self) -> Material:
        """Load the material selected for this analysis (config ``material: active``)."""
        name = self.active_material.active
        with h5py.File(self._materials_file, 'r') as f:
            return self._material(name, f)

    def _material(self, name: str, definitions: h5py.File) -> Material:
        return Material(name, definitions,
                        dmin=self.active_material.dmin,
                        sfacmin=self.active_material.min_sfac_ratio,
                        beam_energy=self.beam_energy)

    @property
    def analysis_id(self) -> str:
        return f'{self.analysis_name}_{self.active_material.active}'
