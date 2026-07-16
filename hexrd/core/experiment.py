"""Typed experiment inputs, loaded from a config file.

:class:`Experiment` holds what every analysis workflow starts from: the
instrument (detector panels, oscillation stage, beam), the image series,
the material selection, and the analysis naming.  Workflow packages extend
it with their own analysis parameters (e.g.
:class:`hexrd.hedm.experiment.HedmExperiment`).
"""
import io
import os
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py
import yaml
from scipy.sparse import csr_array

from hexrd.core.material.material_data import Material
from hexrd.core.extensions import transforms


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
    roi: Optional[list]            # [row, col] offset into the parent frames

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('columns', 0), d.get('rows', 0),
                   d.get('size', [0.0, 0.0]), d.get('roi'))


@dataclass
class Detector:
    name: str
    distortion: Distortion
    pixels: Pixels
    transform: Transform
    buffer: Optional[np.ndarray]   # (2,) edge buffer in mm
    group: Optional[str]           # parent panel for ROI instruments

    @classmethod
    def from_dict(cls, name, d):
        buffer = d.get('buffer')
        return cls(name, Distortion.from_dict(d.get('distortion', {})),
                   Pixels.from_dict(d.get('pixels', {})),
                   Transform.from_dict(d.get('transform', {})),
                   None if buffer is None else np.asarray(buffer, dtype=np.float64),
                   d.get('group'))

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


class RoiImageSeries:
    """One sub-panel's view of a whole-panel :class:`ImageSeries`.

    ROI instruments describe a physical panel as several detector entries,
    each a rectangular region of the shared frames; the parent series is
    decompressed once and every view slices it.
    """

    def __init__(self, panel: str, parent: ImageSeries,
                 roi: list, shape: tuple[int, int]):
        self.panel = panel
        self._parent = parent
        self._roi = roi
        self._shape = shape
        self.omega = parent.omega
        self._images: list[csr_array] | None = None

    @property
    def images(self) -> list[csr_array]:
        if self._images is None:
            r0, c0 = self._roi
            rows, cols = self._shape
            self._images = [image[r0:r0 + rows, c0:c0 + cols]
                            for image in self._parent.images]
        return self._images

    def __len__(self):
        return len(self._parent)


def _merge_config(base: dict, overlay: dict) -> dict:
    """Recursively overlay one config document on another (hexrd multi-doc yml)."""
    merged = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], val)
        else:
            merged[key] = val
    return merged


def _h5_to_dict(group) -> dict:
    """An HDF5 group tree as nested plain dicts, mirroring the yaml layout."""
    out = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _h5_to_dict(value)
        else:
            item = value[()]
            if isinstance(item, bytes):
                item = item.decode()
            elif isinstance(item, np.ndarray):
                # a size-1 dataset is a scalar stored as an array
                item = item.item() if item.size == 1 else item.tolist()
            elif isinstance(item, np.generic):
                item = item.item()
            out[key] = item
    return out


def _load_instrument(path: str) -> dict:
    """An instrument definition from a yaml file or a .hexrd HDF5 archive."""
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            return _h5_to_dict(f['instrument'])
    with open(path) as f:
        return yaml.safe_load(f)


class Experiment:
    """The inputs every analysis workflow shares, loaded from a config file:
    instrument, image series, material selection, analysis naming.

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

        instrument = _load_instrument(self._path(config['instrument']))
        self.detectors = [Detector.from_dict(name, d)
                          for name, d in instrument['detectors'].items()]
        self.beam_energy = instrument.get('beam', {}).get('energy')
        self.oscillation_stage = OscillationStage.from_dict(
            instrument.get('oscillation_stage', {}))

        self.active_material = ActiveMaterial.from_dict(config['material'])
        self.max_workers = _parse_multiprocessing(config.get('multiprocessing', -1))
        detector_of = {d.name: d for d in self.detectors}
        self.image_series_list = []
        for d in config['image_series']['data']:
            panel = d['panel']
            if isinstance(panel, str):
                self.image_series_list.append(
                    ImageSeries(panel, self._path(d['file']), self.max_workers))
                continue
            # a list of panels: shared frames split by each detector's ROI
            parent = ImageSeries(None, self._path(d['file']), self.max_workers)
            for name in panel:
                det = detector_of[name]
                if det.pixels.roi is None:
                    raise ValueError(
                        f'image series panel list needs "pixels: roi" on '
                        f'detector {name}')
                self.image_series_list.append(RoiImageSeries(
                    name, parent, det.pixels.roi,
                    (det.pixels.rows, det.pixels.columns)))

        self.analysis_name = config['analysis_name']
        self.analysis_dir = self._path(self.analysis_name)
        self._materials_file = self._path(config['material']['definitions'])

    def _path(self, name: str) -> str:
        return os.path.join(self.experiment_dir, name)

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
