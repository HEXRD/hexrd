"""Typed experiment inputs, loaded from a config file.

:class:`Experiment` holds what every analysis workflow starts from: the
instrument (detector panels, oscillation stage, beam), the image series,
the material selection, and the analysis naming.  Workflow packages extend
it with their own analysis parameters (e.g.
:class:`hexrd.hedm.experiment.HedmExperiment`).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import yaml

_YamlLoader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)

if TYPE_CHECKING:
    from scipy.sparse import csr_array

from hexrd.core.material.material_data import Material
from hexrd.core.extensions import transforms
from hexrd.core.utils.npz import _NpzReader


@dataclass
class ActiveMaterial:
    definitions: str
    active: str
    dmin: float                # angstrom
    two_theta_width: float     # degrees
    min_sfac_ratio: float
    reset_exclusions: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('definitions'), d.get('active'), d.get('dmin'),
                   d.get('tth_width'), d.get('min_sfac_ratio'),
                   d.get('reset_exclusions', True))


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


@dataclass(frozen=True)
class Beam:
    energy: float
    vector: np.ndarray
    source_distance: float
    energy_correction: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'Beam':
        if not d:
            d = {}
        if 'energy' not in d:
            d = next(iter(d.values()), {})
        angles = d.get('vector', {})
        azimuth = np.radians(angles.get('azimuth', 90.0))
        polar = np.radians(angles.get('polar_angle', 90.0))
        vector = -np.array([np.sin(polar) * np.cos(azimuth),
                            np.cos(polar),
                            np.sin(polar) * np.sin(azimuth)])
        return cls(float(d.get('energy', 0.0)), vector,
                   float(d.get('source_distance', np.inf)),
                   d.get('energy_correction'))


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
    from scipy.sparse import csr_array

    flat = row.astype(np.int64) * shape[1] + col
    if flat.size == 0 or np.all(np.diff(flat) > 0):
        indptr = np.searchsorted(row, np.arange(shape[0] + 1))
        return csr_array((data, col, indptr), shape=shape, dtype=dtype)
    return csr_array((data, (row, col)), shape=shape, dtype=dtype)


def _pixel_chunks(frames, row_of_frame, max_pixels):
    """Pack streamed (pixel IDs, weights) into bounded float64 frame batches."""
    ids, values, counts = [], [], []
    first, total = 0, 0

    def pack(stop):
        return (np.concatenate(ids), np.concatenate(values).astype(np.float64, copy=False),
                np.r_[0, np.cumsum(counts)], np.asarray(row_of_frame[first:stop]))

    stop = 0
    for stop, (pixels, weights) in enumerate(frames, 1):
        if counts and total + len(pixels) > max_pixels:
            if total:
                yield pack(stop - 1)
            ids, values, counts = [], [], []
            first, total = stop - 1, 0
        ids.append(pixels)
        values.append(weights)
        counts.append(len(pixels))
        total += len(pixels)
    if total:
        yield pack(stop)


class ImageSeries:
    """A frame-cache (.npz) of sparse detector frames for one panel, with omegas.

    Construction only indexes the file; the frames themselves are decompressed
    on first access to ``images``, so runs that never look at the frames (e.g.
    with cached eta-omega maps) never pay for them.
    """

    def __init__(self, panel: str, filename: str, max_workers: int = 8):
        self.panel = panel
        self._arrs = _NpzReader(filename, metadata_only=True)
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

    def pixel_chunks(self, row_of_frame, max_pixels=8_000_000):
        """Stream COO pixels into bounded chunks, without materializing CSR frames.

        Decompress only a bounded batch at a time. Keeping the original
        float64 weights also prevents overflow when duplicate uint16 pixels
        contribute to the same map bin.
        """
        def load_frame(i):
            data = self._arrs[f'{i}_data']
            row, col = self._arrs[f'{i}_row'], self._arrs[f'{i}_col']
            if (np.any(row < 0) or np.any(row >= self._shape[0])
                    or np.any(col < 0) or np.any(col >= self._shape[1])):
                raise ValueError('frame-cache pixel index is out of bounds')
            ids = row.astype(np.int64) * self._shape[1] + col
            return ids, data.astype(self._dtype, copy=False)

        workers = max(1, min(self._max_workers, 8))
        if workers == 1:
            yield from _pixel_chunks(map(load_frame, range(len(self))), row_of_frame, max_pixels)
            return
        with ThreadPoolExecutor(workers) as pool:
            frames = (frame for batch in range(0, len(self), workers)
                      for frame in pool.map(load_frame,
                          range(batch, min(batch + workers, len(self)))))
            yield from _pixel_chunks(frames, row_of_frame, max_pixels)

    def __len__(self):
        return self._n_frames

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.images[key[0]][key[1:]].toarray()
        return self.images[key]

    @property
    def metadata(self):
        return {'omega': np.degrees(self.omega)}


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

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.images[key[0]][key[1:]].toarray()
        return self.images[key]

    @property
    def metadata(self):
        return {'omega': np.degrees(self.omega)}


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
    import h5py

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
    with open(path, 'rb') as f:
        # HDF5 signatures can follow a user block at powers of two >= 512.
        offset, size = 0, os.fstat(f.fileno()).st_size
        while offset + 8 <= size:
            f.seek(offset)
            if f.read(8) == b'\x89HDF\r\n\x1a\n':
                import h5py

                with h5py.File(path, 'r') as archive:
                    return _h5_to_dict(archive['instrument'])
            offset = 2 * offset if offset else 512
        f.seek(0)
        return yaml.load(f, Loader=_YamlLoader)


class Experiment:
    """The inputs every analysis workflow shares, loaded from a config file:
    instrument, image series, material selection, analysis naming.

    The config may hold several yaml documents: a base followed by study
    overlays, each a sparse dict merged over the base. ``study``
    selects one by index; the default runs the base document alone.
    """

    def __init__(self, filename: str, study: int | None = None):
        self.experiment_dir = os.path.dirname(filename)
        with open(filename) as f:
            config, *studies = yaml.load_all(f, Loader=_YamlLoader)
        self.studies = list(studies)
        if study is not None:
            for overlay in self.studies[:study + 1]:
                config = _merge_config(config, overlay)
        self.config = config

        instrument = _load_instrument(self._path(config['instrument']))
        self._instrument_config = instrument
        self.detectors = {name: Detector.from_dict(name, d)
                          for name, d in instrument['detectors'].items()}
        self.beam = Beam.from_dict(instrument.get('beam', {}))
        self.oscillation_stage = OscillationStage.from_dict(
            instrument.get('oscillation_stage', {}))

        self.active_material = ActiveMaterial.from_dict(config['material'])
        self.max_workers = _parse_multiprocessing(config.get('multiprocessing', -1))
        self.image_series_list = []
        for d in config['image_series']['data']:
            panel, parent = self._load_image_series(d)
            if isinstance(panel, str):
                self.image_series_list.append(parent)
                continue
            # a list of panels: shared frames split by each detector's ROI
            for name in panel:
                det = self.detectors[name]
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

    def _load_image_series(self, spec):
        panel = spec['panel']
        series = ImageSeries(panel if isinstance(panel, str) else None,
                             self._path(spec['file']), self.max_workers)
        return panel, series

    @property
    def instrument(self):
        if not hasattr(self, '_instrument'):
            from hexrd.core.instrument import HEDMInstrument
            self._instrument = HEDMInstrument(
                instrument_config=self._instrument_config,
                max_workers=self.max_workers,
            )
        return self._instrument

    @property
    def image_series(self):
        if not hasattr(self, '_image_series'):
            from hexrd.core.imageseries.omega import OmegaImageSeries
            self._image_series = {
                series.panel: OmegaImageSeries(series)
                for series in self.image_series_list
            }
        return self._image_series

    def _path(self, name: str) -> str:
        return os.path.join(self.experiment_dir, name)

    def get_materials(self) -> list[Material]:
        """Load every crystal material defined in the materials HDF5 file."""
        import h5py

        with h5py.File(self._materials_file, 'r') as f:
            return [self._material(name, f) for name in f]

    def get_active_material(self) -> Material:
        """Load the material selected for this analysis (config ``material: active``)."""
        import h5py

        name = self.active_material.active
        with h5py.File(self._materials_file, 'r') as f:
            return self._material(name, f)

    def _material(self, name: str, definitions: h5py.File) -> Material:
        return Material(name, definitions,
                        dmin=self.active_material.dmin,
                        sfacmin=self.active_material.min_sfac_ratio,
                        beam_energy=self.beam.energy,
                        reset_exclusions=self.active_material.reset_exclusions)

    @property
    def beam_energy(self) -> float:
        return self.beam.energy

    @property
    def analysis_id(self) -> str:
        return f'{self.analysis_name}_{self.active_material.active}'
