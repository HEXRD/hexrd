"""Create azimuthally segmented powder profiles from composite detector data."""

from __future__ import annotations

import glob
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np
import yaml

from hexrd.core import imageseries
from hexrd.powder.cake import CakeMatrix, export_cake_matrix
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.instrument.hedm_instrument import pixel_resolution


logger = logging.getLogger(__name__)
GEOMETRY_DEFAULTS = dict(
    tth_start=2.0, tth_end=5.0, eta_start=-180.0, eta_end=180.0, refinement=1.0
)


def _interval_count(span: float, width: float, name: str) -> int:
    count = abs(span) / width
    if (
        not np.isfinite(count)
        or count < 1
        or not np.isclose(count, round(count), rtol=1e-10, atol=1e-10)
    ):
        raise ValueError(f"{name} range must be divisible by its interval width")
    return int(round(count))


@dataclass
class CakeDataConfig:
    """Configuration for :func:`cake_data`. Angles are in degrees."""

    data_dir: str | None = None
    output_dir: str | None = None
    sample_name: str | None = None
    instrument: str | None = None
    matrix: str | None = None
    par_file: str | None = None
    output_par: str | None = None
    tth_start: float | None = None
    tth_end: float | None = None
    cake_width: float = 5.0
    eta_start: float | None = None
    eta_end: float | None = None
    ome_start: float = 0.0
    ome_end: float = 360.0
    dome: float = 5.0
    skip_frames: int = 0
    refinement: float | None = None
    start_layer: int = 0
    par_version: int = 5
    max_input_frames: int | None = None
    image_pattern: str = "EIG16M_CdTe_*.h5"
    image_format: str = "eiger-stream-v1"
    image_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> CakeDataConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("cake-data configuration must be a JSON object")
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown cake-data parameters: {sorted(unknown)}")
        return cls(**values)

    @property
    def geometry(self) -> dict[str, float]:
        return {
            name: default if getattr(self, name) is None else getattr(self, name)
            for name, default in GEOMETRY_DEFAULTS.items()
        }

    def validate(self) -> None:
        required = ("data_dir", "output_dir", "sample_name", "par_file")
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"missing required cake-data parameters: {missing}")
        if bool(self.instrument) == bool(self.matrix):
            raise ValueError("provide exactly one of instrument or matrix")
        overrides = [
            name for name in GEOMETRY_DEFAULTS if getattr(self, name) is not None
        ]
        if self.matrix and overrides:
            raise ValueError(
                f"matrix fixes the geometry; remove these parameters: {overrides}"
            )
        for name in (*GEOMETRY_DEFAULTS, 'cake_width', 'ome_start', 'ome_end', 'dome'):
            value = getattr(self, name)
            if value is None and name in GEOMETRY_DEFAULTS:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        for name in ('skip_frames', 'start_layer', 'par_version', 'max_input_frames'):
            value = getattr(self, name)
            if value is None and name == 'max_input_frames':
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cake_width <= 0 or self.dome <= 0 or self.geometry['refinement'] <= 0:
            raise ValueError("cake_width, dome, and refinement must be positive")
        if self.max_input_frames == 0:
            raise ValueError("max_input_frames must be positive")
        _interval_count(self.ome_end - self.ome_start, self.dome, "omega")
        if not self.matrix:
            geometry = self.geometry
            if geometry['tth_start'] >= geometry['tth_end']:
                raise ValueError("tth_start must be less than tth_end")
            if geometry['eta_start'] >= geometry['eta_end']:
                raise ValueError("eta_start must be less than eta_end")
            _interval_count(
                geometry['eta_end'] - geometry['eta_start'], self.cake_width, "eta"
            )
        if self.par_version != 5:
            raise ValueError("cake-data currently supports par version 5 only")
        if not isinstance(self.image_options, dict):
            raise ValueError("image_options must be a JSON object")
        if 'format' in self.image_options:
            raise ValueError("set image_format separately from image_options")
        if self.image_format not in imageseries.load.Registry.adapter_registry:
            raise ValueError(f'unknown image_format: {self.image_format}')
        if self.image_format == 'hdf5' and 'path' not in self.image_options:
            raise ValueError("hdf5 requires image_options with the dataset group path")
        for name in ('sample_name', 'output_par'):
            value = getattr(self, name)
            if value and (Path(value).name != value or value in ('.', '..')):
                raise ValueError(f"{name} must be a filename, without a directory")

    def update(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if value is not None:
                setattr(self, name, value)


def load_instrument(path: str | Path) -> HEDMInstrument:
    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path) as stream:
            return HEDMInstrument(stream)
    with path.open(encoding="utf-8") as stream:
        return HEDMInstrument(yaml.safe_load(stream))


def load_par_file(path: str | Path, version: int = 5) -> dict[str, np.ndarray]:
    if version != 5:
        raise ValueError("cake-data currently supports par version 5 only")
    data = np.loadtxt(path, ndmin=2)
    if not data.size or data.shape[1] <= 24:
        raise ValueError("version-5 par input must contain at least 25 columns")
    numbers = data[:, [3, 6, 24]]
    if (
        not np.isfinite(data[:, [3, 6, 7, 8, 9, 24]]).all()
        or np.any(numbers != np.floor(numbers))
        or np.any(numbers < 0)
        or np.any(data[:, 24] == 0)
    ):
        raise ValueError(
            "par metadata needs finite positions and non-negative integer scan/image numbers and positive frame counts"
        )
    # Read optional fly-axis columns from the same table; numeric JSON keys
    # identify columns regardless of the order in which they were written.
    flips = np.zeros(len(data), dtype=bool)
    json_path = Path(path).with_suffix(".json")
    if json_path.exists():
        with json_path.open(encoding="utf-8") as stream:
            names = json.load(stream)
        if not isinstance(names, dict):
            raise ValueError("par JSON must map column indices to names")
        columns = {name: int(index) for index, name in names.items()}
        if {"fly_axis0_rising", "fly_axis0_falling"} <= columns.keys():
            indices = [columns["fly_axis0_rising"], columns["fly_axis0_falling"]]
            if min(indices) < 0 or max(indices) >= data.shape[1]:
                raise ValueError("par JSON fly-axis columns do not match the par data")
            rising, falling = data[:, indices].T
            if not np.isfinite([rising, falling]).all():
                raise ValueError("par fly-axis values must be finite")
            flips = rising > falling
        else:
            logger.warning("par JSON has no fly-axis columns; omega order is unchanged")
    return {
        "scan_number": data[:, 3].astype(int),
        "image_number": data[:, 6].astype(int),
        "position_key": np.vstack((np.arange(len(data)), data[:, 7:10].T)),
        "num_frames": data[:, 24].astype(int),
        "flip": flips,
    }


def _load_matrix(config: CakeDataConfig, temporary: Path) -> CakeMatrix:
    path = Path(config.matrix) if config.matrix else temporary / 'geometry.hexrdcake'
    if not config.matrix:
        instrument = load_instrument(config.instrument)
        geometry = config.geometry
        tth_stats, eta_stats = pixel_resolution(instrument)
        export_cake_matrix(
            path,
            instrument,
            tth_range=(geometry['tth_start'], geometry['tth_end']),
            eta_range=(geometry['eta_start'], geometry['eta_end']),
            pixel_size=(
                geometry['refinement'] * np.degrees(tth_stats[1]),
                np.degrees(eta_stats[1]),
            ),
        )
    operator = CakeMatrix.load(path)
    if operator.input_layout != 'composite':
        raise ValueError(
            "cake-data needs a composite-layout matrix for its single input image series"
        )
    return operator


def _raw_path(config: CakeDataConfig, image_number: int) -> Path:
    template = config.data_dir
    try:
        directory = template.format(
            sample_name=config.sample_name, image_number=int(image_number)
        )
    except (IndexError, KeyError) as exc:
        raise ValueError(f"unknown data_dir template field: {exc}") from exc
    if "%" in directory:
        try:
            directory = directory % (config.sample_name, int(image_number))
        except TypeError:
            directory = directory % int(image_number)
    matches = sorted(glob.glob(str(Path(directory) / config.image_pattern)))
    if not matches:
        raise FileNotFoundError(f"no {config.image_pattern!r} file under {directory}")
    if len(matches) > 1:
        raise ValueError(f"multiple raw image files match under {directory}: {matches}")
    return Path(matches[0])


def _read_integrated_frames(
    config: CakeDataConfig,
    operator: CakeMatrix,
    raw_path: Path,
    frame_count: int,
    flip: bool = False,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield one full-layout omega average at a time; preserve acquisition order."""
    series = imageseries.open(
        raw_path, format=config.image_format, **config.image_options
    )
    available = max(0, len(series) - config.skip_frames)
    count = min(frame_count, config.max_input_frames or frame_count)
    if count > available:
        raise ValueError(
            f"par file requests {count} frame(s) after skipping {config.skip_frames}, but {raw_path} has only {available}"
        )
    bins = _interval_count(config.ome_end - config.ome_start, config.dome, 'omega')
    if frame_count < bins or frame_count % bins:
        raise ValueError("input frame count is not divisible into omega intervals")
    frames_per_output = frame_count // bins
    if count % frames_per_output:
        raise ValueError("max_input_frames must end on a complete omega interval")
    starts = range(0, count, frames_per_output)
    if flip:
        starts = reversed(starts)
    step = np.copysign(config.dome, config.ome_end - config.ome_start)
    for start in starts:
        integrated = np.zeros(operator.composite_shape, dtype=np.float64)
        for index in range(start, start + frames_per_output):
            image = np.array(
                series[config.skip_frames + index], dtype=np.float64, copy=True
            )
            if image.shape != operator.composite_shape:
                raise ValueError(
                    f"input image shape {image.shape} must match the instrument shape {operator.composite_shape}"
                )
            image[image >= np.iinfo(np.uint32).max] = np.nan
            image /= frames_per_output
            integrated += image
        np.floor(integrated, out=integrated)
        bin_index = start // frames_per_output
        if flip:
            bin_index = bins - 1 - bin_index
        omega = config.ome_start + step * (bin_index + 0.5)
        yield omega, integrated


def _wedge_edges(config: CakeDataConfig, operator: CakeMatrix) -> np.ndarray:
    start, step, count = operator.eta
    stop = start + step * count
    # The saved grid's rounded endpoint can extend a fraction of a pixel beyond
    # the requested range. Do not create an empty extra wedge for that sliver.
    quotient = (stop - start) / config.cake_width
    nearest = round(quotient)
    if nearest and abs(stop - start - nearest * config.cake_width) <= step / 2 + 1e-10:
        nwedges = nearest
    else:
        nwedges = int(np.ceil(quotient))
    edges = start + np.arange(nwedges + 1) * config.cake_width
    edges[-1] = stop
    return edges


def _mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """NaN-aware mean without copying a float array or warning for empty bins."""
    total = np.sum(values, axis=0, where=valid)
    count = np.sum(valid, axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count != 0)


def _profiles(
    cake: np.ma.MaskedArray, bounds: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = cake.data
    valid = ~np.ma.getmaskarray(cake) & ~np.isnan(values)
    profiles = np.array(
        [
            _mean(values[first:last], valid[first:last])
            for first, last in zip(bounds[:-1], bounds[1:])
        ]
    )
    # Give each valid wedge equal weight, as in Chris's newer Eiger script.
    full = _mean(profiles, ~np.isnan(profiles))
    return full, profiles


def cake_data(config: CakeDataConfig) -> tuple[Path, Path]:
    """Run the CakeData workflow and return its HDF5 and par output paths."""
    config.validate()
    metadata = load_par_file(config.par_file, config.par_version)
    layers = range(config.start_layer, len(metadata['scan_number']))
    if not layers:
        raise ValueError("start_layer must select at least one par-file layer")
    raw_paths = {
        layer: _raw_path(config, metadata['image_number'][layer]) for layer in layers
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{config.sample_name}.h5"
    par_path = output_dir / (config.output_par or f"{config.sample_name}.par")
    inputs = {
        Path(config.par_file).resolve(),
        Path(config.par_file).with_suffix(".json").resolve(),
        Path(config.matrix or config.instrument).resolve(),
        *(path.resolve() for path in raw_paths.values()),
    }
    if h5_path.resolve() == par_path.resolve() or inputs & {
        h5_path.resolve(),
        par_path.resolve(),
    }:
        raise ValueError("output paths must differ from each other and all input files")
    for path in (h5_path, par_path):
        if path.is_dir():
            raise ValueError(f'output path is a directory: {path}')
    if config.start_layer and h5_path.exists():
        raise ValueError(
            'start_layer selects a new output; resuming an existing output is not supported'
        )
    outputs = h5_path, par_path
    # Keep previous outputs intact until processing and staging finish.
    with TemporaryDirectory(prefix='.cake-data-', dir=output_dir) as directory:
        temporary = Path(directory)
        operator = _load_matrix(config, temporary)
        edges = _wedge_edges(config, operator)
        bounds = np.searchsorted(operator.eta_centers, edges)
        eta = (edges[:-1] + edges[1:]) / 2
        eta_key = np.vstack((np.arange(len(eta)), eta))
        tth = operator.tth_centers
        h5_path, par_path = temporary / h5_path.name, temporary / par_path.name
        par_rows = []
        with h5py.File(h5_path, 'w') as stream:
            stream['XYZ_key'] = metadata['position_key']
            for layer, xyz in enumerate(metadata['position_key'][1:].T):
                stream[f'Cake/{layer}/xyz'] = xyz
            for layer in layers:
                group = stream[f'Cake/{layer}']
                full_profiles, omega = [], []
                frames = _read_integrated_frames(
                    config,
                    operator,
                    raw_paths[layer],
                    metadata["num_frames"][layer],
                    metadata["flip"][layer],
                )
                for index, (angle, frame) in enumerate(frames):
                    full, wedges = _profiles(operator.apply_composite(frame), bounds)
                    full_profiles.append(full)
                    group[f'Ome/{index}/IvTTH'] = wedges
                    group[f'Ome/{index}/eta_key'] = eta_key
                    omega.append(angle)
                by_omega = np.asarray(full_profiles)
                group['tth'] = tth
                group['IvTTH_Full'] = np.mean(by_omega, axis=0)
                group['IvTTHvW'] = by_omega
                group['ome_key'] = np.vstack((np.arange(len(omega)), omega))
                group['nFrames'] = len(omega)
                scan = metadata["scan_number"][layer]
                image = metadata["image_number"][layer]
                x, y, z = metadata["position_key"][1:, layer]
                par_rows.append(
                    [10, 100, 1000, scan, image, image, x, x, y, z, 20, 0.6]
                )
                logger.info("processed cake-data layer %d", layer)
        par_format = ["%d"] * 6 + ["%.6f"] * 6
        np.savetxt(par_path, np.asarray(par_rows), fmt=par_format, newline=" \n")
        for source, target in zip((h5_path, par_path), outputs):
            source.replace(target)
    logger.info("wrote CakeData outputs %s and %s", *outputs)
    return outputs


def config_as_json(config: CakeDataConfig | None = None) -> str:
    return json.dumps(asdict(config or CakeDataConfig()), indent=2)
