"""Create azimuthally segmented powder profiles from composite detector data."""

from __future__ import annotations

import glob
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, get_args, get_origin, get_type_hints

import h5py
import numpy as np
import yaml

from hexrd.core import imageseries
from hexrd.core.instrument import load_instrument
from hexrd.core.instrument.hedm_instrument import pixel_resolution
from hexrd.powder.cake import CakeMatrix, export_cake_matrix


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


def _option(default: Any, short: str, help: str) -> Any:
    """Attach command-line help to a configuration field."""
    return field(default=default, metadata={"short": short, "help": help})


@dataclass
class CakeDataConfig:
    """Configuration for :func:`cake_data`. Angles are in degrees."""

    data_dir: str | None = _option(None, 'd', 'raw-data directory template')
    output_dir: str | Path | None = _option(None, 'o', 'directory for the output files')
    sample_name: str | None = _option(None, 's', 'sample name and output file stem')
    instrument: str | Path | None = _option(
        None, 'i', 'HEXRD instrument YAML or HDF5 file'
    )
    matrix: str | Path | None = _option(
        None, 'M', 'saved .hexrdcake matrix (replaces instrument and geometry options)'
    )
    par_file: str | Path | None = _option(None, 'p', 'input CHESS par file')
    output_par: str | None = _option(None, 'P', 'output par filename')
    tth_start: float | None = _option(None, 'b', 'starting two-theta angle')
    tth_end: float | None = _option(None, 't', 'ending two-theta angle')
    cake_width: float = _option(5.0, 'c', 'eta wedge width')
    eta_start: float | None = _option(None, 'a', 'starting eta angle')
    eta_end: float | None = _option(None, 'z', 'ending eta angle')
    ome_start: float = _option(0.0, 'w', 'starting omega angle')
    ome_end: float = _option(360.0, 'e', 'ending omega angle')
    dome: float = _option(5.0, 'D', 'omega integration width')
    skip_frames: int = _option(0, 'f', 'number of initial raw frames to skip')
    refinement: float | None = _option(None, 'r', 'two-theta pixel refinement factor')
    start_layer: int = _option(0, 'l', 'first par-file layer to process')
    par_version: int = _option(5, 'v', 'CHESS par-file version')
    max_input_frames: int | None = _option(None, 'm', 'maximum raw frames per layer')
    image_pattern: str = _option(
        'EIG16M_CdTe_*.h5', 'g', 'glob used to locate an image file'
    )
    image_format: str = _option('eiger-stream-v1', 'F', 'HEXRD image-series format')
    image_options: dict[str, Any] = field(
        default_factory=dict,
        metadata={"short": 'O', "help": 'image-series reader options as a JSON object'},
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CakeDataConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("cake-data configuration must be a YAML mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown cake-data parameters: {sorted(unknown, key=str)}"
            )
        return cls(**values)

    @property
    def geometry(self) -> dict[str, float]:
        return {
            name: default if getattr(self, name) is None else getattr(self, name)
            for name, default in GEOMETRY_DEFAULTS.items()
        }

    def validate(self) -> None:
        for name, hint in get_type_hints(CakeDataConfig).items():
            types = get_args(hint) if type(None) in get_args(hint) else (hint,)
            types = tuple(get_origin(t) or t for t in types)
            if float in types:
                types += (int,)
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, types):
                expected = (
                    'a mapping'
                    if dict in types
                    else ' or '.join(t.__name__ for t in types)
                )
                raise ValueError(f'{name} must be {expected}')
        required = ("data_dir", "output_dir", "sample_name", "par_file")
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"missing required cake-data parameters: {missing}")
        if bool(self.instrument) == bool(self.matrix):
            raise ValueError("provide exactly one of instrument or matrix")
        overrides = [n for n in GEOMETRY_DEFAULTS if getattr(self, n) is not None]
        if self.matrix and overrides:
            raise ValueError(
                f"matrix fixes the geometry; remove these parameters: {overrides}"
            )
        geometry = self.geometry
        scales = (self.cake_width, self.dome, geometry['refinement'])
        if not np.isfinite(scales).all() or min(scales) <= 0:
            raise ValueError(
                "cake_width, dome, and refinement must be finite and positive"
            )
        for name in ('skip_frames', 'start_layer', 'max_input_frames'):
            value = getattr(self, name)
            minimum = 1 if name == 'max_input_frames' else 0
            if value is not None and value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        _interval_count(self.ome_end - self.ome_start, self.dome, "omega")
        if not self.matrix:
            if geometry['tth_start'] >= geometry['tth_end']:
                raise ValueError("tth_start must be less than tth_end")
            eta_span = geometry['eta_end'] - geometry['eta_start']
            if eta_span <= 0:
                raise ValueError("eta_start must be less than eta_end")
            _interval_count(eta_span, self.cake_width, "eta")
        if self.par_version != 5:
            raise ValueError("cake-data currently supports par version 5 only")
        if not all(isinstance(key, str) for key in self.image_options):
            raise ValueError('image_options keys must be strings')
        if 'format' in self.image_options:
            raise ValueError("set image_format separately from image_options")
        if self.image_format not in imageseries.load.Registry.adapter_registry:
            raise ValueError(f'unknown image_format: {self.image_format}')
        for name in ('sample_name', 'output_par'):
            value = getattr(self, name)
            if value and (Path(value).name != value or value in ('.', '..')):
                raise ValueError(f"{name} must be a filename, without a directory")


def load_par_file(path: str | Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 25:
        raise ValueError("version-5 par input must contain at least 25 columns")
    counts = data[:, [3, 6, 24]]
    if (
        not np.isfinite(data[:, [3, 6, 7, 8, 9, 24]]).all()
        or np.any(counts != np.floor(counts))
        or np.any(counts < 0)
        or np.any(data[:, 24] == 0)
    ):
        raise ValueError(
            'par scan/image numbers must be integers and frame counts positive'
        )
    # Optional fly-axis columns are named by numeric keys in a JSON sidecar.
    flips = np.zeros(len(data), dtype=bool)
    json_path = Path(path).with_suffix(".json")
    if json_path.exists():
        names = json.loads(json_path.read_text(encoding="utf-8"))
        columns = {name: int(index) for index, name in names.items()}
        try:
            fly_columns = [
                columns[name] for name in ('fly_axis0_rising', 'fly_axis0_falling')
            ]
        except KeyError:
            logger.warning("par JSON has no fly-axis columns; omega order is unchanged")
        else:
            if any(index < 0 or index >= data.shape[1] for index in fly_columns):
                raise ValueError(
                    'fly-axis columns must refer to columns in the par file'
                )
            fly_axis = data[:, fly_columns]
            if not np.isfinite(fly_axis).all():
                raise ValueError('fly-axis values must be finite')
            flips = fly_axis[:, 0] > fly_axis[:, 1]
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
    operator = CakeMatrix(path)
    if operator.input_layout != 'composite':
        raise ValueError("cake-data needs a composite-layout matrix")
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
    """Yield floored omega averages, reversing bins when par metadata requires it."""
    series = imageseries.open(
        raw_path, format=config.image_format, **config.image_options
    )
    count = min(frame_count, config.max_input_frames or frame_count)
    if count > len(series) - config.skip_frames:
        raise ValueError(f"{raw_path} has fewer than the {count} requested frame(s)")
    bins = _interval_count(config.ome_end - config.ome_start, config.dome, 'omega')
    frames_per_output = frame_count // bins
    if frames_per_output < 1 or frame_count % bins or count % frames_per_output:
        raise ValueError("frames must divide into complete omega intervals")
    starts = range(0, count, frames_per_output)
    step = np.copysign(config.dome, config.ome_end - config.ome_start)
    for start in reversed(starts) if flip else starts:
        for index in range(start, start + frames_per_output):
            image = np.array(
                series[config.skip_frames + index], dtype=np.float64, copy=True
            )
            if image.shape != operator.input_shape:
                raise ValueError(
                    f"input image shape {image.shape} must be {operator.input_shape}"
                )
            image[image >= np.iinfo(np.uint32).max] = np.nan
            if frames_per_output > 1:
                image /= frames_per_output
            if index == start:
                integrated = image
            else:
                integrated += image
        np.floor(integrated, out=integrated)
        bin_index = start // frames_per_output
        bin_index = bins - 1 - bin_index if flip else bin_index
        omega = config.ome_start + step * (bin_index + 0.5)
        yield omega, integrated


def _wedge_edges(config: CakeDataConfig, operator: CakeMatrix) -> np.ndarray:
    start, step, count = operator.eta
    stop = start + step * count
    # The saved grid's rounded endpoint can extend up to half a pixel beyond
    # the requested range. Do not create an empty extra wedge for that sliver.
    nwedges = int(np.ceil((stop - start - step / 2) / config.cake_width - 1e-10))
    edges = start + np.arange(nwedges + 1) * config.cake_width
    edges[-1] = stop
    return edges


def _mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """NaN-aware mean without copying a float array or warning for empty bins."""
    total = np.sum(values, axis=0, where=valid)
    count = np.sum(valid, axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count != 0)


def _profiles(
    cakes: Iterable[np.ma.MaskedArray],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each wedge while its small cake is still in cache."""
    profiles = np.array(
        [
            _mean(cake.data, ~np.ma.getmaskarray(cake) & ~np.isnan(cake.data))
            for cake in cakes
        ]
    )
    # Give each valid wedge equal weight, as in Chris's newer Eiger script.
    full = _mean(profiles, ~np.isnan(profiles))
    return full, profiles


def cake_data(config: CakeDataConfig) -> tuple[Path, Path]:
    """Run the CakeData workflow and return its HDF5 and par output paths."""
    config.validate()
    metadata = load_par_file(config.par_file)
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
        Path(config.par_file).with_suffix('.json').resolve(),
        Path(config.matrix or config.instrument).resolve(),
        *(path.resolve() for path in raw_paths.values()),
    }
    if h5_path.resolve() == par_path.resolve() or inputs.intersection(
        (h5_path.resolve(), par_path.resolve())
    ):
        raise ValueError('output paths must differ from each other and all input files')
    if config.start_layer and h5_path.exists():
        raise ValueError(
            'start_layer selects a new output; resuming an existing output is not supported'
        )
    outputs = h5_path, par_path
    if any(path.is_dir() for path in outputs):
        raise ValueError('output paths must be files, not directories')
    # Keep previous outputs intact until processing and staging finish.
    with TemporaryDirectory(prefix='.cake-data-', dir=output_dir) as directory:
        temporary = Path(directory)
        operator = _load_matrix(config, temporary)
        edges = _wedge_edges(config, operator)
        bounds = np.searchsorted(operator.eta_centers, edges)
        wedges = [operator[first:last] for first, last in pairwise(bounds)]
        eta = (edges[:-1] + edges[1:]) / 2
        eta_key = np.vstack((np.arange(len(eta)), eta))
        tth = operator.tth_centers
        staged_h5 = temporary / h5_path.name
        staged_par = temporary / par_path.name
        par_rows = []
        with h5py.File(staged_h5, 'w') as stream:
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
                    full, profiles = _profiles(wedge.apply(frame) for wedge in wedges)
                    full_profiles.append(full)
                    omega_group = group.create_group(f'Ome/{index}')
                    omega_group['IvTTH'] = profiles
                    omega_group['eta_key'] = eta_key
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
                # Preserve the legacy CakeData par column order and placeholders.
                par_rows.append(
                    [10, 100, 1000, scan, image, image, x, x, y, z, 20, 0.6]
                )
                logger.info("processed cake-data layer %d", layer)
        par_format = ["%d"] * 6 + ["%.6f"] * 6
        np.savetxt(staged_par, np.asarray(par_rows), fmt=par_format, newline=" \n")
        for source, target in zip((staged_h5, staged_par), outputs):
            source.replace(target)
    logger.info("wrote CakeData outputs %s and %s", *outputs)
    return outputs


def config_as_yaml(config: CakeDataConfig | None = None) -> str:
    values = asdict(config or CakeDataConfig())
    for name, value in values.items():
        if isinstance(value, Path):
            values[name] = str(value)
    return yaml.safe_dump(values, sort_keys=False)
