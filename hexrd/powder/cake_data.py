"""Create azimuthally segmented powder profiles from detector data"""

from __future__ import annotations

import glob
import json
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

import h5py
import numpy as np
import yaml

from hexrd.core import imageseries
from hexrd.core.config.loader import NumPyIncludeLoader
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.instrument.hedm_instrument import pixel_resolution
from hexrd.powder.cake import CakeMatrix, export_cake_matrix


logger = logging.getLogger(__name__)

# These defaults apply when we build a matrix from an instrument file
GEOMETRY_DEFAULTS = {
    'tth_start': 2.0,
    'tth_end': 5.0,
    'eta_start': -180.0,
    'eta_end': 180.0,
    'refinement': 1.0,
}


def _interval_count(span: float, width: float, name: str) -> int:
    count = abs(span) / width
    if (
        not np.isfinite(count) or
        count < 1 or
        not np.isclose(count, round(count), rtol=1e-10, atol=1e-10)
    ):
        raise ValueError(f'{name} range must be divisible by its interval width')
    return int(round(count))


def _option(default: Any, short: str, help: str) -> Any:
    """Attach command-line help to a configuration field"""
    return field(default=default, metadata={'short': short, 'help': help})


@dataclass
class CakeDataConfig:
    """Configuration for :func:`cake_data`. Angles are in degrees.

    Relative paths resolve against the working directory. ``data_dir`` accepts
    named fields ``{sample_name}`` and ``{image_number:04d}``, or legacy ``%s``
    (sample) and ``%d``/``%04d`` (image number) fields. Use doubled braces in
    named templates and ``%%`` in legacy templates for literal delimiters.

    ``image_pattern`` can also map panel names or ROI group names to file
    globs, such as ``{ff1: '*ff1*.h5', ff2: '*ff2*.h5'}``. These streams share
    the reader options and omega schedule, and must have equal frame counts.
    ``image_flips`` maps those names to HEXRD flip operations, applied to the
    complete parent image before cropping subpanels. Otherwise, inputs must
    already have the orientation expected by the instrument.
    """

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

    image_pattern: str | dict[str, str] = _option(
        'EIG16M_CdTe_*.h5', 'g',
        'image file glob (YAML also accepts a panel/group-to-glob mapping)'
    )
    image_format: str = _option('eiger-stream-v1', 'F', 'HEXRD image-series format')
    image_options: dict[str, Any] = field(
        default_factory=dict,
        metadata={'short': 'O', 'help': 'image-series reader options as a JSON object'},
    )
    image_flips: dict[str, str] = field(
        default_factory=dict,
        metadata={'short': 'X', 'help': 'per-panel/group flips as a JSON object'},
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CakeDataConfig:
        values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(values, dict):
            raise ValueError('cake-data configuration must be a YAML mapping')

        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            msg = f'unknown cake-data parameters: {sorted(unknown, key=str)}'
            raise ValueError(msg)

        return cls(**values)

    @property
    def geometry(self) -> dict[str, float]:
        geometry = {}
        for name, default in GEOMETRY_DEFAULTS.items():
            value = getattr(self, name)
            if value is None:
                value = default

            geometry[name] = value

        return geometry

    def validate(self) -> None:
        for name, types in _field_types().items():
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

        required = ('data_dir', 'output_dir', 'sample_name', 'par_file')
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f'missing required cake-data parameters: {missing}')

        if bool(self.instrument) == bool(self.matrix):
            raise ValueError('provide exactly one of instrument or matrix')

        overrides = [n for n in GEOMETRY_DEFAULTS if getattr(self, n) is not None]
        if self.matrix and overrides:
            msg = f'matrix fixes the geometry; remove these parameters: {overrides}'
            raise ValueError(msg)

        geometry = self.geometry
        scales = (self.cake_width, self.dome, geometry['refinement'])
        if not np.isfinite(scales).all() or min(scales) <= 0:
            msg = 'cake_width, dome, and refinement must be finite and positive'
            raise ValueError(msg)

        minimums = {'skip_frames': 0, 'start_layer': 0, 'max_input_frames': 1}
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if value is not None and value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')

        _interval_count(self.ome_end - self.ome_start, self.dome, 'omega')
        if not self.matrix:
            if geometry['tth_start'] >= geometry['tth_end']:
                raise ValueError('tth_start must be less than tth_end')

            eta_span = geometry['eta_end'] - geometry['eta_start']
            if eta_span <= 0:
                raise ValueError('eta_start must be less than eta_end')
            _interval_count(eta_span, self.cake_width, 'eta')

        if self.par_version != 5:
            raise ValueError('cake-data currently supports par version 5 only')

        if isinstance(self.image_pattern, dict):
            if not self.image_pattern or not all(
                isinstance(name, str) and name and isinstance(pattern, str) and pattern
                for name, pattern in self.image_pattern.items()
            ):
                msg = 'image_pattern must map panel/group names to nonempty file globs'
                raise ValueError(msg)
        elif self.image_flips:
            raise ValueError('image_flips requires a panel/group image_pattern mapping')

        flips = ('y', 'v', 'x', 'h', 'vh', 'hv', 'r180', 't', 'T', 'ccw90',
                 'r90', 'cw90', 'r270')
        for name, flip in self.image_flips.items():
            if name not in self.image_pattern or flip not in flips:
                raise ValueError(f'invalid image_flips entry: {name!r}: {flip!r}')

        if not all(isinstance(key, str) for key in self.image_options):
            raise ValueError('image_options keys must be strings')
        if 'format' in self.image_options:
            raise ValueError('set image_format separately from image_options')
        if self.image_format not in imageseries.load.Registry.adapter_registry:
            raise ValueError(f'unknown image_format: {self.image_format}')

        reader_options = {
            'hdf5': {'path', 'dataname', 'close_when_finished'},
            'eiger-stream-v1': set(),
            'eiger-stream-v2': {'threshold_setting', 'multiplier'},
        }
        if self.image_format in reader_options:
            unknown = self.image_options.keys() - reader_options[self.image_format]
            if unknown:
                msg = f'unknown image_options for {self.image_format}: {sorted(unknown)}'
                raise ValueError(msg)

        if self.image_format == 'eiger-stream-v2':
            threshold = self.image_options.get('threshold_setting', 'threshold_1')
            if threshold == 'man_diff':
                msg = 'cake-data does not support man_diff; select a single threshold'
                raise ValueError(msg)
            if threshold not in ('threshold_1', 'threshold_2'):
                msg = 'threshold_setting must be threshold_1 or threshold_2'
                raise ValueError(msg)

        for name in ('sample_name', 'output_par'):
            value = getattr(self, name)
            if value and (Path(value).name != value or value in ('.', '..')):
                raise ValueError(f'{name} must be a filename, without a directory')


def _field_types() -> dict[str, tuple[type, ...]]:
    """Resolve configuration field types for validation and CLI parsing"""
    result = {}
    for name, hint in get_type_hints(CakeDataConfig).items():
        types = get_args(hint) if get_origin(hint) is UnionType else (hint,)
        result[name] = tuple(get_origin(t) or t for t in types)
    return result


def load_par_file(path: str | Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 25:
        raise ValueError('version-5 par input must contain at least 25 columns')

    counts = data[:, [3, 6, 24]]
    if (
        not np.isfinite(data[:, [3, 6, 7, 8, 9, 24]]).all() or
        np.any(counts != np.floor(counts)) or
        np.any(counts < 0) or
        np.any(data[:, 24] == 0)
    ):
        msg = 'par scan/image numbers must be integers and frame counts positive'
        raise ValueError(msg)

    # Optional fly-axis columns are named by numeric keys in a JSON sidecar
    flips = np.zeros(len(data), dtype=bool)
    json_path = Path(path).with_suffix('.json')
    if json_path.exists():
        names = json.loads(json_path.read_text(encoding='utf-8'))
        columns = {name: int(index) for index, name in names.items()}
        try:
            fly_columns = [
                columns[name] for name in ('fly_axis0_rising', 'fly_axis0_falling')
            ]
        except KeyError:
            logger.warning('par JSON has no fly-axis columns; omega order is unchanged')
        else:
            if any(index < 0 or index >= data.shape[1] for index in fly_columns):
                msg = 'fly-axis columns must refer to columns in the par file'
                raise ValueError(msg)

            fly_axis = data[:, fly_columns]
            if not np.isfinite(fly_axis).all():
                raise ValueError('fly-axis values must be finite')
            flips = fly_axis[:, 0] > fly_axis[:, 1]

    return {
        'scan_number': data[:, 3].astype(int),
        'image_number': data[:, 6].astype(int),
        'position_key': np.vstack((np.arange(len(data)), data[:, 7:10].T)),
        'num_frames': data[:, 24].astype(int),
        'flip': flips,
    }


def _load_matrix(config: CakeDataConfig, temporary: Path) -> CakeMatrix:
    path = Path(config.matrix) if config.matrix else temporary / 'geometry.hexrdcake'
    if not config.matrix:
        if h5py.is_hdf5(config.instrument):
            with h5py.File(config.instrument, 'r') as stream:
                instrument = HEDMInstrument(stream)
        else:
            with open(config.instrument, 'r') as stream:
                instr_config = yaml.load(stream, Loader=NumPyIncludeLoader)
            instrument = HEDMInstrument(instr_config)

        geometry = config.geometry
        tth_stats, eta_stats = pixel_resolution(instrument)
        kwargs = {
            'tth_range': (geometry['tth_start'], geometry['tth_end']),
            'eta_range': (geometry['eta_start'], geometry['eta_end']),
            'pixel_size': (
                geometry['refinement'] * np.degrees(tth_stats[1]),
                np.degrees(eta_stats[1]),
            ),
        }
        export_cake_matrix(path, instrument, **kwargs)

    operator = CakeMatrix(path)
    if operator.input_layout == 'panels' and isinstance(config.image_pattern, str):
        msg = 'panel-layout matrices require an image_pattern mapping by panel or group'
        raise ValueError(msg)

    return operator


def _raw_path(config: CakeDataConfig, image_number: int, pattern: str) -> Path:
    template = config.data_dir
    if '{' in template or '}' in template:
        try:
            directory = template.format(
                sample_name=config.sample_name, image_number=int(image_number)
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f'invalid data_dir template: {exc}') from exc
    else:
        def substitute(match):
            token = match.group()
            if token == '%%':
                return '%'
            if token == '%':
                msg = 'data_dir supports %s and %d fields; escape literal percent as %%'
                raise ValueError(msg)
            value = config.sample_name if token.endswith('s') else int(image_number)
            return token % value

        directory = re.sub(r'%%|%[0-9]*[sd]|%', substitute, template)

    matches = sorted(glob.glob(str(Path(directory) / pattern)))
    if not matches:
        raise FileNotFoundError(f'no {pattern!r} file under {directory}')
    if len(matches) > 1:
        raise ValueError(f'multiple raw image files match under {directory}: {matches}')

    return Path(matches[0])


def _read_integrated_frames(
    config: CakeDataConfig,
    operator: CakeMatrix,
    raw_paths: dict[str | None, Path],
    frame_count: int,
    flip: bool = False,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield floored omega averages, reversing bins when par metadata requires it"""
    series = {}
    for name, path in raw_paths.items():
        source = imageseries.open(path, format=config.image_format, **config.image_options)
        if name in config.image_flips:
            source = imageseries.process.ProcessedImageSeries(
                source, [('flip', config.image_flips[name])]
            )
        series[name] = source

    if len({len(source) for source in series.values()}) != 1:
        raise ValueError('detector image streams must have equal frame counts')

    count = min(frame_count, config.max_input_frames or frame_count)
    for name, source in series.items():
        if count > len(source) - config.skip_frames:
            msg = f'{raw_paths[name]} has fewer than the {count} requested frame(s)'
            raise ValueError(msg)

    bins = _interval_count(config.ome_end - config.ome_start, config.dome, 'omega')
    frames_per_output = frame_count // bins
    if frames_per_output < 1 or frame_count % bins or count % frames_per_output:
        raise ValueError('frames must divide into complete omega intervals')

    starts = range(0, count, frames_per_output)
    step = np.copysign(config.dome, config.ome_end - config.ome_start)
    if flip:
        starts = reversed(starts)

    for start in starts:
        for index in range(start, start + frames_per_output):
            frames = {
                name: source[config.skip_frames + index]
                for name, source in series.items()
            }
            pixels = frames[None] if None in frames else operator.assemble(frames)
            image = np.array(pixels, dtype=np.float64, copy=True)
            if image.shape != operator.input_shape:
                msg = (
                    f'input image shape {image.shape} must be {operator.input_shape}, '
                    'the composite extent defined by the instrument panel ROIs'
                )
                raise ValueError(msg)

            # Invalid Eiger pixels remain invalid throughout omega averaging
            image[image >= np.iinfo(np.uint32).max] = np.nan

            if index == start:
                integrated = image
            else:
                integrated += image

        # Divide once so per-frame rounding cannot lower an integer mean
        if frames_per_output > 1:
            integrated /= frames_per_output
        np.floor(integrated, out=integrated)
        bin_index = start // frames_per_output
        if flip:
            bin_index = bins - 1 - bin_index

        omega = config.ome_start + step * (bin_index + 0.5)
        yield omega, integrated


def _wedge_edges(config: CakeDataConfig, operator: CakeMatrix) -> np.ndarray:
    start, step, count = operator.eta
    stop = start + step * count
    if config.cake_width < step and not np.isclose(
        config.cake_width, step, rtol=1e-10, atol=0
    ):
        raise ValueError(f'cake_width must be at least the matrix eta step ({step})')

    nwedges = int(np.ceil((stop - start) / config.cake_width))
    edges = start + np.arange(nwedges + 1) * config.cake_width
    # Drop an empty final wedge caused by a rounded grid endpoint, but keep
    # one whose first row lies exactly on its left boundary.
    nwedges = np.searchsorted(edges, operator.eta_centers[-1], side='right')
    edges = edges[:nwedges + 1]
    edges[-1] = stop

    return edges


def _mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """NaN-aware mean without copying a float array or warning for empty bins"""
    total = np.sum(values, axis=0, where=valid)
    count = np.sum(valid, axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count != 0)


def _profiles(
    cakes: Iterable[np.ma.MaskedArray],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each wedge while its small cake is still in cache"""
    profiles = []
    for cake in cakes:
        valid = ~np.ma.getmaskarray(cake) & ~np.isnan(cake.data)
        profiles.append(_mean(cake.data, valid))

    profiles = np.asarray(profiles)

    # Give each valid wedge equal weight, as in Chris's newer Eiger script
    full = _mean(profiles, ~np.isnan(profiles))
    return full, profiles


def _write_layer_profiles(group, frames, wedges, tth, eta_key):
    full_profiles = []
    omega = []
    for index, (angle, frame) in enumerate(frames):
        # Process one wedge at a time to keep memory usage down
        full, profiles = _profiles(wedge.apply(frame) for wedge in wedges)
        full_profiles.append(full)
        omega.append(angle)

        omega_group = group.create_group(f'Ome/{index}')
        omega_group['IvTTH'] = profiles
        omega_group['eta_key'] = eta_key

    by_omega = np.asarray(full_profiles)
    group.update({
        'tth': tth,
        'IvTTH_Full': np.mean(by_omega, axis=0),
        'IvTTHvW': by_omega,
        'ome_key': np.vstack((np.arange(len(omega)), omega)),
        'nFrames': len(omega),
    })


def cake_data(config: CakeDataConfig) -> tuple[Path, Path]:
    """Run the CakeData workflow and return its HDF5 and par output paths"""
    config.validate()
    metadata = load_par_file(config.par_file)
    layers = range(config.start_layer, len(metadata['scan_number']))
    if not layers:
        msg = 'start_layer must select at least one par-file layer'
        raise ValueError(msg)

    patterns = config.image_pattern
    if isinstance(patterns, str):
        patterns = {None: patterns}
    raw_paths = {
        layer: {
            name: _raw_path(config, metadata['image_number'][layer], pattern)
            for name, pattern in patterns.items()
        }
        for layer in layers
    }

    output_dir = Path(config.output_dir)
    h5_path = output_dir / f'{config.sample_name}.h5'
    par_path = output_dir / (config.output_par or f'{config.sample_name}.par')
    outputs = h5_path, par_path

    # Check the destinations before processing or replacing any files
    inputs = {
        Path(config.par_file).resolve(),
        Path(config.par_file).with_suffix('.json').resolve(),
        Path(config.matrix or config.instrument).resolve(),
        *(path.resolve() for paths in raw_paths.values() for path in paths.values()),
    }
    if h5_path.resolve() == par_path.resolve() or inputs.intersection(
        (h5_path.resolve(), par_path.resolve())
    ):
        msg = 'output paths must differ from each other and all input files'
        raise ValueError(msg)

    if config.start_layer and h5_path.exists():
        msg = ('start_layer selects a new output. '
               'Resuming an existing output is not supported')
        raise ValueError(msg)

    if any(path.is_dir() for path in outputs):
        msg = 'output paths must be files, not directories'
        raise ValueError(msg)

    # Keep previous outputs intact until processing and staging finish
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cake-data-', dir=output_dir) as directory:
        temporary = Path(directory)
        operator = _load_matrix(config, temporary)
        edges = _wedge_edges(config, operator)
        bounds = np.searchsorted(operator.eta_centers, edges)
        if np.any(np.diff(bounds) == 0):
            raise ValueError('each cake wedge must contain at least one eta row')
        wedges = [operator[first:last] for first, last in pairwise(bounds)]
        eta = (edges[:-1] + edges[1:]) / 2
        eta_key = np.vstack((np.arange(len(eta)), eta))

        staged_h5 = temporary / h5_path.name
        staged_par = temporary / par_path.name
        par_rows = []
        with h5py.File(staged_h5, 'w') as wf:
            wf['XYZ_key'] = metadata['position_key']
            for layer, xyz in enumerate(metadata['position_key'][1:].T):
                wf[f'Cake/{layer}/xyz'] = xyz

            for layer in layers:
                frames = _read_integrated_frames(
                    config,
                    operator,
                    raw_paths[layer],
                    metadata['num_frames'][layer],
                    metadata['flip'][layer],
                )
                group = wf[f'Cake/{layer}']
                _write_layer_profiles(group, frames, wedges, operator.tth_centers, eta_key)

                # Preserve the legacy par column order and placeholders
                scan = metadata['scan_number'][layer]
                image = metadata['image_number'][layer]
                x, y, z = metadata['position_key'][1:, layer]
                par_rows.append([10, 100, 1000, scan, image, image, x, x, y, z, 20, 0.6])
                logger.info(f'Processed cake-data layer {layer}')

        par_format = ['%d'] * 6 + ['%.6f'] * 6
        np.savetxt(staged_par, np.asarray(par_rows), fmt=par_format, newline=' \n')
        for source, target in zip((staged_h5, staged_par), outputs):
            source.replace(target)

    logger.info(f'Wrote CakeData outputs {h5_path} and {par_path}')
    return outputs


def config_as_yaml(config: CakeDataConfig | None = None) -> str:
    if config is None:
        config = CakeDataConfig()

    values = asdict(config)
    for name, value in values.items():
        if isinstance(value, Path):
            values[name] = str(value)

    return yaml.safe_dump(values, sort_keys=False)
