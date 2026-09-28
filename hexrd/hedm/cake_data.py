"""Create azimuthally segmented powder profiles from composite detector data."""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.sparse import coo_matrix

from hexrd.core import imageseries
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.instrument.hedm_instrument import pixel_resolution
from hexrd.core.projections.polar import PolarView


logger = logging.getLogger(__name__)


@dataclass
class CakeDataConfig:
    """Configuration for :func:`cake_data`. Angles are in degrees."""

    data_dir: str | None = None
    output_dir: str | None = None
    sample_name: str | None = None
    instrument: str | None = None
    par_file: str | None = None
    output_par: str | None = None
    tth_start: float = 2.0
    tth_end: float = 5.0
    cake_width: float = 5.0
    eta_start: float = -180.0
    eta_end: float = 180.0
    ome_start: float = 0.0
    ome_end: float = 360.0
    dome: float = 5.0
    skip_frames: int = 0
    refinement: float = 1.0
    start_layer: int = 0
    par_version: int = 5
    max_input_frames: int | None = None
    image_pattern: str = "EIG16M_CdTe_*.h5"
    image_format: str = "eiger-stream-v1"

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

    def validate(self) -> None:
        required = ("data_dir", "output_dir", "sample_name", "instrument", "par_file")
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"missing required cake-data parameters: {missing}")
        if self.tth_start >= self.tth_end:
            raise ValueError("tth_start must be less than tth_end")
        if self.eta_start >= self.eta_end:
            raise ValueError("eta_start must be less than eta_end")
        if self.ome_start == self.ome_end:
            raise ValueError("ome_start and ome_end must differ")
        if self.cake_width <= 0 or self.dome <= 0 or self.refinement <= 0:
            raise ValueError("cake_width, dome, and refinement must be positive")
        if self.skip_frames < 0 or self.start_layer < 0:
            raise ValueError("skip_frames and start_layer must be non-negative")
        if self.max_input_frames is not None and self.max_input_frames <= 0:
            raise ValueError("max_input_frames must be positive")
        eta_count = (self.eta_end - self.eta_start) / self.cake_width
        if not np.isclose(eta_count, round(eta_count)):
            raise ValueError("eta range must be divisible by cake_width")
        if self.par_version != 5:
            raise ValueError("cake-data currently supports par version 5 only")

    def update(self, values: dict) -> None:
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


def load_par_file(path: str | Path, version: int = 5):
    if version != 5:
        raise ValueError("cake-data currently supports par version 5 only")
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] <= 24:
        raise ValueError("version-5 par input must contain at least 25 columns")
    layers = np.arange(data.shape[0])
    return {
        "scan_number": data[:, 3].astype(int),
        "image_number": data[:, 6].astype(int),
        "position_key": np.vstack((layers, data[:, 7:10].T)),
        "num_frames": data[:, 24].astype(int),
    }


def load_flip_flags(path: str | Path, row_count: int) -> np.ndarray:
    json_path = Path(path).with_suffix(".json")
    if not json_path.exists():
        return np.zeros(row_count, dtype=bool)
    with json_path.open(encoding="utf-8") as stream:
        columns = list(json.load(stream).values())
    try:
        rising = columns.index("fly_axis0_rising")
        falling = columns.index("fly_axis0_falling")
    except ValueError:
        logger.warning("par JSON has no fly-axis columns; omega order is unchanged")
        return np.zeros(row_count, dtype=bool)
    data = np.loadtxt(path, ndmin=2)
    return data[:, rising] > data[:, falling]


class CakeMatrix:
    """Sparse bilinear interpolation matrix for a fixed polar geometry."""

    def __init__(self, instrument, tth_range, eta_range, pixel_size):
        self.instrument = instrument
        self.view = PolarView(
            tth_range,
            instrument,
            eta_min=eta_range[0],
            eta_max=eta_range[1],
            pixel_size=pixel_size,
            cache_coordinate_map=True,
        )
        self.mask = self.view._nan_mask.copy()
        self.matrix = self._build_matrix()

    def _build_matrix(self):
        rows = []
        columns = []
        weights = []
        panel_offset = 0
        mapping = self.view._coordinate_mapping
        for name, panel in self.instrument.detectors.items():
            panel_map = mapping[name]
            output_rows = panel_map["on_panel_idx"]
            interp = panel_map["bilinear_interp_dict"]
            sources = (
                interp["i_floor_img"] * panel.cols + interp["j_floor_img"],
                interp["i_floor_img"] * panel.cols + interp["j_ceil_img"],
                interp["i_ceil_img"] * panel.cols + interp["j_floor_img"],
                interp["i_ceil_img"] * panel.cols + interp["j_ceil_img"],
            )
            for source, weight in zip(
                sources, (interp["cc"], interp["fc"], interp["cf"], interp["ff"])
            ):
                rows.append(output_rows)
                columns.append(source + panel_offset)
                weights.append(weight)
            panel_offset += panel.rows * panel.cols
        coordinates = (np.concatenate(rows), np.concatenate(columns))
        return coo_matrix(
            (np.concatenate(weights), coordinates),
            shape=(self.view.ntth * self.view.neta, panel_offset),
        ).tocsr()

    def apply(self, images_by_panel):
        pixels = np.concatenate(
            [
                np.asarray(images_by_panel[name]).ravel()
                for name in self.instrument.detectors
            ]
        )
        result = (self.matrix @ pixels).reshape(self.view.shape)
        return np.ma.array(result, mask=self.mask)


def _raw_path(config: CakeDataConfig, image_number: int) -> Path:
    template = config.data_dir
    try:
        directory = template.format(
            sample_name=config.sample_name, image_number=int(image_number)
        )
    except (IndexError, KeyError):
        directory = template
    if "%" in directory:
        try:
            directory = directory % (config.sample_name, int(image_number))
        except TypeError:
            directory = directory % int(image_number)
    matches = sorted(glob.glob(str(Path(directory) / config.image_pattern)))
    if not matches:
        raise FileNotFoundError(f"no {config.image_pattern!r} file under {directory}")
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple raw image files match under {directory}: {matches}"
        )
    return Path(matches[0])


def _read_integrated_frames(config, instrument, image_number, frame_count):
    raw_path = _raw_path(config, image_number)
    series = imageseries.open(raw_path, format=config.image_format)
    available = max(0, len(series) - config.skip_frames)
    count = frame_count
    if config.max_input_frames is not None:
        count = min(count, config.max_input_frames)
    if count > available:
        raise ValueError(
            f"par file requests {count} frame(s) after skipping "
            f"{config.skip_frames}, but {raw_path} has only {available}"
        )
    omega_range = config.ome_end - config.ome_start
    frames_per_output = int(abs(count / (omega_range / config.dome)))
    if frames_per_output <= 0 or count % frames_per_output:
        raise ValueError("input frame count is not divisible into omega intervals")

    output = []
    for start in range(0, count, frames_per_output):
        panels = {
            name: np.zeros(panel.shape, dtype=np.float64)
            for name, panel in instrument.detectors.items()
        }
        for index in range(start, start + frames_per_output):
            frame = np.asarray(series[config.skip_frames + index])
            for name, panel in instrument.detectors.items():
                if panel.roi is None:
                    raise ValueError(f"panel {name} has no ROI for composite input")
                (r0, r1), (c0, c1) = panel.roi
                image = frame[r0:r1, c0:c1].astype(np.float64, copy=True)
                image[image >= np.iinfo(np.uint32).max] = np.nan
                panels[name] += image / frames_per_output
        output.append({name: np.floor(value) for name, value in panels.items()})
    return output


def _profiles(cake: np.ma.MaskedArray, config: CakeDataConfig, matrix: CakeMatrix):
    eta_centers = np.degrees(matrix.view.angular_grid[0][:, 0])
    wedge_starts = np.arange(config.eta_start, config.eta_end, config.cake_width)
    profiles = []
    for start in wedge_starts:
        selected = (eta_centers >= start) & (eta_centers < start + config.cake_width)
        profiles.append(np.ma.mean(cake[selected], axis=0).filled(np.nan))
    eta = wedge_starts + 0.5 * config.cake_width
    full = np.ma.mean(cake, axis=0).filled(np.nan)
    return full, np.asarray(profiles), eta


def _initialize_output(path: Path, position_key: np.ndarray) -> None:
    with h5py.File(path, "w") as stream:
        stream.create_dataset("XYZ_key", data=position_key)
        cake = stream.create_group("Cake")
        for layer in range(position_key.shape[1]):
            group = cake.create_group(str(layer))
            group.create_dataset("xyz", data=position_key[1:, layer])


def _write_layer(path, layer, tth, full, by_omega, by_eta, eta, omega):
    with h5py.File(path, "r+") as stream:
        group = stream[f"Cake/{layer}"]
        group.create_dataset("tth", data=tth)
        group.create_dataset("IvTTH_Full", data=full)
        group.create_dataset("IvTTHvW", data=by_omega)
        group.create_dataset("ome_key", data=np.vstack((np.arange(len(omega)), omega)))
        group.create_dataset("nFrames", data=len(omega))
        omega_group = group.create_group("Ome")
        for index, values in enumerate(by_eta):
            frame = omega_group.create_group(str(index))
            frame.create_dataset("IvTTH", data=values)
            frame.create_dataset("eta_key", data=np.vstack((np.arange(len(eta)), eta)))


def cake_data(config: CakeDataConfig) -> tuple[Path, Path]:
    """Run the CakeData workflow and return its HDF5 and par output paths."""
    config.validate()
    instrument = load_instrument(config.instrument)
    tth_stats, eta_stats = pixel_resolution(instrument)
    pixel_size = (
        config.refinement * np.degrees(tth_stats[1]),
        np.degrees(eta_stats[1]),
    )
    operator = CakeMatrix(
        instrument,
        (config.tth_start, config.tth_end),
        (config.eta_start, config.eta_end),
        pixel_size,
    )

    metadata = load_par_file(config.par_file, config.par_version)
    flips = load_flip_flags(config.par_file, len(metadata["scan_number"]))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{config.sample_name}.h5"
    par_path = output_dir / (config.output_par or f"{config.sample_name}.par")
    _initialize_output(h5_path, metadata["position_key"])
    tth = np.degrees(operator.view.angular_grid[1][0])
    par_rows = []

    for layer in range(config.start_layer, len(metadata["scan_number"])):
        frames = _read_integrated_frames(
            config,
            instrument,
            metadata["image_number"][layer],
            metadata["num_frames"][layer],
        )
        if flips[layer]:
            frames.reverse()
        full_profiles = []
        eta_profiles = []
        eta = None
        for frame in frames:
            full, wedges, eta = _profiles(operator.apply(frame), config, operator)
            full_profiles.append(full)
            eta_profiles.append(wedges)
        by_omega = np.asarray(full_profiles)
        full = np.mean(by_omega, axis=0)
        omega_step = (
            config.dome if config.ome_end >= config.ome_start else -config.dome
        )
        omega = config.ome_start + omega_step * (np.arange(len(frames)) + 0.5)
        _write_layer(
            h5_path, layer, tth, full, by_omega, np.asarray(eta_profiles), eta, omega
        )
        xyz = metadata["position_key"][1:, layer]
        par_rows.append(
            [
                10,
                100,
                1000,
                metadata["scan_number"][layer],
                metadata["image_number"][layer],
                metadata["image_number"][layer],
                xyz[0],
                xyz[0],
                xyz[1],
                xyz[2],
                20,
                0.6,
            ]
        )

    par_format = ["%d"] * 6 + ["%.6f"] * 6
    np.savetxt(par_path, np.asarray(par_rows), fmt=par_format, newline=" \n")
    logger.info("wrote CakeData outputs %s and %s", h5_path, par_path)
    return h5_path, par_path


def config_as_json(config: CakeDataConfig | None = None) -> str:
    return json.dumps(asdict(config or CakeDataConfig()), indent=2)
