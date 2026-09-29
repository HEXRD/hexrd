"""Write reusable HEXRD polar-caking operators.

The writer deliberately contains only geometry, persistence, and input-layout
concerns.  Scan metadata, dark correction, omega integration, and CakeData
result writing belong to the batch workflow that consumes these files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hexrd.core.utils.panel_buffer import (
    panel_buffer_as_2d_array,
    panel_buffer_from_str,
    valid_panel_buffer_names,
)
from hexrd.core.utils.hdf5 import unwrap_h5_to_dict
from hexrd.core.projections.polar import PolarView


FORMAT_NAME = 'HEXRD cake operator'
FORMAT_VERSION_MAJOR = 2
FORMAT_VERSION_MINOR = 0
REPRESENTATION = 'bilinear-output-block-v1'
SCHEMA_STATUS = 'provisional-2.0'

COMPRESSION = 'gzip'
COMPRESSION_LEVEL = 1
CHUNK_TARGET_BYTES = 1024 * 1024
DEFAULT_ETA_CHUNK_ROWS = 64
MIN_COMPRESSED_BYTES = 4096

STRING_DTYPE = h5py.string_dtype(encoding='utf-8')


class _ExactGridPolarView(PolarView):
    """PolarView that projects a supplied slice of one canonical grid."""

    def __init__(self, *args: Any, angular_grid: Sequence[np.ndarray], **kwargs: Any):
        self._exact_angular_grid = list(angular_grid)
        super().__init__(*args, **kwargs)

    @property
    def angular_grid(self) -> list[np.ndarray]:
        return self._exact_angular_grid


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unknown-source-checkout'


@dataclass(frozen=True)
class CakeOperatorSpec:
    """Polar grid specification, with all angles expressed in degrees."""

    tth_range: tuple[float, float]
    eta_range: tuple[float, float]
    pixel_size: tuple[float, float]

    def __post_init__(self) -> None:
        tth_range = tuple(float(x) for x in self.tth_range)
        eta_range = tuple(float(x) for x in self.eta_range)
        pixel_size = tuple(float(x) for x in self.pixel_size)
        if len(tth_range) != 2 or len(eta_range) != 2 or len(pixel_size) != 2:
            raise ValueError('ranges and pixel_size must each contain two values')
        if not np.isfinite(tth_range + eta_range + pixel_size).all():
            raise ValueError('grid parameters must be finite')
        if tth_range[1] <= tth_range[0]:
            raise ValueError('tth_range must be increasing')
        if eta_range[1] <= eta_range[0]:
            raise ValueError('eta_range must be increasing')
        if pixel_size[0] <= 0 or pixel_size[1] <= 0:
            raise ValueError('pixel sizes must be positive')
        if self._bin_count(tth_range, pixel_size[0]) < 1:
            raise ValueError('tth_range must contain at least one output bin')
        if self._bin_count(eta_range, pixel_size[1]) < 1:
            raise ValueError('eta_range must contain at least one output bin')
        object.__setattr__(self, 'tth_range', tth_range)
        object.__setattr__(self, 'eta_range', eta_range)
        object.__setattr__(self, 'pixel_size', pixel_size)

    @staticmethod
    def _bin_count(bounds: tuple[float, float], step: float) -> int:
        # This exactly matches HEXRD PolarView's grid-size calculation.
        return int(round((bounds[1] - bounds[0]) / step))

    @property
    def ntth(self) -> int:
        return self._bin_count(self.tth_range, self.pixel_size[0])

    @property
    def neta(self) -> int:
        return self._bin_count(self.eta_range, self.pixel_size[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.neta, self.ntth

    @property
    def size(self) -> int:
        return self.neta * self.ntth

    @property
    def tth_edges(self) -> np.ndarray:
        return self.tth_range[0] + self.pixel_size[0] * np.arange(
            self.ntth + 1, dtype=np.float64
        )

    @property
    def eta_edges(self) -> np.ndarray:
        return self.eta_range[0] + self.pixel_size[1] * np.arange(
            self.neta + 1, dtype=np.float64
        )

    @property
    def tth_centers_radians(self) -> np.ndarray:
        return (
            np.radians(self.pixel_size[0] * np.arange(self.ntth))
            + np.radians(self.tth_range[0])
            + 0.5 * np.radians(self.pixel_size[0])
        )

    @property
    def eta_centers_radians(self) -> np.ndarray:
        return (
            np.radians(self.pixel_size[1] * np.arange(self.neta))
            + np.radians(self.eta_range[0])
            + 0.5 * np.radians(self.pixel_size[1])
        )


@dataclass(frozen=True)
class CakeOperatorSummary:
    path: Path
    grid_shape: tuple[int, int]
    panel_count: int
    active_pixel_count: int
    full_panel_pixel_count: int
    group_count: int
    file_size: int
    symbolic_panel_buffers: Mapping[str, str]


def _chunk_shape(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
    if len(shape) == 1:
        records = max(1, CHUNK_TARGET_BYTES // dtype.itemsize)
        return (max(1, min(shape[0], records)),)

    record_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    records = max(1, CHUNK_TARGET_BYTES // max(1, record_bytes))
    return (max(1, min(shape[0], records)), *shape[1:])


def _storage_kwargs(shape: tuple[int, ...], dtype: np.dtype) -> dict[str, Any]:
    size = int(np.prod(shape)) * dtype.itemsize
    if not shape or 0 in shape or size < MIN_COMPRESSED_BYTES:
        return {}
    return {
        'compression': COMPRESSION,
        'compression_opts': COMPRESSION_LEVEL,
        'shuffle': True,
        'fletcher32': True,
        'chunks': _chunk_shape(shape, dtype),
    }


def _write_array(group: h5py.Group, name: str, data: Any) -> h5py.Dataset:
    array = np.asarray(data)
    return group.create_dataset(
        name,
        data=array,
        **_storage_kwargs(array.shape, array.dtype),
    )


def _create_fixed_array(
    group: h5py.Group,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        **_storage_kwargs(shape, dtype),
    )


def _create_appendable_records(
    group: h5py.Group,
    name: str,
    columns: int,
    dtype: np.dtype,
) -> h5py.Dataset:
    chunk_shape = _chunk_shape(
        (max(1, CHUNK_TARGET_BYTES // dtype.itemsize), columns), dtype
    )
    return group.create_dataset(
        name,
        shape=(0, columns),
        maxshape=(None, columns),
        dtype=dtype,
        compression=COMPRESSION,
        compression_opts=COMPRESSION_LEVEL,
        shuffle=True,
        fletcher32=True,
        chunks=chunk_shape,
    )


def _append_records(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if not len(values):
        return
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values


def _write_mapping(group: h5py.Group, mapping: Mapping[str, Any]) -> None:
    """Write a nested instrument mapping without pickle or object arrays."""
    for key, item in mapping.items():
        if item is None:
            continue
        if isinstance(item, Mapping):
            _write_mapping(group.create_group(str(key)), item)
            continue

        array = np.atleast_1d(item)
        if array.dtype.kind in 'SUO':
            strings = np.asarray([str(x) for x in array.flat], dtype=object).reshape(
                array.shape
            )
            if strings.size == 1:
                group.create_dataset(
                    str(key), data=str(strings.flat[0]), dtype=STRING_DTYPE
                )
            else:
                group.create_dataset(str(key), data=strings, dtype=STRING_DTYPE)
        else:
            _write_array(group, str(key), array)


def _mapping_digest(mapping: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(f'{prefix}/{key}', value[key])
            return
        if value is None:
            return
        digest.update(prefix.encode())
        array = np.atleast_1d(value)
        if array.dtype.kind in 'SUO':
            digest.update('\0'.join(str(x) for x in array.flat).encode())
            return
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(np.asarray(contiguous.shape, dtype='<u8').tobytes())
        digest.update(contiguous.tobytes())

    visit('', mapping)
    return digest.hexdigest()


def _normalized_instrument_config(
    instrument: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return an embeddable config, replacing exact known buffers by name."""
    config = deepcopy(instrument.write_config(style='hdf5'))
    replacements: dict[str, str] = {}
    known_names = sorted(valid_panel_buffer_names())

    array_panels = {
        name: panel
        for name, panel in instrument.detectors.items()
        if isinstance(panel.panel_buffer, np.ndarray) and panel.panel_buffer.ndim == 2
    }
    # Composite subpanels can individually be ambiguous because a small ROI
    # may happen to be identical in multiple known buffers. Prefer one known
    # full-detector buffer only when it matches every expanded panel mask.
    for known_name in known_names:
        if not array_panels:
            break
        matches_all = True
        for panel in array_panels.values():
            try:
                known_buffer = panel_buffer_from_str(known_name, panel)
            except (NotImplementedError, RuntimeError):
                matches_all = False
                break
            if not np.array_equal(panel.panel_buffer, known_buffer):
                matches_all = False
                break
        if matches_all:
            for panel_name in array_panels:
                config['detectors'][panel_name]['buffer'] = known_name
                replacements[panel_name] = known_name
            break

    for panel_name, panel in instrument.detectors.items():
        panel_buffer = panel.panel_buffer
        if isinstance(panel_buffer, str):
            replacements[panel_name] = panel_buffer
            continue
        if not isinstance(panel_buffer, np.ndarray) or panel_buffer.ndim != 2:
            continue
        if panel_name in replacements:
            continue

        for known_name in known_names:
            try:
                known_buffer = panel_buffer_from_str(known_name, panel)
            except (NotImplementedError, RuntimeError):
                continue
            if np.array_equal(panel_buffer, known_buffer):
                config['detectors'][panel_name]['buffer'] = known_name
                replacements[panel_name] = known_name
                break

    return config, replacements


def _attribute_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_digest(digest: Any, name: str, dataset: h5py.Dataset) -> None:
    digest.update(name.encode())
    digest.update(dataset.dtype.str.encode())
    digest.update(np.asarray(dataset.shape, dtype='<u8').tobytes())
    if not dataset.size:
        return

    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return

    row_bytes = max(1, int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize)
    rows = max(1, CHUNK_TARGET_BYTES // row_bytes)
    for start in range(0, dataset.shape[0], rows):
        values = np.ascontiguousarray(dataset[start : start + rows])
        digest.update(values.tobytes())


def _payload_digest(root: h5py.Group) -> str:
    digest = hashlib.sha256()
    panel_names = root['input/panel_names'].asstr()[:]
    digest.update('\0'.join(panel_names).encode())
    for digest_name, dataset_name in (
        ('tth_edges', 'grid/tth_edges'),
        ('eta_edges', 'grid/eta_edges'),
        ('output_mask', 'grid/output_mask'),
        ('panel_shapes', 'input/panel_shapes'),
        ('active_pixel_indptr', 'input/active_pixel_indptr'),
        ('active_local_indices', 'input/active_local_indices'),
        ('output_indptr', 'operator/output_indptr'),
        ('source_indices', 'operator/source_indices'),
        ('weights', 'operator/weights'),
    ):
        if dataset_name == 'grid/output_mask' and dataset_name not in root:
            if int(root.attrs['format_version_major']) == 1:
                raise ValueError('legacy operator is missing output_mask')
            continue
        _dataset_digest(digest, digest_name, root[dataset_name])
    adapter = 'input_adapters/composite/active_frame_indices'
    if adapter in root:
        _dataset_digest(digest, 'active_frame_indices', root[adapter])
        _dataset_digest(
            digest,
            'composite_frame_shape',
            root['input_adapters/composite/frame_shape'],
        )
    return digest.hexdigest()


def _integer_dtype(maximum: int) -> np.dtype:
    return np.dtype(np.uint32 if maximum < 2**32 else np.uint64)


def _operator_records_for_chunk(
    instrument: Any,
    spec: CakeOperatorSpec,
    eta_start: int,
    eta_stop: int,
    panel_names: tuple[str, ...],
    panel_offsets: np.ndarray,
    valid_pixels: Mapping[str, np.ndarray],
    used_pixels: Mapping[str, np.ndarray],
    source_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one eta chunk and return output-major interpolation records."""
    eta_edges = spec.eta_edges
    # Keep the arithmetic identical to one monolithic HEXRD PolarView, while
    # materializing only this eta slice of the two-dimensional angular grid.
    angular_grid = np.meshgrid(
        spec.eta_centers_radians[eta_start:eta_stop],
        spec.tth_centers_radians,
        indexing='ij',
    )
    polar = _ExactGridPolarView(
        spec.tth_range,
        instrument,
        eta_min=float(eta_edges[eta_start]),
        eta_max=float(eta_edges[eta_stop]),
        pixel_size=spec.pixel_size,
        cache_coordinate_map=False,
        angular_grid=angular_grid,
    )
    mapping = polar._generate_coordinate_mapping()

    rows_by_panel: list[np.ndarray] = []
    sources_by_panel: list[np.ndarray] = []
    weights_by_panel: list[np.ndarray] = []
    for panel_index, panel_name in enumerate(panel_names):
        panel = instrument.detectors[panel_name]
        panel_map = mapping[panel_name]
        interpolation = panel_map['bilinear_interp_dict']
        rows = panel_map['on_panel_idx']
        sources = np.stack(
            (
                interpolation['i_floor_img'] * panel.cols
                + interpolation['j_floor_img'],
                interpolation['i_floor_img'] * panel.cols + interpolation['j_ceil_img'],
                interpolation['i_ceil_img'] * panel.cols + interpolation['j_floor_img'],
                interpolation['i_ceil_img'] * panel.cols + interpolation['j_ceil_img'],
            ),
            axis=1,
        )
        weights = np.stack(
            (
                interpolation['cc'],
                interpolation['fc'],
                interpolation['cf'],
                interpolation['ff'],
            ),
            axis=1,
        )

        # Drop a complete bilinear group if any source pixel is statically
        # invalid.  The weights are not renormalized.  This keeps invalid raw
        # pixels out of the active input vector and makes the mask policy
        # explicit and reproducible.
        keep = valid_pixels[panel_name].ravel()[sources].all(axis=1)
        rows = rows[keep]
        sources = sources[keep]
        weights = weights[keep]
        if len(sources):
            used_pixels[panel_name][np.unique(sources)] = True
            sources = sources + panel_offsets[panel_index]

        rows_by_panel.append(rows)
        sources_by_panel.append(sources.astype(source_dtype, copy=False))
        weights_by_panel.append(weights.astype(np.float64, copy=False))

    output_count = (eta_stop - eta_start) * spec.ntth
    if not any(len(x) for x in rows_by_panel):
        return (
            np.empty(0, dtype=np.uint32),
            np.empty((0, 4), dtype=source_dtype),
            np.empty((0, 4), dtype=np.float64),
            np.zeros(output_count, dtype=np.uint32),
        )

    rows = np.concatenate(rows_by_panel).astype(np.uint32, copy=False)
    sources = np.concatenate(sources_by_panel)
    weights = np.concatenate(weights_by_panel)
    order = np.argsort(rows, kind='stable')
    rows = rows[order]
    sources = sources[order]
    weights = weights[order]
    counts = np.bincount(rows, minlength=output_count).astype(np.uint32)
    return rows, sources, weights, counts


def _write_composite_adapter(
    root: h5py.Group,
    instrument: Any,
    panel_names: tuple[str, ...],
    panel_shapes: np.ndarray,
    active_pixel_indptr: np.ndarray,
    active_local_indices: np.ndarray,
) -> None:
    panels = [instrument.detectors[name] for name in panel_names]
    if any(panel.roi is None for panel in panels):
        return

    frame_rows = max(int(panel.roi[0][1]) for panel in panels)
    frame_columns = max(int(panel.roi[1][1]) for panel in panels)
    frame_shape = (frame_rows, frame_columns)
    frame_indices = np.empty(
        len(active_local_indices), dtype=_integer_dtype(frame_rows * frame_columns)
    )

    for panel_index, panel in enumerate(panels):
        start = int(active_pixel_indptr[panel_index])
        stop = int(active_pixel_indptr[panel_index + 1])
        local_rows, local_columns = np.divmod(
            active_local_indices[start:stop], int(panel_shapes[panel_index, 1])
        )
        row_start = int(panel.roi[0][0])
        column_start = int(panel.roi[1][0])
        frame_indices[start:stop] = (
            (row_start + local_rows) * frame_columns + column_start + local_columns
        )

    adapters = root.create_group('input_adapters')
    composite = adapters.create_group('composite', track_order=True)
    composite.attrs['kind'] = 'flat-index-v1'
    composite.attrs['frame_order'] = 'C'
    composite.create_dataset('frame_shape', data=frame_shape)
    _write_array(composite, 'active_frame_indices', frame_indices)


def _write_operator(
    stream: h5py.File,
    instrument: Any,
    instrument_config: Mapping[str, Any],
    symbolic_buffers: Mapping[str, str],
    spec: CakeOperatorSpec,
    eta_chunk_rows: int,
    metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
    update_progress: Callable[[int], None] | None,
) -> CakeOperatorSummary:
    panel_names = tuple(instrument.detectors)
    if not panel_names:
        raise ValueError('the instrument has no detector panels')
    panel_shapes = np.asarray(
        [instrument.detectors[name].shape for name in panel_names], dtype=np.uint32
    )
    panel_sizes = np.prod(panel_shapes, axis=1, dtype=np.uint64)
    panel_offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(panel_sizes))
    )
    full_pixel_count = int(panel_offsets[-1])
    source_dtype = _integer_dtype(full_pixel_count)
    max_group_count = spec.size * len(panel_names)
    indptr_dtype = _integer_dtype(max_group_count)

    valid_pixels = {
        name: np.asarray(
            panel_buffer_as_2d_array(instrument.detectors[name]), dtype=bool
        )
        for name in panel_names
    }
    used_pixels = {
        name: np.zeros(int(panel_sizes[i]), dtype=bool)
        for i, name in enumerate(panel_names)
    }

    root = stream.create_group('cake_operator', track_order=True)
    root.attrs['format'] = FORMAT_NAME
    root.attrs['format_version_major'] = FORMAT_VERSION_MAJOR
    root.attrs['format_version_minor'] = FORMAT_VERSION_MINOR
    root.attrs['schema_status'] = SCHEMA_STATUS
    root.attrs['representation'] = REPRESENTATION
    root.attrs['hexrd_version'] = _package_version('hexrd')
    root.attrs['array_order'] = 'eta,tth (C order)'
    root.attrs['compression'] = f'{COMPRESSION}-{COMPRESSION_LEVEL}'
    root.attrs['required_features_json'] = json.dumps(['core-v2', REPRESENTATION])
    optional_features = []
    if all(instrument.detectors[name].roi is not None for name in panel_names):
        optional_features.append('composite-flat-index-v1')
    if symbolic_buffers:
        optional_features.append('symbolic-panel-buffer-v1')
    root.attrs['optional_features_json'] = json.dumps(optional_features)
    root.attrs['instrument_sha256'] = _mapping_digest(instrument_config)
    root.attrs['mask_policy'] = 'drop-four-corner-group-if-any-source-invalid'
    root.attrs['weight_precision'] = 'float64'
    for key, value in metadata.items():
        if str(key) in root.attrs:
            raise ValueError(f'reserved metadata key: {key}')
        root.attrs[str(key)] = _attribute_value(value)

    root.create_dataset(
        'README',
        data=(
            'Self-contained persistent HEXRD cake operator. Unknown optional '
            'features may be ignored; unknown required features must be rejected.'
        ),
        dtype=STRING_DTYPE,
    )

    instrument_group = root.create_group('instrument', track_order=True)
    _write_mapping(instrument_group, instrument_config)
    if instrument.active_beam_name is not None:
        instrument_group.attrs['active_beam'] = instrument.active_beam_name
    if symbolic_buffers:
        instrument_group.attrs['symbolic_panel_buffers_json'] = json.dumps(
            symbolic_buffers, sort_keys=True
        )

    grid = root.create_group('grid', track_order=True)
    grid.attrs['angle_units'] = 'degrees'
    grid.attrs['array_order'] = 'eta,tth (C order)'
    grid.attrs['requested_tth_range'] = spec.tth_range
    grid.attrs['requested_eta_range'] = spec.eta_range
    grid.attrs['pixel_size'] = spec.pixel_size
    _write_array(grid, 'tth_edges', spec.tth_edges)
    _write_array(grid, 'eta_edges', spec.eta_edges)
    grid.create_dataset('shape', data=spec.shape)

    transform = root.create_group('operator', track_order=True)
    transform.attrs['semantics'] = (
        'For output row r and groups g in [output_indptr[r], '
        'output_indptr[r+1]), accumulate sum(weights[g, :] * '
        'active_pixels[source_indices[g, :]]).'
    )
    transform.attrs['output_size'] = spec.size
    transform.attrs['weight_dtype'] = 'float64'
    output_indptr = _create_fixed_array(
        transform, 'output_indptr', (spec.size + 1,), indptr_dtype
    )
    output_indptr[0] = 0
    source_indices = _create_appendable_records(
        transform, 'source_indices', 4, source_dtype
    )
    weights = _create_appendable_records(transform, 'weights', 4, np.dtype(np.float64))

    group_count = 0
    eta_chunk_rows = min(spec.neta, eta_chunk_rows)
    for eta_start in range(0, spec.neta, eta_chunk_rows):
        eta_stop = min(spec.neta, eta_start + eta_chunk_rows)
        rows, chunk_sources, chunk_weights, counts = _operator_records_for_chunk(
            instrument,
            spec,
            eta_start,
            eta_stop,
            panel_names,
            panel_offsets,
            valid_pixels,
            used_pixels,
            source_dtype,
        )
        del rows
        _append_records(source_indices, chunk_sources)
        _append_records(weights, chunk_weights)

        cumulative = np.cumsum(counts, dtype=indptr_dtype) + group_count
        output_start = eta_start * spec.ntth
        output_stop = eta_stop * spec.ntth
        output_indptr[output_start + 1 : output_stop + 1] = cumulative
        group_count += len(chunk_sources)
        if update_progress is not None:
            update_progress(int(80 * eta_stop / spec.neta))

    active_indptr = [0]
    active_local_parts = []
    active_global_parts = []
    for panel_index, panel_name in enumerate(panel_names):
        local = np.flatnonzero(used_pixels[panel_name]).astype(np.uint32)
        active_local_parts.append(local)
        active_global_parts.append(local.astype(np.uint64) + panel_offsets[panel_index])
        active_indptr.append(active_indptr[-1] + len(local))
    active_local_indices = np.concatenate(active_local_parts)
    active_global_indices = np.concatenate(active_global_parts)
    # Keep this tiny panel-boundary array uint64 for compatibility with the
    # v1 CakeOperator loader. The large per-record arrays still use the
    # narrowest safe integer dtype.
    active_pixel_indptr = np.asarray(active_indptr, dtype=np.uint64)

    # Convert streamed full-panel source indices into the compact active input
    # vector in place, without retaining the complete transform in memory.
    record_rows = max(1, CHUNK_TARGET_BYTES // (4 * source_indices.dtype.itemsize))
    for start in range(0, group_count, record_rows):
        stop = min(group_count, start + record_rows)
        full_sources = source_indices[start:stop]
        compact_sources = np.searchsorted(active_global_indices, full_sources)
        if not np.array_equal(active_global_indices[compact_sources], full_sources):
            raise RuntimeError('failed to compact cake operator source indices')
        source_indices[start:stop] = compact_sources.astype(source_dtype, copy=False)
        if update_progress is not None:
            update_progress(80 + int(15 * stop / max(1, group_count)))

    inputs = root.create_group('input', track_order=True)
    inputs.attrs['indexing'] = 'zero-based, panel-local flattened C order'
    inputs.attrs['active_pixel_count'] = len(active_local_indices)
    inputs.attrs['full_panel_pixel_count'] = full_pixel_count
    inputs.create_dataset(
        'panel_names',
        data=np.asarray(panel_names, dtype=object),
        dtype=STRING_DTYPE,
    )
    _write_array(inputs, 'panel_shapes', panel_shapes)
    _write_array(inputs, 'active_pixel_indptr', active_pixel_indptr)
    _write_array(inputs, 'active_local_indices', active_local_indices)

    _write_composite_adapter(
        root,
        instrument,
        panel_names,
        panel_shapes,
        active_pixel_indptr,
        active_local_indices,
    )

    transform.attrs['group_count'] = group_count
    transform.attrs['term_count'] = group_count * 4
    transform.attrs['source_index_dtype'] = str(source_indices.dtype)
    provenance_group = root.create_group('provenance', track_order=True)
    provenance_group.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
    provenance_group.attrs['created_by'] = 'HEXRD'
    provenance_group.attrs['eta_build_chunk_rows'] = eta_chunk_rows
    provenance_group.attrs['postprocessing_included'] = (
        'geometry-and-static-panel-buffers-only'
    )
    for key, value in provenance.items():
        provenance_group.attrs[str(key)] = _attribute_value(value)

    root.attrs['payload_sha256'] = _payload_digest(root)
    stream.flush()
    if update_progress is not None:
        update_progress(100)

    return CakeOperatorSummary(
        path=Path(stream.filename),
        grid_shape=spec.shape,
        panel_count=len(panel_names),
        active_pixel_count=len(active_local_indices),
        full_panel_pixel_count=full_pixel_count,
        group_count=group_count,
        file_size=0,
        symbolic_panel_buffers=dict(symbolic_buffers),
    )


def export_cake_operator(
    path: str | Path,
    instrument: Any,
    spec: CakeOperatorSpec,
    *,
    eta_chunk_rows: int = DEFAULT_ETA_CHUNK_ROWS,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    update_progress: Callable[[int], None] | None = None,
) -> CakeOperatorSummary:
    """Atomically write a self-contained, compressed ``.hexrdcake`` file."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.hexrdcake':
        raise ValueError('cake operator paths must end in .hexrdcake')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if eta_chunk_rows < 1:
        raise ValueError('eta_chunk_rows must be positive')
    path.parent.mkdir(parents=True, exist_ok=True)

    instrument_config, symbolic_buffers = _normalized_instrument_config(instrument)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with h5py.File(temporary_path, 'w', libver='latest') as stream:
            summary = _write_operator(
                stream,
                instrument,
                instrument_config,
                symbolic_buffers,
                spec,
                eta_chunk_rows,
                metadata or {},
                provenance or {},
                update_progress,
            )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return CakeOperatorSummary(
        path=path,
        grid_shape=summary.grid_shape,
        panel_count=summary.panel_count,
        active_pixel_count=summary.active_pixel_count,
        full_panel_pixel_count=summary.full_panel_pixel_count,
        group_count=summary.group_count,
        file_size=path.stat().st_size,
        symbolic_panel_buffers=summary.symbolic_panel_buffers,
    )


def verify_cake_operator(path: str | Path) -> None:
    """Validate the required schema and semantic payload digest."""
    path = Path(path)
    with h5py.File(path, 'r') as stream:
        root = stream['cake_operator']
        if root.attrs['format'] != FORMAT_NAME:
            raise ValueError(f'{path} is not a HEXRD cake operator')
        if int(root.attrs['format_version_major']) not in {1, FORMAT_VERSION_MAJOR}:
            raise ValueError('unsupported .hexrdcake major format version')
        required = set(json.loads(root.attrs['required_features_json']))
        understood = {'core-v1', 'core-v2', REPRESENTATION}
        unknown = required - understood
        if unknown:
            raise ValueError(f'unknown required features: {sorted(unknown)}')
        expected = root.attrs['payload_sha256']
        actual = _payload_digest(root)
        if actual != expected:
            raise ValueError(f'payload digest mismatch: {actual} != {expected}')

        instrument_config: dict[str, Any] = {}
        unwrap_h5_to_dict(root['instrument'], instrument_config)
        expected = root.attrs['instrument_sha256']
        actual = _mapping_digest(instrument_config)
        if actual != expected:
            raise ValueError(f'instrument digest mismatch: {actual} != {expected}')
