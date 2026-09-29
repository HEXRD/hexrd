"""Build a persistent CSR cake matrix directly from bilinear interpolation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.projections.polar import PolarView
from hexrd.core.utils.hdf5 import unwrap_dict_to_h5
from hexrd.core.utils.panel_buffer import panel_buffer_as_2d_array

from .matrix import FORMAT, VERSION, _input_layout

STORAGE = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)


@dataclass(frozen=True)
class CakeOperatorSummary:
    path: Path
    grid_shape: tuple[int, int]
    panel_count: int
    active_pixel_count: int
    full_panel_pixel_count: int
    file_size: int


def _csr_chunk(
    view: PolarView,
    eta_start: int,
    eta_stop: int,
    panel_offsets: Mapping[str, int],
    valid_pixels: Mapping[str, np.ndarray],
    used_pixels: Mapping[str, np.ndarray],
    composite_shape: tuple[int, int] | None,
    source_dtype: type[np.integer],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one eta chunk and return output-major interpolation records."""
    mapping = view._generate_coordinate_mapping(slice(eta_start, eta_stop))

    rows_by_panel: list[np.ndarray] = []
    sources_by_panel: list[np.ndarray] = []
    weights_by_panel: list[np.ndarray] = []
    for panel_name, panel in view.detectors.items():
        panel_map = mapping[panel_name]
        interpolation = panel_map['bilinear_interp_dict']
        rows = panel_map['on_panel_idx']
        sources = np.column_stack(
            [
                interpolation[i + '_img'] * panel.cols + interpolation[j + '_img']
                for i in ('i_floor', 'i_ceil')
                for j in ('j_floor', 'j_ceil')
            ]
        )
        weights = np.column_stack([interpolation[k] for k in ('cc', 'fc', 'cf', 'ff')])

        # Exclude all four weights if any source pixel is buffered.
        keep = valid_pixels[panel_name].ravel()[sources].all(axis=1)
        rows = rows[keep]
        sources = sources[keep]
        weights = weights[keep]
        if len(sources):
            used_pixels[panel_name][np.unique(sources)] = True
            if composite_shape is None:
                sources = sources + panel_offsets[panel_name]
            else:
                r0, c0 = (panel.roi[0][0], panel.roi[1][0]) if panel.roi else (0, 0)
                rr, cc = np.divmod(sources, panel.cols)
                sources = (r0 + rr) * composite_shape[1] + c0 + cc

        rows_by_panel.append(rows)
        sources_by_panel.append(sources.astype(source_dtype, copy=False))
        weights_by_panel.append(weights.astype(np.float64, copy=False))

    output_count = (eta_stop - eta_start) * view.ntth
    rows = np.concatenate(rows_by_panel)
    sources = np.concatenate(sources_by_panel)
    weights = np.concatenate(weights_by_panel)
    order = np.argsort(rows, kind='stable')
    rows = rows[order]
    sources = sources[order]
    weights = weights[order]
    counts = np.bincount(rows, minlength=output_count)
    return sources.ravel(), weights.ravel(), 4 * counts


def export_cake_matrix(
    path: str | Path,
    instrument: HEDMInstrument,
    *,
    tth_range: tuple[float, float],
    eta_range: tuple[float, float],
    pixel_size: tuple[float, float],
    eta_chunk_rows: int = 64,
    overwrite: bool = False,
    update_progress: Callable[[int], None] | None = None,
    input_layout: str = 'auto',
) -> CakeOperatorSummary:
    """Write full-image CSR columns in bounded eta chunks, replacing atomically.

    Composite inputs use C-order raw image pixels. Independent panels use
    complete images concatenated in sorted panel-name order. Contributions
    retain instrument order, including duplicate columns and zero weights.
    Raising from update_progress cancels without replacing the destination.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.hexrdcake':
        raise ValueError('cake matrix paths must end in .hexrdcake')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if eta_chunk_rows < 1:
        raise ValueError('eta_chunk_rows must be positive')
    if (
        not np.isfinite((*tth_range, *eta_range, *pixel_size)).all()
        or min(pixel_size) <= 0
        or tth_range[0] >= tth_range[1]
        or eta_range[0] >= eta_range[1]
    ):
        raise ValueError(
            'grid requires finite increasing ranges and positive pixel sizes'
        )
    view = PolarView(
        tth_range,
        instrument,
        eta_min=eta_range[0],
        eta_max=eta_range[1],
        pixel_size=pixel_size,
        cache_coordinate_map=False,
    )
    if min(view.shape) < 1:
        raise ValueError('grid must contain at least one bin along each axis')
    output_size = view.neta * view.ntth
    panels = instrument.detectors
    input_layout, frame_shape, offsets, input_size = _input_layout(
        instrument, input_layout
    )
    source_dtype = np.int32 if max(input_size, output_size) < 2**31 else np.int64
    valid = {name: panel_buffer_as_2d_array(p) for name, p in panels.items()}
    used = {name: np.zeros(p.rows * p.cols, dtype=bool) for name, p in panels.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{path.name}.', dir=path.parent) as temporary:
        result = Path(temporary) / 'matrix.hexrdcake'
        with h5py.File(result, 'w') as f:
            f.attrs.update(format=FORMAT, version=VERSION)
            snapshot = f.create_group('instrument')
            config = deepcopy(instrument.write_config(style='hdf5'))
            # Store actual array buffers compressed, without trying to identify
            # equivalent named detector masks or changing the live instrument.
            buffers = {}
            for name, panel_config in config['detectors'].items():
                if np.ndim(panel_config.get('buffer')) == 2:
                    buffers[name] = panel_config.pop('buffer')
            unwrap_dict_to_h5(snapshot, config)
            for name, buffer in buffers.items():
                snapshot['detectors'][name].create_dataset(
                    'buffer', data=buffer, **STORAGE
                )
            if instrument.active_beam_name is not None:
                snapshot.attrs['active_beam'] = instrument.active_beam_name
            f['grid/tth'] = [tth_range[0], pixel_size[0], view.ntth]
            f['grid/eta'] = [eta_range[0], pixel_size[1], view.neta]
            matrix = f.create_group('matrix')
            matrix.attrs.update(
                format='csr', input_layout=input_layout, shape=(output_size, input_size)
            )
            data = matrix.create_dataset(
                'data', (0,), maxshape=(None,), dtype='f8', **STORAGE
            )
            indices = matrix.create_dataset(
                'indices', (0,), maxshape=(None,), dtype=source_dtype, **STORAGE
            )
            indptr = matrix.create_dataset(
                'indptr', (output_size + 1,), dtype='i8', **STORAGE
            )
            indptr[0] = 0
            nnz = 0
            for first in range(0, view.neta, eta_chunk_rows):
                last = min(first + eta_chunk_rows, view.neta)
                columns, weights, counts = _csr_chunk(
                    view,
                    first,
                    last,
                    offsets,
                    valid,
                    used,
                    frame_shape,
                    source_dtype,
                )
                end = nnz + len(weights)
                data.resize((end,))
                indices.resize((end,))
                data[nnz:end] = weights
                indices[nnz:end] = columns
                indptr[first * view.ntth + 1 : last * view.ntth + 1] = nnz + np.cumsum(
                    counts
                )
                nnz = end
                if update_progress is not None:
                    update_progress(int(100 * last / view.neta))
        result.replace(path)
    return CakeOperatorSummary(
        path,
        view.shape,
        len(panels),
        sum(int(a.sum()) for a in used.values()),
        sum(a.size for a in used.values()),
        path.stat().st_size,
    )
