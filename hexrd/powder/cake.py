"""Persistent CSR matrices for bilinear polar caking"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.projections.polar import PolarView
from hexrd.core.utils.panel_buffer import panel_buffer_as_2d_array

# Saved matrix format and compression settings
FORMAT = 'HEXRD cake matrix'
VERSION = 'hexrd-cake-matrix-1'
STORAGE = {'compression': 'gzip', 'compression_opts': 1, 'shuffle': True, 'fletcher32': True}


def _input_layout(
    instrument: HEDMInstrument, layout: str = 'auto'
) -> tuple[str, tuple[int, ...], dict[str, tuple[int, int]]]:
    """Resolve the matrix input shared by writer and reader.

    Composite inputs are one C-order raw frame with each panel at its ROI.
    Independent panels are complete images concatenated in sorted name order.
    Returns the layout, the input shape, and each panel's ``(start, stride)``:
    panel pixel (i, j) is matrix column ``start + i * stride + j``.
    """
    panels = instrument.detectors
    if not panels:
        raise ValueError('instrument has no panels')

    rois = [p.roi for p in panels.values()]
    if layout in ('auto', 'composite'):
        overlap = any(
            a is not None and
            b is not None and
            all(a[k][0] < b[k][1] and b[k][0] < a[k][1] for k in (0, 1))
            for a, b in combinations(rois, 2)
        )
        composite = len(panels) == 1 or (
            None not in rois and len({p.group for p in panels.values()}) == 1
        )
        if layout == 'auto':
            layout = 'composite' if composite and not overlap else 'panels'
        elif not composite or overlap:
            msg = 'composite inputs require nonoverlapping ROIs in one detector group'
            raise ValueError(msg)

    if layout == 'composite':
        origins = {
            name: (panel.roi[0][0], panel.roi[1][0]) if panel.roi else (0, 0)
            for name, panel in panels.items()
        }

        shape = tuple(
            max(origins[name][k] + panel.shape[k] for name, panel in panels.items())
            for k in (0, 1)
        )
        columns = {n: (r * shape[1] + c, shape[1]) for n, (r, c) in origins.items()}
    elif layout == 'panels':
        columns = {}
        size = 0
        for name in sorted(panels):
            columns[name] = (size, panels[name].cols)
            size += panels[name].rows * panels[name].cols
        shape = (size,)
    else:
        raise ValueError('input_layout must be auto, composite, or panels')

    return layout, shape, columns


class CakeMatrix:
    """Loaded CSR matrix with its input layout and uniform polar grid.

    Geometry is fixed by the file. Runtime NaNs propagate through the sparse
    product; empty rows are masked. Composite matrices multiply a full raw
    frame directly: ``matrix @ frame.ravel()``. Its shape must exactly match
    the extent of the instrument's panel ROIs, including their offsets.
    Larger raw frames with unused trailing rows or columns are not accepted.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

        with h5py.File(self.path) as rf:
            if rf.attrs.get('format') != FORMAT or rf.attrs.get('version') not in (
                VERSION,
                '1',
                'draft-csr-2',
            ):
                raise ValueError('unsupported cake matrix format or version')

            grids = [rf['grid/tth'][()], rf['grid/eta'][()]]
            for start, step, count in grids:
                if (
                    not np.isfinite([start, step, count]).all() or
                    step <= 0 or
                    count < 1 or
                    count != int(count)
                ):
                    msg = 'grid requires finite bounds, positive steps and integer counts'
                    raise ValueError(msg)

            self.tth, self.eta = (
                (float(start), float(step), int(count)) for start, step, count in grids
            )
            self.shape = (self.eta[2], self.tth[2])

            self.instrument = HEDMInstrument(rf['instrument'])
            matrix = rf['matrix']
            self.input_layout, self.input_shape, self.columns = _input_layout(
                self.instrument, matrix.attrs['input_layout']
            )

            shape = (int(np.prod(self.shape)), int(np.prod(self.input_shape)))
            if matrix.attrs['format'] != 'csr' or tuple(matrix.attrs['shape']) != shape:
                raise ValueError('CSR shape must match the grid and input layout')

            data, indices, indptr = (
                matrix[name][:] for name in ('data', 'indices', 'indptr')
            )

            # Check before SciPy can cast indices or trim trailing entries
            if (
                indices.dtype.kind not in 'iu' or
                indptr.dtype.kind not in 'iu' or
                not np.isfinite(data).all() or
                indptr.size == 0 or
                indptr[-1] != len(data)
            ):
                msg = 'CSR arrays require integer indices, finite weights, and full pointers'
                raise ValueError(msg)
            self.matrix = csr_matrix((data, indices, indptr), shape=shape)

        self.matrix.check_format(full_check=True)
        self.mask = (np.diff(self.matrix.indptr) == 0).reshape(self.shape)

    @classmethod
    def load(cls, path: str | Path) -> CakeMatrix:
        return cls(path)

    @property
    def tth_centers(self) -> np.ndarray:
        start, step, count = self.tth
        return start + step * (np.arange(count) + 0.5)

    @property
    def eta_centers(self) -> np.ndarray:
        start, step, count = self.eta
        return start + step * (np.arange(count) + 0.5)

    def assemble(self, images: Mapping[str, np.ndarray]) -> np.ndarray:
        """Place complete panel images into the matrix input array"""
        if set(images) != set(self.columns):
            raise ValueError('input panel names must match the embedded instrument')

        for name, panel in self.instrument.detectors.items():
            if np.shape(images[name]) != panel.shape:
                msg = f'Panel {name} has the wrong shape'
                raise ValueError(msg)

        if self.input_layout == 'panels':
            return np.concatenate([np.ravel(images[name]) for name in sorted(images)])

        # Place each panel in its region of the full detector image
        pixels = np.zeros(self.input_shape, dtype=np.result_type(*images.values()))
        for name, (start, stride) in self.columns.items():
            row, col = divmod(start, stride)
            rows, cols = self.instrument.detectors[name].shape
            pixels[row:row + rows, col:col + cols] = images[name]

        return pixels

    def __getitem__(self, eta_rows: slice) -> CakeMatrix:
        """Return an eta view sharing weights, indices and instrument geometry"""
        first, last, step = eta_rows.indices(self.shape[0])
        if step != 1:
            raise ValueError('eta rows must be contiguous')

        last = max(first, last)
        result = copy(self)
        start, spacing, _ = self.eta
        result.eta = (start + first * spacing, spacing, last - first)
        result.shape = (last - first, self.shape[1])
        result.mask = self.mask[first:last]

        row_start, row_stop = first * self.shape[1], last * self.shape[1]
        entry_start, entry_stop = self.matrix.indptr[[row_start, row_stop]]
        entries = slice(entry_start, entry_stop)

        # SciPy's tuple constructor can copy small views even with copy=False
        result.matrix = csr_matrix((row_stop - row_start, self.matrix.shape[1]))
        result.matrix.data = self.matrix.data[entries]
        result.matrix.indices = self.matrix.indices[entries]
        result.matrix.indptr = (
            self.matrix.indptr[row_start : row_stop + 1] - entry_start
        )

        return result

    def apply(self, images: np.ndarray | Mapping[str, np.ndarray]) -> np.ma.MaskedArray:
        """Cake a complete raw frame or a mapping of complete panel images"""
        pixels = self.assemble(images) if isinstance(images, Mapping) else images
        if np.shape(pixels) != self.input_shape:
            msg = f'input shape must be {self.input_shape}'
            if self.input_layout == 'composite':
                msg += ', the composite extent defined by the instrument panel ROIs'
            raise ValueError(msg)

        values = self.matrix @ np.ravel(pixels)
        return np.ma.array(values.reshape(self.shape), mask=self.mask)


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
    eta_rows: slice,
    columns: Mapping[str, tuple[int, int]],
    valid_pixels: Mapping[str, np.ndarray],
    used_pixels: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the columns, weights, and row counts for a slice of eta rows"""
    mapping = view._generate_coordinate_mapping(eta_rows)
    rows = []
    sources = []
    weights = []

    for name, panel in view.detectors.items():
        interp = mapping[name]['bilinear_interp_dict']
        panel_sources = np.column_stack(
            [
                interp[i + '_img'] * panel.cols + interp[j + '_img']
                for i in ('i_floor', 'i_ceil')
                for j in ('j_floor', 'j_ceil')
            ]
        )

        # Exclude all four weights if any source pixel is buffered
        keep = valid_pixels[name].ravel()[panel_sources].all(axis=1)
        panel_sources = panel_sources[keep]
        used_pixels[name][panel_sources.ravel()] = True

        start, stride = columns[name]
        i, j = np.divmod(panel_sources, panel.cols)
        rows.append(mapping[name]['on_panel_idx'][keep])
        sources.append(start + i * stride + j)
        weights.append(
            np.column_stack([interp[k] for k in ('cc', 'fc', 'cf', 'ff')])[keep]
        )

    rows = np.concatenate(rows)
    order = np.argsort(rows, kind='stable')
    counts = np.bincount(rows, minlength=(eta_rows.stop - eta_rows.start) * view.ntth)
    return (
        np.concatenate(sources)[order].ravel(),
        np.concatenate(weights)[order].ravel(),
        4 * counts,
    )


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
    """Write a CSR cake matrix in bounded eta chunks, replacing atomically.

    Contributions retain instrument order, including duplicate columns and
    zero weights. Raising from update_progress cancels without replacing the
    destination.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.hexrdcake':
        raise ValueError('cake matrix paths must end in .hexrdcake')

    if path.exists() and not overwrite:
        raise FileExistsError(path)

    if (
        not np.isfinite([*tth_range, *eta_range, *pixel_size]).all() or
        tth_range[0] >= tth_range[1] or
        eta_range[0] >= eta_range[1] or
        min(pixel_size) <= 0
    ):
        msg = 'grid requires finite increasing ranges and positive pixel sizes'
        raise ValueError(msg)

    if eta_chunk_rows <= 0:
        raise ValueError('eta_chunk_rows must be positive')

    view = PolarView(
        tth_range,
        instrument,
        eta_min=eta_range[0],
        eta_max=eta_range[1],
        pixel_size=pixel_size,
    )
    if min(view.shape) < 1:
        raise ValueError('grid requires at least one bin on each axis')

    output_size = view.neta * view.ntth
    panels = instrument.detectors
    input_layout, input_shape, columns = _input_layout(instrument, input_layout)
    input_size = int(np.prod(input_shape))
    index_dtype = np.int32 if max(input_size, output_size) < 2**31 else np.int64

    valid = {name: panel_buffer_as_2d_array(p) for name, p in panels.items()}
    used = {name: np.zeros(p.rows * p.cols, dtype=bool) for name, p in panels.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{path.name}.', dir=path.parent) as temporary:
        result = Path(temporary) / path.name
        with h5py.File(result, 'w') as wf:
            wf.attrs.update(format=FORMAT, version=VERSION)
            instrument.write_config(wf, style='hdf5', **STORAGE)
            wf['grid/tth'] = [tth_range[0], pixel_size[0], view.ntth]
            wf['grid/eta'] = [eta_range[0], pixel_size[1], view.neta]

            matrix = wf.create_group('matrix')
            matrix.attrs.update(
                format='csr', input_layout=input_layout, shape=(output_size, input_size)
            )
            data = matrix.create_dataset(
                'data', (0,), maxshape=(None,), dtype='f8', **STORAGE
            )
            indices = matrix.create_dataset(
                'indices', (0,), maxshape=(None,), dtype=index_dtype, **STORAGE
            )
            indptr = matrix.create_dataset(
                'indptr', (output_size + 1,), dtype='i8', **STORAGE
            )

            indptr[0] = 0
            for first in range(0, view.neta, eta_chunk_rows):
                last = min(first + eta_chunk_rows, view.neta)
                sources, weights, counts = _csr_chunk(
                    view, slice(first, last), columns, valid, used
                )

                nnz = len(data)
                indptr[first * view.ntth + 1 : last * view.ntth + 1] = nnz + np.cumsum(
                    counts
                )
                for dataset, values in ((data, weights), (indices, sources)):
                    dataset.resize((nnz + len(values),))
                    dataset[nnz:] = values

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
