"""Self-contained CSR cake matrices with instrument-defined input layouts.

Geometry generation uses the bounded-memory bilinear builder before
streaming into CSR.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from ._builder import CakeOperatorSummary, export_cake_operator
from .operator import _image_array

FORMAT = 'HEXRD cake matrix'
VERSION = '1'
LEGACY_VERSION = 'draft-csr-1'
STORAGE = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)


def _write_array(group, name, array):
    array = np.asarray(array)
    options = STORAGE if array.nbytes >= 4096 else {}
    return group.create_dataset(name, data=array, **options)


def _integer_array(value, name, ndim=1):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in 'iu' or np.any(array < 0):
        raise ValueError(f'{name} must contain nonnegative integers with ndim={ndim}')
    return array


def _grid(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite [start, step, count]')
    start, step, count = value
    if step <= 0 or count < 1 or count != int(count):
        raise ValueError(f'{name} requires a positive step and integer count')
    return float(start), float(step), int(count)


def _composite_shape(shapes, rois):
    if len(shapes) == 1 and rois[0] is None:
        return shapes[0]
    if any(roi is None for roi in rois):
        raise ValueError('composite inputs require an ROI for every panel')
    return tuple(
        max(roi[i] + shape[i] for shape, roi in zip(shapes, rois)) for i in (0, 1)
    )


def _write_csr_from_blocks(
    source, destination, spec, update_progress=None, input_layout='auto'
):
    """Stream full-image columns; retain original contribution summation order."""
    with h5py.File(source) as old, h5py.File(destination, 'w') as f:
        root = old['cake_operator']
        names = root['input/panel_names'].asstr()[:].tolist()
        offsets = root['input/active_pixel_indptr'][:]
        local = root['input/active_local_indices'][:]
        panels = root['instrument/detectors']
        shapes, rois = [], []
        for name in names:
            pixels = panels[name]['pixels']
            shapes.append(
                tuple(
                    int(np.asarray(pixels[k][()]).item()) for k in ('rows', 'columns')
                )
            )
            rois.append(tuple(pixels['roi'][:]) if 'roi' in pixels else None)
        groups = [
            np.asarray(panels[name]['group'][()]).item()
            if 'group' in panels[name]
            else None
            for name in names
        ]
        overlapping_rois = any(
            a is not None
            and b is not None
            and a[0] < b[0] + sb[0]
            and b[0] < a[0] + sa[0]
            and a[1] < b[1] + sb[1]
            and b[1] < a[1] + sa[1]
            for i, (sa, a) in enumerate(zip(shapes, rois))
            for sb, b in zip(shapes[i + 1 :], rois[i + 1 :])
        )
        if input_layout == 'auto':
            input_layout = (
                'composite'
                if len(names) == 1
                or (
                    all(roi is not None for roi in rois)
                    and len(set(groups)) == 1
                    and not overlapping_rois
                )
                else 'panels'
            )
        if input_layout not in ('composite', 'panels'):
            raise ValueError('input_layout must be auto, composite, or panels')
        if input_layout == 'composite':
            frame_shape = _composite_shape(shapes, rois)
            input_size = int(frame_shape[0]) * int(frame_shape[1])
        else:
            input_size = sum(rows * cols for rows, cols in shapes)
        column_map = np.empty(len(local), dtype=np.int64)
        panel_start = 0
        for i in sorted(range(len(names)), key=lambda i: names[i]):
            first, last = offsets[i : i + 2]
            if input_layout == 'composite':
                r0, c0 = rois[i] if rois[i] is not None else (0, 0)
                rows, cols = np.divmod(local[first:last], shapes[i][1])
                column_map[first:last] = (r0 + rows) * frame_shape[1] + c0 + cols
            else:
                column_map[first:last] = panel_start + local[first:last]
                panel_start += shapes[i][0] * shapes[i][1]
        f.attrs['format'] = FORMAT
        f.attrs['version'] = VERSION
        old.copy(root['instrument'], f, name='instrument')
        grid = f.create_group('grid')
        grid['tth'] = [spec.tth_range[0], spec.pixel_size[0], spec.ntth]
        grid['eta'] = [spec.eta_range[0], spec.pixel_size[1], spec.neta]
        matrix = f.create_group('matrix')
        matrix.attrs['format'] = 'csr'
        matrix.attrs['input_layout'] = input_layout
        matrix.attrs['shape'] = (spec.size, input_size)
        group_count = root['operator/weights'].shape[0]
        terms = 4 * group_count
        index_dtype = (
            np.int32 if max(terms, spec.size, input_size) < 2**31 else np.int64
        )
        datasets = {}
        for name, count, dtype in (
            ('data', terms, np.float64),
            ('indices', terms, index_dtype),
            ('indptr', spec.size + 1, index_dtype),
        ):
            options = STORAGE if count * np.dtype(dtype).itemsize >= 4096 else {}
            datasets[name] = matrix.create_dataset(
                name, shape=(count,), dtype=dtype, **options
            )
        for first in range(0, spec.size + 1, 262144):
            last = min(spec.size + 1, first + 262144)
            datasets['indptr'][first:last] = (
                root['operator/output_indptr'][first:last].astype(np.int64) * 4
            )
        for first in range(0, group_count, 32768):
            last = min(group_count, first + 32768)
            datasets['data'][4 * first : 4 * last] = root['operator/weights'][
                first:last
            ].ravel()
            datasets['indices'][4 * first : 4 * last] = column_map[
                root['operator/source_indices'][first:last].ravel()
            ]
            if update_progress:
                update_progress(85 + int(14 * last / group_count))


def export_cake_matrix(
    path,
    instrument,
    spec,
    *,
    eta_chunk_rows=64,
    overwrite=False,
    update_progress=None,
    input_layout="auto",
):
    """Atomically export the simple CSR format from an instrument and polar grid.

    Composite columns address the full instrument-defined image in C order.
    Separate-panel columns concatenate complete panels in sorted name order.
    Auto selects composite for a single panel or disjoint ROIs in one group;
    use input_layout to explicitly select composite or panels. Contributions
    retain the instrument's original order. Raises from the progress callback
    cancel the operation without replacing the destination.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.hexrdcake':
        raise ValueError('cake matrix paths must end in .hexrdcake')
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f'.{path.name}.', dir=path.parent
    ) as temporary:
        temporary = Path(temporary)
        blocks = temporary / 'blocks.hexrdcake'
        progress = (
            (lambda value: update_progress(int(0.85 * value)))
            if update_progress
            else None
        )
        summary = export_cake_operator(
            blocks,
            instrument,
            spec,
            eta_chunk_rows=eta_chunk_rows,
            update_progress=progress,
        )
        result = temporary / 'matrix.hexrdcake'
        _write_csr_from_blocks(blocks, result, spec, update_progress, input_layout)
        if update_progress:
            update_progress(100)
        os.replace(result, path)
    return CakeOperatorSummary(
        path=path,
        grid_shape=spec.shape,
        panel_count=summary.panel_count,
        active_pixel_count=summary.active_pixel_count,
        full_panel_pixel_count=summary.full_panel_pixel_count,
        group_count=summary.group_count,
        file_size=path.stat().st_size,
        symbolic_panel_buffers=summary.symbolic_panel_buffers,
    )


class CakeMatrix:
    """Loaded CSR matrix with its input mapping and uniform polar grid.

    Geometry is fixed by the archive. Dynamic NaNs propagate through sparse
    multiplication. Profile reduction and image/omega processing belong to
    the caller. Composite files support ``matrix @ frame.ravel()`` directly.
    """

    @classmethod
    def load(cls, path):
        result = cls()
        result.path = Path(path).expanduser().resolve()
        result._instrument = None
        result._frame_shape = None
        result._frame_indices = None
        result._composite_matrix = None
        with h5py.File(result.path) as f:
            if f.attrs.get('format') != FORMAT or f.attrs.get('version') not in (
                VERSION,
                'draft-csr-2',
                LEGACY_VERSION,
            ):
                raise ValueError('unsupported cake matrix format or version')
            result.tth = _grid(f['grid/tth'][:], 'tth')
            result.eta = _grid(f['grid/eta'][:], 'eta')
            result.shape = (result.eta[2], result.tth[2])
            panels = f['instrument/detectors']
            result.panel_names = tuple(sorted(panels))
            if not result.panel_names:
                raise ValueError('embedded instrument has no panels')
            shapes, rois = [], []
            for name in result.panel_names:
                pixels = panels[name]['pixels']
                dimensions = [pixels[key][()].item() for key in ('rows', 'columns')]
                dimensions = _integer_array(dimensions, 'panel shape')
                if np.any(dimensions == 0):
                    raise ValueError('panel dimensions must be positive')
                shapes.append(tuple(int(x) for x in dimensions))
                roi = pixels['roi'][:] if 'roi' in pixels else None
                if roi is not None:
                    roi = _integer_array(roi, 'ROI')
                    if roi.shape != (2,):
                        raise ValueError('ROI must contain [row_start, column_start]')
                    roi = tuple(int(x) for x in roi)
                rois.append(roi)
            result.panel_shapes = tuple(shapes)
            result.rois = tuple(rois)
            m = f['matrix']
            result.input_layout = (
                'compact'
                if f.attrs['version'] == LEGACY_VERSION
                else m.attrs.get('input_layout')
            )
            result.composite_shape = None
            if result.input_layout == 'compact':
                offsets = _integer_array(f['input/panel_offsets'][:], 'panel_offsets')
                local = _integer_array(f['input/pixel_indices'][:], 'pixel_indices')
                if (
                    offsets.shape != (len(shapes) + 1,)
                    or offsets[0] != 0
                    or offsets[-1] != len(local)
                    or np.any(offsets[1:] < offsets[:-1])
                ):
                    raise ValueError(
                        'panel_offsets must span active pixels in panel order'
                    )
                for i, (rows, columns) in enumerate(shapes):
                    segment = local[offsets[i] : offsets[i + 1]]
                    if np.any(segment >= rows * columns) or np.any(
                        segment[1:] <= segment[:-1]
                    ):
                        raise ValueError(
                            'pixel_indices must be sorted, unique, and inside each panel'
                        )
                result.pixel_indices = local
                input_size = len(local)
            elif result.input_layout in ('composite', 'panels'):
                offsets = np.concatenate(([0], np.cumsum([r * c for r, c in shapes])))
                result.pixel_indices = None
                if result.input_layout == 'composite':
                    result.composite_shape = _composite_shape(shapes, rois)
                    input_size = result.composite_shape[0] * result.composite_shape[1]
                else:
                    input_size = int(offsets[-1])
            else:
                raise ValueError('unknown matrix input_layout')
            result.panel_offsets = offsets
            shape = _integer_array(m.attrs['shape'], 'matrix shape')
            if m.attrs.get('format') != 'csr' or tuple(shape) != (
                np.prod(result.shape),
                input_size,
            ):
                raise ValueError(
                    'CSR shape must match the grid and instrument input layout'
                )
            data = m['data'][:]
            indices = _integer_array(m['indices'][:], 'matrix indices')
            indptr = _integer_array(m['indptr'][:], 'matrix indptr')
            if (
                data.ndim != 1
                or data.dtype.kind not in 'fiu'
                or not np.isfinite(data).all()
            ):
                raise ValueError('matrix data must be finite real coefficients')
            if len(indices) != len(data) or np.any(indices >= input_size):
                raise ValueError('matrix indices must address valid inputs')
            if (
                indptr.shape != (int(shape[0]) + 1,)
                or indptr[0] != 0
                or indptr[-1] != len(data)
                or np.any(indptr[1:] < indptr[:-1])
            ):
                raise ValueError('matrix indptr must span coefficients monotonically')
            result.matrix = csr_matrix((data, indices, indptr), shape=tuple(shape))
        result.mask = (np.diff(result.matrix.indptr) == 0).reshape(result.shape)
        return result

    @property
    def instrument(self):
        """Construct the embedded instrument lazily; caking needs no geometry rebuild."""
        if self._instrument is None:
            from hexrd.core.instrument import HEDMInstrument
            from hexrd.core.utils.hdf5 import unwrap_h5_to_dict

            config = {}
            with h5py.File(self.path) as f:
                unwrap_h5_to_dict(f['instrument'], config)
                active_beam = f['instrument'].attrs.get('active_beam')
            self._instrument = HEDMInstrument(config)
            if active_beam is not None:
                self._instrument.active_beam_name = active_beam
        return self._instrument

    @property
    def tth_centers(self):
        start, step, count = self.tth
        return start + step * (np.arange(count) + 0.5)

    @property
    def eta_centers(self):
        start, step, count = self.eta
        return start + step * (np.arange(count) + 0.5)

    def gather_active_pixels(self, images):
        """Assemble panel inputs; legacy compact files still gather active pixels.

        Composite archives expect panels from the same full detector image.
        Batch composite callers should use apply_composite to avoid assembly.
        """
        if set(images) != set(self.panel_names):
            raise ValueError('input panel names must match the embedded instrument')
        arrays = [_image_array(images[name]) for name in self.panel_names]
        for name, array, shape in zip(self.panel_names, arrays, self.panel_shapes):
            if array.shape != shape:
                raise ValueError(f'panel {name} has the wrong shape')
        dtype = np.result_type(*[a.dtype for a in arrays])
        if self.input_layout == 'composite':
            frame = np.zeros(self.composite_shape, dtype=dtype)
            for array, roi in zip(arrays, self.rois):
                r0, c0 = roi if roi is not None else (0, 0)
                frame[r0 : r0 + array.shape[0], c0 : c0 + array.shape[1]] = array
            return frame.ravel()
        if self.input_layout == 'panels':
            return np.concatenate([a.ravel() for a in arrays])
        active = np.empty(len(self.pixel_indices), dtype=dtype)
        for i, array in enumerate(arrays):
            first, last = self.panel_offsets[i : i + 2]
            active[first:last] = array.flat[self.pixel_indices[first:last]]
        return active

    def _composite_indices(self, frame_shape):
        dimensions = _integer_array(frame_shape, 'composite frame shape')
        if dimensions.shape != (2,) or np.any(dimensions == 0):
            raise ValueError('composite frame shape must contain two positive integers')
        frame_shape = tuple(int(x) for x in dimensions)
        if frame_shape != self._frame_shape:
            raw_indices = np.empty(len(self.pixel_indices), dtype=np.intp)
            for i, (shape, roi) in enumerate(zip(self.panel_shapes, self.rois)):
                if roi is None:
                    raise ValueError('composite inputs require an ROI for every panel')
                r0, c0 = roi
                if r0 + shape[0] > frame_shape[0] or c0 + shape[1] > frame_shape[1]:
                    raise ValueError('composite frame is too small for the panel ROIs')
                first, last = self.panel_offsets[i : i + 2]
                rows, cols = np.divmod(self.pixel_indices[first:last], shape[1])
                raw_indices[first:last] = (r0 + rows) * frame_shape[1] + c0 + cols
            self._frame_indices = raw_indices
            self._frame_shape = frame_shape
            self._composite_matrix = None
        return self._frame_indices

    def matrix_for_composite(self, frame_shape):
        """Prepare ordinary CSR for ``C @ frame.ravel()`` without pixel gathering.

        New composite archives already have full-image columns: this validates
        the instrument-defined shape and returns self.matrix without copying.
        Legacy compact archives are remapped once and cached by frame shape;
        prepare those before starting worker threads. Their remapped column
        indices require another array, while weights/row pointers are shared.

        The returned matrix expects the same shape and orientation on every
        call. A contiguous frame's ravel is a view; other layouts may copy.
        """
        if self.input_layout == 'composite':
            dimensions = _integer_array(frame_shape, 'composite frame shape')
            if tuple(dimensions) != self.composite_shape:
                raise ValueError(
                    f'composite frame shape must be {self.composite_shape}'
                )
            return self.matrix
        if self.input_layout == 'panels':
            raise ValueError('matrix expects separate panel inputs')
        raw_indices = self._composite_indices(frame_shape)
        if self._composite_matrix is None:
            matrix = self.matrix
            rows, columns = self._frame_shape
            size = rows * columns
            dtype = np.int64 if max(size, matrix.nnz) >= 2**31 else np.int32
            indices = np.empty(matrix.nnz, dtype=dtype)
            for first in range(0, matrix.nnz, 1048576):
                last = min(first + 1048576, matrix.nnz)
                indices[first:last] = raw_indices[matrix.indices[first:last]]
            self._composite_matrix = csr_matrix(
                (matrix.data, indices, matrix.indptr),
                shape=(matrix.shape[0], size),
                copy=False,
            )
        return self._composite_matrix

    def gather_active_composite_pixels(self, frame):
        """Legacy input adapter; new composite files only flatten the full frame."""
        frame = _image_array(frame)
        if self.input_layout != 'compact':
            self.matrix_for_composite(frame.shape)
            return frame.ravel()
        self._composite_indices(frame.shape)
        return frame.flat[self._frame_indices]

    def apply_active_pixels(self, pixels):
        pixels = _image_array(pixels)
        if pixels.ndim not in (1, 2) or pixels.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                'active pixels must have shape (pixels,) or (pixels, frames)'
            )
        shape = self.shape + pixels.shape[1:]
        values = (self.matrix @ pixels).reshape(shape)
        mask = (
            self.mask
            if pixels.ndim == 1
            else np.broadcast_to(self.mask[..., None], shape)
        )
        return np.ma.array(values, mask=mask)

    def apply(self, images):
        return self.apply_active_pixels(self.gather_active_pixels(images))

    def apply_composite(self, frame):
        """Apply to a full detector image, preparing its column layout once."""
        frame = _image_array(frame)
        matrix = self.matrix_for_composite(frame.shape)
        values = (matrix @ frame.ravel()).reshape(self.shape)
        return np.ma.array(values, mask=self.mask)


def verify_cake_matrix(path):
    """Validate the archive and CSR bounds; the simple format has no stored hashes."""
    CakeMatrix.load(path)
