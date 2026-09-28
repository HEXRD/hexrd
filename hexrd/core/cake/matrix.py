"""Simple, self-contained CSR cake matrices and compact detector inputs.

The provisional format is documented in CAKE_OPERATOR.md. Geometry generation
currently uses the bounded-memory bilinear builder before streaming into CSR.
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
VERSION = 'draft-csr-1'
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


def _write_csr_from_blocks(source, destination, spec, update_progress=None):
    """Convert in bounded chunks; retain original contribution summation order."""
    with h5py.File(source) as old, h5py.File(destination, 'w') as f:
        root = old['cake_operator']
        names = root['input/panel_names'].asstr()[:].tolist()
        old_offsets = root['input/active_pixel_indptr'][:]
        old_local = root['input/active_local_indices'][:]
        order = sorted(range(len(names)), key=lambda i: names[i])
        index_dtype = np.int32 if len(old_local) < 2**31 else np.int64
        permutation = np.concatenate(
            [
                np.arange(old_offsets[i], old_offsets[i + 1], dtype=index_dtype)
                for i in order
            ]
        )
        inverse = np.empty(len(old_local), dtype=index_dtype)
        inverse[permutation] = np.arange(len(old_local), dtype=index_dtype)
        offsets = np.concatenate(
            (
                [0],
                np.cumsum(
                    [int(old_offsets[i + 1] - old_offsets[i]) for i in order],
                    dtype=np.int64,
                ),
            )
        )
        f.attrs['format'] = FORMAT
        f.attrs['version'] = VERSION
        old.copy(root['instrument'], f, name='instrument')
        grid = f.create_group('grid')
        grid['tth'] = [spec.tth_range[0], spec.pixel_size[0], spec.ntth]
        grid['eta'] = [spec.eta_range[0], spec.pixel_size[1], spec.neta]
        inputs = f.create_group('input')
        inputs['panel_offsets'] = offsets
        _write_array(inputs, 'pixel_indices', old_local[permutation])
        matrix = f.create_group('matrix')
        matrix.attrs['format'] = 'csr'
        matrix.attrs['shape'] = (spec.size, len(old_local))
        group_count = root['operator/weights'].shape[0]
        terms = 4 * group_count
        index_dtype = (
            np.int32 if max(terms, spec.size, len(old_local)) < 2**31 else np.int64
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
            datasets['indices'][4 * first : 4 * last] = inverse[
                root['operator/source_indices'][first:last].ravel()
            ]
            if update_progress:
                update_progress(85 + int(14 * last / group_count))


def export_cake_matrix(
    path, instrument, spec, *, eta_chunk_rows=64, overwrite=False, update_progress=None
):
    """Atomically export the simple CSR format from an instrument and polar grid.

    Panel input segments are sorted by detector name. Matrix contributions
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
        _write_csr_from_blocks(blocks, result, spec, update_progress)
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
    the caller. The matrix can also be used directly as ``matrix @ x``.
    """

    @classmethod
    def load(cls, path):
        result = cls()
        result.path = Path(path).expanduser().resolve()
        result._instrument = None
        result._frame_shape = None
        result._frame_indices = None
        with h5py.File(result.path) as f:
            if f.attrs.get('format') != FORMAT or f.attrs.get('version') != VERSION:
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
            offsets = _integer_array(f['input/panel_offsets'][:], 'panel_offsets')
            local = _integer_array(f['input/pixel_indices'][:], 'pixel_indices')
            if (
                offsets.shape != (len(shapes) + 1,)
                or offsets[0] != 0
                or offsets[-1] != len(local)
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError('panel_offsets must span active pixels in panel order')
            for i, (rows, columns) in enumerate(shapes):
                segment = local[offsets[i] : offsets[i + 1]]
                if np.any(segment >= rows * columns) or np.any(
                    segment[1:] <= segment[:-1]
                ):
                    raise ValueError(
                        'pixel_indices must be sorted, unique, and inside each panel'
                    )
            result.panel_offsets = offsets
            result.pixel_indices = local
            m = f['matrix']
            shape = _integer_array(m.attrs['shape'], 'matrix shape')
            if m.attrs.get('format') != 'csr' or tuple(shape) != (
                np.prod(result.shape),
                len(local),
            ):
                raise ValueError('CSR shape must match the grid and compact input')
            data = m['data'][:]
            indices = _integer_array(m['indices'][:], 'matrix indices')
            indptr = _integer_array(m['indptr'][:], 'matrix indptr')
            if (
                data.ndim != 1
                or data.dtype.kind not in 'fiu'
                or not np.isfinite(data).all()
            ):
                raise ValueError('matrix data must be finite real coefficients')
            if len(indices) != len(data) or np.any(indices >= len(local)):
                raise ValueError('matrix indices must address compact inputs')
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
            self._instrument = HEDMInstrument(config)
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
        if set(images) != set(self.panel_names):
            raise ValueError('input panel names must match the embedded instrument')
        arrays = [_image_array(images[name]) for name in self.panel_names]
        active = np.empty(
            len(self.pixel_indices), dtype=np.result_type(*[a.dtype for a in arrays])
        )
        for i, (array, shape) in enumerate(zip(arrays, self.panel_shapes)):
            if array.shape != shape:
                raise ValueError(f'panel {self.panel_names[i]} has the wrong shape')
            first, last = self.panel_offsets[i : i + 2]
            active[first:last] = array.flat[self.pixel_indices[first:last]]
        return active

    def gather_active_composite_pixels(self, frame):
        frame = _image_array(frame)
        if frame.ndim != 2:
            raise ValueError('composite frame must be two-dimensional')
        if frame.shape != self._frame_shape:
            raw_indices = np.empty(len(self.pixel_indices), dtype=np.intp)
            for i, (shape, roi) in enumerate(zip(self.panel_shapes, self.rois)):
                if roi is None:
                    raise ValueError('composite inputs require an ROI for every panel')
                r0, c0 = roi
                if r0 + shape[0] > frame.shape[0] or c0 + shape[1] > frame.shape[1]:
                    raise ValueError('composite frame is too small for the panel ROIs')
                first, last = self.panel_offsets[i : i + 2]
                rows, cols = np.divmod(self.pixel_indices[first:last], shape[1])
                raw_indices[first:last] = (r0 + rows) * frame.shape[1] + c0 + cols
            self._frame_indices = raw_indices
            self._frame_shape = frame.shape
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
        return self.apply_active_pixels(self.gather_active_composite_pixels(frame))


def verify_cake_matrix(path):
    """Validate the archive and CSR bounds; the simple format has no stored hashes."""
    CakeMatrix.load(path)
