"""Load ordinary CSR cake matrices with instrument-defined input layouts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.utils.hdf5 import unwrap_h5_to_dict

FORMAT = 'HEXRD cake matrix'
VERSION = '1'


def _image_array(value: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(
            'masked inputs must be explicitly filled (for example with NaN)'
        )
    array = np.asarray(value)
    if array.dtype.kind not in 'buif' or array.dtype.itemsize > 8:
        raise ValueError('pixels must have a real numeric dtype of at most 64 bits')
    return array.astype(np.float32) if array.dtype == np.float16 else array


def _integer_array(value: object, name: str, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in 'iu' or np.any(array < 0):
        raise ValueError(f'{name} must contain nonnegative integers with ndim={ndim}')
    return array


def _grid(value: object, name: str) -> tuple[float, float, int]:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite [start, step, count]')
    start, step, count = value
    if step <= 0 or count < 1 or count != int(count):
        raise ValueError(f'{name} requires a positive step and integer count')
    return float(start), float(step), int(count)


def _input_layout(
    instrument: HEDMInstrument, layout: str
) -> tuple[str, tuple[int, int] | None, dict[str, int], int]:
    """Resolve raw-frame shape or sorted panel offsets for both writer and reader."""
    panels = instrument.detectors
    if not panels:
        raise ValueError('instrument has no panels')
    rois = [p.roi for p in panels.values()]
    if layout == 'auto':
        overlap = any(
            a is not None
            and b is not None
            and all(a[k][0] < b[k][1] and b[k][0] < a[k][1] for k in (0, 1))
            for i, a in enumerate(rois)
            for b in rois[i + 1 :]
        )
        composite = len(panels) == 1 or (
            all(r is not None for r in rois)
            and len({p.group for p in panels.values()}) == 1
            and not overlap
        )
        layout = 'composite' if composite else 'panels'
    offsets = {}
    size = 0
    for name in sorted(panels):
        offsets[name] = size
        size += panels[name].rows * panels[name].cols
    shape = None
    if layout == 'composite':
        if len(panels) == 1 and rois[0] is None:
            shape = next(iter(panels.values())).shape
        else:
            if any(r is None for r in rois):
                raise ValueError('composite inputs require an ROI for every panel')
            shape = tuple(max(r[k][1] for r in rois) for k in (0, 1))
        size = shape[0] * shape[1]
    elif layout != 'panels':
        raise ValueError('input_layout must be auto, composite, or panels')
    return layout, shape, offsets, size


class CakeMatrix:
    """Loaded CSR matrix with its input mapping and uniform polar grid.

    Geometry is fixed by the archive. Dynamic NaNs propagate through sparse
    multiplication. Profile reduction and image/omega processing belong to
    the caller. Composite files support ``matrix @ frame.ravel()`` directly.
    """

    @classmethod
    def load(cls, path: str | Path) -> CakeMatrix:
        result = cls()
        result.path = Path(path).expanduser().resolve()
        with h5py.File(result.path) as f:
            if f.attrs.get('format') != FORMAT or f.attrs.get('version') not in (
                VERSION,
                'draft-csr-2',
            ):
                raise ValueError('unsupported cake matrix format or version')
            result.tth = _grid(f['grid/tth'][:], 'tth')
            result.eta = _grid(f['grid/eta'][:], 'eta')
            result.shape = (result.eta[2], result.tth[2])
            config = {}
            unwrap_h5_to_dict(f['instrument'], config)
            result.instrument = HEDMInstrument(
                config, active_beam_name=f['instrument'].attrs.get('active_beam')
            )
            m = f['matrix']
            result.input_layout = m.attrs.get('input_layout')
            if result.input_layout not in ('composite', 'panels'):
                raise ValueError('unknown matrix input_layout')
            _, result.composite_shape, offsets, input_size = _input_layout(
                result.instrument, result.input_layout
            )
            result.panel_names = tuple(offsets)
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
            # SciPy may trim trailing storage on construction; reject that first.
            if not indptr.size or indptr[-1] != len(data):
                raise ValueError('matrix indptr must span all coefficients')
            result.matrix = csr_matrix((data, indices, indptr), shape=tuple(shape))
            result.matrix.check_format(full_check=True)
        result.mask = (np.diff(result.matrix.indptr) == 0).reshape(result.shape)
        return result

    @property
    def tth_centers(self) -> np.ndarray:
        start, step, count = self.tth
        return start + step * (np.arange(count) + 0.5)

    @property
    def eta_centers(self) -> np.ndarray:
        start, step, count = self.eta
        return start + step * (np.arange(count) + 0.5)

    def apply(self, images: Mapping[str, np.ndarray]) -> np.ma.MaskedArray:
        """Apply complete panel images in instrument-defined shapes."""
        if set(images) != set(self.panel_names):
            raise ValueError('input panel names must match the embedded instrument')
        arrays = [_image_array(images[name]) for name in self.panel_names]
        panels = [self.instrument.detectors[name] for name in self.panel_names]
        for name, array, panel in zip(self.panel_names, arrays, panels):
            if array.shape != panel.shape:
                raise ValueError(f'panel {name} has the wrong shape')
        if self.input_layout == 'composite':
            frame = np.zeros(self.composite_shape, dtype=np.result_type(*arrays))
            for array, panel in zip(arrays, panels):
                r0, c0 = (panel.roi[0][0], panel.roi[1][0]) if panel.roi else (0, 0)
                frame[r0 : r0 + array.shape[0], c0 : c0 + array.shape[1]] = array
            return self.apply_composite(frame)
        values = self.matrix @ np.concatenate([a.ravel() for a in arrays])
        return np.ma.array(values.reshape(self.shape), mask=self.mask)

    def apply_composite(self, frame: np.ndarray) -> np.ma.MaskedArray:
        """Multiply a full detector image without gathering or rearranging pixels."""
        if self.input_layout != 'composite':
            raise ValueError('matrix expects separate panel inputs')
        frame = _image_array(frame)
        if frame.shape != self.composite_shape:
            raise ValueError(f'composite frame shape must be {self.composite_shape}')
        values = (self.matrix @ frame.ravel()).reshape(self.shape)
        return np.ma.array(values, mask=self.mask)


def verify_cake_matrix(path: str | Path) -> None:
    """Validate the archive and CSR bounds."""
    CakeMatrix.load(path)
