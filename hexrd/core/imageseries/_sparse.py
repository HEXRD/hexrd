"""Pixel sampling from sparse detector frames."""

import numba
import numpy as np


@numba.njit(cache=True, nogil=True)
def _sample_csr(indptr, indices, data, rows, cols, nrows, ncols):
    values = np.zeros(rows.size, dtype=data.dtype)
    for i in range(rows.size):
        row, col = rows[i], cols[i]
        if row < 0:
            row += nrows
        if col < 0:
            col += ncols
        if row < 0 or row >= nrows or col < 0 or col >= ncols:
            raise IndexError('pixel index out of bounds')

        lo, hi = indptr[row], indptr[row + 1]
        end = hi
        while lo < hi:
            mid = (lo + hi) // 2
            if indices[mid] < col:
                lo = mid + 1
            else:
                hi = mid
        # Sum duplicates as scipy sparse indexing does, including CSR
        # arrays whose indices are sorted but have not been coalesced.
        while lo < end and indices[lo] == col:
            values[i] += data[lo]
            lo += 1
    return values


def sample_pixels(frame, rows, cols):
    """Sample paired pixel indices without constructing a sparse result.

    Spot extraction repeatedly requests small angular grids. SciPy's general
    sparse indexing constructs and validates a new sparse array for each grid;
    here only the values are needed. Other index types retain SciPy semantics.
    """
    if (
        isinstance(rows, np.ndarray)
        and isinstance(cols, np.ndarray)
        and rows.shape == cols.shape
        and rows.dtype.kind in 'iu'
        and cols.dtype.kind in 'iu'
        # Mixing uint64 indices with signed CSR offsets promotes arithmetic
        # to float64 in Numba. Keep SciPy's validation for that index type.
        and rows.dtype != np.dtype('uint64')
        and cols.dtype != np.dtype('uint64')
        and frame.has_sorted_indices
        and frame.dtype.char in '?bBhHiIlLqQfdFD'
    ):
        return _sample_csr(
            frame.indptr,
            frame.indices,
            frame.data,
            rows.ravel(),
            cols.ravel(),
            *frame.shape,
        ).reshape(rows.shape)
    result = frame[rows, cols]
    return result.toarray() if hasattr(result, 'toarray') else np.asarray(result)
