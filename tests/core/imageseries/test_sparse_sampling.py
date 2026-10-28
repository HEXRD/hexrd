import numpy as np
import pytest
from scipy.sparse import csr_array

from hexrd.core.imageseries._sparse import sample_pixels


@pytest.mark.parametrize(
    'dtype', ['uint16', 'int32', 'float32', 'float64', 'complex128']
)
@pytest.mark.parametrize('index_dtype', ['int32', 'int64', 'uint32', 'uint64'])
def test_sample_pixels_matches_sparse_indexing(dtype, index_dtype):
    rng = np.random.default_rng(42)
    dense = rng.integers(0, 100, (32, 40)).astype(dtype)
    dense[dense < 70] = 0
    frame = csr_array(dense)
    rows = rng.integers(0, 32, (7, 9)).astype(index_dtype)
    cols = rng.integers(0, 40, (7, 9)).astype(index_dtype)
    # Includes repeated pixels, empty rows, and noncontiguous index grids.
    rows, cols = rows[:, ::2], cols[:, ::2]
    actual = sample_pixels(frame, rows, cols)
    np.testing.assert_array_equal(actual, dense[rows, cols])
    assert actual.shape == rows.shape
    assert actual.dtype == frame.dtype


def test_sample_pixels_duplicates_negative_indices_and_bounds():
    frame = csr_array(
        (np.array([2.0, 3.0, 4.0]), np.array([1, 1, 2]), np.array([0, 2, 2, 3])),
        shape=(3, 4),
    )
    rows, cols = np.array([0, -1, 1]), np.array([1, -2, 0])
    np.testing.assert_array_equal(sample_pixels(frame, rows, cols), [5.0, 4.0, 0.0])
    for row, col in [(3, 0), (0, 4), (-4, 0), (0, -5)]:
        with pytest.raises(IndexError):
            sample_pixels(frame, np.array([row]), np.array([col]))
    assert (
        sample_pixels(frame, np.array([], dtype=int), np.array([], dtype=int)).size == 0
    )


def test_sample_pixels_unsorted_and_slice_fallback():
    frame = csr_array(
        (np.array([4.0, 2.0, 3.0]), np.array([2, 1, 1]), np.array([0, 3, 3])),
        shape=(2, 4),
    )
    assert not frame.has_sorted_indices
    rows, cols = np.array([0, 0, 1]), np.array([1, 2, 0])
    np.testing.assert_array_equal(sample_pixels(frame, rows, cols), [5.0, 4.0, 0.0])
    np.testing.assert_array_equal(
        sample_pixels(frame, slice(None), slice(1, 3)), frame.toarray()[:, 1:3]
    )
