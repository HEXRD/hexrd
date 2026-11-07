import numpy as np
from scipy.sparse import csr_array

from hexrd.hedm.find_orientations import _bin_pixels, _pixel_chunks


def test_bin_pixels_does_not_overflow_uint16_weights():
    # Both pixels land in the same bin in one frame. The sum exceeds uint16.
    frames = [csr_array(np.full((1, 2), 40_000, dtype=np.uint16))]
    ring_mask = np.ones(2, dtype=np.int64)
    eta_bin = np.zeros(2, dtype=np.int64)
    ring_maps = np.zeros((1, 1, 1))

    for ids, vals, offsets, rows in _pixel_chunks(frames, np.array([0])):
        _bin_pixels(offsets, ids, vals, rows, ring_mask, eta_bin, 0.0, ring_maps)

    assert ring_maps[0, 0, 0] == 80_000


def test_pixel_chunks_preserve_duplicate_pixels_and_empty_frames():
    # CSR permits duplicate, unsorted column indices. Keep their individual
    # uint16 values until float64 accumulation; canonicalizing can overflow.
    hot = csr_array((np.array([40_000, 7, 40_000], dtype=np.uint16),
                     np.array([2, 0, 2]), np.array([0, 3, 3])), shape=(2, 3))
    empty = csr_array((2, 3), dtype=np.uint16)
    frames = [empty, hot, empty, hot, empty]
    rows = np.array([0, 2, 3, 5, 6])
    ring_maps = np.zeros((1, 7, 1))
    for ids, vals, offsets, chunk_rows in _pixel_chunks(frames, rows, max_pixels=3):
        _bin_pixels(offsets, ids, vals, chunk_rows, np.ones(6, dtype=np.int64),
                    np.zeros(6, dtype=np.int64), 10., ring_maps)
    assert np.array_equal(ring_maps.ravel(), [0., 0., 80_000., 0., 0., 80_000., 0.])
