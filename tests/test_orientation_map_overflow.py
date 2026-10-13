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
