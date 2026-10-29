"""Native chunk directories and sparse-coordinate lookup."""

import h5py
import hdf5plugin
import numpy as np
import pytest
from hexrd.hedm._chunk_io import ChunkReader
from hexrd.hedm._image_cache import FrameCache
from hexrd.hedm._progress import _cancellation_check


def test_reader_falls_back_for_small_budget_or_unsupported_codec(tmp_path):
    path = tmp_path / 'other.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'gzip', data=np.arange(100)[:, None], chunks=(10, 1), compression='gzip'
        )
        handle.create_dataset('contiguous', data=np.arange(100)[:, None])
    with h5py.File(path, 'r') as handle:
        small = FrameCache(1024)
        small.chunk_workers = 4
        assert ChunkReader.create(handle, small) is None
        cache = FrameCache(32 * 1024**2)
        cache.chunk_workers = 4
        try:
            backend = ChunkReader.create(handle, cache)
            assert not backend.supported(handle['gzip'])
            assert not backend.supported(handle['contiguous'])
        finally:
            cache.close()


def test_cancellation_during_chunk_directory_prevents_decoder_work(tmp_path):
    path = tmp_path / 'directory.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'data', data=np.arange(2000)[:, None], chunks=(2, 1), **hdf5plugin.Blosc()
        )
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    try:
        with h5py.File(path, 'r') as handle:
            backend = ChunkReader.create(handle, cache)

            def cancel():
                raise KeyboardInterrupt

            token = _cancellation_check.set(cancel)
            try:
                with pytest.raises(KeyboardInterrupt):
                    backend.supported(handle['data'])
                assert backend.pool.directory_bytes == 0
            finally:
                _cancellation_check.reset(token)
    finally:
        cache.close()


@pytest.mark.parametrize('dtype', ['u2', 'i2', 'u4', 'i4', 'u8', 'i8'])
@pytest.mark.parametrize('start', [3, 3_000_000_000])
def test_native_pixel_positions_match_sparse_coordinates(dtype, start):
    from hexrd.core.extensions import transforms as native

    rng = np.random.default_rng(823)
    coordinates = np.unique(rng.integers([0, 0], [20, 45], size=(500, 2)), axis=0)
    ptr = np.r_[0, np.cumsum(np.bincount(coordinates[:, 0], minlength=23))]
    size = 17
    ids = np.arange(start // size, (start + len(coordinates) + size - 1) // size)
    chunks = [np.zeros(size, dtype=dtype) for _ in ids]
    for i, (_, col) in enumerate(coordinates):
        chunk, offset = divmod(start + i, size)
        chunks[chunk - ids[0]][offset] = col
    rows = np.repeat(np.arange(23), 47)
    cols = np.tile(np.arange(-1, 46), 23)
    lookup = {(row, col): start + i for i, (row, col) in enumerate(coordinates)}
    expected = np.array(
        [lookup.get((r, c), -1) for r, c in zip(rows, cols)], dtype='i8'
    )
    actual = native._fch5Positions(ptr, start, chunks, ids, size, rows, cols, 45)
    np.testing.assert_array_equal(actual, expected, strict=True)
    # A duplicate in a queried row must use SciPy's original coalescing path.
    chunk, offset = divmod(start + 1, size)
    chunks[chunk - ids[0]][offset] = coordinates[0, 1]
    assert native._fch5Positions(ptr, start, chunks, ids, size, rows, cols, 45) is None
