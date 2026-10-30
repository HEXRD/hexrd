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
