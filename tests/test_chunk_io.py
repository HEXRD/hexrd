"""Parallel decoding must preserve HDF5 bits and keep work and memory bounded."""

import os
import threading
import time

import h5py
import hdf5plugin
import numpy as np
import pytest

from hexrd.hedm._chunk_io import ChunkReader, _blosc
from hexrd.hedm._image_cache import FrameCache
from hexrd.hedm._progress import _cancellation_check


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('dtype', ['u2', 'i4', 'f4', 'f8', '>f8'])
@pytest.mark.parametrize('shuffle', [0, 1, 2])
def test_parallel_chunks_match_hdf5_bits(tmp_path, direct, dtype, shuffle):
    from hexrd.hedm._direct_io import open_direct_hdf5

    if direct and not hasattr(os, 'O_DIRECT'):
        pytest.skip('Linux direct I/O')
    path = tmp_path / 'chunked.h5'
    data = np.arange(130, dtype=dtype).reshape(65, 2)
    if np.dtype(dtype).kind == 'f':
        data[:3] = [[-0.0, np.nan], [np.inf, -np.inf], [0.5, -0.5]]
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'data',
            data=data,
            chunks=(7, 1),
            **hdf5plugin.Blosc(cname='zstd', shuffle=shuffle),
        )
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    handle = open_direct_hdf5(path) if direct else h5py.File(path, 'r')
    cache.files[object()] = handle
    try:
        backend = ChunkReader.create(handle, cache)
        assert backend is not None and backend.supported(handle['data'])
        for column in range(2):
            chunks = backend.chunks(
                handle['data'], column, range(10), cache, object(), 'test'
            )
            try:
                for index, actual, _ in chunks:
                    expected = data[index * 7 : min(index * 7 + 7, len(data)), column]
                    # Check bytes as well as values: NaNs, endian and signed zero.
                    assert actual[: len(expected), 0].tobytes() == expected.tobytes()
                    assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
            finally:
                chunks.close()
        assert backend.pool.executor._max_workers <= 4
    finally:
        pool = cache.chunk_pool
        cache.close()
        assert cache.chunk_pool is None and not cache.reserved_bytes
        assert all(not thread.is_alive() for thread in pool.executor._threads)


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


def test_cancel_joins_active_reads_and_does_not_launch_unbounded_work(
    tmp_path, monkeypatch
):
    path = tmp_path / 'queued.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'data', data=np.arange(1000)[:, None], chunks=(10, 1), **hdf5plugin.Blosc()
        )
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    active, launched = set(), []
    lock = threading.Lock()
    entered = threading.Event()
    with h5py.File(path, 'r') as handle:
        backend = ChunkReader.create(handle, cache)
        original = backend._decode

        def decode(*args):
            with lock:
                active.add(threading.get_ident())
                launched.append(1)
                entered.set()
            try:
                time.sleep(0.04)
                return original(*args)
            finally:
                with lock:
                    active.remove(threading.get_ident())

        monkeypatch.setattr(backend, '_decode', decode)

        def cancel():
            if entered.is_set():
                raise KeyboardInterrupt

        token = _cancellation_check.set(cancel)
        stream = backend.chunks(handle['data'], 0, range(100), cache, object(), 'test')
        start = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                list(stream)
            assert not active
            assert len(launched) <= 4
            assert time.monotonic() - start < 1
        finally:
            _cancellation_check.reset(token)
            stream.close()
            cache.close()


def test_codec_rejects_corrupt_or_wrong_size_payload(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / 'invalid.bin'
    path.write_bytes(b'!' * 80)
    backend = object.__new__(ChunkReader)
    backend.fd = os.open(path, os.O_RDONLY)
    backend.direct, backend.codec = False, _blosc()
    try:
        with pytest.raises(OSError, match='Invalid Blosc'):
            backend._decode(
                SimpleNamespace(size=80, byte_offset=0, filter_mask=0),
                (10, 1),
                np.dtype('f8'),
            )
        with pytest.raises(OSError, match='chunk size'):
            backend._decode(
                SimpleNamespace(size=200, byte_offset=0, filter_mask=0),
                (10, 1),
                np.dtype('f8'),
            )
    finally:
        os.close(backend.fd)


def test_unfiltered_and_unallocated_chunks_keep_hdf5_semantics(tmp_path):
    path = tmp_path / 'sparse.h5'
    values = np.array([-0.0, np.nan, np.inf, 0.5], dtype='f8')
    with h5py.File(path, 'w') as handle:
        dataset = handle.create_dataset(
            'data',
            shape=(12, 1),
            dtype='f8',
            chunks=(4, 1),
            fillvalue=7.5,
            **hdf5plugin.Blosc(),
        )
        dataset[:4, 0] = values
        dataset.id.write_direct_chunk((4, 0), values.tobytes(), filter_mask=1)
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    try:
        with h5py.File(path, 'r') as handle:
            backend = ChunkReader.create(handle, cache)
            stream = backend.chunks(
                handle['data'], 0, range(3), cache, object(), 'test'
            )
            try:
                chunks = list(stream)
                assert chunks[0][1].ravel().tobytes() == values.tobytes()
                assert chunks[1][1].ravel().tobytes() == values.tobytes()
                assert chunks[2][1] is None  # HDF5 supplies its fill value.
            finally:
                stream.close()
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
                assert not backend.pool.executor._threads
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
