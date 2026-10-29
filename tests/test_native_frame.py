"""Native frame sampling must retain CSR bits, byte bounds and cancellation."""

import os
import time

import h5py
import hdf5plugin
import numpy as np
import pytest

from hexrd.hedm._image_cache import FrameCache
from hexrd.hedm._fch5_pixels import Fch5Pixels


@pytest.mark.parametrize(
    'dtype', ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8', 'f4', 'f8']
)
@pytest.mark.parametrize('direct', [False, True])
def test_native_frame_preserves_pixel_bits_and_original_query_order(
    tmp_path, monkeypatch, dtype, direct
):
    from test_fch5_pixels import make_reader

    if direct:
        if not hasattr(os, 'O_DIRECT'):
            pytest.skip('Linux direct I/O')
        monkeypatch.setenv('HEXRD_FIT_GRAINS_DIRECT_IO', '1')
    reader = make_reader(
        tmp_path / 'frame.h5', dtype=dtype, compression=hdf5plugin.Blosc(cname='zstd')
    )
    cache, reference = FrameCache(32 * 1024**2), FrameCache(12000)
    cache.chunk_workers = 4
    rows, cols = np.indices(reader.shape)
    rng = np.random.default_rng(132)
    order = rng.permutation(rows.size)
    rows, cols = rows.ravel()[order], cols.ravel()[order]
    rows, cols = np.r_[rows, rows], np.r_[cols, cols]
    try:
        expected = [
            reader.sample_pixels(frame, rows, cols, reference) for frame in range(5)
        ]

        def no_python_lookup(*args, **kwargs):
            pytest.fail('native frame sampling returned to Python coordinate lookup')

        monkeypatch.setattr(Fch5Pixels, '_positions', no_python_lookup)
        for frame, wanted in enumerate(expected):
            actual = reader.sample_pixels(frame, rows, cols, cache)
            assert actual.dtype == wanted.dtype
            assert actual.tobytes() == wanted.tobytes()
            assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
    finally:
        cache.close()
        reference.close()


def test_native_frame_handles_partial_codec_blocks_and_large_positions(tmp_path):
    from hexrd.hedm._image_cache import FileReader

    height, width = 1200, 2000
    row, col = np.meshgrid(
        np.arange(height, dtype='u2'), np.arange(0, width, 4, dtype='u2'), indexing='ij'
    )
    coords = np.column_stack((row.ravel(), col.ravel()))
    data = np.arange(len(coords), dtype='f8')
    data[::37] = -0.0
    data[::97] = np.nan
    data[::101] = np.inf
    data[::113] = -np.inf
    path = tmp_path / 'large.h5'
    # Sparse allocation tests 64-bit file positions without a large file.
    offset = 3_000_000_000
    with h5py.File(path, 'w') as handle:
        handle['frame_ids'] = np.array([offset, offset + len(coords)], dtype='u8')
        indices = handle.create_dataset(
            'indices',
            shape=(offset + len(coords), 2),
            dtype='u2',
            chunks=(327680, 1),
            **hdf5plugin.Blosc(cname='zstd'),
        )
        values = handle.create_dataset(
            'data',
            shape=(offset + len(coords), 1),
            dtype='f8',
            chunks=(81920, 1),
            **hdf5plugin.Blosc(cname='zstd'),
        )
        indices[offset:] = coords
        values[offset:, 0] = data
    reader = FileReader(
        str(path), 'fch5', {}, (height, width), np.dtype('f8'), 1, estimate=1024**3
    )
    cache, reference = FrameCache(64 * 1024**2), FrameCache(12000)
    cache.chunk_workers = 4
    rows = np.repeat([0, 2, 219, 527, 655, 700, 1100, 1199], 400)
    cols = np.tile(np.arange(0, 2000, 5), 8)
    try:
        expected = reader.sample_pixels(0, rows, cols, reference)
        actual = reader.sample_pixels(0, rows, cols, cache)
        assert actual.tobytes() == expected.tobytes()
        assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
    finally:
        cache.close()
        reference.close()


def test_cancellation_drains_native_jobs_and_reader_can_be_reused(tmp_path):
    from test_fch5_pixels import make_reader
    from hexrd.hedm._chunk_io import ChunkReader

    reader = make_reader(tmp_path / 'cancel.h5', compression=hdf5plugin.Blosc())
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    try:
        sampler = reader._chunk_reader(cache)
        sampler.reporter = None
        sampler.read_bytes = sampler.read_seconds = 0
        ptr = sampler._offsets(0, cache)
        backend = ChunkReader.create(sampler.handle(), cache)
        coordinates, values = backend._spec(sampler.indices), backend._spec(
            sampler.values
        )
        rows, cols = np.indices(reader.shape)
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            if calls == 1:
                time.sleep(0.04)
            else:
                raise KeyboardInterrupt

        def read(check):
            return backend.pool.native_reader.read_pixels(
                backend.fd,
                coordinates,
                values,
                ptr,
                *map(int, sampler.bounds[0]),
                rows.ravel(),
                cols.ravel(),
                reader.shape[1],
                backend.direct,
                8 * 1024**2,
                check,
            )

        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            read(cancel)
        assert calls >= 2 and time.monotonic() - started < 1
        result, status, decoded, chunks = read(lambda: None)
        assert status == 1 and decoded > 0 and chunks > 0
        reference = FrameCache(12000)
        try:
            expected = reader.sample_pixels(0, rows.ravel(), cols.ravel(), reference)
            assert result.tobytes() == expected.tobytes()
        finally:
            reference.close()
    finally:
        cache.close()


def test_corrupt_native_chunk_drains_jobs_and_restored_source_can_be_read(tmp_path):
    from test_fch5_pixels import make_reader
    from hexrd.hedm._chunk_io import ChunkReader

    path = tmp_path / 'corrupt.h5'
    reader = make_reader(path, compression=hdf5plugin.Blosc())
    with h5py.File(path, 'a') as handle:
        coordinates = handle['scan/indices'][()]
        del handle['scan/indices']
        dataset = handle.create_dataset(
            'scan/indices',
            shape=(1024, 2),
            dtype='u2',
            chunks=(1024, 1),
            **hdf5plugin.Blosc(cname='zstd'),
        )
        dataset[: len(coordinates)] = coordinates
    cache, reference = FrameCache(32 * 1024**2), FrameCache(12000)
    cache.chunk_workers = 4
    try:
        sampler = reader._chunk_reader(cache)
        sampler.reporter = None
        sampler.read_bytes = sampler.read_seconds = 0
        sampler._offsets(0, cache)
        backend = ChunkReader.create(sampler.handle(), cache)
        sampler.chunk_reader = backend
        _, _, offsets, _, masks = backend._spec(sampler.indices)
        chunk = int(sampler.bounds[0, 0]) // sampler.coord_chunk
        assert masks[chunk, 1] == 0, 'fixture must actually use Blosc'
        position = int(offsets[chunk, 1])
        fd = os.open(path, os.O_RDWR)
        try:
            original = os.pread(fd, 16, position)
            os.pwrite(fd, b'!' * 16, position)
            with pytest.raises(OSError, match='native FCH5'):
                reader.sample_pixels(0, [1, 4, 8, 12], [2, 5, 9, 13], cache)
            os.pwrite(fd, original, position)
            actual = reader.sample_pixels(0, [1, 4, 8, 12], [2, 5, 9, 13], cache)
            expected = reader.sample_pixels(0, [1, 4, 8, 12], [2, 5, 9, 13], reference)
            assert actual.tobytes() == expected.tobytes()
            assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
        finally:
            os.close(fd)
    finally:
        cache.close()
        reference.close()
