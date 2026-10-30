"""Compact row indexes must be exact, bounded and cancellable."""

import hdf5plugin
import numpy as np
import pytest

from hexrd.hedm._image_cache import FrameCache
from hexrd.hedm._progress import _cancellation_check


@pytest.mark.parametrize('height', [1, 43, 32767])
def test_native_reader_row_boundaries_and_disorder(tmp_path, height):
    import h5py
    from hexrd.hedm._chunk_io import ChunkReader

    rows = np.repeat(np.array([0, height - 1], dtype='u2'), 19)
    path = tmp_path / 'boundaries.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'rows', data=rows[:, None], chunks=(7, 1),
            **hdf5plugin.Blosc(cname='zstd'),
        )
        invalid = rows.copy()
        invalid[10] = 65535
        handle.create_dataset(
            'invalid', data=invalid[:, None], chunks=(7, 1),
            **hdf5plugin.Blosc(cname='zstd'),
        )
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 2
    try:
        with h5py.File(path, 'r') as handle:
            reader = ChunkReader.create(handle, cache)
            status, offsets, _ = reader.row_index(handle['rows'], 0, len(rows), height)
            assert status == 1
            np.testing.assert_array_equal(
                offsets, np.searchsorted(rows, np.arange(height + 1)).astype('i4'),
                strict=True,
            )
            assert reader.row_index(handle['invalid'], 0, len(rows), height)[0] == 0
    finally:
        cache.close()


@pytest.mark.parametrize('position', [0, 1, 7, 8, 9, 15, 16, 31, 32])
@pytest.mark.parametrize('value', [0, 65535])
def test_native_reader_rejects_disordered_or_out_of_range_rows(tmp_path, position, value):
    import h5py
    from hexrd.hedm._chunk_io import ChunkReader

    rows = np.ones(40, dtype='u2')
    rows[position] = value
    path = tmp_path / 'invalid.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'rows', data=rows[:, None], chunks=(7, 1),
            **hdf5plugin.Blosc(cname='zstd'),
        )
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 2
    try:
        with h5py.File(path, 'r') as handle:
            reader = ChunkReader.create(handle, cache)
            status, _, _ = reader.row_index(handle['rows'], 0, len(rows), 4)
            assert bool(status) == (position == 0 and value == 0)
    finally:
        cache.close()


@pytest.mark.parametrize('chunk_size', [7, 16, 131])
def test_native_index_reads_partial_chunks_and_checks_cancellation(
    tmp_path, chunk_size
):
    import h5py
    from hexrd.hedm._chunk_io import ChunkReader

    rng = np.random.default_rng(813)
    rows = np.repeat(np.arange(43, dtype='u2'), rng.integers(0, 60, 43))
    stored = np.r_[np.full(5, 65535, dtype='u2'), rows, np.zeros(11, dtype='u2')]
    path = tmp_path / 'batch.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'rows',
            data=stored[:, None],
            chunks=(chunk_size, 1),
            **hdf5plugin.Blosc(cname='zstd')
        )
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    try:
        with h5py.File(path, 'r') as handle:
            reader = ChunkReader.create(handle, cache)
            spec = reader._spec(handle['rows'])

            def read(start, stop, check=lambda: None):
                return reader.pool.native_reader.read_rows(
                    reader.fd, spec, start, stop, 43, False, check
                )

            status, offsets, decoded = read(5, 5 + len(rows))
            assert status == 1 and decoded >= rows.nbytes
            np.testing.assert_array_equal(
                offsets, np.searchsorted(rows, np.arange(44)).astype('i4'), strict=True
            )
            assert read(0, 5 + len(rows))[0] == 0
            status, empty, decoded = read(5, 5)
            assert status == 1 and decoded == 0 and not empty.any()
            calls = 0

            def cancel():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt

            with pytest.raises(KeyboardInterrupt):
                read(5, 5 + len(rows), cancel)
            # Reuse after an interrupted read must release buffers and locks.
            assert read(5, 5 + len(rows))[0] == 1
            with pytest.raises(ValueError, match='dimensions'):
                read(-1, 0)
            with pytest.raises(ValueError, match='dimensions'):
                read(0, len(spec[2]) * chunk_size + 1)
    finally:
        cache.close()


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('malformed', [None, 'rows', 'columns', 'duplicate'])
def test_native_indexes_match_hdf5_in_any_frame_order(
    tmp_path, direct, malformed, monkeypatch
):
    import os
    from test_fch5_pixels import make_reader

    if direct:
        if not hasattr(os, 'O_DIRECT'):
            pytest.skip('Linux direct I/O')
        monkeypatch.setenv('HEXRD_FIT_GRAINS_DIRECT_IO', '1')
    reader = make_reader(
        tmp_path / 'rows.h5',
        malformed=malformed,
        compression=hdf5plugin.Blosc(cname='zstd'),
    )
    cache, reference = FrameCache(64 * 1024**2), FrameCache(12000)
    cache.chunk_workers = 4
    rows, cols = np.indices(reader.shape)
    try:
        with reader.prefetch_frame_indexes(np.array([2, 0, 3, 1, 4]), cache):
            for frame in [2, 0, 3, 1, 4, 3, 0]:
                actual = reader.sample_pixels(frame, rows.ravel(), cols.ravel(), cache)
                expected = reader.sample_pixels(
                    frame, rows.ravel(), cols.ravel(), reference
                )
                if expected is None:
                    assert actual is None
                else:
                    assert actual.tobytes() == expected.tobytes()
                assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
    finally:
        cache.close()
        reference.close()


def test_native_prefetch_handles_skipped_frames_and_cancelled_scopes(tmp_path):
    from test_fch5_pixels import make_reader

    reader = make_reader(tmp_path / 'scope.h5', compression=hdf5plugin.Blosc())
    cache = FrameCache(64 * 1024**2)
    cache.chunk_workers = 4
    frames = np.array([2, 0, 3, 1, 4])
    try:
        with reader.prefetch_frame_indexes(frames, cache):
            for frame in [3, 4]:
                actual = reader.sample_pixels(frame, [4], [9], cache)
                assert actual[0] == (7 if frame == 3 else 0)
        assert reader._chunk_reader(cache).row_indexes is None
        cache.clear_frames()

        def cancel():
            raise KeyboardInterrupt

        token = _cancellation_check.set(cancel)
        try:
            with pytest.raises(KeyboardInterrupt):
                with reader.prefetch_frame_indexes(frames, cache):
                    reader.sample_pixels(0, [4], [9], cache)
        finally:
            _cancellation_check.reset(token)
        assert reader._chunk_reader(cache).row_indexes is None
        with reader.prefetch_frame_indexes(frames, cache):
            assert reader.sample_pixels(0, [4], [9], cache)[0] == 7
    finally:
        cache.close()


def test_prefetch_scope_can_be_released_after_its_reader_closes(tmp_path):
    from test_fch5_pixels import make_reader

    reader = make_reader(tmp_path / 'owner.h5', compression=hdf5plugin.Blosc())
    cache = FrameCache(64 * 1024**2)
    cache.chunk_workers = 4
    with reader.prefetch_frame_indexes(np.array([2, 0, 3, 1, 4]), cache):
        cache.close()
    # Native row jobs must already be drained before descriptors are closed.
    assert not cache.files and cache.chunk_pool is None
