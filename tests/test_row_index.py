"""Compact row indexes must be exact, bounded and cancellable."""

import hdf5plugin
import numpy as np
import pytest

from hexrd.hedm._image_cache import FrameCache
from hexrd.hedm._progress import _cancellation_check


@pytest.mark.parametrize('chunk_size', [1, 7, 8, 9, 16, 31, 131])
def test_native_row_index_matches_searchsorted_across_partial_chunks(chunk_size):
    from hexrd.core.extensions import transforms as native

    rng = np.random.default_rng(813)
    height = 43
    rows = np.repeat(np.arange(height, dtype='u2'), rng.integers(0, 60, height))
    # Padding is deliberately invalid and must not participate in validation.
    stored = np.r_[np.full(5, 65535, dtype='u2'), rows, np.zeros(11, dtype='u2')]
    offsets = np.full(height + 1, len(rows), dtype='i4')
    next_row, previous = 0, -1
    start, stop = 5, 5 + len(rows)
    first = start
    while first < stop:
        chunk = first // chunk_size * chunk_size
        end = min(stop, chunk + chunk_size)
        next_row, previous = native._fch5RowOffsets(
            stored[chunk : chunk + chunk_size],
            first - chunk,
            end - chunk,
            offsets,
            first - start,
            previous,
            next_row,
        )
        assert next_row >= 0
        first = end
    expected = np.searchsorted(rows, np.arange(height + 1)).astype('i4')
    np.testing.assert_array_equal(offsets, expected, strict=True)


@pytest.mark.parametrize('position', [0, 1, 7, 8, 9, 15, 16, 31, 32])
@pytest.mark.parametrize('value', [0, 65535])
def test_native_row_validation_rejects_disorder_and_unsigned_overflow(position, value):
    from hexrd.core.extensions import transforms as native

    rows = np.ones(40, dtype='u2')
    rows[position] = value
    # A leading zero is canonical; all other changes violate ordering or bounds.
    offsets = np.full(5, len(rows), dtype='i4')
    next_row, _ = native._fch5RowOffsets(rows, 0, len(rows), offsets, 0, -1, 0)
    assert (next_row >= 0) == (position == 0 and value == 0)


def test_native_row_index_checks_writable_output_and_overflow():
    from hexrd.core.extensions import transforms as native

    rows = np.array([0, 0, 1, 1], dtype='u2')
    with pytest.raises(ValueError, match='int32'):
        native._fch5RowOffsets(rows, 0, 4, np.zeros(3, dtype='i8'), 0, -1, 0)
    output = np.zeros(3, dtype='i4')
    with pytest.raises(ValueError, match='bounds'):
        native._fch5RowOffsets(rows, 0, 4, output, 2**63 - 1, -1, 0)
    output.flags.writeable = False
    with pytest.raises(ValueError, match='int32'):
        native._fch5RowOffsets(rows, 0, 4, output, 0, -1, 0)
    unaligned = np.ndarray((3,), dtype='i4', buffer=bytearray(13), offset=1)
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match='int32'):
        native._fch5RowOffsets(rows, 0, 4, unaligned, 0, -1, 0)


@pytest.mark.parametrize('height', [1, 32767])
def test_native_row_index_handles_signed_limit_and_empty_frames(height):
    from hexrd.core.extensions import transforms as native

    rows = np.full(40, height - 1, dtype='u2')
    offsets = np.full(height + 1, len(rows), dtype='i4')
    next_row, previous = native._fch5RowOffsets(rows, 0, len(rows), offsets, 0, -1, 0)
    assert next_row == height and previous == height - 1
    np.testing.assert_array_equal(offsets, np.searchsorted(rows, np.arange(height + 1)))
    empty = np.zeros(height + 1, dtype='i4')
    assert native._fch5RowOffsets(rows[:0], 0, 0, empty, 0, -1, 0) == (0, -1)
    assert not empty.any()


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
