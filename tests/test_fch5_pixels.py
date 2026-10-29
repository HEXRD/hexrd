"""Partial FCH5 reads must produce the existing CSR path's exact pixels."""

import pickle
import os

import h5py
import numpy as np
import pytest

from hexrd.hedm._image_cache import FileReader, FrameCache, StreamingView, View


def make_reader(
    path, dtype='float64', malformed=None, large_offset=False, compression=None
):
    shape = (14, 18)
    row, col = np.meshgrid([1, 4, 8, 12], [2, 5, 9, 13], indexing='ij')
    indices = np.column_stack((row.ravel(), col.ravel())).astype('uint16')
    values = np.arange(len(indices), dtype=dtype) + 1
    if np.dtype(dtype).kind == 'f':
        values[:4] = [-0.0, np.nan, np.inf, -np.inf]
    if malformed == 'duplicate':
        indices[2] = indices[1]
    elif malformed == 'columns':
        indices[[1, 2]] = indices[[2, 1]]
    elif malformed == 'rows':
        indices[[0, -1]] = indices[[-1, 0]]
    bounds = np.zeros((5, 2), dtype='u8')
    coords, intensities = [], []
    cursor = 0
    for frame in [2, 0, 3, 1]:
        coords.append(indices)
        intensities.append(values)
        bounds[frame] = cursor, cursor + len(values)
        cursor += len(values)
    bounds[4] = cursor, cursor  # Empty frame, with trailing empty rows too.
    offset = 3_000_000_000 if large_offset else 0
    bounds += offset
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('scan')
        group['frame_ids'] = bounds.ravel()
        coordinates = group.create_dataset(
            'indices',
            shape=(offset + cursor, 2),
            dtype='u2',
            chunks=(7, 1),
            **(compression or {}),
        )
        intensities_dataset = group.create_dataset(
            'data',
            shape=(offset + cursor, 1),
            dtype=dtype,
            chunks=(5, 1),
            **(compression or {}),
        )
        coordinates[offset:] = np.concatenate(coords)
        intensities_dataset[offset:] = np.concatenate(intensities)[:, None]
    return FileReader(
        str(path), 'fch5', {'path': 'scan'}, shape, np.dtype(dtype), 5, estimate=1024**3
    )


@pytest.mark.parametrize('dtype', ['uint16', 'int32', 'float32', 'float64'])
@pytest.mark.parametrize('large_offset', [False, True])
@pytest.mark.parametrize(
    'direct',
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not hasattr(os, 'O_DIRECT'), reason='Linux direct I/O'
            ),
        ),
    ],
)
def test_chunk_sampling_matches_whole_frame_across_boundaries(
    tmp_path, dtype, large_offset, direct, monkeypatch
):
    if direct:
        monkeypatch.setenv('HEXRD_FIT_GRAINS_DIRECT_IO', '1')
    reader = make_reader(tmp_path / 'frames.h5', dtype, large_offset=large_offset)
    rng = np.random.default_rng(32)
    rows, cols = np.indices(reader.shape)
    rows, cols = rows.ravel(), cols.ravel()
    selected = rng.permutation(len(rows))
    rows, cols = (
        np.r_[rows[selected], rows[selected]],
        np.r_[cols[selected], cols[selected]],
    )
    partial, whole = FrameCache(12000), FrameCache(0)
    try:
        for frame in [0, 4, 3, 1, 2, 0]:
            actual = reader.sample_pixels(frame, rows, cols, partial)
            reference = StreamingView(View(reader), whole)
            # Disable only the new access path for the independent CSR reference.
            from hexrd.core.extensions import transforms as native
            from hexrd.core.imageseries._sparse import sample_pixels

            expected = reference._sample_frame(frame, rows, cols, native, sample_pixels)
            np.testing.assert_array_equal(actual, expected, strict=True)
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
            assert partial.bytes <= partial.max_bytes
            assert partial.metadata_bytes <= partial.max_bytes // 4
    finally:
        partial.close()
        whole.close()


@pytest.mark.parametrize('malformed', ['duplicate', 'columns', 'rows'])
def test_noncanonical_coordinates_fall_back_to_existing_csr(tmp_path, malformed):
    reader = make_reader(tmp_path / 'frames.h5', malformed=malformed)
    cache = FrameCache(12000)
    try:
        assert reader.sample_pixels(0, np.array([1]), np.array([5]), cache) is None
        # Exercise the actual dispatch and compare to the established sampler.
        ids, offsets = np.array([[0]]), np.array([0, 1])
        actual = StreamingView(View(reader), cache).gather(
            ids, offsets, np.array([1]), np.array([5])
        )
        from hexrd.core.imageseries._sparse import sample_pixels

        expected = sample_pixels(cache.frame(reader, 0), np.array([1]), np.array([5]))
        np.testing.assert_array_equal(actual, expected)
    finally:
        cache.close()


def test_chunk_gather_preserves_roi_mapping_and_clipping(tmp_path):
    reader = make_reader(tmp_path / 'frames.h5')
    partial, whole = FrameCache(12000), FrameCache(0)
    view = View(reader, roi=(1, 2, 9, 12), mapping=np.array([3, 0, 2, 1, 4]))
    ids = np.array([[0, 1, 4], [3, 0, 2]])
    offsets = np.array([0, 4, 6])
    rows, cols = np.array([-2, 0, 3, 20, 7, 3]), np.array([0, 3, 7, 30, 11, 3])
    try:
        actual = StreamingView(view, partial).gather(ids, offsets, rows, cols)
        reader.estimate = 0
        expected = StreamingView(view, whole).gather(ids, offsets, rows, cols)
        np.testing.assert_array_equal(actual, expected, strict=True)
    finally:
        partial.close()
        whole.close()


def test_indexes_survive_chunk_churn_inside_total_cache_budget():
    cache = FrameCache(10240)
    reader = object()
    index = cache.array(reader, 'index', 0, lambda: np.zeros(768, 'u1'), metadata=True)
    for chunk in range(20):
        cache.array(reader, 'chunk', chunk, lambda: np.zeros(7168, 'u1'))
        assert cache.bytes <= cache.max_bytes
    retained = cache.array(
        reader, 'index', 0, lambda: pytest.fail('index evicted'), metadata=True
    )
    assert retained is index
    cache.array(reader, 'index', 1, lambda: np.zeros(768, 'u1'), metadata=True)
    assert len(cache.metadata) == 1  # Metadata has its own quota within the total.
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.bytes == restored.metadata_bytes == 0
    assert not restored.frames and not restored.metadata
    cache.close()
    assert cache.bytes == cache.metadata_bytes == 0


def test_cancel_stops_before_next_chunk_read(tmp_path):
    from hexrd.hedm._progress import Dashboard, FIT_STAGES

    reader = make_reader(tmp_path / 'frames.h5')
    cache = FrameCache(12000)
    dashboard = Dashboard(FIT_STAGES)
    dashboard.start()
    dashboard.cancelled.set()
    try:
        with pytest.raises(KeyboardInterrupt):
            reader.sample_pixels(0, np.array([1]), np.array([5]), cache)
        assert not cache.frames and not cache.metadata
    finally:
        cache.close()
        dashboard.close()


def test_missing_pixels_do_not_read_intensity_chunks(tmp_path):
    reader = make_reader(tmp_path / 'frames.h5')
    cache = FrameCache(12000)
    try:
        actual = reader.sample_pixels(
            0, np.array([1, 4, 8]), np.array([0, 0, 0]), cache
        )
        np.testing.assert_array_equal(actual, np.zeros(3))
        assert not any(key[1] == 'values' for key in cache.frames)
        actual = reader.sample_pixels(
            0, np.array([], dtype='i8'), np.array([], dtype='i8'), cache
        )
        assert actual.shape == (0,) and actual.dtype == reader.dtype
    finally:
        cache.close()


def test_reads_follow_file_offsets_instead_of_omega_order(tmp_path):
    reader = make_reader(tmp_path / 'frames.h5')
    cache = FrameCache(12000)
    seen = []
    original = reader.sample_pixels

    def record(index, rows, cols, cache):
        seen.append(index)
        return original(index, rows, cols, cache)

    reader.sample_pixels = record
    try:
        actual = StreamingView(View(reader), cache).gather(
            np.array([[0, 1, 2, 3, 4]]), np.array([0, 1]), np.array([4]), np.array([9])
        )
        assert seen == [2, 0, 3, 1, 4]
        np.testing.assert_array_equal(actual, [7.0, 7.0, 7.0, 7.0, 0.0])
    finally:
        cache.close()


@pytest.mark.skipif(not hasattr(os, 'O_DIRECT'), reason='Linux direct I/O')
def test_direct_reader_handles_unaligned_reads_eof_and_close(tmp_path):
    from hexrd.hedm._direct_io import DirectFile

    payload = bytes(range(256)) * 65 + b'end'
    path = tmp_path / 'direct.bin'
    path.write_bytes(payload)
    with DirectFile(path) as source:
        for start, count in [(0, 8), (4093, 17), (15, 9000), (len(payload) - 7, 32)]:
            assert source.seek(start) == start
            target = bytearray(count)
            read = source.readinto(target)
            assert bytes(target[:read]) == payload[start : start + count]
        source.seek(-3, os.SEEK_END)
        assert source.read(10) == b'end'
        assert source.read(1) == b''
        assert source.readinto(bytearray()) == 0
        with pytest.raises(ValueError):
            source.seek(-1)
    source.close()
    with pytest.raises(ValueError):
        source.read(1)


def test_direct_reader_fails_if_direct_io_is_unavailable(tmp_path, monkeypatch):
    from hexrd.hedm._direct_io import DirectFile

    monkeypatch.delattr(os, 'O_DIRECT', raising=False)
    with pytest.raises(OSError, match='require Linux'):
        DirectFile(tmp_path / 'input')


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('malformed', [None, 'rows', 'columns', 'duplicate'])
def test_parallel_pixel_queries_preserve_csr_and_fallback(
    tmp_path, direct, malformed, monkeypatch
):
    import hdf5plugin
    from hexrd.core.extensions import transforms as native
    from hexrd.core.imageseries._sparse import sample_pixels

    if direct:
        if not hasattr(os, 'O_DIRECT'):
            pytest.skip('Linux direct I/O')
        monkeypatch.setenv('HEXRD_FIT_GRAINS_DIRECT_IO', '1')
    reader = make_reader(
        tmp_path / 'parallel.h5',
        malformed=malformed,
        compression=hdf5plugin.Blosc(cname='zstd'),
    )
    cache, reference = FrameCache(32 * 1024**2), FrameCache(0)
    cache.chunk_workers = 4
    rows, cols = np.indices(reader.shape)
    rows, cols = rows.ravel(), cols.ravel()
    try:
        for frame in [0, 4, 3, 0]:
            actual = reader.sample_pixels(frame, rows, cols, cache)
            if malformed and frame != 4:
                assert actual is None
            else:
                expected = StreamingView(View(reader), reference)._sample_frame(
                    frame, rows, cols, native, sample_pixels
                )
                assert actual.tobytes() == expected.tobytes()
            assert cache.chunk_pool is not None
            assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
    finally:
        cache.close()
        reference.close()
