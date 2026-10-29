"""Reusing source pixels must preserve values, coordinates and byte bounds."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.hedm._pixel_cache import PixelCache


def source(dtype='f8', shape=(14, 18)):
    return type('Reader', (), {'dtype': np.dtype(dtype), 'shape': shape})()


def read_cached(cache, reader, frame, keys):
    """Exercise the query API and supply zeros only for unknown source pixels."""
    keys = np.asarray(keys, dtype='u8')
    query = cache.query(
        reader,
        frame,
        (keys // reader.shape[1]).astype('i8'),
        (keys % reader.shape[1]).astype('i8'),
    )
    found = ~np.isin(keys, query.missing_keys)
    return found, query.finish(np.zeros(query.missing_count, dtype=reader.dtype))


@pytest.mark.parametrize('dtype', ['i2', 'u2', 'f4', 'f8'])
@pytest.mark.parametrize('compressed', [False, True])
def test_pixels_reused_in_original_order_with_exact_float_bits(
    dtype, compressed, monkeypatch
):
    if not compressed:
        from hexrd.hedm import _pixel_cache

        monkeypatch.setattr(_pixel_cache, '_codec', lambda: None)
    reader = source(dtype)
    cache = PixelCache()
    cache.set_limit(8192)
    keys = np.array([7, 1, 7, 9, 3], dtype='u4')
    values = np.arange(5, dtype=dtype)
    values[2] = values[0]
    if reader.dtype.kind == 'f':
        values[:] = [-0.0, np.nan, -0.0, np.inf, -np.inf]
        if reader.dtype == np.dtype('f8'):
            values.view('u8')[1] = 0x7FF8000000001234
    unique, indices = np.unique(keys, return_index=True)
    cache.remember(reader, 0, unique, values[indices])
    selected = np.array([3, 7, 2, 1, 7, 9], dtype='u4')
    found, actual = read_cached(cache, reader, 0, selected)
    np.testing.assert_array_equal(found, [True, True, False, True, True, True])
    expected = values[[4, 0, 1, 0, 3]]
    assert actual[found].tobytes() == expected.tobytes()
    cache.remember(reader, 0, np.array([2], dtype='u4'), np.array([8], dtype=dtype))
    found, actual = read_cached(cache, reader, 0, selected)
    assert found.all() and actual[2] == 8
    assert cache.bytes <= cache.max_bytes
    keys, values, _ = cache._arrays(cache.frames[reader, 0])
    assert not keys.flags.writeable and not values.flags.writeable
    # Separate source identities and frames can never share measurements.
    assert not read_cached(cache, source(dtype), 0, selected)[0].any()
    assert not read_cached(cache, reader, 1, selected)[0].any()


def test_packed_coordinates_do_not_round_or_collide_above_float_precision():
    cache = PixelCache()
    reader = source(shape=(2**27, 2**27))
    rows, cols = np.array([2**27 - 1] * 2), np.array([1, 2])
    keys = cache.query(reader, 0, rows, cols).missing_keys
    assert keys.dtype == np.dtype('u8')
    assert keys[1] - keys[0] == 1
    assert cache.query(reader, 0, [-1], [0]) is None
    assert cache.query(reader, 0, [0], [2**27]) is None
    assert cache.query(source(shape=(2**40, 2**40)), 0, [0], [0]) is None


@pytest.mark.parametrize('dtype', ['u2', 'u8', 'f4', 'f8'])
@pytest.mark.parametrize('shape', [(14, 18), (2**27, 2**27)])
def test_native_query_matches_catalogue_bits_with_duplicates_and_missing_pixels(
    dtype, shape
):
    reader = source(dtype, shape)
    cache = PixelCache()
    cache.set_limit(8192)
    rows = np.array([shape[0] - 1, 0, shape[0] - 1, 1], dtype='i8')
    cols = np.array([2, 1, 3, 4], dtype='i8')
    keys = cache.query(reader, 0, rows, cols).missing_keys
    values = np.array([7, 8, 9, 10], dtype=dtype)
    if reader.dtype.kind == 'f':
        values[:] = [-0.0, np.nan, np.inf, -np.inf]
    elif dtype == 'u8':
        values[:] = np.array([2**63 + 1, 2**63 + 2, 2**63 + 3, 2**63 + 4], dtype='u8')
    cache.remember(reader, 0, keys, values[np.lexsort((cols, rows))])
    indices = [2, 0, 2, 3, 1, 0]
    selected_rows = np.r_[rows[indices], 2]
    selected_cols = np.r_[cols[indices], 5]
    query = cache.query(reader, 0, selected_rows, selected_cols)
    assert query is not None and query.unique_count == 5 and query.missing_count == 1
    np.testing.assert_array_equal(query.rows, [2])
    np.testing.assert_array_equal(query.cols, [5])
    missing = np.array([11], dtype=dtype)
    actual = query.finish(missing)
    expected = np.r_[values[indices], missing]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    cache.remember(reader, 0, query.missing_keys, missing)
    cached = cache.query(reader, 0, selected_rows, selected_cols)
    assert cached.missing_count == 0
    assert cached.finish().tobytes() == expected.tobytes()
    assert cache.bytes <= cache.max_bytes


def test_native_query_rejects_invalid_coordinates_and_handles_empty_input():
    cache = PixelCache()
    reader = source()
    assert cache.query(reader, 0, [-1], [0]) is None
    assert cache.query(reader, 0, [0], [18]) is None
    assert cache.query(source(shape=(2**40, 2**40)), 0, [0], [0]) is None
    # Keep the established floating-input behavior in the generic fallback.
    assert cache.query(reader, 0, [0.5], [0.5]) is None
    query = cache.query(reader, 0, np.array([], dtype='i8'), np.array([], dtype='i8'))
    assert query.unique_count == query.missing_count == 0
    assert query.finish().shape == (0,)


def test_capacity_does_not_thrash_frames_before_next_tolerance_pass():
    reader = source()
    cache = PixelCache()
    cache.set_limit(2300)
    keys, values = np.arange(4, dtype='u4'), np.arange(4, dtype='f8')
    cache.remember(reader, 0, keys, values)
    cache.remember(reader, 1, keys, values)
    for frame in range(2, 100):
        cache.remember(reader, frame, keys, values)
        assert cache.bytes <= cache.max_bytes
    assert read_cached(cache, reader, 0, keys)[0].all()
    assert read_cached(cache, reader, 1, keys)[0].all()
    cache.set_limit(1150)
    assert len(cache.frames) == 1 and cache.bytes <= 1150
    cache.set_limit(0)
    assert cache.bytes == 0 and not cache.frames


def test_file_reader_reuses_pixels_across_windows_without_more_source_reads(tmp_path):
    from test_fch5_pixels import make_reader
    from hexrd.hedm._image_cache import FrameCache

    reader = make_reader(tmp_path / 'frames.h5')
    cache = FrameCache(12000)
    cache.pixel_cache = PixelCache()
    cache.pixel_cache.set_limit(8192)
    rows = np.array([1, 1, 4, 8, 12, 1])
    cols = np.array([2, 2, 5, 9, 13, 2])
    try:
        expected = reader.sample_pixels(0, rows, cols, cache)
        sampler = reader._chunk_reader(cache)
        original = sampler.sample

        def no_read(*args):
            pytest.fail('cached source pixels were read again')

        sampler.sample = no_read
        actual = reader.sample_pixels(0, rows[::-1], cols[::-1], cache)
        assert actual.tobytes() == expected[::-1].tobytes()
        sampler.sample = original
        # New coordinates still read the original source. The old pixel data
        # survives clearing decoder payloads between measurement groups.
        cache.clear_payload()
        actual = reader.sample_pixels(0, np.r_[rows, 0], np.r_[cols, 0], cache)
        assert actual[:-1].tobytes() == expected.tobytes() and actual[-1] == 0
    finally:
        cache.close()
    assert cache.pixel_cache is None


def test_cancellation_discards_retained_pixels_and_restores_fit_context(
    tmp_path, monkeypatch
):
    from test_fch5_pixels import make_reader
    from hexrd.hedm import fitgrains
    from hexrd.hedm._staged_spots import fit_staged, StagedCancelled
    from hexrd.hedm._image_cache import FrameCache, BoundedSeries, View
    from hexrd.core.imageseries.omega import OmegaImageSeries
    from hexrd.core.instrument._spotutils import (
        sparse_frame_view,
        clear_sparse_frame_views,
    )
    from hexrd.hedm._progress import _cancellation_check

    reader = make_reader(tmp_path / 'frames.h5')
    cache = FrameCache(12000)
    series = OmegaImageSeries(
        BoundedSeries(
            View(reader),
            cache,
            {'omega': np.column_stack((np.arange(5), np.arange(1, 6)))},
            5,
        )
    )

    class Instrument:
        def pull_spots(self):
            return sparse_frame_view(series).extract(
                np.array([[0, 1, 2, 3, 4]]),
                np.array([0, 4]),
                np.array([1, 4, 8, 12]),
                np.array([2, 5, 9, 13]),
                np.array([[5, 2, 2]]),
                5,
            )

    def steps(grain):
        yield ((), {})
        pytest.fail('cancellation should happen during measurement')

    monkeypatch.setattr(fitgrains, '_fit_grain_steps', steps)
    params = {
        'grains_table': np.zeros((1, 21)),
        'imgser_dict': {'panel': series},
        'instrument': Instrument(),
        'staged_workspace': 16 * 1024**2,
        'tth_tol': [1],
    }
    observed = []

    def cancel():
        if cache.pixel_cache is not None and cache.pixel_cache.bytes:
            observed.append(cache.pixel_cache.bytes)
            return True
        return False

    try:
        with pytest.raises(StagedCancelled):
            fit_staged(params, SimpleNamespace(update=lambda n: None), cancel)
        assert observed and cache.pixel_cache.bytes == 0
        assert _cancellation_check.get() is None
        assert not hasattr(fitgrains, 'paramMP')
    finally:
        cache.close()
        clear_sparse_frame_views()


@pytest.mark.parametrize('count, distinct', [(0, 1), (65536, 17), (65536, 65536)])
@pytest.mark.parametrize('shape', [(257, 263), (2**20, 2**20)])
def test_native_query_scatter_matches_numpy_at_tile_limit(count, distinct, shape):
    rng = np.random.default_rng(802)
    reader = source(shape=shape, dtype='f8')
    cache = PixelCache()
    cache.set_limit(2 * 1024**2)
    possible = rng.integers(0, shape[0] * shape[1], distinct, dtype='u8')
    selected = possible[rng.integers(0, distinct, count)]
    rows = (selected // shape[1]).astype('i8')
    cols = (selected % shape[1]).astype('i8')
    query = cache.query(reader, 0, rows, cols)
    assert query is not None
    expected_keys = np.unique(selected)
    np.testing.assert_array_equal(query.missing_keys, expected_keys)
    values = expected_keys.astype('f8')
    if len(values):
        values[::3] = -0.0
        values[1::3] = np.nan
    expected = values[np.searchsorted(expected_keys, selected)]
    assert query.finish(values).tobytes() == expected.tobytes()
    cache.remember(reader, 0, query.missing_keys, values)
    cached = cache.query(reader, 0, rows[::-1], cols[::-1])
    assert cached.missing_count == 0
    assert cached.finish().tobytes() == expected[::-1].tobytes()


def test_sorted_admission_owns_buffers_and_matches_generic_merge():
    reader = source(shape=(1, 65536))
    cache = PixelCache()
    cache.set_limit(60000)
    keys = np.arange(4096, dtype='u4')
    values = keys.astype('f8')
    expected = values.copy()
    cache.remember(reader, 0, keys, values)
    keys[:] = 0
    values[:] = -1
    found, actual = read_cached(cache, reader, 0, np.arange(4096, dtype='u4'))
    assert found.all() and actual.tobytes() == expected.tobytes()
    cache.remember(
        reader,
        0,
        np.array([42, 4096], dtype='u4'),
        np.array([-0.0, np.nan]),
    )
    expected[42] = -0.0
    expected = np.r_[expected, np.nan]
    found, actual = read_cached(cache, reader, 0, np.arange(4097, dtype='u4'))
    assert found.all() and actual.tobytes() == expected.tobytes()
    assert cache.bytes <= cache.max_bytes


@pytest.mark.parametrize('dtype', ['u8', 'f4', 'f8'])
def test_native_sorted_merge_matches_original_overwrite_bits(dtype):
    from hexrd.core.extensions import transforms as native

    keys_a = np.array([0, 1, 5, 6], dtype='u4')
    keys_b = np.array([1, 3, 6], dtype='u8')
    a = np.array([3, 4, 5, 6], dtype=dtype)
    b = np.array([7, 8, 9], dtype=dtype)
    if a.dtype.kind == 'f':
        a[0] = -0.0
        b[:] = [np.nan, np.inf, -np.inf]
    else:
        a += np.uint64(2**63)
        b += np.uint64(2**63)
    keys, values = native._mergePixelEntries(keys_a, a, keys_b, b)
    np.testing.assert_array_equal(
        keys, np.array([0, 1, 3, 5, 6], dtype='u8'), strict=True
    )
    assert (
        values.tobytes()
        == np.array([a[0], b[0], b[1], a[2], b[2]], dtype=dtype).tobytes()
    )


@pytest.mark.parametrize('compressed', [False, True])
def test_strided_input_entries_keep_exact_values(compressed, monkeypatch):
    if not compressed:
        from hexrd.hedm import _pixel_cache

        monkeypatch.setattr(_pixel_cache, '_codec', lambda: None)
    reader = source(shape=(1, 128))
    cache = PixelCache()
    cache.set_limit(8192)
    keys, values = np.arange(64, dtype='u4')[::2], np.arange(64, dtype='f8')[::2]
    cache.remember(reader, 0, keys, values)
    found, actual = read_cached(cache, reader, 0, keys)
    assert found.all() and actual.tobytes() == values.tobytes()


def test_array_codec_checks_types_sizes_and_corrupt_input():
    codec = PixelCache().codec
    if codec is None:
        pytest.skip('native Blosc codec is optional')
    assert codec.encode(np.array([object()])) is None
    assert codec.encode(np.zeros(1024**2 + 1, dtype='u1')) is None
    assert codec.encode(np.arange(12, dtype='f8')[::2]) is None
    with pytest.raises(ValueError, match='dtype'):
        codec.decode(b'invalid', np.dtype('O'), 1)
    with pytest.raises(ValueError, match='dimensions'):
        codec.decode(b'invalid', np.dtype('f8'), 1024**2)
    with pytest.raises(RuntimeError, match='data'):
        codec.decode(b'invalid', np.dtype('f8'), 1)
