"""Lossless catalogue compression must preserve bits and the byte budget."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.core.extensions import transforms as native
from hexrd.hedm._chunk_io import _blosc
from hexrd.hedm._pixel_cache import PixelCache
from hexrd.hedm._progress import _cancellation_check


@pytest.mark.parametrize('missing', ['extension', 'codec'])
def test_packing_unavailable_keeps_original_bounded_cache(monkeypatch, missing):
    from hexrd.hedm import _chunk_io

    if missing == 'extension':
        monkeypatch.delattr(native, '_BloscReader')
    else:
        monkeypatch.setattr(_chunk_io, '_blosc', lambda: None)
    reader = type('Reader', (), {'dtype': np.dtype('f8')})()
    cache = PixelCache(compression_min_bytes=0)
    cache.set_limit(60000)
    keys = np.arange(4096, dtype='u4')
    values = keys.astype('f8')
    cache.remember(reader, 0, keys, values)
    cache.remember(reader, 1, keys, values)
    assert not cache.packing and cache.bytes <= cache.max_bytes
    assert len(cache.frames) == 1
    found, actual = cache.lookup(reader, 0, keys)
    assert found.all() and actual.tobytes() == values.tobytes()


@pytest.mark.parametrize('algorithm', ['zstd', 'lz4'])
@pytest.mark.parametrize('dtype', ['u2', 'i4', 'u8', 'f4', 'f8', '>f8'])
def test_native_packed_pixels_round_trip_bits_and_ownership(dtype, algorithm):
    codec = native._BloscReader(_blosc())
    keys = 2**63 + np.arange(4096, dtype='u8') * 3
    values = np.arange(4096, dtype=dtype)
    if values.dtype.kind == 'f':
        values[:4] = [-0.0, np.nan, np.inf, -np.inf]
        if dtype == 'f8':
            values.view('u8')[4] = 0x7FF8000000001234
    if dtype == 'u8':
        values += np.uint64(2**63)
    expected_keys, expected_values = keys.tobytes(), values.tobytes()
    packed = codec.pack_pixels(keys, values, algorithm)
    assert packed is not None
    assert packed.storage_bytes < 1024 + keys.nbytes + values.nbytes
    assert packed.decoded_bytes == keys.nbytes + values.nbytes
    keys[:] = 0
    values[:] = 0
    del codec
    restored_keys, restored_values = packed.restore()
    assert restored_keys.tobytes() == expected_keys
    assert restored_values.tobytes() == expected_values
    assert restored_values.dtype == np.dtype(dtype)
    assert not restored_keys.flags.writeable and not restored_values.flags.writeable


def test_packed_frame_limit_is_checked_before_encoding():
    codec = native._BloscReader(_blosc())
    keys = np.arange(65537, dtype='u8')
    values = np.zeros(65537, dtype='f8')
    assert codec.pack_pixels(keys, values) is None
    assert codec.pack_pixels(keys[:-1], values[:-1]) is not None
    assert codec.pack_pixels(keys.astype('i8'), values) is None


@pytest.mark.parametrize('compress', [False, True])
def test_sorted_admission_owns_buffers_and_matches_generic_merge(compress):
    reader = type('Reader', (), {'dtype': np.dtype('f8')})()
    cache = PixelCache(compress=compress, compression_min_bytes=0)
    cache.set_limit(60000)
    keys = np.arange(4096, dtype='u4')
    values = keys.astype('f8')
    expected = values.copy()
    cache.remember(reader, 0, keys, values, sorted_unique=True)
    if compress:
        cache._enable_packing()
        assert cache.packing
    keys[:] = 0
    values[:] = -1
    found, actual = cache.lookup(reader, 0, np.arange(4096, dtype='u4'))
    assert found.all() and actual.tobytes() == expected.tobytes()
    cache.remember(
        reader,
        0,
        np.array([42, 4096], dtype='u4'),
        np.array([-0.0, np.nan]),
        sorted_unique=True,
    )
    expected[42] = -0.0
    expected = np.r_[expected, np.nan]
    found, actual = cache.lookup(reader, 0, np.arange(4097, dtype='u4'))
    assert found.all() and actual.tobytes() == expected.tobytes()
    assert cache.bytes <= cache.max_bytes


def test_pressure_compresses_existing_frames_before_rejecting_new_frames():
    reader = SimpleNamespace(shape=(128, 128), dtype=np.dtype('f8'))
    # Reader identity is a catalogue key.
    reader = type('Reader', (), vars(reader))()
    cache = PixelCache(compression_min_bytes=0)
    cache.set_limit(100000)
    keys, values = np.arange(4096, dtype='u4'), np.arange(4096, dtype='f8')
    values[:4] = [-0.0, np.nan, np.inf, -np.inf]
    try:
        for frame in range(20):
            cache.remember(reader, frame, keys, values)
            assert cache.bytes <= cache.max_bytes
        assert cache.packing and len(cache.frames) == 20
        assert all(item[1] is None for item in cache.frames.values())
        for frame in range(20):
            found, actual = cache.lookup(reader, frame, keys[::-1])
            assert found.all() and actual.tobytes() == values[::-1].tobytes()
        # A query and its subsequent merge share one decoded old entry.
        query = cache.query(reader, 0, [0, 32], [0, 0])
        assert query.missing_count == 1
        assert cache._decoded_source is cache.frames[reader, 0][0]
        missing = np.array([81.0], dtype='f8')
        query.finish(missing)
        cache.remember(reader, 0, query.missing_keys, missing)
        assert cache.query(reader, 0, [32], [0]).finish()[0] == 81
        cache.set_limit(4000)
        assert cache.bytes <= 4000
    finally:
        cache.clear()
    assert not cache.frames and cache.bytes == 0 and cache._decoded is None


def test_cancelling_catalogue_migration_preserves_entries_and_byte_counts():
    reader = type('Reader', (), {'dtype': np.dtype('f8')})()
    cache = PixelCache(compression_min_bytes=0)
    cache.set_limit(1000000)
    keys, values = np.arange(4096, dtype='u4'), np.arange(4096, dtype='f8')
    for frame in range(10):
        cache.remember(reader, frame, keys, values)

    def cancel():
        if any(item[1] is None for item in cache.frames.values()):
            raise KeyboardInterrupt

    token = _cancellation_check.set(cancel)
    try:
        with pytest.raises(KeyboardInterrupt):
            cache.set_limit(500000)
    finally:
        _cancellation_check.reset(token)
    assert cache.bytes == sum(item[2] for item in cache.frames.values())
    for frame in range(10):
        found, actual = cache.lookup(reader, frame, keys)
        assert found.all() and actual.tobytes() == values.tobytes()
    cache.clear()


@pytest.mark.parametrize('dtype', ['u8', 'f4', 'f8'])
def test_native_sorted_merge_matches_original_overwrite_bits(dtype):
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
