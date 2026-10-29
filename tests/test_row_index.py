"""Frame-index read-ahead must be exact, bounded and cancellable."""

import threading
import time

import hdf5plugin
import numpy as np
import pytest

from hexrd.hedm._image_cache import FrameCache
from hexrd.hedm._progress import _cancellation_check


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('malformed', [None, 'rows', 'columns', 'duplicate'])
def test_prefetched_indexes_match_existing_row_validation(
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
    cache, reference = FrameCache(32 * 1024**2), FrameCache(12000)
    cache.chunk_workers = 4
    frames = np.array([2, 0, 3, 1, 4])
    rows, cols = np.indices(reader.shape)
    rows, cols = rows.ravel(), cols.ravel()
    try:
        sampler = reader._chunk_reader(cache)
        with reader.prefetch_frame_indexes(frames, cache):
            assert sampler.row_queue is not None
            for frame in frames:
                actual = reader.sample_pixels(frame, rows, cols, cache)
                expected = reader.sample_pixels(frame, rows, cols, reference)
                if expected is None:
                    assert actual is None
                else:
                    assert actual.tobytes() == expected.tobytes()
                assert len(sampler.row_queue.pending) <= cache.chunk_pool.workers
                assert cache.bytes + cache.reserved_bytes <= cache.max_bytes
        assert sampler.row_queue is None
    finally:
        cache.close()
        reference.close()


def test_skipping_frames_with_reused_pixels_does_not_break_index_order(tmp_path):
    from test_fch5_pixels import make_reader

    reader = make_reader(tmp_path / 'rows.h5', compression=hdf5plugin.Blosc())
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    try:
        with reader.prefetch_frame_indexes([2, 0, 3, 1, 4], cache):
            for frame in [3, 4]:
                actual = reader.sample_pixels(frame, [4], [9], cache)
                assert actual[0] == (7 if frame == 3 else 0)
    finally:
        cache.close()


def test_cancel_joins_index_jobs_before_returning(tmp_path, monkeypatch):
    from test_fch5_pixels import make_reader
    from hexrd.hedm import _row_index

    reader = make_reader(tmp_path / 'rows.h5', compression=hdf5plugin.Blosc())
    cache = FrameCache(32 * 1024**2)
    cache.chunk_workers = 4
    active = set()
    lock, started = threading.Lock(), threading.Event()
    original = _row_index._build

    def build(*args):
        with lock:
            active.add(threading.get_ident())
            started.set()
        try:
            time.sleep(0.03)
            return original(*args)
        finally:
            with lock:
                active.remove(threading.get_ident())

    monkeypatch.setattr(_row_index, '_build', build)

    def cancel():
        if started.is_set():
            raise KeyboardInterrupt

    token = _cancellation_check.set(cancel)
    begin = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            with reader.prefetch_frame_indexes([0, 1, 2, 3, 4], cache):
                reader.sample_pixels(0, [4], [9], cache)
        assert not active and time.monotonic() - begin < 1
        assert reader._chunk_reader(cache).row_queue is None
    finally:
        _cancellation_check.reset(token)
        cache.close()
