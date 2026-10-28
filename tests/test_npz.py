from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

from hexrd.core.extensions import transforms
from hexrd.core.utils.npz import _NpzReader


def test_concurrent_first_frame_reads_wait_for_complete_index(tmp_path, monkeypatch):
    path = tmp_path / 'frames.npz'
    np.savez(path, nframes=2, **{'0_data': np.array([1.0]), '1_data': np.array([2.0])})
    reader = _NpzReader(path, metadata_only=True)
    original = transforms._npzIndex
    indexing = threading.Event()
    resume = threading.Event()
    main_thread = threading.current_thread()

    def delayed_index(buffer, metadata_only):
        if not metadata_only and threading.current_thread() is not main_thread:
            indexing.set()
            assert resume.wait(10)
        return original(buffer, metadata_only)

    monkeypatch.setattr(transforms, '_npzIndex', delayed_index)
    with reader, ThreadPoolExecutor(1) as pool:
        first = pool.submit(reader.__getitem__, '0_data')
        try:
            assert indexing.wait(10)
            # The worker has started building the full directory, but its
            # result is not available. The second reader must not treat the
            # existing metadata-only directory as complete.
            np.testing.assert_array_equal(reader['1_data'], [2.0])
        finally:
            resume.set()
        np.testing.assert_array_equal(first.result(timeout=10), [1.0])
