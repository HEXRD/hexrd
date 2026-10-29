"""Bounded read-ahead of compact FCH5 row indexes, without worker HDF5 calls."""

from concurrent.futures import CancelledError, TimeoutError
from collections import namedtuple
from contextlib import contextmanager
import threading
import os

import numpy as np

from hexrd.hedm._progress import check_cancelled

IndexResult = namedtuple('IndexResult', 'array decoded_bytes')


def wait_index(future):
    while True:
        check_cancelled()
        try:
            return future.result(timeout=0.025)
        except TimeoutError:
            if future.done():
                raise


def _build(reader, spec, start, stop, height, cancelled):
    shape, dtype, addresses, sizes, masks = spec
    pool = reader.pool
    source = getattr(pool.local, 'buffer', None)
    if source is None:
        import mmap

        source = pool.local.buffer = mmap.mmap(-1, pool.max_chunk + 8192)
    pixels = getattr(pool.local, 'row_buffer', None)
    if pixels is None or pixels.shape != shape:
        pixels = pool.local.row_buffer = np.empty(shape, dtype=dtype)
    result = np.full(height + 1, stop - start, dtype='i4')
    invalid = np.array([-1], dtype='i8')
    first, previous, next_row = start, -1, 0
    decoded_bytes = 0
    while first < stop:
        if cancelled.is_set():
            raise CancelledError()
        chunk, offset = divmod(first, shape[0])
        count = pool.native_reader.read(
            reader.fd,
            int(addresses[chunk, 0]),
            int(sizes[chunk, 0]),
            source,
            pixels,
            int(masks[chunk, 0]),
            -1,
            -1,
            reader.direct,
        )
        if count < 0:
            if count <= -1000:
                error = -1000 - count
                raise OSError(error, os.strerror(error))
            raise OSError('Cannot read FCH5 row-coordinate chunk')
        decoded_bytes += count
        end = min(stop, (chunk + 1) * shape[0])
        rows = pixels[offset : offset + end - first, 0]
        if rows[0] < previous or rows[-1] >= height or np.any(rows[1:] < rows[:-1]):
            return IndexResult(invalid, decoded_bytes)
        last = int(rows[-1]) + 1
        if last > next_row:
            result[next_row:last] = (
                first
                - start
                + np.searchsorted(rows, np.arange(next_row, last, dtype=dtype))
            )
            next_row = last
        previous, first = int(rows[-1]), end
    result.flags.writeable = False
    return IndexResult(result, decoded_bytes)


class IndexQueue:
    def __init__(self, sampler, spec, frames, cache):
        self.sampler, self.spec, self.cache = sampler, spec, cache
        self.frames = iter(map(int, frames))
        self.pending = {}
        self.current = None
        self.last_frame = None
        self.cancelled = threading.Event()

    def _submit(self):
        check_cancelled()
        try:
            frame = next(self.frames)
        except StopIteration:
            return False
        sampler, cache = self.sampler, self.cache
        start, stop = map(int, sampler.bounds[frame])
        shape, _, addresses, sizes, masks = self.spec
        first, last = start // shape[0], (stop + shape[0] - 1) // shape[0]
        eligible = (
            0 <= start <= stop <= min(len(sampler.indices), len(sampler.values))
            and stop - start <= np.iinfo('i4').max
            and last - first <= 128
        )
        if eligible:
            eligible = (
                np.all(addresses[first:last, 0] >= 0)
                and np.all(sizes[first:last, 0] > 0)
                and np.all(masks[first:last, 0] <= 1)
            )
        key = sampler.reader, 'row-offsets', frame
        self.pending[frame] = (
            sampler.chunk_reader.pool.executor.submit(
                _build,
                sampler.chunk_reader,
                self.spec,
                start,
                stop,
                sampler.reader.shape[0],
                self.cancelled,
            )
            if eligible and key not in cache.metadata
            else None
        )
        return True

    def take(self, frame):
        if frame == self.last_frame:
            return self.current
        while self.pending:
            check_cancelled()
            index = next(iter(self.pending))
            previous = self.current
            self.current = self.pending.pop(index)
            # At most one skipped index is being drained in addition to the
            # fixed look-ahead. Do not accumulate abandoned frame jobs.
            if previous is not None:
                wait_index(previous)
            self._submit()
            self.last_frame = index
            if index == frame:
                return self.current
        return None

    def close(self):
        self.cancelled.set()
        pending = [self.current, *self.pending.values()]
        for future in pending:
            if future is not None:
                future.cancel()
        for future in pending:
            if future is not None:
                try:
                    future.result()
                except BaseException:
                    pass
        self.pending.clear()
        self.current = None


@contextmanager
def prefetch_rows(sampler, frames, cache):
    from hexrd.hedm._chunk_io import ChunkReader

    if sampler.chunk_reader is None:
        sampler.chunk_reader = ChunkReader.create(sampler.handle(), cache)
    reader = sampler.chunk_reader
    queue = None
    try:
        if (
            reader is not None
            and reader.pool.native_reader is not None
            and sampler.indices.dtype == np.dtype('u2')
            and sampler.reader.shape[0] <= 32767
            and reader.supported(sampler.indices)
        ):
            queue = IndexQueue(sampler, reader._spec(sampler.indices), frames, cache)
            for _ in range(reader.pool.workers):
                if not queue._submit():
                    break
            sampler.row_queue = queue
        yield
    finally:
        if queue is not None:
            queue.close()
            sampler.row_queue = None
