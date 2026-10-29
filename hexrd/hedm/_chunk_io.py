"""Bounded parallel reads of immutable, Blosc-compressed FCH5 chunks.

Only the coordinator calls HDF5. Workers use positional reads and Blosc's
reentrant C context API, so neither HDF5's global lock nor Python's GIL
serializes disk access and decompression. Unsupported layouts use h5py.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections import namedtuple
import ctypes
from functools import lru_cache
import mmap
import os
from pathlib import Path
import threading

import numpy as np

from hexrd.hedm._progress import check_cancelled

_MIB = 1024**2
ChunkInfo = namedtuple('ChunkInfo', 'byte_offset size filter_mask')
SelectedChunk = namedtuple('SelectedChunk', 'array complete decoded_bytes')


@lru_cache(maxsize=1)
def _blosc():
    import hdf5plugin

    # Use the codec already shipped with HEXRD's HDF5 dependency. Absence of
    # these public C-Blosc symbols simply disables this optional fast path.
    for path in Path(hdf5plugin.PLUGINS_PATH).glob('*blosc.*'):
        try:
            lib = ctypes.CDLL(str(path))
            for symbol in (
                'blosc_cbuffer_validate',
                'blosc_decompress_ctx',
                'blosc_cbuffer_sizes',
                'blosc_cbuffer_metainfo',
                'blosc_getitem',
            ):
                getattr(lib, symbol)
            return lib
        except (OSError, AttributeError):
            continue
    return None


class ChunkPool:
    """One small executor shared by all detectors in a coordinator."""

    max_chunk = _MIB
    directory_budget = 8 * _MIB

    def __init__(self, cache, workers):
        # Leave at least a quarter of the cache for indexes and retained data.
        # Bound codec scratch as well as inputs/results, without multiplying
        # this allowance by the number of open detector files.
        # Native frame jobs and Python row lookahead share their active I/O
        # limit, but each retains its own reusable aligned source workspace.
        per_worker = 8 * self.max_chunk + 16384
        # Also cover the bounded native query's sort/inverse/missing buffers
        # and the coordinator's borrowed rows, columns and sampled result.
        fixed = self.directory_budget + 8 * self.max_chunk + 16384
        capacity = max(1, (cache.max_bytes * 3 // 4 - fixed) // per_worker)
        self.workers = min(workers, os.cpu_count() or 1, capacity)
        # Includes decoded results, compressed input, direct-I/O alignment and
        # one result borrowed by the consumer. This is INSIDE the cache budget.
        self.reserved_bytes = fixed + self.workers * per_worker
        self.directory_bytes = 0
        self.local = threading.local()
        from hexrd.core.extensions import transforms as native

        self.native_reader = native._BloscReader(_blosc())
        self.native_reader.configure(self.workers)
        cache.reserved_bytes += self.reserved_bytes
        cache._evict(0)
        self.executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix='hexrd-chunk'
        )

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.native_reader.close()


class ChunkReader:
    def __init__(self, handle, pool):
        self.pool = pool
        self.specs = {}
        direct = getattr(handle, '_hexrd_direct_file', None)
        self.direct = direct is not None
        self.fd = direct.fd if self.direct else handle.id.get_vfd_handle()

    @classmethod
    def create(cls, handle, cache):
        workers = cache.chunk_workers
        if os.name != 'posix' or workers < 2 or cache.max_bytes < 32 * _MIB:
            return None
        if handle.driver not in ('sec2', 'fileobj') or handle.userblock_size:
            return None
        if handle.driver == 'fileobj' and not hasattr(handle, '_hexrd_direct_file'):
            return None
        codec = _blosc()
        if codec is None:
            return None
        if cache.chunk_pool is None:
            cache.chunk_pool = ChunkPool(cache, workers)
        return cls(handle, cache.chunk_pool)

    def _spec(self, dataset):
        if dataset.name in self.specs:
            return self.specs[dataset.name]
        if (
            dataset.ndim != 2
            or dataset.chunks is None
            or dataset.chunks[1] != 1
            or dataset.dtype.kind not in 'iuf'
            or dataset.is_virtual
            or dataset.external
        ):
            return None
        size = int(np.prod(dataset.chunks)) * dataset.dtype.itemsize
        filters = dataset.id.get_create_plist()
        supported = (
            size <= self.pool.max_chunk
            and filters.get_nfilters() == 1
            and filters.get_filter(0)[0] == 32001
        )
        count = (len(dataset) + dataset.chunks[0] - 1) // dataset.chunks[0]
        shape = (count, dataset.shape[1])
        directory_size = count * dataset.shape[1] * 16
        if (
            not supported
            or not hasattr(dataset.id, 'chunk_iter')
            or self.pool.directory_bytes + directory_size > self.pool.directory_budget
        ):
            self.specs[dataset.name] = None
            return None
        offsets = np.full(shape, -1, dtype='i8')
        sizes = np.zeros(shape, dtype='u4')
        masks = np.zeros(shape, dtype='u4')
        visited = 0
        chunk_size = dataset.chunks[0]

        def record(info):
            nonlocal visited
            if visited % 512 == 0:
                check_cancelled()
            visited += 1
            index, column = info.chunk_offset
            index //= chunk_size
            offsets[index, column] = info.byte_offset
            sizes[index, column] = info.size
            masks[index, column] = info.filter_mask

        # One B-tree traversal, rather than a disk-address lookup for every
        # pixel query in every tolerance pass. No pixel data is scanned here.
        try:
            dataset.id.chunk_iter(record)
        except (AttributeError, NotImplementedError):
            self.specs[dataset.name] = None
            return None
        self.pool.directory_bytes += directory_size
        offsets.flags.writeable = sizes.flags.writeable = masks.flags.writeable = False
        spec = dataset.chunks, dataset.dtype, offsets, sizes, masks
        self.specs[dataset.name] = spec
        return spec

    def supported(self, dataset):
        return self._spec(dataset) is not None

    def _decode(self, info, shape, dtype, selected=None):
        """No HDF5 calls, mutable file position, or Python callback in workers."""
        expected = shape[0] * shape[1] * dtype.itemsize
        if (
            selected is not None
            and not 0 <= selected[0] <= selected[1] <= shape[0] * shape[1]
        ):
            raise ValueError('Selected FCH5 pixels exceed their chunk')
        size, offset = int(info.size), int(info.byte_offset)
        if size <= 0 or size > expected + 16:
            raise OSError('Invalid FCH5 compressed chunk size')
        result = np.empty(shape, dtype=dtype)
        source = getattr(self.pool.local, 'buffer', None)
        if source is None:
            source = self.pool.local.buffer = mmap.mmap(-1, self.pool.max_chunk + 8192)
        first, last = selected if selected is not None else (-1, -1)
        decoded_bytes = self.pool.native_reader.read(
            self.fd,
            offset,
            size,
            source,
            result,
            info.filter_mask,
            first,
            last,
            self.direct,
        )
        if decoded_bytes < 0:
            if decoded_bytes <= -1000:
                error = -1000 - decoded_bytes
                raise OSError(error, os.strerror(error))
            message = {
                -1: 'Truncated FCH5 chunk',
                -2: 'Invalid Blosc FCH5 chunk',
                -3: 'Cannot decompress Blosc FCH5 chunk',
            }[decoded_bytes]
            raise OSError(message)
        result.flags.writeable = False
        if selected is not None:
            result = SelectedChunk(result, decoded_bytes == expected, decoded_bytes)
        return result

    def chunks(self, dataset, column, indices, cache, reader, kind, selected=None):
        """Yield requested chunks in order, with a fixed-size rolling queue."""
        pending = {}
        future = None
        iterator = iter(indices)
        shape, dtype, offsets, sizes, masks = self._spec(dataset)

        def submit():
            check_cancelled()
            try:
                index = int(next(iterator))
            except StopIteration:
                return False
            if (reader, kind, index) in cache.frames:
                pending[index] = None
            else:
                info = ChunkInfo(
                    int(offsets[index, column]),
                    int(sizes[index, column]),
                    int(masks[index, column]),
                )
                # Unallocated chunks have the dataset's fill value. Let HDF5
                # handle them, without making any HDF5 call from a worker.
                pending[index] = (
                    self.pool.executor.submit(
                        self._decode,
                        info,
                        shape,
                        dtype,
                        selected.get(index) if selected is not None else None,
                    )
                    if info.byte_offset >= 0 and info.size
                    else None
                )
            return True

        try:
            for _ in range(self.pool.workers):
                if not submit():
                    break
            while pending:
                check_cancelled()
                index = next(iter(pending))
                future = pending.pop(index)
                value = None
                if future is not None:
                    while True:
                        check_cancelled()
                        try:
                            value = future.result(timeout=0.025)
                            break
                        except TimeoutError:
                            if future.done():
                                raise
                            continue
                submit()
                yield index, value
                # Do not retain the previously yielded result while waiting.
                value = future = None
        finally:
            if future is not None:
                pending[-1] = future
            for future in pending.values():
                if future is not None:
                    future.cancel()
            # Running jobs own only one bounded disk read and codec call. Join
            # them before HDF5/file descriptors can be closed by cancellation.
            for future in pending.values():
                if future is not None:
                    try:
                        future.result()
                    except BaseException:
                        pass
