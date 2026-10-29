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
from time import monotonic

import numpy as np

from hexrd.hedm._progress import check_cancelled

_MIB = 1024**2
ChunkInfo = namedtuple('ChunkInfo', 'byte_offset size filter_mask')


@lru_cache(maxsize=1)
def _blosc():
    import hdf5plugin

    # Use the codec already shipped with HEXRD's HDF5 dependency. Absence of
    # these public C-Blosc symbols simply disables this optional fast path.
    for path in Path(hdf5plugin.PLUGINS_PATH).glob('*blosc.*'):
        try:
            lib = ctypes.CDLL(str(path))
            lib.blosc_cbuffer_validate.argtypes = (
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            )
            lib.blosc_cbuffer_validate.restype = ctypes.c_int
            lib.blosc_decompress_ctx.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            )
            lib.blosc_decompress_ctx.restype = ctypes.c_int
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
        per_worker = 6 * self.max_chunk + 16384
        fixed = self.directory_budget + self.max_chunk + 16384
        capacity = max(1, (cache.max_bytes * 3 // 4 - fixed) // per_worker)
        self.workers = min(workers, os.cpu_count() or 1, capacity)
        # Includes decoded results, compressed input, direct-I/O alignment and
        # one result borrowed by the consumer. This is INSIDE the cache budget.
        self.reserved_bytes = fixed + self.workers * per_worker
        self.directory_bytes = 0
        self.local = threading.local()
        cache.reserved_bytes += self.reserved_bytes
        cache._evict(0, None)
        self.executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix='hexrd-chunk'
        )

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class ChunkReader:
    def __init__(self, handle, pool, codec):
        self.pool, self.codec = pool, codec
        self.specs = {}
        direct = getattr(handle, '_hexrd_direct_file', None)
        self.direct = direct is not None
        self.fd = direct.fd if self.direct else handle.id.get_vfd_handle()

    @classmethod
    def create(cls, handle, cache):
        workers = cache.chunk_workers
        if workers < 2 or cache.max_bytes < 32 * _MIB:
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
        return cls(handle, cache.chunk_pool, codec)

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
        spec = dataset.chunks, dataset.dtype, offsets, sizes, masks
        self.specs[dataset.name] = spec
        return spec

    def supported(self, dataset):
        return self._spec(dataset) is not None

    def _decode(self, info, shape, dtype):
        """No HDF5 calls, mutable file position, or Python callback in workers."""
        started = monotonic()
        expected = shape[0] * shape[1] * dtype.itemsize
        size, offset = int(info.size), int(info.byte_offset)
        if size <= 0 or size > expected + 16:
            raise OSError('Invalid FCH5 compressed chunk size')
        result = np.empty(shape, dtype=dtype)
        if self.direct:
            first = offset // 4096 * 4096
            skip = offset - first
            allocated = ((skip + size + 4095) // 4096) * 4096
            source = getattr(self.pool.local, 'buffer', None)
            if source is None:
                source = self.pool.local.buffer = mmap.mmap(
                    -1, self.pool.max_chunk + 8192
                )
            # Reuse one aligned input buffer per worker. Limit each preadv to
            # the actual compressed extent, even when the buffer is larger.
            with memoryview(source)[:allocated] as target:
                count = os.preadv(self.fd, [target], first)
            if count < skip + size:
                raise OSError('Truncated FCH5 chunk')
            address = ctypes.addressof(ctypes.c_char.from_buffer(source, skip))
            self._decompress(address, size, expected, info.filter_mask, result)
        else:
            source = os.pread(self.fd, size, offset)
            if len(source) != size:
                raise OSError('Truncated FCH5 chunk')
            address = ctypes.cast(ctypes.c_char_p(source), ctypes.c_void_p)
            self._decompress(address, size, expected, info.filter_mask, result)
        result.flags.writeable = False
        return result, monotonic() - started

    def _decompress(self, source, size, expected, mask, result):
        target = result.ctypes.data
        if mask == 1:  # HDF5 explicitly skipped its only filter.
            if size != expected:
                raise OSError('Invalid unfiltered FCH5 chunk size')
            ctypes.memmove(target, source, expected)
            return
        if mask != 0:
            raise OSError('Invalid FCH5 filter mask')
        decoded = ctypes.c_size_t()
        if (
            self.codec.blosc_cbuffer_validate(source, size, ctypes.byref(decoded))
            or decoded.value != expected
        ):
            raise OSError('Invalid Blosc FCH5 chunk')
        # One codec thread per job avoids nested thread-pool oversubscription.
        count = self.codec.blosc_decompress_ctx(source, target, expected, 1)
        if count != expected:
            raise OSError('Cannot decompress Blosc FCH5 chunk')

    def chunks(self, dataset, column, indices, cache, reader, kind):
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
                    self.pool.executor.submit(self._decode, info, shape, dtype)
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
                value, seconds = None, 0.0
                if future is not None:
                    while True:
                        check_cancelled()
                        try:
                            value, seconds = future.result(timeout=0.025)
                            break
                        except TimeoutError:
                            if future.done():
                                raise
                            continue
                submit()
                yield index, value, seconds
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
