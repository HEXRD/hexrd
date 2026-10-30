"""Metadata and budget ownership for the single native FCH5 decoder pool."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from hexrd.hedm._progress import check_cancelled

_MIB = 1024**2


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
    """Shared native decoder and its cache reservation."""

    max_chunk = _MIB
    directory_budget = 8 * _MIB

    def __init__(self, cache, workers):
        # Native readers own one aligned input and one decoded chunk per
        # worker. Directories, pixel queries and row-index scratch are shared.
        per_worker = 2 * self.max_chunk + 8192
        fixed = self.directory_budget + 16 * self.max_chunk
        capacity = max(1, (cache.max_bytes * 3 // 4 - fixed) // per_worker)
        self.workers = min(workers, os.cpu_count() or 1, capacity)
        # Includes decoded results, compressed input, direct-I/O alignment and
        # one result borrowed by the consumer. This is INSIDE the cache budget.
        self.reserved_bytes = fixed + self.workers * per_worker
        self.directory_bytes = 0
        from hexrd.core.extensions import transforms as native

        self.native_reader = native._BloscReader(_blosc())
        self.native_reader.configure(self.workers)
        cache.reserved_bytes += self.reserved_bytes
        cache._evict(0)

    def close(self):
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
        if os.name != 'posix' or workers < 1 or cache.max_bytes < 32 * _MIB:
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

    def row_index(self, dataset, start, stop, height):
        spec = self._spec(dataset)
        if spec is None:
            return 2, None, 0
        status, offsets, decoded = self.pool.native_reader.read_rows(
            self.fd, spec, start, stop, height, self.direct, check_cancelled
        )
        self.check(status)
        return status, offsets, decoded

    def prefetch(self, dataset, bounds, frames, height, cache, reader):
        spec = self._spec(dataset)
        if spec is None:
            return None
        missing = np.array(
            [
                int(f)
                for f in frames
                if (reader, 'row-offsets', int(f)) not in cache.metadata
                and 0 <= int(bounds[f, 0]) <= int(bounds[f, 1]) <= len(dataset)
                and int(bounds[f, 1]) - int(bounds[f, 0]) <= np.iinfo('i4').max
            ],
            dtype='i8',
        )
        if not missing.size:
            return None
        return self.pool.native_reader.prefetch_rows(
            self.fd, spec, bounds, missing, height, self.direct
        )

    @staticmethod
    def check(status):
        if status <= -1000:
            error = -1000 - status
            raise OSError(error, os.strerror(error))
        if status < 0:
            raise OSError('Cannot read native FCH5 image data')
