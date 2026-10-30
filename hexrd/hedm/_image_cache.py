"""Bounded, process-local image access for grain fitting.

File readers are descriptors until used in a worker. One byte-bounded LRU is shared
by all detector/ROI views in that worker. Extraction pins one source frame, not
all frames referenced by a patch batch. No intermediate files are written.
"""

from collections import OrderedDict
from dataclasses import dataclass
import multiprocessing
import os
import zipfile

from hexrd.hedm._fit_progress import current_reporter

import numpy as np
from scipy import sparse

from hexrd.core.imageseries.imageseriesabc import ImageSeriesABC

_MIB = 1024**2


def _array_storage(array):
    owner = array
    seen = set()
    while getattr(owner, 'base', None) is not None and id(owner) not in seen:
        seen.add(id(owner))
        owner = owner.base
    size = getattr(owner, 'nbytes', None)
    if size is None and isinstance(owner, (bytes, bytearray)):
        size = len(owner)
    return id(owner), max(array.nbytes, size or 0)


def frame_bytes(frame):
    arrays = (
        (frame.data, frame.indices, frame.indptr)
        if sparse.issparse(frame)
        else (np.asarray(frame),)
    )
    owners = dict(_array_storage(array) for array in arrays)
    return 1024 + sum(owners.values())


class FrameCache:
    """An aggregate byte LRU, not a separate allowance for each panel."""

    def __init__(self, max_bytes, max_open_files=8):
        self.max_bytes = max(0, int(max_bytes))
        self.max_open_files = max_open_files
        self._reset()

    def _reset(self):
        self.pid = os.getpid()
        self.frames = OrderedDict()
        self.metadata = OrderedDict()
        self.metadata_bytes = 0
        self.metadata_limit = None
        self.files = OrderedDict()
        self.chunk_workers = getattr(self, 'chunk_workers', 0)
        self.chunk_pool = None
        self.reserved_bytes = 0
        self.pixel_cache = None
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def _process(self):
        if self.pid != os.getpid():
            self.close()
            self._reset()

    def opened(self, reader):
        self._process()
        handle = self.files.pop(reader, None)
        if handle is None:
            while len(self.files) >= self.max_open_files:
                _, old = self.files.popitem(last=False)
                old.close()
            handle = reader.open()
        self.files[reader] = handle
        return handle

    def _evict(self, size):
        while self.bytes + size + self.reserved_bytes > self.max_bytes:
            store = self.frames if self.frames else self.metadata
            _, (_, old_size) = store.popitem(last=False)
            self.bytes -= old_size
            if store is self.metadata:
                self.metadata_bytes -= old_size

    def _cached(self, key, load, *, metadata=False, retain=True):
        """Cache decoded chunks and compact indexes under one byte allowance.

        Protect small row indexes from intensity chunk churn, with a separate
        quota inside (never in addition to) the total cache budget.
        """
        self._process()
        store = self.metadata if metadata else self.frames
        item = store.pop(key, None)
        if item is not None:
            self.hits += 1
            store[key] = item
            return item[0]
        self.misses += 1
        value = load()
        if not retain:
            return value
        size = frame_bytes(value)
        quota = min(
            self.max_bytes - self.reserved_bytes,
            (
                min(self.max_bytes, self.metadata_limit)
                if metadata and self.metadata_limit is not None
                else (
                    min(128 * _MIB, self.max_bytes // 4) if metadata else self.max_bytes
                )
            ),
        )
        if size <= quota:
            if metadata:
                while self.metadata_bytes + size > quota:
                    _, (_, old_size) = store.popitem(last=False)
                    self.bytes -= old_size
                    self.metadata_bytes -= old_size
            self._evict(size)
            store[key] = (value, size)
            self.bytes += size
            if metadata:
                self.metadata_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value

    def array(self, reader, kind, index, load, *, metadata=False):
        value = self._cached((reader, kind, int(index)), load, metadata=metadata)
        value.flags.writeable = False
        return value

    def frame(self, reader, index):
        index = int(index)
        if index < 0:
            index += reader.nframes
        if not 0 <= index < reader.nframes:
            raise IndexError('image frame index out of bounds')

        def load():
            reporter = current_reporter()
            if (
                reporter is not None
                and not reader.resident
                and not isinstance(reader, ProcessedReader)
            ):
                reporter.read_started()
            frame = reader.read(index, self)
            if tuple(frame.shape) != tuple(reader.shape) or frame.dtype != reader.dtype:
                raise ValueError('image frame shape/dtype differs from its metadata')
            return frame

        return self._cached((reader, index), load, retain=not reader.resident)

    def clear_payload(self):
        self.frames.clear()
        self.bytes = self.metadata_bytes

    def clear_frames(self):
        self.clear_payload()
        self.metadata.clear()
        self.metadata_bytes = 0
        self.bytes = 0

    def close(self):
        if self.pixel_cache is not None:
            self.pixel_cache.clear()
            self.pixel_cache = None
        if self.chunk_pool is not None:
            self.chunk_pool.close()
            self.chunk_pool = None
        self.reserved_bytes = 0
        self.clear_frames()
        for file in self.files.values():
            file.close()
        self.files.clear()

    def __getstate__(self):
        return self.max_bytes, self.max_open_files, self.chunk_workers

    def __setstate__(self, state):
        self.max_bytes, self.max_open_files, self.chunk_workers = state
        self._reset()


@dataclass(eq=False)
class FileReader:
    filename: str
    fmt: str
    args: dict
    shape: tuple
    dtype: np.dtype
    nframes: int
    estimate: int | None = None
    peak_frame: int = 0
    resident: bool = False

    def _chunk_reader(self, cache):
        if self.fmt != 'fch5' or (self.estimate or 0) <= max(
            64 * _MIB, cache.max_bytes
        ):
            return None
        from hexrd.hedm._fch5_pixels import Fch5Pixels

        handle = cache.opened(self)
        sampler = getattr(handle, '_hexrd_pixel_sampler', None)
        if sampler is None:
            sampler = Fch5Pixels(self, handle)
            handle._hexrd_pixel_sampler = sampler
        return sampler

    def sample_pixels(self, index, rows, cols, cache):
        if (self, int(index)) in cache.frames:
            return None
        sampler = self._chunk_reader(cache)
        if sampler is None:
            return None
        pixels = cache.pixel_cache
        if pixels is None or not pixels.max_bytes:
            return sampler.sample(int(index), rows, cols, cache)
        query = pixels.query(self, int(index), rows, cols)
        if query is None:
            return sampler.sample(int(index), rows, cols, cache)
        if not query.missing_count:
            return query.finish()
        values = sampler.sample(int(index), query.rows, query.cols, cache)
        if values is None:
            return None
        result = query.finish(values)
        pixels.remember(self, int(index), query.missing_keys, values)
        return result

    def read_positions(self, indices, cache):
        sampler = self._chunk_reader(cache)
        return sampler.bounds[indices, 0] if sampler else None

    def prefetch_frame_indexes(self, frames, cache):
        from contextlib import nullcontext

        sampler = self._chunk_reader(cache)
        return sampler.prefetch(frames, cache) if sampler else nullcontext()

    def open(self):
        if self.fmt == 'npz':
            return zipfile.ZipFile(self.filename)
        if self.fmt == 'fch5':
            import h5py
            import hdf5plugin  # Register frame-cache codecs.

            if os.environ.get('HEXRD_FIT_GRAINS_DIRECT_IO') == '1':
                from hexrd.hedm._direct_io import open_direct_hdf5

                return open_direct_hdf5(self.filename, rdcc_nbytes=_MIB)
            return h5py.File(self.filename, 'r', rdcc_nbytes=_MIB)
        from hexrd.core import imageseries

        kwargs = dict(self.args)
        if self.fmt == 'hdf5':
            kwargs['rdcc_nbytes'] = _MIB
        return _OpenSeries(imageseries.open(self.filename, self.fmt, **kwargs))

    def read(self, index, cache):
        from hexrd.core.experiment import _frame_csr
        from hexrd.core.utils.npz import _read_array

        handle = cache.opened(self)
        if self.fmt == 'npz':
            # ZIP member reads are bounded; do not mmap the entire compressed
            # file in every worker or retain all decompressed NPY members.
            arrays = [
                _read_array(handle.read(f'{index}_{field}.npy'), copy=False)
                for field in ('data', 'row', 'col')
            ]
            return _frame_csr(*arrays, self.shape, self.dtype)
        if self.fmt == 'fch5':
            group = handle[self.args['path']] if self.args.get('path') else handle
            bounds = group['frame_ids']
            start, stop = (
                bounds[2 * index : 2 * index + 2] if bounds.ndim == 1 else bounds[index]
            )
            indices = group['indices'][int(start) : int(stop)]
            values = group['data'][int(start) : int(stop), 0]
            return _frame_csr(
                values, indices[:, 0], indices[:, 1], self.shape, self.dtype
            )
        return handle.series[index]


class _OpenSeries:
    def __init__(self, series):
        self.series = series

    def close(self):
        series, self.series = self.series, None
        close = getattr(series._adapter, 'close', None)
        if close is not None:
            close()


@dataclass(eq=False)
class ResidentReader:
    frames: object
    shape: tuple
    dtype: np.dtype
    nframes: int
    estimate: int | None = None
    peak_frame: int = 0
    resident: bool = True

    def read(self, index, cache):
        return self.frames[index]


class _OneFrame(ImageSeriesABC):
    def __init__(self, frame, metadata):
        self.frame, self.metadata = frame, metadata

    def __getitem__(self, index):
        return self.frame

    def __len__(self):
        return 1

    def get_region(self, index, region):
        result = self.frame[slice(*region[0]), slice(*region[1])]
        return result.toarray() if sparse.issparse(result) else result


@dataclass(eq=False)
class ProcessedReader:
    parent: object
    operations: list
    metadata: dict
    nframes: int
    shape: tuple = ()
    dtype: np.dtype | None = None
    estimate: int | None = None
    peak_frame: int = 0
    resident: bool = False

    def read(self, index, cache):
        from hexrd.core.imageseries.process import ProcessedImageSeries

        frame = self.parent.frame(cache, index, original=True)
        processor = ProcessedImageSeries(
            _OneFrame(frame, self.metadata), self.operations
        )
        return processor[0]


@dataclass
class View:
    reader: object
    roi: tuple | None = None  # row, column, height, width in reader coordinates
    mapping: object = None
    dense: bool = False

    @property
    def shape(self):
        return tuple(self.roi[2:]) if self.roi else tuple(self.reader.shape)

    @property
    def dtype(self):
        return self.reader.dtype

    def frame(self, cache, index, original=False):
        actual = int(self.mapping[index]) if self.mapping is not None else int(index)
        result = cache.frame(self.reader, actual)
        if self.roi is not None:
            r, c, h, w = self.roi
            result = result[r : r + h, c : c + w]
        if original:
            if self.dense and sparse.issparse(result):
                return result.toarray()
            if not self.dense and not sparse.issparse(result):
                return sparse.csr_array(result)
        return result


class BoundedSeries(ImageSeriesABC):
    def __init__(self, view, cache, metadata, nframes):
        self.view, self.cache, self._metadata, self.nframes = (
            view,
            cache,
            metadata,
            nframes,
        )

    def __len__(self):
        return self.nframes

    @property
    def shape(self):
        return self.view.shape

    @property
    def dtype(self):
        return self.view.dtype

    @property
    def metadata(self):
        return self._metadata

    def __getitem__(self, key):
        from hexrd.core.imageseries._sparse import sample_pixels

        if isinstance(key, tuple):
            frame = self.view.frame(self.cache, key[0])
            if len(key) == 3 and sparse.issparse(frame):
                return sample_pixels(frame, key[1], key[2])
            result = frame[key[1:]]
            return result.toarray() if sparse.issparse(result) else result
        return self.view.frame(self.cache, key, original=True)

    def get_region(self, index, region):
        result = self.view.frame(self.cache, index)[
            slice(*region[0]), slice(*region[1])
        ]
        return result.toarray() if sparse.issparse(result) else result

    def native_view(self):
        if self.dtype not in [
            np.dtype(t) for t in ('u1', 'i2', 'u2', 'i4', 'u4', 'f4', 'f8')
        ]:
            return None
        return StreamingView(self.view, self.cache)


class FrameSampler:
    """Borrow one frame across sampling tiles; release it before the next read."""

    def __init__(self, reader, cache):
        from hexrd.core.extensions import transforms as native
        from hexrd.core.imageseries._sparse import sample_pixels

        self.native, self.sample_pixels = native, sample_pixels
        self.reader, self.cache = reader, cache
        self.index = self.frame = self.sparse = None

    def __call__(self, index, rows, cols):
        if self.index != index:
            self.index, self.frame, self.sparse = index, None, None
        if self.frame is None:
            direct = getattr(self.reader, 'sample_pixels', None)
            values = direct(index, rows, cols, self.cache) if direct else None
            if values is not None:
                return values
            self.frame = self.cache.frame(self.reader, index)
            if sparse.issparse(self.frame) and self.frame.has_sorted_indices:
                self.sparse = self.native._SparseFrames([self.frame])
        if self.sparse is not None:
            return self.sparse.gather(
                np.zeros((1, 1), dtype='i8'),
                np.array([0, len(rows)], dtype='i8'),
                rows,
                cols,
            )
        if sparse.issparse(self.frame):
            return self.sample_pixels(self.frame, rows, cols)
        return self.frame[rows, cols]


class StreamingView:
    def __init__(self, view, cache):
        self.view, self.cache, self.dtype = view, cache, view.dtype

    @property
    def defer_result(self):
        from hexrd.hedm._staged_spots import _extraction_context

        context = _extraction_context.get()
        return context.defer_result if context is not None else None

    def extract(self, frame_ids, offsets, rows, cols, shapes, threshold):
        from hexrd.hedm._staged_spots import (
            Window,
            measure_windows,
            _extraction_context,
        )

        windows, stats, size = Window.capture(
            self, frame_ids, offsets, rows, cols, shapes, threshold
        )
        context = _extraction_context.get()
        if context is None:
            measure_windows(windows, self.cache)
        else:
            context.capture(windows, size)
        return stats


class _Readers:
    """Describe source files without retaining adapters or decoded payloads."""

    def __init__(self, cache):
        self.cache, self.objects, self.files = cache, {}, {}
        self.readers = []
        self.spawn_safe = True

    def file(self, filename, fmt, args, shape, dtype, nframes):
        filename = os.path.realpath(os.fspath(filename))
        key = (filename, fmt, repr(args))
        if key in self.files:
            return self.files[key]
        reader = FileReader(
            filename, fmt, args, tuple(shape), np.dtype(dtype), int(nframes)
        )
        row_bytes = (int(shape[0]) + 1) * 8 + 1024
        dense_bytes = int(np.prod(shape)) * (reader.dtype.itemsize + 8) + row_bytes
        reader.estimate = dense_bytes * nframes
        reader.peak_frame = dense_bytes
        if fmt == 'npz':
            with zipfile.ZipFile(filename) as archive:
                # Includes NPY headers and both coordinates: a conservative
                # estimate of the compact CSR storage, before decoding pixels.
                sizes = [
                    sum(
                        archive.getinfo(f'{i}_{field}.npy').file_size
                        for field in ('data', 'row', 'col')
                    )
                    + row_bytes
                    for i in range(nframes)
                ]
            reader.estimate = sum(sizes)
            reader.peak_frame = max(sizes, default=0)
        elif fmt == 'fch5':
            import h5py

            with h5py.File(filename, 'r', rdcc_nbytes=_MIB) as handle:
                group = handle[args['path']] if args.get('path') else handle
                sizes = np.diff(group['frame_ids'][()].reshape(-1, 2), axis=1).ravel()
                stride = (
                    group['data'].dtype.itemsize + 2 * group['indices'].dtype.itemsize
                )
                reader.estimate = int(sizes.sum()) * stride + nframes * row_bytes
                reader.peak_frame = int(sizes.max(initial=0)) * stride + row_bytes
                chunks = sum(
                    int(np.prod(group[name].chunks or (0,)))
                    * group[name].dtype.itemsize
                    for name in ('indices', 'data')
                )
                reader.peak_frame = max(reader.peak_frame, chunks)
        elif fmt == 'hdf5':
            import h5py

            with h5py.File(filename, 'r', rdcc_nbytes=_MIB) as handle:
                dataset = handle[
                    '/'.join((args['path'], args.get('dataname', 'images')))
                ]
                chunk_bytes = (
                    int(np.prod(dataset.chunks or (0,))) * dataset.dtype.itemsize
                )
                reader.peak_frame = max(reader.peak_frame, chunk_bytes)
        self.files[key] = reader
        self.readers.append(reader)
        return reader

    def view(self, series, descriptor=None):
        from hexrd.core.experiment import ImageSeries, RoiImageSeries
        from hexrd.core.imageseries.baseclass import ImageSeries as AdapterSeries
        from hexrd.core.imageseries.omega import OmegaImageSeries
        from hexrd.core.imageseries.process import ProcessedImageSeries
        from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter
        from hexrd.core.imageseries.load.array import ArrayImageSeriesAdapter
        from hexrd.core.imageseries.load.hdf5 import HDF5ImageSeriesAdapter
        from hexrd.hedm.experiment import _AdapterImageSeries
        from dataclasses import replace

        if id(series) in self.objects:
            return self.objects[id(series)]
        loaded = getattr(series, '_images', None)
        if (
            loaded is None
            and type(series) is FrameCacheImageSeriesAdapter
            and series._framelist_was_loaded
        ):
            loaded = series._framelist
        if loaded is not None:
            # Respect caller edits to already loaded frames. These buffers are
            # caller-owned, and cannot be evicted by this invocation's cache.
            reader = ResidentReader(
                loaded,
                loaded[0].shape,
                loaded[0].dtype,
                len(loaded),
                sum(frame_bytes(f) for f in loaded),
                max(frame_bytes(f) for f in loaded),
            )
            self.readers.append(reader)
            self.spawn_safe = False
            view = View(reader, dense=type(series) is FrameCacheImageSeriesAdapter)
        elif isinstance(series, RoiImageSeries):
            parent = self.view(series._parent)
            r, c = series._roi
            if parent.roi:
                r, c = r + parent.roi[0], c + parent.roi[1]
            view = replace(parent, roi=(r, c, *series._shape))
        elif type(series) in (AdapterSeries, OmegaImageSeries):
            view = self.view(series._adapter, descriptor)
        elif type(series) is ProcessedImageSeries:
            parent = self.view(series._imser)
            if series._hasframelist:
                indices = np.asarray(series._frames)
                mapping = indices if parent.mapping is None else parent.mapping[indices]
                parent = replace(parent, mapping=mapping)
            builtin = {
                series.DARK: ProcessedImageSeries._subtract_dark,
                series.FLIP: ProcessedImageSeries._flip,
                series.RECT: ProcessedImageSeries._rectangle,
                series.ADD: ProcessedImageSeries._add,
                series.GAUSS_LAPLACE: ProcessedImageSeries._gauss_laplace,
                series.ADD_ROW: ProcessedImageSeries._add_row,
                series.ADD_COL: ProcessedImageSeries._add_col,
            }
            if any(
                name not in builtin
                or getattr(series._opdict[name], '__self__', None) is not series
                or getattr(series._opdict[name], '__func__', None) is not builtin[name]
                for name, _ in series.oplist
            ):
                raise ValueError(
                    'Lazy fitting requires standard ProcessedImageSeries operations'
                )
            if len(series.oplist) == 1 and series.oplist[0][0] == series.RECT:
                (r0, r1), (c0, c1) = series.oplist[0][1]
                # Match NumPy slicing, including negative and oversized bounds.
                r0, r1, _ = slice(r0, r1).indices(parent.shape[0])
                c0, c1, _ = slice(c0, c1).indices(parent.shape[1])
                r, c = parent.roi[:2] if parent.roi else (0, 0)
                view = replace(
                    parent,
                    roi=(r + r0, c + c0, max(0, r1 - r0), max(0, c1 - c0)),
                    dense=True,
                )
            else:
                reader = ProcessedReader(
                    parent, series.oplist, series.metadata, len(series)
                )
                # Shape/dtype can change with processing; sample exactly one
                # safely loaded frame instead of invoking the eager adapter.
                sample = reader.read(0, self.cache)
                reader.shape, reader.dtype = sample.shape, sample.dtype
                reader.peak_frame = frame_bytes(sample)
                reader.estimate = reader.peak_frame * len(series)
                self.readers.append(reader)
                view = View(reader, dense=True)
        elif type(series) is _AdapterImageSeries:
            view = replace(self.view(series._series, series._source), dense=False)
        elif type(series) is ImageSeries:
            reader = self.file(
                series._arrs._filename,
                'npz',
                {},
                series._shape,
                series._dtype,
                len(series),
            )
            view = View(reader)
        elif type(series) is FrameCacheImageSeriesAdapter:
            filename, fmt, args = series._fname, 'npz', {}
            if series._style == 'fch5':
                fmt = 'fch5'
                if not isinstance(filename, (str, os.PathLike)):
                    args['path'] = filename.name
                    filename = filename.file.filename
                elif series._path:
                    args['path'] = series._path
            elif series._from_yml:
                filename = os.path.join(os.path.dirname(filename), series._cache)
            view = View(
                self.file(filename, fmt, args, series.shape, series.dtype, len(series)),
                dense=True,
            )
        elif type(series) is ArrayImageSeriesAdapter:
            reader = ResidentReader(
                series._data,
                series.shape,
                series.dtype,
                len(series),
                series._data.nbytes,
                int(np.prod(series.shape)) * series.dtype.itemsize,
            )
            self.readers.append(reader)
            self.spawn_safe = False
            view = View(reader, dense=True)
        elif type(series) is HDF5ImageSeriesAdapter:
            prefix = '_HDF5ImageSeriesAdapter'
            filename = getattr(series, prefix + '__h5name')
            args = {
                'path': getattr(series, prefix + '__path'),
                'dataname': getattr(series, prefix + '__dataname'),
            }
            view = View(
                self.file(
                    filename, 'hdf5', args, series.shape, series.dtype, len(series)
                ),
                dense=True,
            )
        elif descriptor is not None:
            filename, fmt, args = descriptor
            view = View(
                self.file(filename, fmt, args, series.shape, series.dtype, len(series)),
                dense=True,
            )
        elif getattr(series, 'format', None) == 'image-files':
            view = View(
                self.file(
                    series._fname,
                    'image-files',
                    {},
                    series.shape,
                    series.dtype,
                    len(series),
                ),
                dense=True,
            )
        else:
            # Registered third-party adapters can participate through the
            # descriptor supplied by HedmExperiment; silently using an unknown
            # object's __getitem__ could reintroduce whole-scan loading.
            raise TypeError(f'No lazy file descriptor for {type(series).__name__}')
        self.objects[id(series)] = view
        return view


class ImagePlan:
    """One fit invocation owns the lazy cache and closes it on every exit."""

    def __init__(self):
        self.cache = None
        self.lazy = False
        self.context = None
        self.grain_workspace = 0

    def prepare(self, series, settings, workers, *, force_process=False):
        import psutil
        from hexrd.core import constants
        from hexrd.core.imageseries.omega import OmegaImageSeries
        from hexrd.hedm._progress import diagnostic

        self.context = constants.mp_context
        if settings.mode == 'eager':
            diagnostic('Image access', 'Eager (requested)')
            diagnostic('Image cache total', 'Unbounded (eager)')
            diagnostic('Image cache / worker', '—')
            return series, workers
        # Allow lightweight experiment doubles used by downstream callers.
        if not all(isinstance(s, OmegaImageSeries) for s in series.values()):
            if settings.mode == 'lazy':
                raise TypeError('Lazy fitting requires omega image series')
            return series, workers
        available = psutil.virtual_memory().available
        requested = (
            int(settings.max_size_mb * _MIB)
            if settings.max_size_mb is not None
            else min(64 * 1024**3, available // 4)
        )
        budget = min(requested, available // 2)
        self.cache = FrameCache(0)  # Metadata processing must not retain frames.
        builder = _Readers(self.cache)
        views = {name: builder.view(s) for name, s in series.items()}
        estimate = sum(r.estimate for r in builder.readers)
        diagnostic('Decoded images estimate', f'{estimate / 1024**3:.2f} GiB')
        # Leave headroom for temporary COO/CSR decode buffers in the eager path.
        if settings.mode == 'auto' and 2 * estimate <= budget:
            self.close()
            diagnostic('Image access', 'Eager (small scan)')
            diagnostic('Image cache total', 'Unbounded (eager)')
            diagnostic('Image cache / worker', '—')
            return series, workers
        self.lazy = True
        peak = max((r.peak_frame for r in builder.readers), default=0)
        # A budget is not an RSS limit. Reserve frame decoder workspace and the
        # fixed Python/native fitting footprint independently of retained data.
        room = max(0, available * 3 // 4 - budget)
        workers = min(workers, max(1, room // max(1, peak * 4 + 512 * _MIB)))
        self.cache.max_bytes = budget // max(1, workers)
        self.cache.chunk_workers = min(
            settings.read_workers, max(1, (os.cpu_count() or 1) // max(1, workers))
        )
        # Reuse the staged coordinator within each grain for large FCH5 scans.
        # Leave small caches and whole-frame formats on their existing path;
        # compact row indexes and native decoding must still fit comfortably.
        if self.cache.max_bytes >= 512 * _MIB and any(
            getattr(r, 'fmt', None) == 'fch5' and r.estimate > self.cache.max_bytes
            for r in builder.readers
        ):
            self.grain_workspace = 512 * _MIB
        if builder.spawn_safe:
            # No inherited HDF5 handles and no pickle of the eager image graph.
            self.context = multiprocessing.get_context('spawn')
        elif self.context.get_start_method() != 'fork' and (
            workers > 1 or force_process
        ):
            raise ValueError(
                'Lazy fitting of resident arrays in a worker requires fork; use file-backed images or a serial fit without cancellation'
            )
        result = {
            name: OmegaImageSeries(
                BoundedSeries(views[name], self.cache, s.metadata, len(s))
            )
            for name, s in series.items()
        }
        diagnostic('Image access', 'Lazy (bounded image cache)')
        if any(getattr(r, 'fmt', None) == 'fch5' for r in builder.readers):
            diagnostic(
                'FCH5 reads',
                (
                    'Direct (bypass filesystem cache)'
                    if os.environ.get('HEXRD_FIT_GRAINS_DIRECT_IO') == '1'
                    else 'Buffered'
                ),
            )
        diagnostic(
            'Image cache total', f'{self.cache.max_bytes * workers / _MIB:.0f} MiB'
        )
        diagnostic('Image cache / worker', f'{self.cache.max_bytes / _MIB:.0f} MiB')
        return result, int(workers)

    def close(self):
        if self.cache is not None:
            self.cache.close()
