"""Read unchanged spot pixels from FCH5 without decoding entire frames.

FCH5 stores row coordinates, column coordinates and intensities in independent
HDF5 chunks. A compact row index lets us read only the columns and intensities
needed by a spot window. All cached arrays share the worker's byte budget.
Unsorted coordinates or duplicates use the existing full-frame CSR path so
that its coalescing order and numerical results are preserved.
"""

from contextlib import contextmanager
import weakref

import numpy as np

from hexrd.hedm._fit_progress import current_reporter
from hexrd.hedm._progress import check_cancelled


class Fch5Pixels:
    def __init__(self, reader, handle):
        self.reader = reader
        group = handle[reader.args['path']] if reader.args.get('path') else handle
        self.indices, self.values = group['indices'], group['data']
        self.bounds = group['frame_ids'][()].reshape(-1, 2)
        self.coord_chunk = (self.indices.chunks or (524288,))[0]
        self.value_chunk = (
            self.values.chunks or (max(1, 1024**2 // self.values.dtype.itemsize),)
        )[0]
        self.handle = weakref.ref(handle)
        self.chunk_reader = None
        self._borrowed = None
        self.row_indexes = None

    @contextmanager
    def prefetch(self, frames, cache):
        from hexrd.hedm._chunk_io import ChunkReader

        if self.chunk_reader is None:
            self.chunk_reader = ChunkReader.create(self.handle(), cache)
        backend = self.chunk_reader
        self.row_indexes = (
            backend.prefetch(
                self.indices,
                self.bounds,
                frames,
                self.reader.shape[0],
                cache,
                self.reader,
            )
            if backend
            else None
        )
        try:
            yield
        finally:
            if self.row_indexes is not None:
                self.row_indexes.close()
                self.row_indexes = None

    def _read(self, dataset, first, last, column):
        if self.reporter is not None and not self.read_bytes:
            self.reporter.read_started()
        result = dataset[first:last, column]
        self.read_bytes += result.nbytes
        return result

    def _offsets(self, frame, cache):
        def load():
            start, stop = map(int, self.bounds[frame])
            invalid = np.array([-1], dtype=np.int64)
            if (
                not 0 <= start <= stop <= min(len(self.indices), len(self.values))
                or self.indices.dtype.kind not in 'iu'
            ):
                return invalid
            from hexrd.hedm._chunk_io import ChunkReader

            if self.chunk_reader is None:
                self.chunk_reader = ChunkReader.create(self.handle(), cache)
            if self.chunk_reader is not None:
                if self.reporter is not None and not self.read_bytes:
                    self.reporter.read_started()
                result = (
                    self.row_indexes.get(frame, check_cancelled)
                    if self.row_indexes is not None
                    else (2, None, 0)
                )
                if result[0] == 2:
                    result = self.chunk_reader.row_index(
                        self.indices, start, stop, self.reader.shape[0]
                    )
                status, result, decoded = result
                self.chunk_reader.check(status)
                self.read_bytes += decoded
                if status != 2:
                    return result if status else np.array([-1], dtype='i8')
            height = self.reader.shape[0]
            dtype = np.int32 if stop - start <= np.iinfo(np.int32).max else np.int64
            result = np.full(height + 1, stop - start, dtype=dtype)
            next_row, previous = 0, -1
            # Stream row coordinates in bounded chunks. Avoid promoting a
            # whole uint16 frame to int64 inside np.searchsorted.
            first = start
            while first < stop:
                check_cancelled()
                # Align to storage chunks, avoiding decoding the same boundary
                # chunk twice when the HDF5 cache can hold only one chunk.
                end = min(stop, (first // self.coord_chunk + 1) * self.coord_chunk)
                rows = self._chunk('rows', first // self.coord_chunk, cache)[
                    first % self.coord_chunk : first % self.coord_chunk + end - first
                ]
                if (
                    rows[0] < 0
                    or rows[0] < previous
                    or rows[-1] >= height
                    or np.any(rows[1:] < rows[:-1])
                ):
                    return invalid
                last = int(rows[-1]) + 1
                if last > next_row:
                    targets = np.arange(next_row, last, dtype=rows.dtype)
                    result[next_row:last] = (
                        first - start + np.searchsorted(rows, targets)
                    )
                    next_row = last
                previous = int(rows[-1])
                first = end
            return result

        return cache.array(self.reader, 'row-offsets', frame, load, metadata=True)

    def _chunk(self, kind, index, cache):
        key = kind, int(index)
        if self._borrowed is not None and self._borrowed[0] == key:
            return self._borrowed[1]
        self._borrowed = None
        dataset = self.values if kind == 'values' else self.indices
        chunk = self.value_chunk if kind == 'values' else self.coord_chunk
        column = 1 if kind == 'columns' else 0
        first = int(index) * chunk
        value = cache.array(
            self.reader,
            kind,
            index,
            lambda: self._read(
                dataset, first, min(first + chunk, len(dataset)), column
            ),
        )
        self._borrowed = key, value
        return value

    def _columns(self, first, last, cache):
        chunks = []
        while first < last:
            index, offset = divmod(first, self.coord_chunk)
            values = self._chunk('columns', index, cache)
            count = min(last - first, len(values) - offset)
            chunks.append(values[offset : offset + count])
            first += count
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _positions(self, ptr, start, rows, cols, needed, cache):
        from hexrd.core.extensions import transforms as native

        lookup = native._fch5Positions
        # Pin at most 16 MiB of coordinate chunks during one native call. This
        # allowance is deducted from the same cache while the arrays are live.
        # A compressed edge chunk owns its full decoded allocation even when
        # only a short slice lies inside the dataset's logical extent.
        size = len(needed) * (1024 + self.coord_chunk * self.indices.dtype.itemsize)
        if size > min(
            16 * 1024**2, cache.max_bytes // 4, cache.max_bytes - cache.reserved_bytes
        ):
            return NotImplemented
        arrays = []
        cache.reserved_bytes += size
        cache._evict(0)
        try:
            arrays = [self._chunk('columns', index, cache) for index in needed]
            return lookup(
                ptr,
                start,
                arrays,
                np.asarray(needed, dtype='i8'),
                self.coord_chunk,
                rows,
                cols,
                self.reader.shape[1],
            )
        finally:
            arrays.clear()
            cache.reserved_bytes -= size

    def sample(self, frame, rows, cols, cache):
        try:
            return self._sample(frame, rows, cols, cache)
        finally:
            self._borrowed = None

    def _sample(self, frame, rows, cols, cache):
        self.reporter = current_reporter()
        self.read_bytes = 0
        rows, cols = np.asarray(rows).ravel(), np.asarray(cols).ravel()
        if not rows.size:
            return np.zeros(rows.size, dtype=self.reader.dtype)
        ptr = self._offsets(frame, cache)
        if ptr[0] < 0:
            return None
        result = self._sample_native(frame, ptr, rows, cols, cache)
        if result is not NotImplemented:
            return result
        output = np.zeros(rows.size, dtype=self.reader.dtype)
        start = int(self.bounds[frame, 0])
        order = np.argsort(rows, kind='stable')
        edges = np.r_[0, np.flatnonzero(np.diff(rows[order])) + 1, len(rows)]
        queries, positions = [], []
        requested_rows = rows[order[edges[:-1]]]
        needed = set()
        for row in requested_rows:
            first, last = start + int(ptr[row]), start + int(ptr[row + 1])
            needed.update(
                range(
                    first // self.coord_chunk,
                    (last + self.coord_chunk - 1) // self.coord_chunk,
                )
                if first < last
                else ()
            )
        needed = sorted(needed)
        found = self._positions(ptr, start, rows[order], cols[order], needed, cache)
        if found is None:
            return None
        if found is not NotImplemented:
            valid = found >= 0
            queries, positions = [order[valid]], [found[valid]]
        else:
            for left, right in zip(edges[:-1], edges[1:]):
                check_cancelled()
                selected = order[left:right]
                row = int(rows[selected[0]])
                first, last = start + int(ptr[row]), start + int(ptr[row + 1])
                if first == last:
                    continue
                columns = self._columns(first, last, cache)
                if (
                    columns[0] < 0
                    or columns[-1] >= self.reader.shape[1]
                    or np.any(columns[1:] <= columns[:-1])
                ):
                    return None
                indices = np.searchsorted(columns, cols[selected])
                valid = indices < len(columns)
                valid[valid] &= columns[indices[valid]] == cols[selected[valid]]
                queries.append(selected[valid])
                positions.append(first + indices[valid])
        if not positions:
            return output
        queries = np.concatenate(queries)
        positions, inverse = np.unique(np.concatenate(positions), return_inverse=True)
        if not positions.size:
            return output
        values = np.empty(len(positions), dtype=self.reader.dtype)
        chunks = positions // self.value_chunk
        edges = np.r_[0, np.flatnonzero(np.diff(chunks)) + 1, len(chunks)]
        for left, right in zip(edges[:-1], edges[1:]):
            if left == right:
                continue
            chunk = self._chunk('values', chunks[left], cache)
            values[left:right] = chunk[positions[left:right] % self.value_chunk]
        # CSR sampling adds into zero, including for stored negative zero.
        output[queries] += values[inverse]
        return output

    def _sample_native(self, frame, ptr, rows, cols, cache):
        from hexrd.hedm._chunk_io import ChunkReader

        if rows.size > 65536 or self.indices.dtype != np.dtype('u2'):
            return NotImplemented
        if self.chunk_reader is None:
            self.chunk_reader = ChunkReader.create(self.handle(), cache)
        backend = self.chunk_reader
        if backend is None:
            return NotImplemented
        coordinates, values = backend._spec(self.indices), backend._spec(self.values)
        if coordinates is None or values is None:
            return NotImplemented
        workspace = min(
            16 * 1024**2, cache.max_bytes // 4, cache.max_bytes - cache.reserved_bytes
        )
        if workspace < 65536:
            return NotImplemented
        cache.reserved_bytes += workspace
        cache._evict(0)
        try:
            if self.reporter is not None and not self.read_bytes:
                self.reporter.read_started()
            result, status, decoded, reads = backend.pool.native_reader.read_pixels(
                backend.fd,
                coordinates,
                values,
                ptr,
                *map(int, self.bounds[frame]),
                rows,
                cols,
                self.reader.shape[1],
                backend.direct,
                workspace,
                check_cancelled,
            )
            self.read_bytes += decoded
            cache.misses += reads
            if status == 2:
                return NotImplemented
            backend.check(status)
            return result
        finally:
            cache.reserved_bytes -= workspace
