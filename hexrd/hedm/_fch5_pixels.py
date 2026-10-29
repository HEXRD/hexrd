"""Read unchanged spot pixels from FCH5 without decoding entire frames.

FCH5 stores row coordinates, column coordinates and intensities in independent
HDF5 chunks. A compact row index lets us read only the columns and intensities
needed by a spot window. All cached arrays share the worker's byte budget.
Unsorted coordinates or duplicates use the existing full-frame CSR path so
that its coalescing order and numerical results are preserved.
"""

from time import monotonic
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
        self.row_queue = None

    @contextmanager
    def _sequence(self, kind, indices, cache, selected=None):
        """Consume ordered chunks once, overlapping a bounded number of reads."""
        from hexrd.hedm._chunk_io import ChunkReader, SelectedChunk

        dataset = self.values if kind == 'values' else self.indices
        column = 1 if kind == 'columns' else 0
        chunk = self.value_chunk if kind == 'values' else self.coord_chunk
        if self.chunk_reader is None:
            self.chunk_reader = ChunkReader.create(self.handle(), cache)
        parallel = self.chunk_reader
        stream = (
            parallel.chunks(
                dataset, column, indices, cache, self.reader, kind, selected
            )
            if parallel is not None and parallel.supported(dataset)
            else ((int(index), None, 0.0) for index in indices)
        )
        current, value = -1, None

        def get(index):
            nonlocal current, value
            index = int(index)
            if current == index:
                return value
            while current < index:
                check_cancelled()
                started = monotonic()
                current, decoded, _ = next(stream)
                self.read_seconds += monotonic() - started
                first = current * chunk
                complete = True
                decoded_bytes = None
                if isinstance(decoded, SelectedChunk):
                    decoded_bytes = decoded.decoded_bytes
                    decoded, complete = decoded.array, decoded.complete

                def load():
                    if decoded is None:
                        return self._read(
                            dataset, first, min(first + chunk, len(dataset)), column
                        )
                    result = decoded[: min(chunk, len(dataset) - first), 0]
                    if self.reporter is not None and not self.read_bytes:
                        self.reporter.read_started()
                    self.read_bytes += (
                        result.nbytes if decoded_bytes is None else decoded_bytes
                    )
                    return result

                # Incomplete arrays are private to this query. Only requested
                # row spans / intensity positions are read from them. A later
                # query must never mistake them for fully decoded cache data.
                if not complete:
                    cache.misses += 1
                    if self.reporter is not None:
                        self.reporter.data['cache_misses'] += 1
                value = (
                    cache.array(self.reader, kind, current, load)
                    if complete
                    else load()
                )
            if current != index:
                raise RuntimeError('FCH5 chunk requests must be ordered')
            return value

        try:
            yield get
        finally:
            stream.close()

    def _read(self, dataset, first, last, column):
        if self.reporter is not None and not self.read_bytes:
            self.reporter.read_started()
        started = monotonic()
        result = dataset[first:last, column]
        self.read_seconds += monotonic() - started
        self.read_bytes += result.nbytes
        return result

    def _offsets(self, frame, cache):
        future = self.row_queue.take(frame) if self.row_queue is not None else None

        def load():
            if future is not None:
                from hexrd.hedm._row_index import wait_index

                if self.reporter is not None and not self.read_bytes:
                    self.reporter.read_started()
                started = monotonic()
                result = wait_index(future)
                self.read_seconds += monotonic() - started
                self.read_bytes += result.decoded_bytes
                return result.array
            start, stop = map(int, self.bounds[frame])
            invalid = np.array([-1], dtype=np.int64)
            if (
                not 0 <= start <= stop <= min(len(self.indices), len(self.values))
                or self.indices.dtype.kind not in 'iu'
            ):
                return invalid
            height = self.reader.shape[0]
            dtype = np.int32 if stop - start <= np.iinfo(np.int32).max else np.int64
            result = np.full(height + 1, stop - start, dtype=dtype)
            next_row, previous = 0, -1
            # Stream row coordinates in bounded chunks. Avoid promoting a
            # whole uint16 frame to int64 inside np.searchsorted.
            first = start
            with self._sequence(
                'rows',
                range(
                    start // self.coord_chunk,
                    (stop + self.coord_chunk - 1) // self.coord_chunk,
                ),
                cache,
            ) as get:
                while first < stop:
                    check_cancelled()
                    # Align to storage chunks, avoiding decoding the same boundary
                    # chunk twice when the HDF5 cache can hold only one chunk.
                    end = min(stop, (first // self.coord_chunk + 1) * self.coord_chunk)
                    rows = get(first // self.coord_chunk)[
                        first % self.coord_chunk : first % self.coord_chunk
                        + end
                        - first
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
        dataset = self.indices if kind == 'columns' else self.values
        chunk = self.coord_chunk if kind == 'columns' else self.value_chunk
        column = 1 if kind == 'columns' else 0
        first = int(index) * chunk
        return cache.array(
            self.reader,
            kind,
            index,
            lambda: self._read(
                dataset, first, min(first + chunk, len(dataset)), column
            ),
        )

    def _columns(self, first, last, cache, get=None):
        chunks = []
        while first < last:
            index, offset = divmod(first, self.coord_chunk)
            values = (
                get(index) if get is not None else self._chunk('columns', index, cache)
            )
            count = min(last - first, len(values) - offset)
            chunks.append(values[offset : offset + count])
            first += count
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _positions(self, ptr, start, rows, cols, needed, cache, selected=None):
        from hexrd.core.extensions import transforms as native

        lookup = getattr(native, '_fch5Positions', None)
        # Pin at most 16 MiB of coordinate chunks during one native call. This
        # allowance is deducted from the same cache while the arrays are live.
        # A compressed edge chunk owns its full decoded allocation even when
        # only a short slice lies inside the dataset's logical extent.
        size = len(needed) * (1024 + self.coord_chunk * self.indices.dtype.itemsize)
        if lookup is None or size > min(
            16 * 1024**2, cache.max_bytes // 4, cache.max_bytes - cache.reserved_bytes
        ):
            return NotImplemented
        arrays = []
        cache.reserved_bytes += size
        cache._evict(0, self.reporter)
        try:
            with self._sequence('columns', needed, cache, selected) as get:
                arrays = [get(index) for index in needed]
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
        self.reporter = current_reporter()
        self.read_bytes = self.read_seconds = 0
        try:
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
            column_spans = {}
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
                while first < last:
                    index, offset = divmod(first, self.coord_chunk)
                    end = min(last, (index + 1) * self.coord_chunk)
                    previous = column_spans.get(index, (offset, offset + end - first))
                    column_spans[index] = (
                        min(offset, previous[0]),
                        max(offset + end - first, previous[1]),
                    )
                    first = end
            needed = sorted(needed)
            found = self._positions(
                ptr, start, rows[order], cols[order], needed, cache, column_spans
            )
            if found is None:
                return None
            if found is not NotImplemented:
                valid = found >= 0
                queries, positions = [order[valid]], [found[valid]]
            else:
                with self._sequence('columns', needed, cache, column_spans) as get:
                    for left, right in zip(edges[:-1], edges[1:]):
                        check_cancelled()
                        selected = order[left:right]
                        row = int(rows[selected[0]])
                        first, last = start + int(ptr[row]), start + int(ptr[row + 1])
                        if first == last:
                            continue
                        columns = self._columns(first, last, cache, get)
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
            positions, inverse = np.unique(
                np.concatenate(positions), return_inverse=True
            )
            if not positions.size:
                return output
            values = np.empty(len(positions), dtype=self.reader.dtype)
            chunks = positions // self.value_chunk
            edges = np.r_[0, np.flatnonzero(np.diff(chunks)) + 1, len(chunks)]
            value_spans = {
                int(chunks[left]): (
                    int(positions[left] % self.value_chunk),
                    int(positions[right - 1] % self.value_chunk) + 1,
                )
                for left, right in zip(edges[:-1], edges[1:])
                if left < right
            }
            with self._sequence(
                'values', chunks[edges[:-1]], cache, value_spans
            ) as get:
                for left, right in zip(edges[:-1], edges[1:]):
                    if left == right:
                        continue
                    chunk = get(chunks[left])
                    values[left:right] = chunk[positions[left:right] % self.value_chunk]
            # CSR sampling adds into zero, including for stored negative zero.
            output[queries] += values[inverse]
            return output
        finally:
            if self.reporter is not None:
                self.reporter.read_finished(self.read_bytes, self.read_seconds)
                self.reporter.cache(cache)

    def _sample_native(self, frame, ptr, rows, cols, cache):
        from hexrd.hedm._chunk_io import ChunkReader

        if rows.size > 65536 or self.indices.dtype != np.dtype('u2'):
            return NotImplemented
        if self.chunk_reader is None:
            self.chunk_reader = ChunkReader.create(self.handle(), cache)
        backend = self.chunk_reader
        read = (
            getattr(backend.pool.native_reader, 'read_pixels', None)
            if backend
            else None
        )
        if read is None:
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
        cache._evict(0, self.reporter)
        started = monotonic()
        try:
            if self.reporter is not None and not self.read_bytes:
                self.reporter.read_started()
            result, status, decoded, reads = read(
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
            if self.reporter is not None:
                self.reporter.data['cache_misses'] += reads
            if status == 2:
                return NotImplemented
            if status < 0:
                if status <= -1000:
                    import os

                    error = -1000 - status
                    raise OSError(error, os.strerror(error))
                raise OSError('Cannot read native FCH5 pixel chunks')
            return result
        finally:
            self.read_seconds += monotonic() - started
            cache.reserved_bytes -= workspace
