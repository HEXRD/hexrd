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

    @contextmanager
    def _sequence(self, kind, indices, cache):
        """Consume ordered chunks once, overlapping a bounded number of reads."""
        from hexrd.hedm._chunk_io import ChunkReader

        dataset = self.values if kind == 'values' else self.indices
        column = 1 if kind == 'columns' else 0
        chunk = self.value_chunk if kind == 'values' else self.coord_chunk
        if self.chunk_reader is None:
            self.chunk_reader = ChunkReader.create(self.handle(), cache)
        parallel = self.chunk_reader
        stream = (
            parallel.chunks(dataset, column, indices, cache, self.reader, kind)
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

                def load():
                    if decoded is None:
                        return self._read(
                            dataset, first, min(first + chunk, len(dataset)), column
                        )
                    result = decoded[: min(chunk, len(dataset) - first), 0]
                    if self.reporter is not None and not self.read_bytes:
                        self.reporter.read_started()
                    self.read_bytes += result.nbytes
                    return result

                value = cache.array(self.reader, kind, current, load)
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
        def load():
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

    def _positions(self, ptr, start, rows, cols, needed, cache):
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
            with self._sequence('columns', needed, cache) as get:
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
            output = np.zeros(rows.size, dtype=self.reader.dtype)
            if not rows.size:
                return output
            ptr = self._offsets(frame, cache)
            if ptr[0] < 0:
                return None
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
                with self._sequence('columns', needed, cache) as get:
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
            with self._sequence('values', chunks[edges[:-1]], cache) as get:
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
