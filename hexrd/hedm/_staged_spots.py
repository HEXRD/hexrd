"""Exact spot measurements prepared in frame order before each grain-fit pass.

Only the original angular windows are segmented. Temporary pixel support is
retained until an omega window is complete; the unchanged native spot-statistics
kernel then produces its measurements. Global detector centroids are not used.
"""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from hexrd.hedm._fit_progress import current_reporter, set_reporter
from hexrd.hedm._image_cache import FrameSampler, StreamingView, _array_storage
from hexrd.hedm._progress import check_cancelled

_extraction_context = ContextVar('spot_extraction_context', default=None)


class StagedCancelled(Exception):
    pass


class _PlanTooLarge(Exception):
    pass


def _compact(array):
    array = np.asarray(array)
    if not array.size:
        return array.copy()
    low, high = int(array.min()), int(array.max())
    dtype = next(
        t
        for t in ('i2', 'i4', 'i8')
        if np.iinfo(t).min <= low and high <= np.iinfo(t).max
    )
    return array.astype(dtype, copy=True)


@dataclass
class Window:
    view: StreamingView
    frames: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    shape: np.ndarray
    threshold: object
    stats: tuple
    inverse: np.ndarray | None = None

    @property
    def reader(self):
        return self.view.view.reader

    @property
    def bytes(self):
        return self.rows.size * self.frames.size * self.view.dtype.itemsize

    @property
    def expansion_bytes(self):
        return (
            0
            if self.inverse is None
            else int(self.shape.prod()) * self.view.dtype.itemsize
        )

    @classmethod
    def capture(cls, original, frames, offsets, rows, cols, shapes, threshold):
        from hexrd.core.extensions import transforms as native

        check_cancelled()
        n = len(shapes)
        if (
            frames.ndim != 2
            or frames.shape[0] != n
            or shapes.shape != (n, 3)
            or offsets.shape != (n + 1,)
            or rows.ndim != 1
            or cols.ndim != 1
            or offsets[0] != 0
            or offsets[-1] != len(rows)
            or len(rows) != len(cols)
            or np.any(shapes <= 0)
            or np.any(shapes[:, 0] != frames.shape[1])
            or np.any(shapes[:, 1:] > len(rows))
            or np.any(np.diff(offsets) != np.prod(shapes[:, 1:], axis=1))
        ):
            raise ValueError('invalid window dimensions')
        sum_dtype = np.empty(0, dtype=original.dtype).sum().dtype
        stats = (
            np.zeros((n, 3)),
            np.zeros(n, dtype=sum_dtype),
            np.zeros(n, dtype=original.dtype),
            np.zeros(n, dtype=bool),
        )
        view = original.view
        rows = np.asarray(rows, dtype='i8').clip(0, view.shape[0] - 1)
        cols = np.asarray(cols, dtype='i8').clip(0, view.shape[1] - 1)
        if view.roi is not None:
            rows += view.roi[0]
            cols += view.roi[1]
        rows, cols = _compact(rows), _compact(cols)
        rows.flags.writeable = cols.flags.writeable = False
        actual = np.array(
            frames if view.mapping is None else np.asarray(view.mapping)[frames],
            dtype='i8',
            copy=True,
        )
        actual[actual < 0] += view.reader.nframes
        shapes = shapes.copy()
        windows = []
        arrays = [actual, shapes, *stats]
        for i, (first, last) in enumerate(zip(offsets[:-1], offsets[1:])):
            r, c = rows[first:last], cols[first:last]
            inverse = None
            query = native._makePixelQuery(
                r, c, *view.reader.shape, original.dtype, None
            )
            if query is not None and query.unique_count <= len(r) * 3 // 4:
                # The source-pixel map is omega-independent. Keep its exact
                # inverse so repeated polar subpixels need no repeated reads.
                r, c, inverse = map(_compact, (query.rows, query.cols, query.inverse))
                r.flags.writeable = c.flags.writeable = inverse.flags.writeable = False
                arrays.append(inverse)
            arrays.extend((r, c))
            windows.append(
                cls(
                    original,
                    actual[i],
                    r,
                    c,
                    shapes[i : i + 1],
                    threshold,
                    tuple(a[i : i + 1] for a in stats),
                    inverse,
                )
            )
        size = 2048 + sum(dict(_array_storage(a) for a in arrays).values())
        return windows, stats, size


class MeasurementPlan:
    def __init__(self, instrument, request, max_bytes=None):
        self.instrument, self.request = instrument, request
        self.windows = []
        self.max_bytes, self.bytes = max_bytes, 0

    def _reserve(self, size):
        if self.max_bytes is not None and self.bytes + size > self.max_bytes:
            raise _PlanTooLarge()
        self.bytes += size

    def capture(self, windows, size):
        self._reserve(size)
        self.windows.extend(windows)

    def defer_result(self, finish, size):
        from hexrd.core.instrument._spotutils import _DeferredSpotData

        self._reserve(size)
        return _DeferredSpotData(finish)

    @contextmanager
    def _context(self):
        token = _extraction_context.set(self)
        reporter = current_reporter()
        set_reporter(None)
        try:
            yield
        finally:
            set_reporter(reporter)
            _extraction_context.reset(token)

    def prepare(self):
        check_cancelled()
        overflow = False
        try:
            with self._context():
                self.fallback_result = self.instrument.pull_spots(
                    *self.request[0], **self.request[1]
                )
        except _PlanTooLarge:
            self.windows.clear()
            self.bytes, overflow = 0, True
        if overflow:
            check_cancelled()
            self.fallback_result = self.instrument.pull_spots(
                *self.request[0], **self.request[1]
            )
        return self

    def result(self):
        from hexrd.core.instrument._spotutils import _DeferredSpotData, _SpotData

        if (
            isinstance(self.fallback_result, tuple)
            and len(self.fallback_result) == 2
            and isinstance(self.fallback_result[1], dict)
            and any(
                isinstance(x, _DeferredSpotData)
                for x in self.fallback_result[1].values()
            )
        ):
            # Resolve captured geometry without simulating the grain or making
            # its resampling grids again. Normalize invalid IDs across panels
            # only after all panels' real validity masks have been measured.
            signal, output, invalid_id = [], {}, -100
            for name, records in self.fallback_result[1].items():
                packed = True
                if isinstance(records, _DeferredSpotData):
                    packed = records.packed
                    records, _ = records.finish()
                    if records is None:
                        records = []
                if isinstance(records, _SpotData):
                    bad = records.peak_id < 0
                    count = int(np.count_nonzero(bad))
                    records.peak_id[bad] = invalid_id - np.arange(count)
                    invalid_id -= count
                    signal.extend(~bad)
                    output[name] = records if packed else list(records)
                else:
                    for record in records:
                        valid = record[0] >= 0
                        signal.append(valid)
                        if not valid:
                            record[0] = invalid_id
                            invalid_id -= 1
                    output[name] = records
            self.fallback_result = None
            return signal, output
        return self.fallback_result


class MeasurementGroup:
    """Bound planned geometry, event tables and live pixel windows together."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.plans, self.windows = [], []
        self.cost = min(128 * 1024**2, max_bytes // 4)
        self.frontiers = {}
        self.frame_ranks = {}
        self.peak_pixels = self.expansion = 0
        self.frame_visits = 0

    def add(self, plan):
        incoming = plan.windows
        expansion = max(
            self.expansion, max((p.expansion_bytes for p in incoming), default=0)
        )
        cost = (
            plan.bytes
            + sum(1024 + 64 * len(p.frames) for p in incoming)
            + expansion
            - self.expansion
        )
        frontiers = {r: x.copy() for r, x in self.frontiers.items()}
        frame_ranks = self.frame_ranks.copy()
        for p in incoming:
            if p.reader not in frontiers:
                frontiers[p.reader] = np.zeros(p.reader.nframes + 1, dtype='i8')
            if p.reader not in frame_ranks:
                ranks = None
                if (
                    hasattr(p.reader, 'read_positions')
                    and p.reader.nframes * 16 <= self.max_bytes // 16
                ):
                    positions = p.reader.read_positions(
                        np.arange(p.reader.nframes), p.view.cache
                    )
                    if positions is not None:
                        ranks = np.empty(p.reader.nframes, dtype='i8')
                        ranks[np.argsort(positions, kind='stable')] = np.arange(
                            p.reader.nframes
                        )
                frame_ranks[p.reader] = ranks
                if ranks is not None:
                    cost += ranks.nbytes
            delta = frontiers[p.reader]
            ranks = frame_ranks[p.reader]
            positions = ranks[p.frames] if ranks is not None else p.frames
            delta[int(positions.min())] += p.bytes
            delta[int(positions.max()) + 1] -= p.bytes
        peak = max(
            (int(x.cumsum().max(initial=0)) for x in frontiers.values()), default=0
        )
        if self.cost + cost + peak > self.max_bytes:
            return False
        self.plans.append(plan)
        self.windows.extend(incoming)
        self.cost += cost
        self.expansion = expansion
        self.frontiers, self.peak_pixels = frontiers, peak
        self.frame_ranks = frame_ranks
        for cache in {window.view.cache for window in incoming}:
            pixels = cache.pixel_cache
            if pixels is not None:
                pixels.set_limit(max(0, self.max_bytes - self.cost - peak - 1024**2))
        return True

    def measure(self, cache, cancel=check_cancelled):
        cache.max_bytes = min(128 * 1024**2, self.max_bytes // 4)
        cache.metadata_limit = cache.max_bytes // 2
        try:
            self.frame_visits += measure_windows(
                self.windows, cache, cancel, self.frame_ranks
            )
        finally:
            cache.clear_payload()


def _tiles(events, items):
    """Bound coordinates and sampled values to 65,536 pixels per read."""
    event = offset = 0
    while event < len(events):
        pieces, total = [], 0
        while event < len(events) and total < 65536:
            _, index, omega = events[event]
            stop = min(items[index].rows.size, offset + 65536 - total)
            pieces.append((index, omega, offset, stop))
            total += stop - offset
            offset = stop
            if stop == items[index].rows.size:
                event += 1
                offset = 0
        yield pieces


def measure_windows(windows, cache, cancel=check_cancelled, frame_ranks=None):
    """One bounded coordinator for both immediate and staged extraction."""
    from hexrd.core.extensions import transforms as native

    readers, visits = {}, 0
    for window in windows:
        readers.setdefault(window.reader, []).append(window)
    reporter = current_reporter()
    for reader, items in readers.items():
        cancel()
        events = np.empty((sum(len(w.frames) for w in items), 3), dtype='i8')
        cursor = 0
        for index, w in enumerate(items):
            if index % 64 == 0:
                cancel()
            stop = cursor + len(w.frames)
            events[cursor:stop, 0] = w.frames
            events[cursor:stop, 1] = index
            events[cursor:stop, 2] = np.arange(len(w.frames))
            cursor = stop
        ranks = (frame_ranks or {}).get(reader)
        positions = ranks[events[:, 0]] if ranks is not None else events[:, 0]
        if frame_ranks is None and hasattr(reader, 'read_positions'):
            physical = reader.read_positions(events[:, 0], cache)
            if physical is not None:
                positions = physical
        events = events[np.lexsort((events[:, 0], positions))]
        edges = np.r_[0, np.flatnonzero(np.diff(events[:, 0])) + 1, len(events)]
        if reporter is not None:
            reporter.batch(len(edges) - 1)
        prefetch = getattr(reader, 'prefetch_frame_indexes', None)
        live, remaining = {}, [len(w.frames) for w in items]
        sample, dtype = FrameSampler(reader, cache), items[0].view.dtype
        with prefetch(events[edges[:-1], 0], cache) if prefetch else nullcontext():
            for start, stop in zip(edges[:-1], edges[1:]):
                cancel()
                completed = False
                for pieces in _tiles(events[start:stop], items):
                    cancel()
                    rows, cols = [
                        np.concatenate(
                            [getattr(items[i], axis)[lo:hi] for i, _, lo, hi in pieces],
                            dtype='i8',
                        )
                        for axis in ('rows', 'cols')
                    ]
                    values = sample(int(events[start, 0]), rows, cols)
                    if values.shape != rows.shape or values.dtype != dtype:
                        raise ValueError(
                            'sampled pixels do not match image shape/dtype'
                        )
                    cursor = 0
                    for index, omega, lo, hi in pieces:
                        w = items[index]
                        if index not in live:
                            live[index] = np.empty(
                                (len(w.frames), len(w.rows)), dtype=dtype
                            )
                        block = live[index][omega, lo:hi]
                        block[:] = values[cursor : cursor + hi - lo]
                        if not w.view.view.dense and dtype.kind == 'f':
                            block[block == 0] = 0
                        cursor += hi - lo
                        del block
                        if hi == len(w.rows):
                            remaining[index] -= 1
                            if remaining[index] == 0:
                                cancel()
                                pixels = live.pop(index)
                                if w.inverse is not None:
                                    pixels = np.take(pixels, w.inverse, axis=1)
                                pixels = pixels.ravel()
                                stats = native._spotStats(
                                    pixels,
                                    np.array([0, len(pixels)], dtype='i8'),
                                    w.shape,
                                    w.threshold,
                                )
                                for dest, value in zip(w.stats, stats):
                                    dest[:] = value
                                del pixels, stats
                                completed = True
                    del rows, cols, values
                if reporter is not None:
                    if completed:
                        reporter.phase('Segmenting peak windows')
                    reporter.sampled_frame()
                visits += 1
    return visits


def fit_staged(params, bar, cancel_callback=None):
    """Advance grains by pass without repeating any optimizer work."""
    from hexrd.hedm import fitgrains as fitting
    from hexrd.hedm._progress import activity, diagnostic, _cancellation_check

    def requested_cancel():
        if cancel_callback is not None and cancel_callback():
            raise StagedCancelled()

    fitting.fit_grain_FF_init(params)
    ids = [int(x) for x in params['grains_table'][:, 0]]
    steps = {i: fitting._fit_grain_steps(i) for i in ids}
    pending, last_spots, pass_indices, results = {}, {}, {}, {}
    cache = next(iter(params['imgser_dict'].values()))._adapter.cache
    if params.get('reuse_pixels', True):
        from hexrd.hedm._pixel_cache import PixelCache

        cache.pixel_cache = PixelCache()
    else:
        if cache.pixel_cache is not None:
            cache.pixel_cache.clear()
        cache.pixel_cache = None
    budget = params['staged_workspace']
    groups_done = 0
    frames_done = windows_done = 0
    cancelled = False

    def advance(grain, response=None):
        check_cancelled()
        params['progress_pass'] = pass_indices.get(grain, 0)
        if grain in last_spots:
            params['pending_spot_data'] = last_spots.pop(grain)
        try:
            pending[grain] = steps[grain].send(response)
        except StopIteration as done:
            pending.pop(grain, None)
            results[grain] = done.value
            fitting._write_pending_spots(grain)
            fitting._grain_event(
                *fitting._result_event(
                    done.value, len(params['tth_tol']), params.get('progress_pass', 0)
                )
            )
            bar.update(1)
        finally:
            pass_indices[grain] = params.get('progress_pass', 0)
            if 'pending_spot_data' in params:
                last_spots[grain] = params.pop('pending_spot_data')

    def finish_group(group, responses):
        nonlocal groups_done, frames_done, windows_done
        if not group.plans:
            return
        check_cancelled()
        activity(f'Segment peaks: {len(group.plans)} grains · shared frame reads')
        group.measure(cache, check_cancelled)
        for plan in group.plans:
            check_cancelled()
            responses[plan.grain] = plan.result()
        groups_done += 1
        frames_done += group.frame_visits
        windows_done += len(group.windows)
        diagnostic('Staged groups complete', groups_done)
        diagnostic('Staged frame visits', frames_done)
        diagnostic('Peak windows prepared', windows_done)
        if cache.pixel_cache is not None:
            diagnostic(
                'Source pixel reuse',
                f'{cache.pixel_cache.hits:,}/{cache.pixel_cache.queries:,} queries',
            )
            diagnostic(
                'Source pixel cache', f'{cache.pixel_cache.bytes / 1024**2:.1f} MiB'
            )

    cancel_token = _cancellation_check.set(
        requested_cancel if cancel_callback is not None else None
    )
    try:
        for grain in ids:
            advance(grain)
        while pending:
            responses = {}
            group = MeasurementGroup(budget)
            for grain, request in pending.items():
                check_cancelled()
                activity(f'Planning peak windows: grain {grain}')
                plan = MeasurementPlan(
                    params['instrument'], request, max_bytes=budget
                ).prepare()
                plan.grain = grain
                if not group.add(plan):
                    finish_group(group, responses)
                    group = MeasurementGroup(budget)
                    if not group.add(plan):
                        # A single unusually large grain must not defeat the
                        # workspace allowance. Its existing bounded reader is
                        # still exact; it simply cannot share this scan.
                        responses[grain] = params['instrument'].pull_spots(
                            *request[0], **request[1]
                        )
                        continue
            finish_group(group, responses)
            del group, plan
            for grain in list(pending):
                advance(grain, responses.pop(grain))
        return [results[i] for i in ids]
    except (KeyboardInterrupt, StagedCancelled):
        cancelled = True
        raise
    finally:
        try:
            for grain, step in steps.items():
                step.close()
                if not cancelled and grain in last_spots:
                    try:
                        check_cancelled()
                    except (KeyboardInterrupt, StagedCancelled):
                        cancelled = True
                    if not cancelled:
                        params['pending_spot_data'] = last_spots.pop(grain)
                        fitting._write_pending_spots(grain)
        finally:
            if cache.pixel_cache is not None:
                cache.pixel_cache.clear()
            cache.clear_payload()
            fitting.fit_grain_FF_cleanup()
            _cancellation_check.reset(cancel_token)
