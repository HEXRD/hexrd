"""Exact spot measurements prepared in frame order before each grain-fit pass.

Only the original angular windows are segmented. Temporary pixel support is
retained until an omega window is complete; the unchanged native spot-statistics
kernel then produces its measurements. Global detector centroids are not used.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy import sparse

from hexrd.core.instrument._spotutils import _extraction_context
from hexrd.hedm._fit_progress import current_reporter, set_reporter
from hexrd.hedm._image_cache import StreamingView
from hexrd.hedm._progress import check_cancelled


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


def _signature(view, arrays, threshold):
    digest = hashlib.sha256()
    source = getattr(view, 'view', None)
    scope = (
        id(view) if source is None else (id(source.reader), source.roi, source.dense)
    )
    if source is not None and source.mapping is not None:
        arrays = (*arrays, np.asarray(source.mapping))
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(repr((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast('B'))
    return scope, repr(threshold), digest.digest()


@dataclass
class Batch:
    view: StreamingView
    frames: np.ndarray
    offsets: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    shapes: np.ndarray
    threshold: object
    signature: tuple
    stats: tuple

    @property
    def bytes(self):
        return 2048 + sum(
            a.nbytes
            for a in (
                self.frames,
                self.offsets,
                self.rows,
                self.cols,
                self.shapes,
                *self.stats,
            )
        )


class _DeferredView:
    def __init__(self, context, original):
        self.context, self.original, self.dtype = context, original, original.dtype

    def extract(self, frames, offsets, rows, cols, shapes, threshold):
        check_cancelled()
        arrays = (frames, offsets, rows, cols, shapes)
        signature = _signature(self.original, arrays, threshold)
        context = self.context
        if context.replaying:
            if context.position >= len(context.batches):
                raise RuntimeError('Unplanned staged spot window')
            batch = context.batches[context.position]
            context.position += 1
            if signature != batch.signature:
                raise RuntimeError(
                    'Staged spot window changed; refusing measurement reuse'
                )
            return batch.stats
        n = len(shapes)
        sum_dtype = np.empty(0, dtype=self.dtype).sum().dtype
        stats = (
            np.zeros((n, 3)),
            np.zeros(n, dtype=sum_dtype),
            np.zeros(n, dtype=self.dtype),
            np.zeros(n, dtype=bool),
        )
        batch = Batch(
            self.original,
            frames.copy(),
            offsets.copy(),
            _compact(rows),
            _compact(cols),
            shapes.copy(),
            threshold,
            signature,
            stats,
        )
        if (
            context.max_bytes is not None
            and context.bytes + batch.bytes > context.max_bytes
        ):
            raise _PlanTooLarge()
        context.bytes += batch.bytes
        context.batches.append(batch)
        # Planning does no image processing or fitting. Invalid placeholders
        # avoid constructing measured positions until the real stats exist.
        return stats


class MeasurementPlan:
    def __init__(self, instrument, request, max_bytes=None):
        self.instrument, self.request = instrument, request
        self.batches = []
        self.replaying = False
        self.position = 0
        self.max_bytes, self.bytes = max_bytes, 0

    def wrap(self, view):
        return _DeferredView(self, view) if isinstance(view, StreamingView) else view

    @contextmanager
    def _context(self, *, planning=False):
        token = _extraction_context.set(self)
        reporter = current_reporter()
        if planning:
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
            with self._context(planning=True):
                self.fallback_result = self.instrument.pull_spots(
                    *self.request[0], **self.request[1]
                )
        except _PlanTooLarge:
            self.batches.clear()
            self.bytes, overflow = 0, True
        if overflow:
            check_cancelled()
            self.fallback_result = self.instrument.pull_spots(
                *self.request[0], **self.request[1]
            )
        return self

    def result(self):
        if not self.batches:
            return self.fallback_result
        self.fallback_result = None
        self.replaying, self.position = True, 0
        with self._context():
            result = self.instrument.pull_spots(*self.request[0], **self.request[1])
        if self.position != len(self.batches):
            raise RuntimeError('Staged measurement replay omitted windows')
        return result


@dataclass
class Patch:
    batch: Batch
    index: int
    frames: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    pixels: object = None
    remaining: int = 0

    @property
    def reader(self):
        return self.batch.view.view.reader

    @property
    def bytes(self):
        return len(self.rows) * len(self.frames) * self.batch.view.dtype.itemsize


def patches(plan):
    result = []
    for batch in plan.batches:
        mapping = batch.view.view.mapping
        for i, frame_ids in enumerate(batch.frames):
            actual = (
                np.asarray(mapping)[frame_ids] if mapping is not None else frame_ids
            )
            actual = np.asarray(actual, dtype=np.int64).copy()
            actual[actual < 0] += batch.view.view.reader.nframes
            first, last = batch.offsets[i : i + 2]
            result.append(
                Patch(
                    batch,
                    i,
                    actual,
                    batch.rows[first:last],
                    batch.cols[first:last],
                    remaining=len(actual),
                )
            )
    return result


class MeasurementGroup:
    """Bound planned geometry, event tables and live pixel windows together."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.plans, self.patches = [], []
        self.cost = min(128 * 1024**2, max_bytes // 4)
        self.frontiers = {}
        self.peak_pixels = 0
        self.frame_visits = 0

    def add(self, plan):
        incoming = patches(plan)
        cost = sum(b.bytes for b in plan.batches) + sum(
            1024 + 64 * len(p.frames) for p in incoming
        )
        frontiers = {r: x.copy() for r, x in self.frontiers.items()}
        for p in incoming:
            if p.reader not in frontiers:
                frontiers[p.reader] = np.zeros(p.reader.nframes + 1, dtype='i8')
            delta = frontiers[p.reader]
            delta[int(p.frames.min())] += p.bytes
            delta[int(p.frames.max()) + 1] -= p.bytes
        peak = max(
            (int(x.cumsum().max(initial=0)) for x in frontiers.values()), default=0
        )
        if self.cost + cost + peak > self.max_bytes:
            return False
        self.plans.append(plan)
        self.patches.extend(incoming)
        self.cost += cost
        self.frontiers, self.peak_pixels = frontiers, peak
        return True

    def measure(self, cache, cancel=check_cancelled):
        try:
            self._measure(cache, cancel)
        finally:
            for patch in self.patches:
                patch.pixels = None
            cache.clear_payload()

    def _measure(self, cache, cancel):
        from hexrd.core.extensions import transforms as native
        from hexrd.core.imageseries._sparse import sample_pixels

        readers = {}
        for p in self.patches:
            readers.setdefault(p.reader, []).append(p)
        reporter = current_reporter()
        for reader, items in readers.items():
            cancel()
            events = np.empty((sum(len(p.frames) for p in items), 3), dtype='i8')
            cursor = 0
            for index, p in enumerate(items):
                end = cursor + len(p.frames)
                events[cursor:end, 0] = p.frames
                events[cursor:end, 1] = index
                events[cursor:end, 2] = np.arange(len(p.frames))
                cursor = end
            order = np.argsort(events[:, 0], kind='stable')
            events = events[order]
            del order
            edges = np.r_[0, np.flatnonzero(np.diff(events[:, 0])) + 1, len(events)]
            if reporter is not None:
                reporter.batch(len(edges) - 1)
            for first, last in zip(edges[:-1], edges[1:]):
                cancel()
                occurrences = events[first:last]
                row_parts, col_parts = [], []
                for _, patch_index, _ in occurrences:
                    p = items[patch_index]
                    view = p.batch.view.view
                    rows = p.rows.astype('i8').clip(0, view.shape[0] - 1)
                    cols = p.cols.astype('i8').clip(0, view.shape[1] - 1)
                    if view.roi is not None:
                        rows += view.roi[0]
                        cols += view.roi[1]
                    row_parts.append(rows)
                    col_parts.append(cols)
                counts = np.array([len(x) for x in row_parts])
                sparse_query = counts.sum() <= 65536
                combined = None
                cache.clear_payload()
                self.frame_visits += 1
                cache.max_bytes = min(128 * 1024**2, self.max_bytes // 4)
                cache.metadata_limit = cache.max_bytes // 2
                if sparse_query and hasattr(reader, 'sample_pixels'):
                    combined = reader.sample_pixels(
                        int(events[first, 0]),
                        np.concatenate(row_parts),
                        np.concatenate(col_parts),
                        cache,
                    )
                frame = frame_view = None
                if combined is None:
                    frame = cache.frame(reader, int(events[first, 0]))
                    frame_view = (
                        native._SparseFrames([frame])
                        if sparse.issparse(frame) and frame.has_sorted_indices
                        else None
                    )
                cursor = 0
                for event, rows, cols in zip(occurrences, row_parts, col_parts):
                    cancel()
                    _, patch_index, omega_index = event
                    p = items[patch_index]
                    view = p.batch.view.view
                    if combined is not None:
                        values = combined[cursor : cursor + len(rows)]
                        cursor += len(rows)
                    elif frame_view is not None:
                        values = frame_view.gather(
                            np.zeros((1, 1), dtype='i8'),
                            np.array([0, len(rows)], dtype='i8'),
                            rows,
                            cols,
                        )
                    else:
                        values = (
                            sample_pixels(frame, rows, cols)
                            if sparse.issparse(frame)
                            else frame[rows, cols]
                        )
                        if not view.dense and not sparse.issparse(frame):
                            values = values.copy()
                            values[values == 0] = 0
                    if p.pixels is None:
                        p.pixels = np.empty(
                            (len(p.frames), len(rows)), dtype=p.batch.view.dtype
                        )
                    p.pixels[omega_index] = values
                    p.remaining -= 1
                    if not p.remaining:
                        if reporter is not None:
                            reporter.phase('Segmenting peak windows')
                        shape = p.batch.shapes[p.index : p.index + 1]
                        stats = native._spotStats(
                            p.pixels.ravel(),
                            np.array([0, p.pixels.size], dtype='i8'),
                            shape,
                            p.batch.threshold,
                        )
                        for dest, source in zip(p.batch.stats, stats):
                            dest[p.index] = source[0]
                        p.pixels = None
                del (
                    frame_view,
                    frame,
                    combined,
                    row_parts,
                    col_parts,
                    rows,
                    cols,
                    values,
                )
                cache.clear_payload()
                if reporter is not None:
                    reporter.cache(cache)
                    reporter.sampled_frame()
        if any(p.remaining for p in self.patches):
            raise RuntimeError('Incomplete staged pixel support')


def fit_staged(params, bar, cancel_callback=None):
    """Advance grains by pass without repeating any optimizer work."""
    from hexrd.hedm import fitgrains as fitting
    from hexrd.hedm._progress import activity, diagnostic, _cancellation_check

    def cancel():
        check_cancelled()

    def requested_cancel():
        if cancel_callback is not None and cancel_callback():
            raise StagedCancelled()

    fitting.fit_grain_FF_init(params)
    ids = [int(x) for x in params['grains_table'][:, 0]]
    steps = {i: fitting._fit_grain_steps(i) for i in ids}
    pending, last_spots, pass_indices, results = {}, {}, {}, {}
    cache = next(iter(params['imgser_dict'].values()))._adapter.cache
    budget = params['staged_workspace']
    groups_done = 0
    frames_done = windows_done = 0
    cancelled = False

    def advance(grain, response=None):
        cancel()
        params['progress_pass'] = pass_indices.get(grain, 0)
        if grain in last_spots:
            params['pending_spot_data'] = last_spots.pop(grain)
        try:
            pending[grain] = steps[grain].send(response)
        except StopIteration as done:
            pending.pop(grain, None)
            results[grain] = done.value
            fitting._write_pending_spots(grain)
            state = 'Insufficient reflections' if np.isinf(done.value[2]) else 'Done'
            fitting._grain_event(
                grain,
                state,
                params.get('progress_pass', 0),
                len(params['tth_tol']),
                done.value[1],
                done.value[2],
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
        cancel()
        activity(f'Segment peaks: {len(group.plans)} grains · shared frame reads')
        group.measure(cache, cancel)
        for plan in group.plans:
            cancel()
            responses[plan.grain] = plan.result()
        groups_done += 1
        frames_done += group.frame_visits
        windows_done += len(group.patches)
        diagnostic('Staged groups complete', groups_done)
        diagnostic('Staged frame visits', frames_done)
        diagnostic('Peak windows prepared', windows_done)

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
                cancel()
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
                        cancel()
                    except (KeyboardInterrupt, StagedCancelled):
                        cancelled = True
                    if not cancelled:
                        params['pending_spot_data'] = last_spots.pop(grain)
                        fitting._write_pending_spots(grain)
        finally:
            cache.clear_payload()
            fitting.fit_grain_FF_cleanup()
            _cancellation_check.reset(cancel_token)
