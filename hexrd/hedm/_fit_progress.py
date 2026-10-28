"""Sampled fit telemetry. Counters never participate in numerical decisions.

Worker snapshots contain cumulative counters, so a full IPC queue can drop an
update without losing all the work counted since the previous update. At most
one latest snapshot per worker is retained by the dashboard.
"""

from contextvars import ContextVar
import os
from queue import Full
from time import monotonic

_reporter = ContextVar('fit_progress_reporter', default=None)


def current_reporter():
    return _reporter.get()


def set_reporter(reporter):
    _reporter.set(reporter)


class FitReporter:
    interval = 0.2

    def __init__(self, queue=None, *, lazy=False, cache_limit=0):
        self.queue = queue
        self.last_sent = -float('inf')
        self.data = dict(
            kind='fit_metrics',
            worker=os.getpid(),
            sequence=0,
            grain=None,
            state='Waiting',
            phase='Waiting',
            pass_index=0,
            passes=0,
            pass_done=0,
            pass_total=0,
            batch_done=0,
            batch_total=0,
            reflections=0,
            signals=0,
            passes_done=0,
            passes_skipped=0,
            refits_done=0,
            evaluations=0,
            frame_reads=0,
            frame_hits=0,
            cache_misses=0,
            decoded_bytes=0,
            read_seconds=0.0,
            cache_bytes=0,
            cache_limit=cache_limit,
            evictions=0,
            lazy=lazy,
            phase_started=monotonic(),
            updated=monotonic(),
        )

    def publish(self, *, force=False):
        now = monotonic()
        if not force and now - self.last_sent < self.interval:
            return
        self.data['updated'] = now
        self.data['sequence'] += 1
        snapshot = dict(self.data)
        if self.queue is None:
            from hexrd.hedm._progress import report_fit_metrics

            report_fit_metrics(snapshot)
        else:
            try:
                self.queue.put_nowait(snapshot)
            except Full:
                return
        self.last_sent = now

    def phase(self, phase, *, force=False):
        if phase != self.data['phase']:
            self.data['phase'] = phase
            self.data['phase_started'] = monotonic()
        self.publish(force=force)

    def grain(self, grain, state, index, passes):
        if grain != self.data['grain']:
            self.data.update(pass_done=0, pass_total=0, batch_done=0, batch_total=0)
        self.data.update(grain=int(grain), state=state, pass_index=index, passes=passes)
        if state == 'Extracting':
            self.data.update(pass_done=0, pass_total=0, batch_done=0, batch_total=0)
        self.phase(state, force=True)

    def panel(self, total):
        self.data['pass_total'] += int(total)
        self.phase('Building patches', force=True)

    def reflections(self, count, signals=0):
        self.data['pass_done'] += int(count)
        self.data['reflections'] += int(count)
        self.data['signals'] += int(signals)
        self.publish()

    def batch(self, total):
        self.data.update(batch_done=0, batch_total=int(total))
        self.phase('Read/sample frames', force=True)

    def sampled_frame(self):
        self.data['batch_done'] += 1
        self.publish()

    def cache(self, cache):
        self.data.update(cache_bytes=cache.bytes, cache_limit=cache.max_bytes)

    def read_started(self):
        self.data['frame_reads'] += 1
        self.phase('Read/decode frame')

    def read_finished(self, size, seconds):
        self.data['decoded_bytes'] += int(size)
        self.data['read_seconds'] += seconds

    def cache_hit(self):
        self.data['frame_hits'] += 1

    def evaluation(self):
        self.data['evaluations'] += 1
        # An objective is cheap; avoid a clock call for every evaluation.
        if self.data['evaluations'] % 8 == 0:
            self.publish()

    def pass_complete(self, skipped=0):
        self.data['passes_skipped'] += skipped
        self.data['passes_done'] += 1
        self.publish(force=True)

    def refit_complete(self):
        self.data['refits_done'] += 1
        self.publish(force=True)
