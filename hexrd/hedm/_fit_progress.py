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

    def __init__(self, queue=None):
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
            frame_reads=0,
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
        self.data['phase'] = phase
        self.publish(force=force)

    def grain(self, grain, state, index, passes):
        if grain != self.data['grain'] or state == 'Extracting':
            self.data.update(pass_done=0, pass_total=0, batch_done=0, batch_total=0)
        self.data.update(grain=int(grain), state=state, pass_index=index, passes=passes)
        self.phase(state, force=True)

    def panel(self, total):
        self.data['pass_total'] += int(total)
        self.phase('Building patches', force=True)

    def reflections(self, count):
        self.data['pass_done'] += int(count)
        self.publish()

    def batch(self, total):
        self.data.update(batch_done=0, batch_total=int(total))
        self.phase('Read/sample frames', force=True)

    def sampled_frame(self):
        self.data['batch_done'] += 1
        self.publish()

    def read_started(self):
        self.data['frame_reads'] += 1
        self.phase('Read/decode frame')
