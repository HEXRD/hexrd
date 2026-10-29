"""Shared terminal progress for HEDM workflows; no scientific state lives here."""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import threading
from time import monotonic

logger = logging.getLogger('hexrd.hedm')
_dashboard = ContextVar('hedm_dashboard', default=None)
_cancellation_check = ContextVar('hedm_cancellation_check', default=None)
STAGES = (
    'Configuration',
    'Material',
    'Eta/omega maps',
    'Fibers',
    'Sampling',
    'Scoring',
    'Clustering',
    'Write results',
)
FIT_STAGES = ('Configuration', 'Material', 'Load images', 'Fit grains', 'Write results')


@dataclass
class _Stage:
    status: str = 'pending'
    started: float = 0.0
    elapsed: float = 0.0
    count: int = 0
    total: int = 0
    unit: str = ''
    detail: str = ''


@contextmanager
def stage(name):
    dashboard = _dashboard.get()
    started = monotonic()
    logger.info('%s started', name)
    item = None
    if dashboard:
        dashboard.check_cancelled()
        with dashboard.lock:
            # Indexing can be invoked by fit-grains when estimates are absent.
            dashboard.stages.setdefault(name, _Stage())
            item = dashboard.active = dashboard.stages[name]
            item.count = item.total = 0
            item.detail = item.unit = ''
            item.status, item.started = 'running', started
    try:
        yield
        if dashboard:
            dashboard.check_cancelled()
    except BaseException as exc:
        logger.error('%s failed: %s', name, str(exc) or type(exc).__name__)
        if item:
            with dashboard.lock:
                item.status = 'failed'
        raise
    else:
        logger.info('%s complete (%.2fs)', name, monotonic() - started)
    finally:
        if item:
            with dashboard.lock:
                item.elapsed = monotonic() - started
                if item.status == 'running':
                    item.status = 'done'


@contextmanager
def progress_bar(**kwargs):
    dashboard = _dashboard.get()
    if not dashboard:
        from tqdm import tqdm

        if dashboard is False:
            kwargs['disable'] = True
        with tqdm(**kwargs) as bar:
            yield bar
    else:
        with dashboard.lock:
            dashboard.active.total = kwargs['total']
            dashboard.active.unit = kwargs['unit']
        yield dashboard


def check_cancelled():
    dashboard = _dashboard.get()
    if dashboard:
        dashboard.check_cancelled()
    callback = _cancellation_check.get()
    if callback is not None:
        token = _cancellation_check.set(None)
        try:
            callback()
        finally:
            _cancellation_check.reset(token)


def current_dashboard():
    return _dashboard.get() or None


def detach_dashboard():
    """Workers report through IPC, never through inherited UI objects/locks."""
    _dashboard.set(False)
    for log in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
        if isinstance(log, logging.Logger):
            for handler in list(log.handlers):
                if isinstance(handler, Dashboard):
                    log.removeHandler(handler)


def report_grain(event):
    """Publish a compact worker event in the parent process only."""
    dashboard = current_dashboard()
    if dashboard is None:
        return
    grain_id, state, pass_index, passes, completeness, chisq = event
    with dashboard.lock:
        dashboard.grains.pop(grain_id, None)
        dashboard.grains[grain_id] = event
        while len(dashboard.grains) > 32:
            dashboard.grains.pop(next(iter(dashboard.grains)))
        if (
            state in ('Done', 'Insufficient reflections')
            and grain_id not in dashboard.finished_grains
        ):
            dashboard.finished_grains.add(grain_id)
            dashboard.diagnostics['Grains finished'] = str(
                len(dashboard.finished_grains)
            )
            if state == 'Insufficient reflections':
                dashboard.diagnostics['Insufficient reflections'] = str(
                    int(dashboard.diagnostics.get('Insufficient reflections', '0')) + 1
                )


def report_fit_metrics(snapshot):
    dashboard = current_dashboard()
    if dashboard is None:
        return
    with dashboard.lock:
        old = dashboard.fit_workers.get(snapshot['worker'])
        if old is None or snapshot['sequence'] > old['sequence']:
            dashboard.fit_workers[snapshot['worker']] = snapshot
            grain = snapshot['grain']
            if grain is not None:
                dashboard.fit_grain_metrics.pop(grain, None)
                dashboard.fit_grain_metrics[grain] = snapshot
                while len(dashboard.fit_grain_metrics) > 32:
                    dashboard.fit_grain_metrics.pop(
                        next(iter(dashboard.fit_grain_metrics))
                    )


def activity(message):
    dashboard = _dashboard.get()
    if dashboard:
        dashboard.check_cancelled()
        with dashboard.lock:
            dashboard.active.detail = message


def report_hkls(role, hkls):
    hkls = tuple(map(tuple, hkls))
    text = ', '.join('[' + ' '.join(map(str, hkl)) + ']' for hkl in hkls)
    logger.info('%s HKLs: %s', role, text or 'none', extra={'reflection': True})
    dashboard = _dashboard.get()
    if dashboard:
        with dashboard.lock:
            dashboard.hkls[role] = hkls


def report_peaks(hkl, count):
    logger.info('Seed HKL %s: %d peaks', hkl, count, extra={'reflection': True})
    dashboard = _dashboard.get()
    if dashboard:
        with dashboard.lock:
            dashboard.peaks[tuple(hkl)] = count


def diagnostic(name, value):
    logger.info('%s: %s', name, value)
    dashboard = _dashboard.get()
    if dashboard:
        with dashboard.lock:
            dashboard.diagnostics[name] = str(value)


@contextmanager
def quiet_progress(quiet):
    token = _dashboard.set(False if quiet else _dashboard.get())
    try:
        yield
    finally:
        _dashboard.reset(token)


class Dashboard(logging.Handler):
    """Thread-safe progress shared by the workflow and the terminal app."""

    disable = False  # Same progress interface as tqdm.

    @staticmethod
    def supported(stream):
        return stream.isatty() and os.environ.get('TERM') != 'dumb'

    def __init__(self, stages=STAGES):
        super().__init__()
        self.stages = {name: _Stage() for name in stages}
        self.active = None
        self.message = ''
        self.diagnostics = {}
        self.hkls = {}
        self.peaks = {}
        self.fit_workers = {}
        self.fit_grain_metrics = {}
        self.grains = {}
        self.finished_grains = set()
        self.events = deque(maxlen=500)
        self.started = monotonic()
        self.cancelled = threading.Event()
        self.token = None

    def start(self):
        self.started = monotonic()
        self.token = _dashboard.set(self)

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise KeyboardInterrupt

    def emit(self, record):
        if getattr(record, 'reflection', False):
            return
        with self.lock:
            self.events.append(self.format(record))
            message = ' '.join(record.getMessage().split())
            if not message.startswith('==='):
                self.message = message

    def update(self, count):
        self.check_cancelled()
        with self.lock:
            self.active.count += count

    def close(self):
        if self.token is not None:
            _dashboard.reset(self.token)
            self.token = None
        super().close()
