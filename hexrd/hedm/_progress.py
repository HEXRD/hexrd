"""Terminal-only presentation for find-orientations; no scientific state lives here."""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import threading
from time import monotonic


_dashboard = ContextVar('find_orientations_dashboard', default=None)
STAGES = ('Configuration', 'Material', 'Eta/omega maps', 'Fibers',
          'Sampling', 'Scoring', 'Clustering', 'Write results')


@dataclass
class _Stage:
    status: str = 'pending'
    started: float = 0.
    elapsed: float = 0.
    count: int = 0
    total: int = 0
    unit: str = ''
    detail: str = ''


@contextmanager
def stage(name):
    dashboard = _dashboard.get()
    if not dashboard:
        yield
        return
    dashboard.check_cancelled()
    item = dashboard.stages[name]
    with dashboard.lock:
        dashboard.active = item
        item.status, item.started = 'running', monotonic()
    try:
        yield
        dashboard.check_cancelled()
    except BaseException:
        with dashboard.lock:
            item.status = 'failed'
        raise
    finally:
        with dashboard.lock:
            item.elapsed = monotonic() - item.started
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


def activity(message):
    dashboard = _dashboard.get()
    if dashboard:
        with dashboard.lock:
            dashboard.active.detail = message


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

    def __init__(self):
        super().__init__()
        self.stages = {name: _Stage() for name in STAGES}
        self.active = None
        self.message = ''
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
