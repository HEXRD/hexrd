"""Terminal-only presentation for find-orientations; no scientific state lives here."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import shutil
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
    item = dashboard.stages[name]
    with dashboard.lock:
        dashboard.active = item
        item.status, item.started = 'running', monotonic()
    try:
        yield
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
    """Refresh a fixed terminal panel, retaining the final view on exit."""

    disable = False  # Same progress interface as tqdm.

    @staticmethod
    def supported(stream):
        try:
            '─✓◒'.encode(stream.encoding or 'utf-8')
        except UnicodeEncodeError:
            return False
        return (stream.isatty() and os.environ.get('TERM') != 'dumb'
                and shutil.get_terminal_size().lines >= 14)

    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.stages = {name: _Stage() for name in STAGES}
        self.active = None
        self.message = ''
        self.lines = 0
        self.started = monotonic()
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        self.token = _dashboard.set(self)
        self.stream.write('\x1b[?25l')
        self.render()
        self.thread = threading.Thread(target=self._refresh, daemon=True)
        self.thread.start()

    def _refresh(self):
        while not self.stopped.wait(0.1):
            self.render()

    def emit(self, record):
        with self.lock:
            message = ' '.join(record.getMessage().split())
            if not message.startswith('==='):
                self.message = message

    def update(self, count):
        with self.lock:
            self.active.count += count

    def render(self):
        with self.lock:
            now = monotonic()
            width = max(1, min(shutil.get_terminal_size().columns - 1, 100))
            lines = [('36;1', f'HEXRD  /  FIND ORIENTATIONS     {now - self.started:,.1f}s'),
                     ('2', '─' * width)]
            for name, item in self.stages.items():
                elapsed = now - item.started if item.status == 'running' else item.elapsed
                icon = {'pending': '·', 'running': '◒◐◓◑'[int(now * 8) % 4],
                        'done': '✓', 'failed': '✗'}[item.status]
                detail = item.detail or item.status
                if item.total:
                    fraction = min(item.count / item.total, 1.)
                    filled = int(12 * fraction)
                    detail = (f'{"━" * filled}{"─" * (12 - filled)} {fraction:4.0%}'
                              f'  {item.count:,}/{item.total:,} {item.unit}')
                color = {'pending': '2', 'running': '36', 'done': '32', 'failed': '31'}[item.status]
                lines.append((color, f' {icon}  {name:<16} {elapsed:7.1f}s  {detail}'))
            lines.extend([('2', '─' * width), ('', self.message)])
            self.stream.write((f'\x1b[{self.lines}A' if self.lines else '') + ''.join(
                f'\r\x1b[2K\x1b[{color or "0"}m{text[:width]}\x1b[0m\n'
                for color, text in lines))
            self.stream.flush()
            self.lines = len(lines)

    def close(self):
        if self.thread is not None:
            self.stopped.set()
            self.thread.join()
            _dashboard.reset(self.token)
            try:
                self.render()
            finally:
                self.stream.write('\x1b[?25h')
                self.stream.flush()
                self.thread = None
        super().close()
