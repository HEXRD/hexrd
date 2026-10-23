"""Textual presentation, imported only for interactive find-orientations runs."""

from dataclasses import replace
import threading
from time import monotonic

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, RichLog, Static


class OrientationDashboard(App):
    TITLE = 'HEXRD'
    SUB_TITLE = 'FIND ORIENTATIONS'
    AUTO_FOCUS = '#events'
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ('l', 'logs', 'Toggle log'),
        ('f', 'follow', 'Follow active stage'),
        Binding('q,ctrl+c,ctrl+q', 'cancel', 'Cancel', key_display='q', priority=True),
    ]
    CSS = """
    Screen { background: #0b1120; color: #dbe7f5; }
    Header { background: #121e32; color: #6de0d2; }
    Footer { background: #121e32; }
    #config { height: 1; padding: 0 2; color: #91a5be; }
    #metrics { height: 3; margin: 0 1; }
    .metric { width: 1fr; padding: 0 2; border-left: tall #24425a; }
    .caption { color: #91a5be; }
    .value { text-style: bold; color: #6de0d2; }
    #panels { height: 12; margin: 0 1; }
    #workflow { width: 3fr; border: round #24425a; padding: 0 1; }
    DataTable { background: #0b1120; height: 1fr; }
    DataTable > .datatable--header { background: #121e32; color: #91a5be; }
    #inspection { width: 2fr; border: round #24425a; padding: 0 2; }
    #stage-name { color: #6de0d2; text-style: bold; margin-bottom: 1; }
    #detail { height: 3; color: #91a5be; }
    #counts { height: 2; }
    ProgressBar { width: 100%; }
    ProgressBar Bar { width: 1fr; }
    ProgressBar Bar > .bar--bar { color: #24425a; }
    ProgressBar Bar > .bar--complete { color: #6de0d2; }
    #events { height: 1fr; min-height: 3; margin: 0 1; border: round #24425a; }
    RichLog { background: #0d1628; }
    #status { height: 1; padding: 0 2; color: #91a5be; }
    .compact #inspection { padding: 0 1; }
    .compact ProgressBar #eta { display: none; }
    """

    def __init__(self, dashboard, workflow, config):
        super().__init__()
        self.dashboard = dashboard
        self.workflow = workflow
        self.config = config
        self.finished = threading.Event()
        self.failure = None
        self.selected = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon='◆')
        yield Static(self.config, id='config', markup=False)
        with Horizontal(id='metrics'):
            for name, caption in (('elapsed', 'ELAPSED'), ('completed', 'STAGES COMPLETE'),
                                  ('throughput', 'THROUGHPUT')):
                with Vertical(classes='metric'):
                    yield Label(caption, classes='caption')
                    yield Static('—', id=name, classes='value', markup=False)
        with Horizontal(id='panels'):
            with Vertical(id='workflow'):
                yield DataTable(id='stages', cursor_type='row')
            with Vertical(id='inspection'):
                yield Static('Preparing workflow', id='stage-name', markup=False)
                yield Static('', id='detail', markup=False)
                yield Static('', id='counts', markup=False)
                yield ProgressBar(id='progress', show_eta=True)
        yield RichLog(id='events', max_lines=500, wrap=True, min_width=1)
        yield Static('Select a stage to inspect it · f returns to live progress',
                     id='status', markup=False)
        yield Footer()

    def on_mount(self):
        self.query_one('#workflow').border_title = 'WORKFLOW'
        self.query_one('#inspection').border_title = 'LIVE PROGRESS'
        self.query_one('#events').border_title = 'EVENT LOG'
        table = self.query_one(DataTable)
        for column in ('Stage', 'State', 'Elapsed'):
            table.add_column(column, key=column, width=9 if column == 'State' else None)
        for name in self.dashboard.stages:
            table.add_row(name, 'Waiting', '—', key=name)
        self.set_interval(0.1, self.refresh_progress)
        self.run_worker(self.run_workflow, thread=True, exit_on_error=False)

    def run_workflow(self):
        try:
            self.workflow()
        except BaseException as exc:
            self.failure = exc
        finally:
            self.finished.set()

    def on_resize(self, event):
        self.screen.set_class(event.size.width < 90, 'compact')

    def refresh_progress(self):
        with self.dashboard.lock:
            stages = {name: replace(item) for name, item in self.dashboard.stages.items()}
            events = list(self.dashboard.events)
            self.dashboard.events.clear()
        now = monotonic()
        active = next((name for name, item in stages.items() if item.status == 'running'), None)
        completed = sum(item.status == 'done' for item in stages.values())
        self.query_one('#elapsed', Static).update(f'{now - self.dashboard.started:,.1f} s')
        self.query_one('#completed', Static).update(f'{completed} / {len(stages)}')
        table = self.query_one(DataTable)
        if self.selected is None and active and not table.has_focus:
            table.move_cursor(row=list(stages).index(active))
        for name, item in stages.items():
            color = dict(pending='dim', running='cyan', done='green', failed='red')[item.status]
            state = dict(pending='Waiting', running='● Running',
                         done='✓ Done', failed='✗ Failed')[item.status]
            elapsed = now - item.started if item.status == 'running' else item.elapsed
            table.update_cell(name, 'State', Text(state, style=color))
            table.update_cell(name, 'Elapsed', f'{elapsed:.1f}s' if item.started else '—')
        name = self.selected or active or next(reversed(stages))
        item = stages[name]
        elapsed = now - item.started if item.status == 'running' else item.elapsed
        self.query_one('#stage-name', Static).update(name)
        self.query_one('#detail', Static).update(item.detail or item.status.capitalize())
        self.query_one('#counts', Static).update(
            f'{item.count:,} / {item.total:,} {item.unit}' if item.total else '—')
        self.query_one(ProgressBar).update(
            total=item.total or (1 if item.status == 'done' else None),
            progress=item.count if item.total else int(item.status == 'done'))
        rate = item.count / elapsed if elapsed and item.count else 0
        self.query_one('#throughput', Static).update(f'{rate:,.0f} {item.unit}/s' if rate else '—')
        log = self.query_one(RichLog)
        for event in events:
            log.write(Text(event))
        if self.finished.is_set():
            self.call_after_refresh(self.exit)

    def on_data_table_row_selected(self, event):
        self.selected = str(event.row_key.value)
        self.query_one('#inspection').border_title = 'STAGE DETAILS'
        self.refresh_progress()

    def action_follow(self):
        self.selected = None
        self.set_focus(None)
        self.query_one('#inspection').border_title = 'LIVE PROGRESS'
        self.refresh_progress()

    def action_logs(self):
        log = self.query_one(RichLog)
        log.display = not log.display

    def action_cancel(self):
        self.dashboard.cancelled.set()
        self.query_one('#status', Static).update('Cancelling · waiting for current work to finish…')
