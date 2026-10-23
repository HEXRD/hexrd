"""Textual presentation, imported only for interactive find-orientations runs."""

from dataclasses import replace
import threading
from time import monotonic

from rich.text import Text
from rich.traceback import Traceback
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Label, ProgressBar, RichLog, Static


class OrientationDashboard(App):
    TITLE = 'HEXRD'
    SUB_TITLE = 'FIND ORIENTATIONS'
    AUTO_FOCUS = '#events'
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ('l', 'logs', 'Toggle log'),
        ('f', 'follow', 'Follow active stage'),
        ('space', 'pause_log', 'Pause log'),
        Binding('q,ctrl+c,ctrl+q', 'cancel', 'Cancel / close', key_display='q', priority=True),
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
    #panels { height: 11; margin: 0 1; }
    #workflow { width: 3fr; border: round #24425a; padding: 0 1; }
    DataTable { background: #0b1120; height: 1fr; }
    DataTable > .datatable--header { background: #121e32; color: #91a5be; }
    #inspection { width: 2fr; border: round #24425a; padding: 0 2; }
    #stage-name { color: #6de0d2; text-style: bold; margin-bottom: 1; }
    #detail { height: 1fr; color: #91a5be; overflow-y: auto; }
    #counts { height: 2; }
    ProgressBar { width: 100%; }
    ProgressBar Bar { width: 1fr; }
    ProgressBar Bar > .bar--bar { color: #24425a; }
    ProgressBar Bar > .bar--complete { color: #6de0d2; }
    #data { height: 1fr; min-height: 5; margin: 0 1; }
    #reflections { width: 2fr; border: round #24425a; padding: 0 1; }
    #events { width: 3fr; height: 1fr; border: round #24425a; }
    RichLog { background: #0d1628; }
    #status { height: 1; padding: 0 2; color: #91a5be; }
    #status.success { color: #6de0d2; }
    #status.failed { color: #ff8080; }
    #controls { height: 3; padding: 0 1; }
    Button { width: auto; min-width: 12; margin-right: 1; }
    .compact #inspection { padding: 0 1; }
    .compact ProgressBar #eta { display: none; }
    .short #panels { height: 9; }
    """

    def __init__(self, dashboard, workflow, config):
        super().__init__()
        self.dashboard = dashboard
        self.workflow = workflow
        self.config = config
        self.finished = threading.Event()
        self.failure = None
        self.selected = None
        self.ended = None
        self.presented = False
        self.reflection_rows = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon='◆')
        yield Static(self.config, id='config', markup=False)
        with Horizontal(id='metrics'):
            for name, caption in (('elapsed', 'ELAPSED'), ('completed', 'STAGES COMPLETE'),
                                  ('throughput', 'THROUGHPUT')):
                with Vertical(classes='metric'):
                    yield Label(caption, id=name + '-caption', classes='caption')
                    yield Static('—', id=name, classes='value', markup=False)
        with Horizontal(id='panels'):
            with Vertical(id='workflow'):
                yield DataTable(id='stages', cursor_type='row')
            with Vertical(id='inspection'):
                yield Static('Preparing workflow', id='stage-name', markup=False)
                yield Static('', id='detail', markup=False)
                yield Static('', id='counts', markup=False)
                yield ProgressBar(id='progress', show_eta=True)
        with Horizontal(id='data'):
            yield DataTable(id='reflections', cursor_type='none')
            yield RichLog(id='events', max_lines=500, wrap=True, min_width=1)
        yield Static('Select a stage to inspect it · f returns to live progress',
                     id='status', markup=False)
        with Horizontal(id='controls'):
            yield Button('Follow stage', id='follow')
            yield Button('Pause log', id='pause_log')
            yield Button('Hide log', id='logs')
            yield Button('Cancel run', id='cancel', variant='error')
        yield Footer()

    def on_mount(self):
        self.query_one('#workflow').border_title = 'WORKFLOW'
        self.query_one('#inspection').border_title = 'LIVE PROGRESS'
        self.query_one('#events').border_title = 'EVENT LOG'
        reflections = self.query_one('#reflections', DataTable)
        reflections.border_title = 'REFLECTIONS'
        reflections.border_subtitle = 'Peaks: — not measured'
        reflections.tooltip = (
            'Miller indices (h k l). All listed rings are scored. Checked seed '
            'rings also generate trial orientations. Peaks are counted during '
            'seed peak finding and are unavailable when cached fibers are reused.')
        for column in ('h', 'k', 'l', 'Seed', 'Peaks'):
            reflections.add_column(column, key=column)
        table = self.query_one('#stages', DataTable)
        for column in ('Stage', 'State', 'Elapsed'):
            table.add_column(column, key=column, width=9 if column == 'State' else None)
        for name in self.dashboard.stages:
            table.add_row(name, 'Waiting', '—', key=name)
        self.timer = self.set_interval(0.1, self.refresh_progress)
        self.run_worker(self.run_workflow, thread=True, exit_on_error=False)

    def run_workflow(self):
        try:
            self.workflow()
        except BaseException as exc:
            self.failure = exc
        finally:
            self.ended = monotonic()
            self.finished.set()

    def on_resize(self, event):
        self.screen.set_class(event.size.width < 90, 'compact')
        self.screen.set_class(event.size.height < 28, 'short')

    def refresh_progress(self):
        with self.dashboard.lock:
            stages = {name: replace(item) for name, item in self.dashboard.stages.items()}
            events = list(self.dashboard.events)
            self.dashboard.events.clear()
            diagnostics = dict(self.dashboard.diagnostics)
            hkls = dict(self.dashboard.hkls)
            peaks = dict(self.dashboard.peaks)
        self.refresh_reflections(hkls, peaks)
        now = self.ended or monotonic()
        active = next((name for name, item in stages.items() if item.status == 'running'), None)
        completed = sum(item.status == 'done' for item in stages.values())
        self.query_one('#elapsed', Static).update(f'{now - self.dashboard.started:,.1f} s')
        self.query_one('#completed', Static).update(f'{completed} / {len(stages)}')
        table = self.query_one('#stages', DataTable)
        for name, item in stages.items():
            color = dict(pending='dim', running='cyan', done='green', failed='red')[item.status]
            state = dict(pending='Waiting', running='● Running',
                         done='✓ Done', failed='✗ Failed')[item.status]
            elapsed = now - item.started if item.status == 'running' else item.elapsed
            table.update_cell(name, 'State', Text(state, style=color))
            table.update_cell(name, 'Elapsed', f'{elapsed:.1f}s' if item.started else '—')
        latest = next((name for name in reversed(stages) if stages[name].started), None)
        name = self.selected or active or latest or next(iter(stages))
        if self.selected is None and not table.has_focus:
            table.move_cursor(row=list(stages).index(name))
        item = stages[name]
        elapsed = now - item.started if item.status == 'running' else item.elapsed
        summary = self.finished.is_set() and self.failure is None and self.selected is None
        detail = item.detail or item.status.capitalize()
        if summary:
            detail = '\n'.join(f'{key}: {value}' for key, value in diagnostics.items())
            detail = detail or self.dashboard.message or 'Workflow complete'
        self.query_one('#stage-name', Static).update('INDEXING SUMMARY' if summary else name)
        self.query_one('#detail', Static).update(detail)
        counts = self.query_one('#counts', Static)
        counts.display = not summary
        counts.update(
            f'{item.count:,} / {item.total:,} {item.unit}' if item.total else '—')
        bar = self.query_one(ProgressBar)
        bar.display = not summary and not (
            self.finished.is_set() and not item.total and item.status != 'done')
        bar.update(
            total=item.total or (1 if item.status == 'done' else None),
            progress=item.count if item.total else int(item.status == 'done'))
        rate = item.count / elapsed if elapsed and item.count else 0
        self.query_one('#throughput-caption', Label).update(
            'BEST COMPLETENESS' if self.finished.is_set() else 'THROUGHPUT')
        self.query_one('#throughput', Static).update(
            diagnostics.get('Best trial', '—') if self.finished.is_set() else
            f'{rate:,.0f} {item.unit}/s' if rate else '—')
        log = self.query_one(RichLog)
        for event in events:
            log.write(Text(event))
        if self.finished.is_set() and not self.presented:
            self.presented = True
            self.timer.pause()
            failed = self.failure is not None
            self.sub_title = 'FAILED' if failed else 'COMPLETE'
            status = self.query_one('#status', Static)
            status.add_class('failed' if failed else 'success')
            if isinstance(self.failure, KeyboardInterrupt):
                self.sub_title = 'CANCELLED'
                message = 'Workflow cancelled'
            elif failed:
                message = f'{type(self.failure).__name__}: {self.failure}'
                if isinstance(self.failure, Exception):
                    log.display = True
                    self.query_one('#logs', Button).label = 'Hide log'
                    log.write(Traceback.from_exception(
                        type(self.failure), self.failure, self.failure.__traceback__),
                        scroll_end=True)
            else:
                message = self.dashboard.message or 'Workflow complete'
            status.update(f'{message} · q to close')
            button = self.query_one('#cancel', Button)
            button.label, button.variant, button.disabled = 'Close', 'primary', False

    def refresh_reflections(self, hkls, peaks):
        rings = tuple(dict.fromkeys(hkl for group in hkls.values() for hkl in group))
        seeds = set(hkls.get('Seed', ()))
        rows = {hkl: (*(str(index) for index in hkl), '✓' if hkl in seeds else '—',
                      f'{peaks[hkl]:,}' if hkl in peaks else '—') for hkl in rings}
        if rows == self.reflection_rows:
            return
        table = self.query_one('#reflections', DataTable)
        if self.reflection_rows is None or rings != tuple(self.reflection_rows):
            table.clear()
            for row in rows.values():
                table.add_row(*row, key=' '.join(row[:3]))
        else:
            for row in rows.values():
                key = ' '.join(row[:3])
                table.update_cell(key, 'Peaks', row[-1], update_width=True)
                table.update_cell(key, 'Seed', row[-2])
        table.border_title = f'REFLECTIONS · {len(rings)} rings' if hkls else 'REFLECTIONS'
        table.border_subtitle = 'Peaks: — not measured' if hkls else 'Awaiting ring selection'
        self.reflection_rows = rows

    def on_data_table_row_selected(self, event):
        if event.data_table.id != 'stages':
            return
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
        self.query_one('#logs', Button).label = 'Hide log' if log.display else 'Show log'

    def action_pause_log(self):
        log = self.query_one(RichLog)
        log.auto_scroll = not log.auto_scroll
        self.query_one('#pause_log', Button).label = 'Pause log' if log.auto_scroll else 'Follow log'
        if log.auto_scroll:
            log.scroll_end(animate=False)

    async def on_button_pressed(self, event: Button.Pressed):
        await self.run_action(event.button.id)

    def action_cancel(self):
        if self.finished.is_set():
            self.exit()
            return
        self.dashboard.cancelled.set()
        self.query_one('#cancel', Button).disabled = True
        self.query_one('#status', Static).update('Cancelling · waiting for current work to finish…')
