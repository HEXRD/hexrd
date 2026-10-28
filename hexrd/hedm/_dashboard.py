"""Shared Textual presentation, imported only for interactive HEDM runs."""

from collections import deque
from dataclasses import replace
from contextvars import copy_context
import multiprocessing as mp
import threading
from time import monotonic

import psutil
from rich.text import Text
from rich.traceback import Traceback
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    RichLog,
    Static,
)


def _memory_usage():
    """Resident bytes for HEXRD and its children, apportioning shared pages when supported."""
    process = psutil.Process()
    used = 0
    for worker in [process, *process.children(recursive=True)]:
        try:
            try:
                memory = worker.memory_full_info()
            except psutil.AccessDenied:
                memory = worker.memory_info()
            used += getattr(memory, 'pss', memory.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return used, psutil.virtual_memory().total


class WorkflowDashboard(App):
    TITLE = 'HEXRD'
    SUB_TITLE = 'FIND ORIENTATIONS'
    AUTO_FOCUS = '#events'
    ENABLE_COMMAND_PALETTE = False
    SUMMARY_TITLE = 'INDEXING SUMMARY'
    COMPLETION_CAPTION = 'BEST COMPLETENESS'
    COMPLETION_KEY = 'Best trial'
    BINDINGS = [
        ('l', 'logs', 'Toggle log'),
        ('f', 'follow', 'Follow active stage'),
        ('space', 'pause_log', 'Pause log'),
        Binding(
            'q,ctrl+c,ctrl+q',
            'cancel',
            'Cancel / close',
            key_display='q',
            priority=True,
        ),
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
    #memory { height: 1; margin: 0 2; }
    #memory-label { width: auto; padding-right: 2; color: #91a5be; }
    #memory-bar { width: 1fr; }
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
        self._startup_lock = mp.get_context('spawn').RLock()
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
            for name, caption in (
                ('elapsed', 'ELAPSED'),
                ('completed', 'STAGES COMPLETE'),
                ('throughput', 'THROUGHPUT'),
            ):
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
        yield from self.compose_work_metrics()
        with Horizontal(id='memory'):
            yield Static('HEXRD RAM', id='memory-label', markup=False)
            yield ProgressBar(id='memory-bar', total=100, show_eta=False)
        with Horizontal(id='data'):
            yield DataTable(id='reflections', cursor_type='none')
            yield RichLog(id='events', max_lines=500, wrap=True, min_width=1)
        yield Static(
            'Select a stage to inspect it · f returns to live progress',
            id='status',
            markup=False,
        )
        with Horizontal(id='controls'):
            yield Button('Follow stage', id='follow')
            yield Button('Pause log', id='pause_log')
            yield Button('Hide log', id='logs')
            yield Button('Cancel run', id='cancel', variant='error')
        yield Footer()

    def compose_work_metrics(self):
        return ()

    def on_mount(self):
        self.query_one('#workflow').border_title = 'WORKFLOW'
        self.query_one('#inspection').border_title = 'LIVE PROGRESS'
        self.query_one('#events').border_title = 'EVENT LOG'
        self.setup_data_table()
        table = self.query_one('#stages', DataTable)
        for column in ('Stage', 'State', 'Elapsed'):
            table.add_column(column, key=column, width=9 if column == 'State' else None)
        for name in self.dashboard.stages:
            table.add_row(name, 'Waiting', '—', key=name)
        self.timer = self.set_interval(0.1, self.refresh_progress)
        self.query_one('#memory').tooltip = (
            'Resident RAM used by HEXRD and its child processes / total system RAM. '
            'Shared pages are apportioned where supported; otherwise RSS is summed.'
        )
        self.refresh_memory()
        self.set_interval(2, self.refresh_memory)
        # Own the workflow thread explicitly. It is cooperatively cancelled
        # through Dashboard, and must not outlive an asyncio executor wrapper
        # that can still be shutting down after the terminal has closed.
        self.workflow_thread = threading.Thread(
            target=copy_context().run,
            args=(self.run_workflow,),
            name='HEXRD workflow',
            daemon=True,
        )
        self.workflow_thread.start()

    def on_unmount(self):
        if not self.finished.is_set():
            self.dashboard.cancelled.set()
        else:
            self.workflow_thread.join()

    def setup_data_table(self):
        reflections = self.query_one('#reflections', DataTable)
        reflections.border_title = 'REFLECTIONS'
        reflections.border_subtitle = 'Peaks: — not measured'
        reflections.tooltip = (
            'Miller indices (h k l). All listed rings are scored. Checked seed '
            'rings also generate trial orientations. Peaks are counted during '
            'seed peak finding and are unavailable when cached fibers are reused.'
        )
        for column in ('h', 'k', 'l', 'Seed', 'Peaks'):
            reflections.add_column(column, key=column)

    def refresh_memory(self):
        used, total = _memory_usage()
        self.query_one('#memory-label', Static).update(
            f'HEXRD RAM  {used / 1024**3:.2f} / {total / 1024**3:.1f} GiB'
        )
        self.query_one('#memory-bar', ProgressBar).update(total=total, progress=used)

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
            stages = {
                name: replace(item) for name, item in self.dashboard.stages.items()
            }
            events = list(self.dashboard.events)
            self.dashboard.events.clear()
            diagnostics = dict(self.dashboard.diagnostics)
            hkls = dict(self.dashboard.hkls)
            peaks = dict(self.dashboard.peaks)
        self.refresh_reflections(hkls, peaks)
        now = self.ended or monotonic()
        active = next(
            (name for name, item in stages.items() if item.status == 'running'), None
        )
        completed = sum(item.status == 'done' for item in stages.values())
        self.query_one('#elapsed', Static).update(
            f'{now - self.dashboard.started:,.1f} s'
        )
        self.query_one('#completed', Static).update(f'{completed} / {len(stages)}')
        table = self.query_one('#stages', DataTable)
        for name, item in stages.items():
            if name not in table.rows:
                table.add_row(name, 'Waiting', '—', key=name)
            color = dict(pending='dim', running='cyan', done='green', failed='red')[
                item.status
            ]
            state = dict(
                pending='Waiting', running='● Running', done='✓ Done', failed='✗ Failed'
            )[item.status]
            elapsed = now - item.started if item.status == 'running' else item.elapsed
            table.update_cell(name, 'State', Text(state, style=color))
            table.update_cell(
                name, 'Elapsed', f'{elapsed:.1f}s' if item.started else '—'
            )
        latest = next((name for name in reversed(stages) if stages[name].started), None)
        name = self.selected or active or latest or next(iter(stages))
        if self.selected is None and not table.has_focus:
            table.move_cursor(row=list(stages).index(name))
        item = stages[name]
        elapsed = now - item.started if item.status == 'running' else item.elapsed
        summary = (
            self.finished.is_set() and self.failure is None and self.selected is None
        )
        detail = item.detail or item.status.capitalize()
        if summary:
            detail = '\n'.join(f'{key}: {value}' for key, value in diagnostics.items())
            detail = detail or self.dashboard.message or 'Workflow complete'
        self.query_one('#stage-name', Static).update(
            self.SUMMARY_TITLE if summary else name
        )
        self.query_one('#detail', Static).update(detail)
        counts = self.query_one('#counts', Static)
        counts.display = not summary
        counts.update(
            f'{item.count:,} / {item.total:,} {item.unit}' if item.total else '—'
        )
        bar = self.query_one('#progress', ProgressBar)
        bar.display = not summary and not (
            self.finished.is_set() and not item.total and item.status != 'done'
        )
        bar.update(
            total=item.total or (1 if item.status == 'done' else None),
            progress=item.count if item.total else int(item.status == 'done'),
        )
        rate = item.count / elapsed if elapsed and item.count else 0
        self.query_one('#throughput-caption', Label).update(
            self.COMPLETION_CAPTION if self.finished.is_set() else 'THROUGHPUT'
        )
        self.query_one('#throughput', Static).update(
            diagnostics.get(self.COMPLETION_KEY, '—')
            if self.finished.is_set()
            else f'{rate:,.0f} {item.unit}/s' if rate else '—'
        )
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
                    log.write(
                        Traceback.from_exception(
                            type(self.failure), self.failure, self.failure.__traceback__
                        ),
                        scroll_end=True,
                    )
            else:
                message = self.dashboard.message or 'Workflow complete'
            status.update(f'{message} · q to close')
            button = self.query_one('#cancel', Button)
            button.label, button.variant, button.disabled = 'Close', 'primary', False

    def refresh_reflections(self, hkls, peaks):
        rings = tuple(dict.fromkeys(hkl for group in hkls.values() for hkl in group))
        seeds = set(hkls.get('Seed', ()))
        rows = {
            hkl: (
                *(str(index) for index in hkl),
                '✓' if hkl in seeds else '—',
                f'{peaks[hkl]:,}' if hkl in peaks else '—',
            )
            for hkl in rings
        }
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
        table.border_title = (
            f'REFLECTIONS · {len(rings)} rings' if hkls else 'REFLECTIONS'
        )
        table.border_subtitle = (
            'Peaks: — not measured' if hkls else 'Awaiting ring selection'
        )
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
        self.query_one('#logs', Button).label = (
            'Hide log' if log.display else 'Show log'
        )

    def action_pause_log(self):
        log = self.query_one(RichLog)
        log.auto_scroll = not log.auto_scroll
        self.query_one('#pause_log', Button).label = (
            'Pause log' if log.auto_scroll else 'Follow log'
        )
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
        self.query_one('#status', Static).update(
            'Cancelling · waiting for current work to finish…'
        )


class OrientationDashboard(WorkflowDashboard):
    """Find-orientations presentation of the shared workflow dashboard."""


class FitGrainsDashboard(WorkflowDashboard):
    """Grain fitting uses the same navigation, log, memory and cancellation UI."""

    SUB_TITLE = 'FIT GRAINS'
    SUMMARY_TITLE = 'FITTING SUMMARY'
    COMPLETION_CAPTION = 'GRAINS FITTED'
    COMPLETION_KEY = 'Grains finished'
    # Keep the shared palette and navigation while exposing fit-specific work.
    CSS = WorkflowDashboard.CSS + """
    #reflections { width: 3fr; }
    #events { width: 2fr; }
    #fit-metrics { height: 9; margin: 0 1; border: round #24425a; padding: 0 1; }
    .fit-metric { width: 1fr; padding: 0 1; }
    .fit-metric Static { height: 1; color: #91a5be; }
    .fit-metric ProgressBar { height: 1; }
    .fit-metric .fit-title { color: #6de0d2; text-style: bold; }
    #fit-reads, #fit-cache { height: 2; }
    .fit-short #panels { height: 8; }
    .fit-short #fit-metrics { height: 8; }
    .fit-short #fit-read-time { display: none; }
    """

    def compose_work_metrics(self):
        with Horizontal(id='fit-metrics'):
            with Vertical(classes='fit-metric'):
                yield Static('PASSES & SOLVER', classes='fit-title')
                yield Static('Passes: waiting', id='fit-passes-label', markup=False)
                yield ProgressBar(id='fit-passes', show_eta=False)
                yield Static('Optimizer evaluations: 0', id='fit-solver', markup=False)
                yield Static('Workers: waiting', id='fit-workers', markup=False)
            with Vertical(classes='fit-metric'):
                yield Static('EXTRACTION', classes='fit-title')
                yield Static('Reflections: waiting', id='fit-spots-label', markup=False)
                yield ProgressBar(id='fit-spots', show_eta=False)
                yield Static(
                    'Frame batches: waiting', id='fit-frames-label', markup=False
                )
                yield ProgressBar(id='fit-frames', show_eta=False)
                yield Static('Total tested: 0', id='fit-tested', markup=False)
            with Vertical(classes='fit-metric'):
                yield Static('IMAGE READS & CACHE', classes='fit-title')
                yield Static('Reads: waiting', id='fit-reads', markup=False)
                yield Static('Cache: waiting', id='fit-cache', markup=False)
                yield Static('Read/decode: waiting', id='fit-read-time', markup=False)
                yield Static(
                    'Latest activity: waiting', id='fit-activity', markup=False
                )

    def on_resize(self, event):
        super().on_resize(event)
        self.screen.set_class(event.size.height < 40, 'fit-short')

    def setup_data_table(self):
        table = self.query_one('#reflections', DataTable)
        table.border_title = 'GRAINS · RECENT ACTIVITY'
        table.border_subtitle = 'Waiting for fitting'
        table.tooltip = (
            'Latest 32 grains. Work shows frame sampling within a patch batch, '
            'then reflections processed in the current pass. Age is time since '
            'the worker last reported activity, not an estimate of time remaining.'
        )
        for column in (
            'Grain',
            'State',
            'Pass',
            'Completeness',
            'Chi-square',
            'Work',
            'Age',
        ):
            table.add_column(column, key=column)
        self.query_one('#fit-metrics').border_title = 'FIT ACTIVITY'
        self.query_one('#fit-spots-label').tooltip = (
            'Progress within workers’ current passes. These totals reset for the '
            'next pass. Total tested counts reflections across all passes.'
        )
        self.query_one('#fit-frames-label').tooltip = (
            'Source-frame groups sampled within current lazy extraction batches. '
            'The same source frame may be read again after eviction or by another worker.'
        )
        self.query_one('#fit-reads').tooltip = (
            'Cache misses and decoded image bytes, not physical disk bytes or disk bandwidth.'
        )
        self.query_one('#fit-cache').tooltip = (
            'Last reported retained image buffers across workers. Decoder/fitting '
            'workspace is additional; the HEXRD RAM bar shows process memory.'
        )
        self.query_one('#fit-activity').tooltip = (
            'Last: time since any worker reported activity. Oldest: longest '
            'interval without an update among active workers. These are not ETAs.'
        )
        self._fit_history = deque()
        self._fit_generation = 0.0

    def refresh_reflections(self, hkls, peaks):
        with self.dashboard.lock:
            rows = tuple(reversed(tuple(self.dashboard.grains.values())))
            grains = dict(self.dashboard.fit_grain_metrics)
            workers = list(self.dashboard.fit_workers.values())
            planned = self.dashboard.fit_passes_planned
            generation = self.dashboard.fit_started
        now = self.ended or monotonic()
        if generation != self._fit_generation:
            self._fit_history.clear()
            self._fit_generation = generation
        # Update existing cells instead of rebuilding the table on every heartbeat.
        table = self.query_one('#reflections', DataTable)
        desired = {str(row[0]) for row in rows}
        for key in list(table.rows):
            if str(key.value) not in desired:
                table.remove_row(key)
        for grain, state, index, passes, completeness, chisq in rows:
            snapshot = grains.get(grain, {})
            terminal = state in ('Done', 'Insufficient reflections', 'Failed')
            phase = state if terminal else snapshot.get('phase', state)
            if snapshot.get('state') == 'Extracting' and snapshot.get('batch_total'):
                work = f"frames {snapshot['batch_done']:,}/{snapshot['batch_total']:,}"
            elif snapshot.get('pass_total'):
                work = f"spots {snapshot['pass_done']:,}/{snapshot['pass_total']:,}"
            else:
                work = '—'
            age = (
                '—'
                if terminal or not snapshot
                else f"{max(0., now-snapshot['updated']):.1f}s"
            )
            values = [
                str(grain),
                phase,
                f'{index}/{passes}' if index else '—',
                f'{completeness:.1%}' if completeness is not None else '—',
                f'{chisq:.5g}' if chisq is not None else '—',
                work,
                age,
            ]
            if str(grain) not in table.rows:
                table.add_row(*values, key=str(grain))
            else:
                for column, value in zip(table.columns, values):
                    if table.get_cell(str(grain), column) != value:
                        table.update_cell(str(grain), column, value, update_width=True)
        table.border_subtitle = (
            'Age = time since worker activity' if rows else 'Waiting for fitting'
        )
        self._refresh_fit_metrics(workers, planned, now)

    def _refresh_fit_metrics(self, workers, planned, now):
        def total(name):
            return sum(worker.get(name, 0) for worker in workers)

        done, skipped = total('passes_done'), total('passes_skipped')
        self.query_one('#fit-passes-label', Static).update(
            f'Passes: {done:,} / {planned:,}'
            + (f' · skipped: {skipped:,}' if skipped else '')
        )
        self.query_one('#fit-passes', ProgressBar).update(
            total=planned or None, progress=done + skipped
        )
        self.query_one('#fit-solver', Static).update(
            f"Solver evals: {total('evaluations'):,} · refits: {total('refits_done'):,}"
        )
        terminal = ('Done', 'Insufficient reflections', 'Failed')
        active = [w for w in workers if w['state'] not in terminal]
        extracting = sum(w['state'] == 'Extracting' for w in active)
        fitting = sum(w['state'] in ('Fitting', 'Refitting') for w in active)
        self.query_one('#fit-workers', Static).update(
            f'Workers: {extracting} extract / {fitting} fit'
        )
        spots, expected = total('pass_done'), total('pass_total')
        self.query_one('#fit-spots-label', Static).update(
            f'Reflections: {spots:,} / {expected:,}'
        )
        self.query_one('#fit-spots', ProgressBar).update(
            total=expected or None, progress=spots
        )
        lazy = any(w['lazy'] for w in workers)
        self.query_one('#fit-frames', ProgressBar).display = lazy
        sampled, frames = total('batch_done'), total('batch_total')
        self.query_one('#fit-frames-label', Static).update(
            f'Frame batches: {sampled:,} / {frames:,}'
            if lazy
            else 'Frames: eager/shared images'
        )
        self.query_one('#fit-frames', ProgressBar).update(
            total=frames or None, progress=sampled
        )
        self.query_one('#fit-tested', Static).update(
            f"Tested: {total('reflections'):,} · signal: {total('signals'):,}"
        )
        reads, decoded = total('frame_reads'), total('decoded_bytes')
        self._fit_history.append((now, reads, decoded))
        while len(self._fit_history) > 1 and now - self._fit_history[0][0] > 2:
            self._fit_history.popleft()
        first, old_reads, old_decoded = self._fit_history[0]
        elapsed = now - first
        rate = (reads - old_reads) / elapsed if elapsed else 0.0
        mib_rate = (decoded - old_decoded) / elapsed / 1024**2 if elapsed else 0.0
        lazy = any(w['lazy'] for w in workers)
        decoded_text = (
            f'{decoded/1024**3:.2f} GiB'
            if decoded >= 1024**3
            else f'{decoded/1024**2:.1f} MiB'
        )
        cache_unit = 'GiB' if total('cache_limit') >= 1024**3 else 'MiB'
        cache_scale = 1024**3 if cache_unit == 'GiB' else 1024**2
        self.query_one('#fit-reads', Static).update(
            f'Reads: {reads:,} · {rate:.1f} frames/s\nDecoded: {decoded_text} · {mib_rate:.1f} MiB/s'
            if lazy
            else 'Images: eager/shared (no lazy reads)'
        )
        hits = total('frame_hits')
        misses = total('cache_misses')
        hit_rate = hits / (hits + misses) if hits + misses else 0.0
        self.query_one('#fit-cache', Static).update(
            f"Cache: {total('cache_bytes')/cache_scale:.2f} / {total('cache_limit')/cache_scale:.2f} {cache_unit}\n"
            f"Hits: {hit_rate:.1%} · evictions: {total('evictions'):,}"
            if lazy
            else 'Cache: eager/shared'
        )
        self.query_one('#fit-read-time', Static).update(
            f"Read/decode time: {total('read_seconds'):.1f}s (sum)"
        )
        if not workers:
            self.query_one('#fit-reads', Static).update('Reads: waiting for workers')
            self.query_one('#fit-cache', Static).update('Cache: waiting for workers')
        newest = max((w['updated'] for w in workers), default=None)
        oldest = max((now - w['updated'] for w in active), default=0.0)
        self.query_one('#fit-activity', Static).update(
            f'Last: {max(0., now-newest):.1f}s ago · oldest: {max(0., oldest):.1f}s'
            if newest is not None
            else 'Latest activity: waiting for workers'
        )
