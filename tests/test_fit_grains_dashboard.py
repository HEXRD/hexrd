import argparse
import asyncio
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.hedm._progress import (
    Dashboard,
    FIT_STAGES,
    progress_bar,
    report_grain,
    stage,
)


@pytest.mark.parametrize(
    'quiet,no_tui,terminal,expected',
    [
        (False, False, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (False, False, False, False),
    ],
)
def test_cli_dashboard_selection(monkeypatch, quiet, no_tui, terminal, expected):
    from hexrd.hedm.cli import fit_grains as cli
    from hexrd.hedm import _dashboard as ui

    seen = []
    monkeypatch.setattr(cli.sys, 'stdin', SimpleNamespace(isatty=lambda: terminal))
    monkeypatch.setattr(Dashboard, 'supported', lambda stream: terminal)
    monkeypatch.setattr(cli, '_execute', lambda *args: seen.append('workflow'))

    class App:
        failure = None

        def __init__(self, dashboard, workflow, config):
            assert tuple(dashboard.stages) == FIT_STAGES
            self.workflow = workflow

        def run(self):
            seen.append('dashboard')
            self.workflow()

    monkeypatch.setattr(ui, 'FitGrainsDashboard', App)
    args = SimpleNamespace(quiet=quiet, no_tui=no_tui, yml='config.yml')
    cli.execute(args, None)
    assert seen == (['dashboard', 'workflow'] if expected else ['workflow'])


def test_cli_parser_has_batch_escape_hatch():
    from hexrd.hedm.cli.fit_grains import configure_parser

    parser = argparse.ArgumentParser()
    configure_parser(parser.add_subparsers())
    assert parser.parse_args(['fit-grains', 'config.yml', '--no-tui']).no_tui


def test_cli_failure_restores_logging_and_progress(monkeypatch):
    from hexrd.hedm.cli import fit_grains as cli
    from hexrd.hedm._progress import current_dashboard

    log = logging.getLogger('hexrd')
    initial = log.level, log.propagate, list(log.handlers)
    monkeypatch.setattr(
        cli, '_run', lambda *args: (_ for _ in ()).throw(ValueError('bad config'))
    )
    dashboard = Dashboard(FIT_STAGES)
    args = SimpleNamespace(quiet=False, no_tui=False, debug=False)
    with pytest.raises(ValueError, match='bad config'):
        cli._execute(args, None, dashboard)
    assert (log.level, log.propagate, log.handlers) == initial
    assert current_dashboard() is None


def test_grain_events_are_bounded_and_counted_once():
    dashboard = Dashboard(FIT_STAGES)
    dashboard.start()
    try:
        for i in range(80):
            report_grain((i, 'Done', 3, 3, 1.0, 0.0))
            report_grain((i, 'Done', 3, 3, 1.0, 0.0))
        assert len(dashboard.grains) == 32
        assert dashboard.diagnostics['Grains finished'] == '80'
        with stage('Fit grains'), progress_bar(total=80, unit='grains') as bar:
            bar.update(80)
        assert dashboard.stages['Fit grains'].count == 80
    finally:
        dashboard.close()


def test_fit_dashboard_completion():
    from hexrd.hedm._dashboard import FitGrainsDashboard
    from textual.widgets import DataTable, Static

    dashboard = Dashboard(FIT_STAGES)

    def workflow():
        dashboard.start()
        try:
            with stage('Fit grains'), progress_bar(total=2, unit='grains') as bar:
                report_grain((7, 'Extracting', 1, 3, None, None))
                report_grain((7, 'Done', 3, 3, 0.95, 0.002))
                report_grain((9, 'Insufficient reflections', 1, 3, 0.1, np.inf))
                bar.update(2)
        finally:
            dashboard.close()

    app = FitGrainsDashboard(dashboard, workflow, 'fit.yml')

    async def interact():
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.3)
            assert app.failure is None and app.finished.is_set()
            table = app.query_one('#reflections', DataTable)
            assert table.get_row('7')[:5] == ['7', 'Done', '3/3', '95.0%', '0.002']
            assert table.get_row('9')[1] == 'Insufficient reflections'
            assert 'FITTING SUMMARY' in str(
                app.query_one('#stage-name', Static).render()
            )
            assert str(app.query_one('#throughput', Static).render()) == '2'
            await pilot.press('q')

    asyncio.run(interact())


def test_fit_dashboard_cancellation():
    from hexrd.hedm._dashboard import FitGrainsDashboard
    from textual.widgets import Button

    dashboard = Dashboard(FIT_STAGES)
    started = threading.Event()

    def workflow():
        dashboard.start()
        try:
            with stage('Fit grains'), progress_bar(total=2, unit='grains'):
                report_grain((4, 'Extracting', 1, 3, None, None))
                started.set()
                dashboard.cancelled.wait(5)
                dashboard.check_cancelled()
        finally:
            dashboard.close()

    app = FitGrainsDashboard(dashboard, workflow, 'fit.yml')

    async def interact():
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.2)
            assert started.is_set()
            await pilot.press('q')
            await pilot.pause(0.2)
            assert isinstance(app.failure, KeyboardInterrupt)
            assert app.sub_title == 'CANCELLED'
            assert str(app.query_one('#cancel', Button).label) == 'Close'
            await pilot.press('q')

    asyncio.run(interact())
    assert not app.workflow_thread.is_alive()


def test_dashboard_cancel_terminates_fit_pool(tmp_path, monkeypatch):
    from hexrd.hedm import fitgrains
    from hexrd.hedm.xrdutil import utils

    dashboard = Dashboard(FIT_STAGES)
    events = []
    future = SimpleNamespace(
        ready=lambda: False,
        wait=lambda _: dashboard.cancelled.set(),
    )

    def make_pool(*args):
        assert args[2][0]['progress_queue'] is not None
        return SimpleNamespace(
            map_async=lambda *a, **kw: future,
            terminate=lambda: events.append('terminate'),
            join=lambda: events.append('join'),
        )

    # Use a real context queue, but deterministic pool completion/cancellation.
    monkeypatch.setattr(fitgrains.constants.mp_context, 'Pool', make_pool)
    monkeypatch.setattr(fitgrains, 'sparse_frame_view', lambda _: object())
    monkeypatch.setattr(utils, '_expand_pixels', lambda *args: None)
    monkeypatch.setattr(utils, '_compute_max', lambda *args: None)
    experiment = SimpleNamespace(
        image_series={'d': SimpleNamespace(omega=np.array([[0.0, 1.0]]))},
        instrument=SimpleNamespace(detectors={}),
        max_workers=2,
        analysis_dir=tmp_path,
        find_orientations=SimpleNamespace(eta=SimpleNamespace(range=[[-np.pi, np.pi]])),
        fit_grains=SimpleNamespace(
            threshold=75,
            npdiv=2,
            refit=None,
            tolerance=SimpleNamespace(tth=[0.1], eta=[1.0], omega=[1.0]),
        ),
        get_active_material=lambda: SimpleNamespace(
            _material=SimpleNamespace(planeData=None)
        ),
    )
    grains = np.zeros((2, 21))
    grains[:, 0] = [0, 1]
    dashboard.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            fitgrains.fit_grains(experiment, grains, write_spots_files=False)
    finally:
        dashboard.close()
    assert events == ['terminate', 'join']
    assert dashboard.stages['Fit grains'].status == 'failed'
