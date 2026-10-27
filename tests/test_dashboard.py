"""Dashboard startup must work before multiprocessing has been initialized."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_memory_usage_counts_only_hexrd_process_tree(monkeypatch):
    from hexrd.hedm import _dashboard as ui

    process, child, exited = Mock(), Mock(), Mock()
    process.children.return_value = [child, exited]
    process.memory_full_info.return_value = SimpleNamespace(pss=100, rss=200)
    child.memory_full_info.side_effect = ui.psutil.AccessDenied()
    child.memory_info.return_value = SimpleNamespace(rss=50)
    exited.memory_full_info.side_effect = ui.psutil.NoSuchProcess(123)
    monkeypatch.setattr(ui.psutil, 'Process', lambda: process)
    monkeypatch.setattr(ui.psutil, 'virtual_memory', lambda: SimpleNamespace(total=1000))
    assert ui._memory_usage() == (150, 1000)
    process.children.assert_called_once_with(recursive=True)


@pytest.mark.parametrize('start_method', ['default', 'spawn'])
def test_first_dashboard_run_initializes_multiprocessing(start_method):
    # An existing resource tracker hides the startup bug, so each case must
    # use a fresh interpreter and exercise Textual's real stderr capture.
    script = '''
import asyncio
import multiprocessing as mp
import sys

if sys.argv[1] != 'default':
    mp.set_start_method(sys.argv[1])

from hexrd.hedm._dashboard import OrientationDashboard
from hexrd.hedm._progress import Dashboard, stage

assert 'numba' not in sys.modules
dashboard = Dashboard()

def workflow():
    dashboard.start()
    try:
        with stage('Material'):
            assert sys.stderr.fileno() == -1, 'Textual must capture stderr'
            import numba
            numba.set_num_threads(1)
            assert numba.get_num_threads() == 1
            # Workflow locks must also work when Numba does not create its
            # own multiprocessing initialization lock (e.g. with spawn).
            with mp.get_context().RLock():
                pass
    finally:
        dashboard.close()

app = OrientationDashboard(dashboard, workflow, 'test.yml')

async def interact():
    async with app.run_test() as pilot:
        while not app.finished.is_set():
            await asyncio.sleep(0.05)
        await pilot.pause(0.2)
        await pilot.press('q')

asyncio.run(interact())
assert app.finished.is_set(), 'Workflow must run on the first launch'
if app.failure is not None:
    raise app.failure
assert dashboard.stages['Material'].status == 'done'
'''
    result = subprocess.run(
        [sys.executable, '-c', script, start_method],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
