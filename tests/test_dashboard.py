"""Dashboard startup must work before multiprocessing has been initialized."""

import subprocess
import sys

import pytest


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
