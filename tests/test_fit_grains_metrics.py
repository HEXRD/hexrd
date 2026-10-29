"""Live fitting progress must move before any whole grain finishes."""

import asyncio
from queue import Queue
import threading
from time import monotonic

import numpy as np
import pytest
from scipy.sparse import csr_array

from hexrd.hedm._fit_progress import FitReporter, current_reporter, set_reporter
from hexrd.hedm._progress import (
    Dashboard,
    FIT_STAGES,
    report_fit_metrics,
    report_grain,
    stage,
    progress_bar,
)


@pytest.fixture(autouse=True)
def reset_reporter():
    set_reporter(None)
    yield
    set_reporter(None)


def test_frame_progress_is_reported_before_extraction_completes():
    from hexrd.hedm._image_cache import FrameCache, StreamingView, View

    queue = Queue()
    reporter = FitReporter(queue, lazy=True)
    reporter.interval = 0
    set_reporter(reporter)
    reporter.grain(7, 'Extracting', 1, 3)
    reporter.panel(1)

    class Reader:
        shape, dtype, nframes, resident = (3, 4), np.dtype('uint16'), 8, False

        def read(self, index, cache):
            assert current_reporter().data['frame_reads'] == index + 1
            assert current_reporter().data['batch_done'] == index
            return csr_array(np.full(self.shape, index + 1, dtype=self.dtype))

    cache = FrameCache(0)
    view = StreamingView(View(Reader()), cache)
    result = view.gather(
        np.arange(8).reshape(1, 8), np.array([0, 1]), np.array([1]), np.array([2])
    )
    np.testing.assert_array_equal(result, np.arange(1, 9))
    snapshots = []
    while not queue.empty():
        snapshots.append(queue.get_nowait())
    assert any(s['batch_done'] == 4 and s['frame_reads'] == 4 for s in snapshots)
    assert snapshots[-1]['batch_done'] == snapshots[-1]['batch_total'] == 8
    assert snapshots[-1]['reflections'] == 0
    assert snapshots[-1]['decoded_bytes'] > 0
    assert snapshots[-1]['cache_bytes'] == 0
    assert snapshots[-1]['read_seconds'] >= 0


def test_dropped_updates_recover_cumulative_counts_without_blocking():
    queue = Queue(maxsize=1)
    reporter = FitReporter(queue, lazy=True)
    reporter.interval = 0
    reporter.grain(7, 'Extracting', 1, 3)
    for _ in range(40):
        reporter.reflections(3, 2)
        reporter.evaluation()
    reporter.pass_complete(skipped=2)
    assert queue.qsize() == 1
    queue.get_nowait()
    reporter.publish(force=True)
    recovered = queue.get_nowait()
    assert recovered['reflections'] == 120
    assert recovered['signals'] == 80
    assert recovered['evaluations'] == 40
    assert recovered['passes_done'] == 1
    assert recovered['passes_skipped'] == 2


def test_parent_retains_bounded_grain_details_and_ignores_stale_updates():
    dashboard = Dashboard(FIT_STAGES)
    dashboard.start()
    try:
        reporter = FitReporter()
        for grain in range(100):
            reporter.grain(grain, 'Done', 3, 3)
        assert len(dashboard.fit_grain_metrics) == 32
        assert len(dashboard.fit_workers) == 1
        latest = dict(reporter.data)
        stale = {**latest, 'sequence': latest['sequence'] - 1, 'grain': -1}
        report_fit_metrics(stale)
        assert dashboard.fit_workers[latest['worker']]['grain'] == 99
        assert -1 not in dashboard.fit_grain_metrics
    finally:
        dashboard.close()


@pytest.mark.parametrize('size', [(120, 38), (190, 55)])
def test_dashboard_shows_grain_progress_without_fit_activity_panel(size, tmp_path):
    from hexrd.hedm._dashboard import FitGrainsDashboard
    from textual.widgets import DataTable, Button

    dashboard = Dashboard(FIT_STAGES)
    release = threading.Event()
    started = threading.Event()

    def workflow():
        dashboard.start()
        try:
            dashboard.fit_passes_planned = 39
            with stage('Fit grains'), progress_bar(total=13, unit='grains'):
                reporter = FitReporter(lazy=True)
                reporter.grain(12, 'Extracting', 1, 3)
                reporter.data.update(
                    pass_total=500,
                    pass_done=23,
                    batch_total=200,
                    batch_done=71,
                    frame_reads=64,
                    frame_hits=36,
                    cache_misses=64,
                    decoded_bytes=512 * 1024**2,
                    cache_bytes=1024**3,
                    cache_limit=4 * 1024**3,
                    evaluations=32,
                    read_seconds=5.0,
                )
                reporter.publish(force=True)
                report_grain((12, 'Extracting', 1, 3, None, None))
                started.set()
                assert release.wait(10)
        finally:
            dashboard.close()

    app = FitGrainsDashboard(dashboard, workflow, 'metrics.yml')

    async def interact():
        async with app.run_test(size=size) as pilot:
            try:
                await pilot.pause(0.3)
                assert started.is_set() and not app.finished.is_set()
                assert dashboard.stages['Fit grains'].count == 0
                assert len(app.query('#fit-metrics')) == 0
                assert (
                    app.query_one('#reflections', DataTable).get_row('12')[5]
                    == 'frames 71/200'
                )
                assert app.query_one('#cancel', Button).region.bottom <= size[1]
                assert (
                    app.query_one('#panels').region.bottom
                    == app.query_one('#memory').region.y
                )
                app.save_screenshot(f'metrics-{size[0]}x{size[1]}.svg', str(tmp_path))
            finally:
                release.set()
            await pilot.pause(0.2)
            await pilot.press('q')

    asyncio.run(interact())
    assert app.failure is None


def test_final_result_recovers_snapshot_lost_to_a_full_progress_queue(monkeypatch):
    from hexrd.hedm import fitgrains

    dashboard = Dashboard(FIT_STAGES)
    dashboard.start()
    queue = Queue(maxsize=1)
    reporter = FitReporter(queue, lazy=True)
    reporter.interval = 0
    set_reporter(reporter)
    reporter.grain(4, 'Extracting', 1, 3)  # Fills the queue.
    expected = (4, 1.0, 0.01, np.zeros(12))

    def fit(grain):
        for _ in range(3):
            reporter.reflections(100, 80)
            reporter.pass_complete()
        reporter.grain(grain, 'Done', 3, 3)  # Last update is dropped too.
        return expected

    monkeypatch.setattr(fitgrains, 'fit_grain_FF_reduced', fit)
    try:
        result, snapshot = fitgrains._fit_grain_with_metrics(4)
        assert result is expected
        old = queue.get_nowait()
        assert old['passes_done'] == 0
        report_fit_metrics(snapshot)
        final = dashboard.fit_workers[snapshot['worker']]
        assert final['passes_done'] == 3 and final['reflections'] == 300
        assert final['state'] == 'Done'
    finally:
        dashboard.close()


def test_processing_does_not_double_count_source_reads():
    from hexrd.hedm._image_cache import FrameCache, ProcessedReader, View, frame_bytes

    queue = Queue()
    reporter = FitReporter(queue, lazy=True)
    set_reporter(reporter)

    class Source:
        shape, dtype, nframes, resident = (3, 4), np.dtype('uint16'), 1, False

        def read(self, index, cache):
            return np.ones(self.shape, dtype=self.dtype)

    processed = ProcessedReader(
        View(Source(), dense=True), [('add', 1)], {}, 1, (3, 4), np.dtype('float32')
    )
    cache = FrameCache(100000)
    np.testing.assert_array_equal(cache.frame(processed, 0), 2)
    assert reporter.data['frame_reads'] == 1
    assert reporter.data['cache_misses'] == 2
    assert reporter.data['decoded_bytes'] == frame_bytes(
        np.ones((3, 4), dtype='uint16')
    )
    cache.frame(processed, 0)
    assert reporter.data['frame_reads'] == 1 and reporter.data['frame_hits'] == 1
    cache.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_batch_progress_does_not_enable_detailed_telemetry_unless_requested(enabled):
    from hexrd.hedm import fitgrains

    queue = Queue()
    fitgrains.fit_grain_FF_init({'progress_queue': queue, 'collect_metrics': enabled})
    try:
        assert (current_reporter() is not None) is enabled
        fitgrains._grain_event(1, 'Extracting', 1, 3)
        event = queue.get_nowait()
        assert isinstance(event, dict) if enabled else isinstance(event, tuple)
    finally:
        fitgrains.fit_grain_FF_cleanup()
