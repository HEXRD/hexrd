"""Frame-major segmentation must reproduce each original angular window."""

from collections import Counter

import numpy as np
import pytest
from scipy import sparse

from hexrd.core.instrument._spotutils import clear_sparse_frame_views, sparse_frame_view
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.hedm._image_cache import BoundedSeries, FrameCache, View
from hexrd.hedm._staged_spots import MeasurementPlan, MeasurementGroup


@pytest.mark.parametrize('dtype', ['uint16', 'int32', 'float32', 'float64'])
@pytest.mark.parametrize('as_sparse', [False, True])
@pytest.mark.parametrize('physical_order', [False, True])
def test_staging_is_exact_and_reads_each_shared_frame_once(
    dtype, as_sparse, physical_order
):
    rng = np.random.default_rng(408)
    data = rng.integers(0, 100, (11, 17, 21)).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        data[:, 4, 6] = -0.0
        data[:, 5, 5] = np.nan
    data[data < 40] = 0
    calls = []

    class Reader:
        shape, nframes, resident = (17, 21), 11, False

        def read(self, index, cache):
            calls.append(index)
            return sparse.csr_array(data[index]) if as_sparse else data[index]

    reader = Reader()
    if physical_order:
        reader.read_positions = lambda frames, cache: (10 - frames)
    reader.dtype = np.dtype(dtype)
    cache = FrameCache(0)
    view = View(reader, roi=(2, 3, 12, 14), mapping=np.arange(11)[::-1], dense=True)
    series = OmegaImageSeries(
        BoundedSeries(
            view,
            cache,
            {'omega': np.column_stack((np.arange(11), np.arange(1, 12)))},
            11,
        )
    )
    frames = np.array([[0, 10, 1], [1, 1, 9]])
    shapes = np.array([[3, 5, 7], [3, 9, 11]])
    offsets = np.r_[0, np.cumsum(np.prod(shapes[:, 1:], axis=1))]
    rows = rng.integers(-3, 16, offsets[-1])
    cols = rng.integers(-3, 18, offsets[-1])

    class Instrument:
        def pull_spots(self, threshold):
            return sparse_frame_view(series).extract(
                frames, offsets, rows, cols, shapes, threshold
            )

    clear_sparse_frame_views()
    instrument = Instrument()
    try:
        expected = [instrument.pull_spots(x) for x in (75, 90)]
        calls.clear()
        plans = [MeasurementPlan(instrument, ((x,), {})).prepare() for x in (75, 90)]
        assert not calls, 'planning read image pixels'
        group = MeasurementGroup(16 * 1024**2)
        assert all(group.add(p) for p in plans)
        group.measure(cache)
        assert Counter(calls) == Counter({0: 1, 1: 1, 9: 1, 10: 1})
        assert calls == ([10, 9, 1, 0] if physical_order else [0, 1, 9, 10])
        before = list(calls)
        for plan, reference in zip(plans, expected):
            for actual, wanted in zip(plan.result(), reference):
                np.testing.assert_array_equal(actual, wanted, strict=True)
        assert calls == before, 'measurement replay read image pixels'
        assert cache.bytes == 0 and cache.max_bytes <= group.max_bytes // 4
        assert all(p.pixels is None for p in group.patches)
    finally:
        cache.close()
        clear_sparse_frame_views()


def test_planning_without_native_windows_uses_existing_result():
    result = ([], {})

    class Instrument:
        def pull_spots(self):
            return result

    plan = MeasurementPlan(Instrument(), ((), {})).prepare()
    assert plan.result() is result
    group = MeasurementGroup(1)
    assert group.add(plan)


def test_tiled_dense_sampling_borrows_one_frame_larger_than_the_cache():
    from hexrd.core.extensions import transforms as native

    line = (1 + np.arange(80000) % 17).astype('f4')
    calls = []

    class Reader:
        shape, nframes, dtype, resident = (100, 80000), 1, line.dtype, False

        def read(self, index, cache):
            calls.append(index)
            return np.broadcast_to(line, self.shape)

    cache = FrameCache(0)
    series = BoundedSeries(View(Reader(), dense=True), cache, {}, 1)

    class Instrument:
        def pull_spots(self):
            return sparse_frame_view(series).extract(
                np.array([[0], [0]]),
                np.array([0, 40000, 80000]),
                np.zeros(80000, dtype='i4'),
                np.tile(np.arange(40000), 2),
                np.array([[1, 1, 40000], [1, 1, 40000]]),
                0,
            )

    clear_sparse_frame_views()
    try:
        plan = MeasurementPlan(Instrument(), ((), {})).prepare()
        group = MeasurementGroup(8 * 1024**2)
        assert group.add(plan)
        group.measure(cache)
        assert calls == [0]
        expected = native._spotStats(
            line[:40000],
            np.array([0, 40000], dtype='i8'),
            np.array([[1, 1, 40000]], dtype='i8'),
            0,
        )
        for actual, wanted in zip(plan.batches[0].stats, expected):
            np.testing.assert_array_equal(
                actual, np.repeat(wanted, 2, axis=0), strict=True
            )
    finally:
        cache.close()
        clear_sparse_frame_views()


def test_source_coordinates_are_prepared_once_even_after_rejected_group():
    from hexrd.hedm._staged_spots import patches

    class Reader:
        shape, nframes, dtype, resident = (80000, 80000), 1, np.dtype('u2'), False

    cache = FrameCache(0)
    series = BoundedSeries(View(Reader(), roi=(40000, 50000, 2, 3)), cache, {}, 1)
    rows, cols = np.array([-1, 0, 10]), np.array([-2, 1, 20])

    class Instrument:
        def pull_spots(self):
            return sparse_frame_view(series).extract(
                np.array([[0]]),
                np.array([0, 3]),
                rows,
                cols,
                np.array([[1, 1, 3]]),
                0,
            )

    clear_sparse_frame_views()
    try:
        plan = MeasurementPlan(Instrument(), ((), {})).prepare()
        signature = plan.batches[0].signature
        assert not MeasurementGroup(1).add(plan)
        (first,) = patches(plan)
        (second,) = patches(plan)
        np.testing.assert_array_equal(first.rows, [40000, 40000, 40001])
        np.testing.assert_array_equal(first.cols, [50000, 50001, 50002])
        assert first.rows.dtype == first.cols.dtype == np.dtype('i4')
        assert np.shares_memory(first.rows, second.rows)
        assert np.shares_memory(first.cols, second.cols)
        assert not first.rows.flags.writeable and not first.cols.flags.writeable
        assert signature == plan.batches[0].signature
        np.testing.assert_array_equal(rows, [-1, 0, 10])
    finally:
        cache.close()
        clear_sparse_frame_views()


def test_workspace_accounts_for_window_overlap_in_physical_frame_order():
    from hexrd.hedm._staged_spots import Batch
    from hexrd.hedm._image_cache import StreamingView

    class Reader:
        shape, nframes, dtype, resident = (2, 2), 4, np.dtype('f8'), False

        def read_positions(self, frames, cache):
            return np.array([0, 2, 1, 3])[frames]

    cache = FrameCache(0)
    reader = Reader()
    batch = Batch(
        StreamingView(View(reader), cache),
        np.array([[0, 1], [2, 3]]),
        np.array([0, 4, 8]),
        np.tile([0, 0, 1, 1], 2),
        np.tile([0, 1, 0, 1], 2),
        np.array([[2, 2, 2], [2, 2, 2]]),
        0,
        (),
        (np.zeros((2, 3)), np.zeros(2), np.zeros(2), np.zeros(2, bool)),
    )
    plan = MeasurementPlan(None, None)
    plan.batches = [batch]
    group = MeasurementGroup(16 * 1024**2)
    assert group.add(plan)
    # The windows are disjoint in omega order, but overlap in file order.
    assert group.peak_pixels == 2 * (2 * 2 * 2 * reader.dtype.itemsize)
    cache.close()


@pytest.mark.parametrize('packed', [False, True])
def test_captured_panel_results_do_not_repeat_geometry_and_number_bad_peaks(packed):
    from hexrd.core.instrument._spotutils import _DeferredSpotData, _SpotData

    calls = []

    def records(ids):
        n = len(ids)
        return _SpotData(
            np.array(ids),
            np.arange(n),
            np.ones((n, 3), 'i8'),
            np.ones(n),
            np.ones(n),
            np.ones((n, 3)),
            np.ones((n, 3)),
            np.ones((n, 2)),
            np.ones((n, 2)),
        )

    class Instrument:
        def pull_spots(self):
            calls.append('predict')
            first = _DeferredSpotData(lambda: (records([-100, 1, -101]), -102))
            second = _DeferredSpotData(lambda: (records([-100, 1]), -101))
            first.packed = second.packed = packed
            return [], {'first': first, 'second': second}

    plan = MeasurementPlan(Instrument(), ((), {})).prepare()
    signal, result = plan.result()
    assert calls == ['predict']
    assert signal == [False, True, False, False, True]
    assert [record[0] for record in result['first']] == [-100, 1, -101]
    assert [record[0] for record in result['second']] == [-102, 1]
    assert isinstance(result['first'], _SpotData if packed else list)


def test_workspace_rejection_and_cancellation_do_not_retain_pixels():
    from hexrd.hedm._staged_spots import Batch
    from hexrd.hedm._image_cache import StreamingView

    class Reader:
        shape, nframes, resident, dtype = (3, 3), 3, False, np.dtype('f8')

        def read(self, index, cache):
            return np.ones(self.shape)

    cache = FrameCache(0)
    view = StreamingView(View(Reader(), dense=True), cache)
    plan = MeasurementPlan(None, None)
    plan.batches = [
        Batch(
            view,
            np.array([[0, 1, 2]]),
            np.array([0, 9]),
            np.repeat(np.arange(3), 3),
            np.tile(np.arange(3), 3),
            np.array([[3, 3, 3]]),
            0.5,
            (),
            (np.zeros((1, 3)), np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool)),
        )
    ]
    assert not MeasurementGroup(100).add(plan)
    group = MeasurementGroup(1024**2)
    assert group.add(plan)
    count = 0

    def cancel():
        nonlocal count
        count += 1
        if count == 4:
            raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            group.measure(cache, cancel)
        assert all(p.pixels is None for p in group.patches)
        assert cache.bytes == 0 and cache.max_bytes <= group.max_bytes // 4
    finally:
        cache.close()


def test_changed_window_refuses_replay():
    from hexrd.hedm._staged_spots import _DeferredView

    class ViewStub:
        dtype = np.dtype('f8')

    view = ViewStub()
    plan = MeasurementPlan(None, None)
    deferred = _DeferredView(plan, view)
    args = [
        np.array([[0, 1]]),
        np.array([0, 1]),
        np.array([0]),
        np.array([0]),
        np.array([[2, 1, 1]]),
    ]
    deferred.extract(*args, 10.0)
    plan.replaying = True
    args[2][0] = 1
    with pytest.raises(RuntimeError, match='window changed'):
        deferred.extract(*args, 10.0)


@pytest.mark.parametrize('cancel_run', [False, 1, 2])
@pytest.mark.parametrize('fallback', [False, True])
def test_staged_driver_prepares_pass_before_fitting_and_restores_grain_state(
    monkeypatch, cancel_run, fallback
):
    from hexrd.hedm import fitgrains
    from hexrd.hedm._staged_spots import fit_staged
    from hexrd.hedm._progress import _cancellation_check

    log, written = [], {}

    class Reader:
        shape, nframes, resident, dtype = (3, 3), 3, False, np.dtype('f8')

        def read(self, index, cache):
            log.append(('read', index))
            return np.full(self.shape, index + 10.0)

    cache = FrameCache(0)
    series = OmegaImageSeries(
        BoundedSeries(
            View(Reader(), dense=True),
            cache,
            {'omega': np.column_stack((np.arange(3), np.arange(1, 4)))},
            3,
        )
    )

    class Instrument:
        def pull_spots(self, grain, iteration):
            return sparse_frame_view(series).extract(
                np.array([[0, 1, 2]]),
                np.array([0, 9]),
                np.repeat(np.arange(3), 3),
                np.tile(np.arange(3), 3),
                np.array([[3, 3, 3]]),
                5.0,
            )

    params = {
        'grains_table': np.zeros((2, 21)),
        'imgser_dict': {'panel': series},
        'instrument': Instrument(),
        'staged_workspace': 16 * 1024**2,
        'tth_tol': [1, 1],
    }
    params['grains_table'][:, 0] = [0, 1]

    def steps(grain):
        for iteration in (1, 2):
            measured = yield ((grain, iteration), {})
            log.append(('fit', grain, iteration))
            fitgrains.paramMP['pending_spot_data'] = (grain, iteration, measured)
        return grain, 1.0, 0.0, np.zeros(12)

    def write(grain):
        item = fitgrains.paramMP.pop('pending_spot_data', None)
        if item is not None:
            written[grain] = item[:2]

    monkeypatch.setattr(fitgrains, '_fit_grain_steps', steps)
    monkeypatch.setattr(fitgrains, '_write_pending_spots', write)

    class Bar:
        count = 0

        def update(self, n):
            self.count += n

    bar = Bar()
    if fallback:
        monkeypatch.setattr(MeasurementGroup, 'add', lambda *args: False)
    clear_sparse_frame_views()
    try:
        if cancel_run:
            from hexrd.hedm._staged_spots import StagedCancelled

            with pytest.raises(StagedCancelled):
                threshold = 1 if cancel_run == 1 else (6 if fallback else 3) + 1
                fit_staged(
                    params, bar, lambda: sum(x[0] == 'read' for x in log) >= threshold
                )
            assert bar.count == 0 and not written
            assert not hasattr(fitgrains, 'paramMP')
            assert cache.bytes == 0
            assert _cancellation_check.get() is None
            return
        result = fit_staged(params, bar)
        assert [x[0] for x in result] == [0, 1]
        reads = [('read', 0), ('read', 1), ('read', 2)] * (2 if fallback else 1)
        assert log == reads + [('fit', 0, 1), ('fit', 1, 1)] + reads + [
            ('fit', 0, 2),
            ('fit', 1, 2),
        ]
        assert written == {0: (0, 2), 1: (1, 2)}
        assert bar.count == 2
        assert not hasattr(fitgrains, 'paramMP')
        assert _cancellation_check.get() is None
    finally:
        cache.close()
        clear_sparse_frame_views()


def test_recreated_shared_panel_roi_has_stable_exact_measurement_scope():
    from hexrd.core.imageseries.process import ProcessedImageSeries

    class Reader:
        shape, nframes, resident, dtype = (6, 8), 3, False, np.dtype('f8')

        def read(self, index, cache):
            return np.arange(48, dtype='f8').reshape(self.shape) + index

    cache = FrameCache(0)
    series = OmegaImageSeries(
        BoundedSeries(
            View(Reader(), dense=True),
            cache,
            {'omega': np.column_stack((np.arange(3), np.arange(1, 4)))},
            3,
        )
    )

    class Instrument:
        def pull_spots(self):
            crop = OmegaImageSeries(
                ProcessedImageSeries(series, [('rectangle', [[2, 6], [3, 8]])])
            )
            return sparse_frame_view(crop).extract(
                np.array([[0, 1]]),
                np.array([0, 4]),
                np.array([0, 1, 2, 3]),
                np.array([1, 2, 3, 4]),
                np.array([[2, 2, 2]]),
                10.0,
            )

    instrument = Instrument()
    clear_sparse_frame_views()
    try:
        expected = instrument.pull_spots()
        plan = MeasurementPlan(instrument, ((), {})).prepare()
        assert plan.batches, 'shared panel ROI unexpectedly used image fallback'
        group = MeasurementGroup(1024**2)
        assert group.add(plan)
        group.measure(cache)
        for actual, wanted in zip(plan.result(), expected):
            np.testing.assert_array_equal(actual, wanted, strict=True)
    finally:
        cache.close()
        clear_sparse_frame_views()


@pytest.mark.parametrize('value', ['grain', 'staged'])
def test_measurement_mode_configuration(value):
    from hexrd.hedm.experiment import FitGrains

    assert (
        FitGrains.from_dict({'measurement_mode': value}, lambda p: p).measurement_mode
        == value
    )


def test_invalid_measurement_mode_is_rejected():
    from hexrd.hedm.experiment import FitGrains

    with pytest.raises(ValueError, match='measurement_mode'):
        FitGrains.from_dict({'measurement_mode': 'approximate'}, lambda p: p)


def test_single_grain_geometry_overflow_uses_exact_bounded_fallback():
    from hexrd.hedm._staged_spots import _DeferredView

    class Source:
        dtype = np.dtype('f8')

        def extract(self, *args):
            return ('actual measurements',)

    source = Source()

    class Instrument:
        def pull_spots(self):
            from hexrd.core.instrument._spotutils import _extraction_context

            context = _extraction_context.get()
            view = _DeferredView(context, source) if context is not None else source
            return view.extract(
                np.array([[0, 1]]),
                np.array([0, 1]),
                np.array([0]),
                np.array([0]),
                np.array([[2, 1, 1]]),
                10.0,
            )

    plan = MeasurementPlan(Instrument(), ((), {}), max_bytes=1).prepare()
    assert not plan.batches and plan.bytes == 0
    assert plan.result() == ('actual measurements',)


def test_large_float_windows_share_full_frame_reads_without_changing_moments():
    calls, sampled = [], []
    data = np.random.default_rng(12).uniform(0, 100, (3, 200, 200)).astype('f4')

    class Reader:
        shape, nframes, resident, dtype = (200, 200), 3, False, np.dtype('f4')

        def read(self, index, cache):
            calls.append(index)
            return data[index]

        def sample_pixels(self, *args):
            sampled.append(True)
            return None

    reader = Reader()
    cache = FrameCache(0)
    series = OmegaImageSeries(
        BoundedSeries(
            View(reader, dense=True),
            cache,
            {'omega': np.column_stack((np.arange(3), np.arange(1, 4)))},
            3,
        )
    )
    rows, cols = np.indices((200, 200))

    class Instrument:
        def pull_spots(self, threshold):
            return sparse_frame_view(series).extract(
                np.array([[2, 0, 1]]),
                np.array([0, 40000]),
                rows.ravel(),
                cols.ravel(),
                np.array([[3, 200, 200]]),
                threshold,
            )

    clear_sparse_frame_views()
    instrument = Instrument()
    try:
        reference = [instrument.pull_spots(t) for t in (10, 50)]
        sampled.clear()
        plans = [MeasurementPlan(instrument, ((t,), {})).prepare() for t in (10, 50)]
        group = MeasurementGroup(16 * 1024**2)
        assert all(group.add(p) for p in plans)
        calls.clear()
        group.measure(cache)
        assert calls == [0, 1, 2]
        # Try the bounded pixel reader once per frame, then share its full
        # fallback frame across every tile when partial sampling is unavailable.
        assert len(sampled) == 3
        for p, wanted in zip(plans, reference):
            for actual, expected in zip(p.result(), wanted):
                np.testing.assert_array_equal(actual, expected, strict=True)
    finally:
        cache.close()
        clear_sparse_frame_views()


@pytest.mark.parametrize('dense', [False, True])
def test_shared_roi_preserves_sparse_zero_conversion(dense):
    from hexrd.core.imageseries.process import ProcessedImageSeries

    class Reader:
        shape, nframes, resident, dtype = (6, 8), 2, False, np.dtype('f8')

        def read(self, index, cache):
            frame = np.ones(self.shape)
            frame[3, 4] = -0.0
            return frame

    cache = FrameCache(0)
    series = OmegaImageSeries(
        BoundedSeries(
            View(Reader(), dense=dense),
            cache,
            {'omega': np.array([[0.0, 1.0], [1.0, 2.0]])},
            2,
        )
    )
    crop = OmegaImageSeries(
        ProcessedImageSeries(series, [('rectangle', [[2, 6], [3, 8]])])
    )
    clear_sparse_frame_views()
    try:
        expected = crop[0]
        expected = expected.toarray() if sparse.issparse(expected) else expected
        rows, cols = np.indices(crop.shape)
        actual = (
            sparse_frame_view(crop)
            .gather(
                np.array([[0]]), np.array([0, rows.size]), rows.ravel(), cols.ravel()
            )
            .reshape(crop.shape)
        )
        np.testing.assert_array_equal(actual, expected, strict=True)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    finally:
        cache.close()
        clear_sparse_frame_views()


def test_cancellation_callback_can_check_dashboard_without_recursing():
    from hexrd.hedm._progress import _cancellation_check, check_cancelled

    token = _cancellation_check.set(check_cancelled)
    try:
        check_cancelled()
    finally:
        _cancellation_check.reset(token)


def test_optimizer_polls_staged_cancellation_before_residual_evaluation():
    from hexrd.core.instrument import HEDMInstrument
    from hexrd.hedm.fitting.grains import fitGrain
    from hexrd.hedm._progress import _cancellation_check
    from hexrd.hedm._staged_spots import StagedCancelled

    def cancel():
        raise StagedCancelled()

    token = _cancellation_check.set(cancel)
    try:
        with pytest.raises(StagedCancelled):
            fitGrain(
                np.r_[np.zeros(6), np.ones(3), np.zeros(3)],
                HEDMInstrument(),
                {},
                np.eye(3),
                0.2,
            )
    finally:
        _cancellation_check.reset(token)


def test_direct_reader_polls_staged_cancellation_before_reading(tmp_path):
    import os

    if not hasattr(os, 'O_DIRECT'):
        pytest.skip('Linux direct I/O')
    from hexrd.hedm._direct_io import DirectFile
    from hexrd.hedm._progress import _cancellation_check
    from hexrd.hedm._staged_spots import StagedCancelled

    path = tmp_path / 'input.bin'
    path.write_bytes(b'0' * 4096)

    def cancel():
        raise StagedCancelled()

    with DirectFile(path) as source:
        token = _cancellation_check.set(cancel)
        try:
            with pytest.raises(StagedCancelled):
                source.readinto(bytearray(1))
            assert source.tell() == 0
        finally:
            _cancellation_check.reset(token)


def test_metadata_survives_payload_release_within_the_same_allowance():
    cache = FrameCache(10240)
    cache.metadata_limit = 6000
    reader = object()
    index = cache.array(reader, 'index', 0, lambda: np.zeros(4000, 'u1'), metadata=True)
    cache.array(reader, 'pixels', 0, lambda: np.zeros(3000, 'u1'))
    assert cache.bytes <= cache.max_bytes
    cache.clear_payload()
    assert cache.bytes == cache.metadata_bytes == index.nbytes + 1024
    assert (
        cache.array(
            reader,
            'index',
            0,
            lambda: pytest.fail('index was discarded'),
            metadata=True,
        )
        is index
    )
    cache.close()
    assert cache.bytes == cache.metadata_bytes == 0
