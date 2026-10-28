import gc
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_array

from hexrd.core.experiment import ImageSeries, RoiImageSeries
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.core.instrument._spotutils import (
    _SpotData,
    clear_sparse_frame_views,
    sparse_frame_view,
)
from hexrd.hedm import fitgrains


@pytest.fixture
def sparse_series(tmp_path):
    path = tmp_path / 'frames.npz'
    arrays = dict(
        nframes=2,
        shape=np.array([4, 5]),
        dtype=np.bytes_('int32'),
        omega=np.array([[0.0, 1.0], [1.0, 2.0]]),
    )
    for frame in range(2):
        arrays.update(
            {
                f'{frame}_row': np.array([0, 1, 3], dtype=np.int32),
                f'{frame}_col': np.array([0, 1, 4], dtype=np.int32),
                f'{frame}_data': np.array([10, 20, 30], dtype=np.int32) + frame,
            }
        )
    np.savez(path, **arrays)
    clear_sparse_frame_views()
    yield ImageSeries('panel', str(path), max_workers=1), path
    clear_sparse_frame_views()


def test_roi_native_view_avoids_copying_images(sparse_series):
    parent, _ = sparse_series
    roi = RoiImageSeries('roi', parent, [1, 1], (2, 3))
    view = sparse_frame_view(OmegaImageSeries(roi))
    assert roi._images is None
    assert parent.images[0].indices.dtype == np.dtype('int32')
    assert parent.images[0].indptr.dtype == np.dtype('int32')
    actual = view.gather(
        np.array([[0, 1]]), np.array([0, 3]), np.array([0, 0, 1]), np.array([0, 2, 1])
    )
    np.testing.assert_array_equal(actual, [20, 0, 0, 21, 0, 0])


def test_new_snapshot_honors_replaced_and_materialized_frames(sparse_series):
    parent, _ = sparse_series
    sparse_frame_view(parent)
    parent.images[0] = csr_array(np.full((4, 5), 55, dtype=np.int32))
    clear_sparse_frame_views()
    actual = sparse_frame_view(parent).gather(
        np.array([[0]]), np.array([0, 1]), np.array([1]), np.array([1])
    )
    np.testing.assert_array_equal(actual, [55])
    roi = RoiImageSeries('roi', parent, [1, 1], (2, 3))
    roi._images = [csr_array(np.full((2, 3), 99, dtype=np.int32)) for _ in range(2)]
    actual = sparse_frame_view(roi).gather(
        np.array([[0]]), np.array([0, 1]), np.array([0]), np.array([0])
    )
    np.testing.assert_array_equal(actual, [99])


def test_processed_images_do_not_bypass_operations(sparse_series):
    from hexrd.core import imageseries

    _, path = sparse_series
    series = imageseries.open(str(path), 'frame-cache')
    dark = ProcessedImageSeries(series, [('dark', np.ones((4, 5)))])
    assert sparse_frame_view(dark) is None
    cropped = ProcessedImageSeries(series, [('rectangle', [[1, 3], [1, 4]])])
    actual = sparse_frame_view(cropped).gather(
        np.array([[0]]), np.array([0, 1]), np.array([0]), np.array([0])
    )
    np.testing.assert_array_equal(actual, [20])
    subset = ProcessedImageSeries(
        series, [('rectangle', [[1, 3], [1, 4]])], frame_list=[1]
    )
    assert sparse_frame_view(subset) is None


@pytest.mark.parametrize('dtype', ['uint16', 'int32', 'int64'])
def test_frame_csr_uses_compact_buffers_without_mutating_inputs(dtype):
    from hexrd.core.experiment import _frame_csr

    row = np.array([0, 1, 3], dtype=dtype)
    col = np.array([0, 1, 4], dtype=dtype)
    data = np.array([10, 20, 30], dtype=np.int32)
    row.flags.writeable = col.flags.writeable = False
    actual = _frame_csr(data, row, col, (4, 5), data.dtype)
    expected = csr_array((data, (row, col)), shape=(4, 5))
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    assert actual.indptr.dtype == actual.indices.dtype == np.dtype('int32')


def test_frame_csr_fallback_coalesces_duplicates():
    from hexrd.core.experiment import _frame_csr

    row = np.array([1, 0, 1, 0], dtype=np.int32)
    col = np.array([2, 1, 2, 0], dtype=np.int32)
    data = np.array([2, 3, 4, 5], dtype=np.int32)
    actual = _frame_csr(data, row, col, (3, 4), data.dtype)
    np.testing.assert_array_equal(
        actual.toarray(), csr_array((data, (row, col)), shape=(3, 4)).toarray()
    )


@pytest.mark.parametrize('row,col', [([-1], [0]), ([0], [5]), ([4], [0])])
def test_frame_csr_rejects_out_of_bounds(row, col):
    from hexrd.core.experiment import _frame_csr

    with pytest.raises(ValueError):
        _frame_csr(np.array([1]), np.array(row), np.array(col), (4, 5), np.int64)


def test_packed_saturation_filter_keeps_float64_comparisons(tmp_path):
    spots = _SpotData(
        np.array([0, 1]),
        np.array([0, 0]),
        np.ones((2, 3)),
        np.array([100, 100]),
        np.array([75, 76], dtype=np.float32),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    instrument = SimpleNamespace(
        detectors={'d': SimpleNamespace(saturation_level=np.nextafter(75.0, 76.0))}
    )
    packed, count, completeness = fitgrains.determine_valid_reflections(
        {'d': spots}, instrument, tmp_path
    )
    legacy, reference_count, reference_completeness = (
        fitgrains.determine_valid_reflections({'d': list(spots)}, instrument, tmp_path)
    )
    assert packed['d'].peak_id.tolist() == [legacy['d'][0][0]] == [0]
    assert count == reference_count
    assert completeness == reference_completeness


@pytest.mark.parametrize('mode', ['success', 'canceled', 'constructor_error'])
def test_pool_preserves_gc_state_and_cleans_up(tmp_path, monkeypatch, mode):
    image = SimpleNamespace(omega=np.array([[0.0, 1.0]]))
    experiment = SimpleNamespace(
        image_series={'d': image},
        instrument=SimpleNamespace(detectors={}),
        find_orientations=SimpleNamespace(
            eta=SimpleNamespace(range=np.array([[-np.pi, np.pi]]))
        ),
        max_workers=2,
        analysis_dir=tmp_path,
        fit_grains=SimpleNamespace(
            threshold=75,
            npdiv=2,
            refit=None,
            tolerance=SimpleNamespace(tth=[0.1], eta=[1.0], omega=[1.0]),
        ),
        get_active_material=lambda: SimpleNamespace(
            _material=SimpleNamespace(planeData=object())
        ),
    )
    grains = np.zeros((2, 21))
    grains[:, 0] = [0, 1]
    initial = gc.get_freeze_count(), gc.isenabled()
    events = []

    def create_pool(*args):
        assert gc.get_freeze_count() >= initial[0]
        if initial[0] == 0:
            assert gc.get_freeze_count() > 0
        if mode == 'constructor_error':
            raise RuntimeError('pool creation failed')
        result = SimpleNamespace(
            ready=lambda: mode != 'canceled', get=lambda: [], wait=lambda _: None
        )
        return SimpleNamespace(
            map_async=lambda *args, **kwargs: result,
            close=lambda: events.append('close'),
            terminate=lambda: events.append('terminate'),
            join=lambda: events.append('join'),
        )

    monkeypatch.setattr(
        fitgrains.constants,
        'mp_context',
        SimpleNamespace(get_start_method=lambda: 'fork', Pool=create_pool),
    )
    monkeypatch.setattr(fitgrains, 'sparse_frame_view', lambda _: object())
    from hexrd.hedm.xrdutil import utils

    monkeypatch.setattr(utils, '_expand_pixels', lambda *args: None)
    monkeypatch.setattr(utils, '_compute_max', lambda *args: None)
    if mode == 'constructor_error':
        with pytest.raises(RuntimeError, match='pool creation failed'):
            fitgrains.fit_grains(experiment, grains, write_spots_files=False)
    else:
        result = fitgrains.fit_grains(
            experiment,
            grains,
            write_spots_files=False,
            check_if_canceled_func=(lambda: True) if mode == 'canceled' else None,
        )
        assert result is None if mode == 'canceled' else result == []
        assert events[-2:] == ['terminate', 'join']
    assert (gc.get_freeze_count(), gc.isenabled()) == initial


@pytest.mark.parametrize('fails', [False, True])
def test_spot_tables_publish_last_complete_extraction_on_exit(
    tmp_path, monkeypatch, fails
):
    params = {
        'instrument': SimpleNamespace(detectors={'d': object(), 'empty': object()}),
        'analysis_dirname': str(tmp_path),
        'spots_filename': 'spots_%05d.out',
        'tth_tol': [0.1, 0.05],
    }
    record = [
        0,
        1,
        np.array([1, -2, 3]),
        100,
        75,
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.3, 0.4]),
        np.array([2.0, 3.0]),
        np.array([4.0, 5.0]),
    ]

    def execute(grain_id):
        params['pending_spot_data'] = {'d': [record]}
        assert not (tmp_path / 'd' / 'spots_00004.out').exists()
        if fails:
            raise RuntimeError('fit failed after extraction')
        return grain_id, 1.0, 0.01, np.zeros(12)

    monkeypatch.setattr(fitgrains, '_fit_grain_FF_reduced', execute)
    fitgrains.fit_grain_FF_init(params)
    try:
        if fails:
            with pytest.raises(RuntimeError, match='fit failed'):
                fitgrains.fit_grain_FF_reduced(4)
        else:
            fitgrains.fit_grain_FF_reduced(4)
        assert 'pending_spot_data' not in params
        assert len((tmp_path / 'd' / 'spots_00004.out').read_text().splitlines()) == 2
        assert (
            len((tmp_path / 'empty' / 'spots_00004.out').read_text().splitlines()) == 1
        )
    finally:
        fitgrains.fit_grain_FF_cleanup()
