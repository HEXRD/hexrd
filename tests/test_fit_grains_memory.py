"""Bounded fitting must never enter a frame-cache adapter's bulk loader."""

import pickle

import h5py
import numpy as np
import pytest
from scipy import sparse

from hexrd.core import imageseries
from hexrd.core.experiment import ImageSeries, RoiImageSeries
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.hedm._image_cache import FrameCache, ImagePlan, frame_bytes
from hexrd.hedm.experiment import ImageCacheSettings, _AdapterImageSeries


@pytest.fixture
def sources(tmp_path):
    rng = np.random.default_rng(13)
    data = rng.integers(0, 100, (7, 12, 15), dtype=np.uint16)
    data[data < 70] = 0
    omega = np.column_stack((np.arange(7), np.arange(1, 8))).astype(float)
    arrays = dict(
        nframes=7, shape=data.shape[1:], dtype=np.bytes_('uint16'), omega=omega
    )
    for i, frame in enumerate(data):
        row, col = frame.nonzero()
        arrays.update({f'{i}_row': row, f'{i}_col': col, f'{i}_data': frame[row, col]})
    npz = tmp_path / 'frames.npz'
    np.savez_compressed(npz, **arrays)
    fch5 = tmp_path / 'frames.fch5'
    with h5py.File(fch5, 'w') as f:
        f.attrs['HEXRD_FRAMECACHE_VERSION'] = 1
        f['shape'], f['nframes'], f['dtype'] = data.shape[1:], 7, np.bytes_('uint16')
        f.create_group('metadata')['omega'] = omega
        indices = [np.column_stack(frame.nonzero()) for frame in data]
        counts = np.r_[0, np.cumsum([len(x) for x in indices])]
        f['frame_ids'] = np.column_stack((counts[:-1], counts[1:])).ravel()
        f['indices'] = np.concatenate(indices)
        f['data'] = np.concatenate(
            [frame[tuple(ix.T)] for frame, ix in zip(data, indices)]
        )[:, None]
    hdf = tmp_path / 'frames.h5'
    with h5py.File(hdf, 'w') as f:
        group = f.create_group('entry')
        group['images'] = data
        group.attrs['omega'] = omega
    import fabio
    import yaml

    for i, frame in enumerate(data):
        fabio.tifimage.TifImage(data=frame).write(str(tmp_path / f'{i}.tif'))
    tiff = tmp_path / 'tiff.yml'
    tiff.write_text(
        yaml.safe_dump(
            {
                'image-files': {'directory': str(tmp_path), 'files': '*.tif'},
                'meta': {'omega': '++np.array', 'omega-array': omega.tolist()},
                'options': {},
            }
        )
    )
    return data, npz, fch5, hdf, tiff


def forbidden_bulk(*args, **kwargs):
    raise AssertionError('lazy fitting called the whole-scan loader')


@pytest.mark.parametrize('fmt', ['core-npz', 'npz', 'fch5', 'hdf5', 'tiff'])
def test_formats_never_bulk_load_and_pickle_without_frames(sources, monkeypatch, fmt):
    data, npz, fch5, hdf, tiff = sources
    from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter

    monkeypatch.setattr(FrameCacheImageSeriesAdapter, '_load_framelist', forbidden_bulk)
    monkeypatch.setattr(ImageSeries, 'images', property(forbidden_bulk))
    if fmt == 'core-npz':
        source = ImageSeries('panel', str(npz))
    else:
        filename, kind, args = {
            'npz': (npz, 'frame-cache', {}),
            'fch5': (fch5, 'frame-cache', {'style': 'fch5'}),
            'hdf5': (hdf, 'hdf5', {'path': 'entry'}),
            'tiff': (tiff, 'image-files', {}),
        }[fmt]
        source = _AdapterImageSeries('panel', str(filename), kind, args)
    plan = ImagePlan()
    try:
        actual, workers = plan.prepare(
            {'panel': OmegaImageSeries(source)}, ImageCacheSettings('lazy', 0.003), 10
        )
        assert plan.lazy and workers == 10
        assert plan.context.get_start_method() == 'spawn'
        for i in [4, 2, 0, 6, 4]:
            np.testing.assert_array_equal(actual['panel'][i].toarray(), data[i])
            assert plan.cache.bytes <= plan.cache.max_bytes
        assert len(plan.cache.frames) < len(data)
        restored = pickle.loads(pickle.dumps(actual))
        assert restored['panel']._adapter.cache.bytes == 0
        np.testing.assert_array_equal(restored['panel'][3].toarray(), data[3])
        restored['panel']._adapter.cache.close()
    finally:
        plan.close()
    assert not plan.cache.frames and not plan.cache.files


@pytest.mark.parametrize(
    'operations',
    [
        [('rectangle', [[1, 9], [2, 12]])],
        [('dark', np.ones((12, 15))), ('flip', 't'), ('rectangle', [[1, 9], [2, 10]])],
        [('rectangle', [[1, 9], [2, 12]]), ('add', -1.5)],
    ],
)
def test_processing_frame_selection_and_roi_share_one_budget(
    sources, monkeypatch, operations
):
    data, path, *_ = sources
    source = imageseries.open(str(path), 'frame-cache')
    processed = ProcessedImageSeries(source, operations, frame_list=[5, 2, 0])
    expected = [processed[i].copy() for i in range(3)]
    # A fresh source ensures the test cannot pass by reusing the eager cache.
    processed = ProcessedImageSeries(
        imageseries.open(str(path), 'frame-cache'), operations, frame_list=[5, 2, 0]
    )
    from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter

    monkeypatch.setattr(FrameCacheImageSeriesAdapter, '_load_framelist', forbidden_bulk)
    root = ImageSeries('panel', str(path))
    roi = RoiImageSeries('roi', root, [1, 2], (8, 10))
    plan = ImagePlan()
    try:
        actual, _ = plan.prepare(
            {'a': OmegaImageSeries(processed), 'b': OmegaImageSeries(roi)},
            ImageCacheSettings('lazy', 0.005),
            1,
        )
        assert actual['a']._adapter.cache is actual['b']._adapter.cache
        for i in range(3):
            np.testing.assert_array_equal(actual['a'][i], expected[i])
            np.testing.assert_array_equal(actual['b'][i].toarray(), data[i, 1:9, 2:12])
            assert plan.cache.bytes <= plan.cache.max_bytes
        roots = [key[0] for key in plan.cache.frames]
        assert len({r.filename for r in roots if hasattr(r, 'filename')}) <= 1
    finally:
        plan.close()


@pytest.mark.parametrize('dtype', ['uint16', 'int32', 'float32', 'float64'])
def test_streaming_batches_match_native_while_holding_only_one_frame(dtype):
    from hexrd.core.extensions import transforms as native
    from hexrd.hedm._image_cache import View, StreamingView

    rng = np.random.default_rng(200)
    data = rng.integers(0, 100, (20, 17, 21)).astype(dtype)
    data[data < 70] = 0
    frames = [sparse.csr_array(frame) for frame in data]

    class Reader:
        shape, nframes, resident = (17, 21), 20, False

        def read(self, index, cache):
            return sparse.csr_array(data[index])

    reader = Reader()
    reader.dtype = data.dtype
    cache = FrameCache(frame_bytes(frames[0]) + 50)
    lazy = StreamingView(View(reader, roi=(2, 3, 12, 14)), cache)
    eager = native._SparseFrames(frames).view(2, 3, 12, 14)
    shapes = np.array([[3, 5, 7], [3, 9, 11]])
    offsets = np.r_[0, np.cumsum(np.prod(shapes[:, 1:], axis=1))]
    ids = np.array([[0, 2, 19], [1, 1, 7]])
    rows = rng.integers(-3, 16, offsets[-1])
    cols = rng.integers(-3, 18, offsets[-1])
    np.testing.assert_array_equal(
        lazy.gather(ids, offsets, rows, cols), eager.gather(ids, offsets, rows, cols)
    )
    for threshold in (-1, 0, 75, 110):
        for actual, expected in zip(
            lazy.extract(ids, offsets, rows, cols, shapes, threshold),
            eager.extract(ids, offsets, rows, cols, shapes, threshold),
        ):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
    assert cache.peak_bytes <= cache.max_bytes
    assert len(cache.frames) <= 1


def test_cache_charges_owners_and_never_pickles_payloads():
    owner = np.zeros((1024, 1024), dtype=np.uint16)
    view = owner[:1, :1]
    assert frame_bytes(view) >= owner.nbytes
    cache = FrameCache(3 * owner.nbytes)

    class Reader:
        shape, dtype, nframes, resident = view.shape, view.dtype, 10, False

        def read(self, index, cache):
            return np.zeros_like(owner)[:1, :1]

    reader = Reader()
    for i in range(10):
        cache.frame(reader, i)
        assert cache.bytes <= cache.max_bytes
    assert len(cache.frames) == 2  # entry overhead is included too
    snapshot = pickle.dumps(cache)
    assert len(snapshot) < 200
    restored = pickle.loads(snapshot)
    assert not restored.frames and restored.max_bytes == cache.max_bytes


def test_auto_keeps_small_scan_fast_and_honors_resident_edits(sources):
    _, path, *_ = sources
    source = ImageSeries('panel', str(path))
    omega = OmegaImageSeries(source)
    plan = ImagePlan()
    try:
        result, _ = plan.prepare({'a': omega}, ImageCacheSettings(), 1)
        assert result['a'] is omega and not plan.lazy
        assert source._images is None
        source.images[0] = sparse.csr_array(np.full(source._shape, 99, dtype=np.uint16))
        result, _ = plan.prepare({'a': omega}, ImageCacheSettings('lazy', 0), 1)
        np.testing.assert_array_equal(result['a'][0].toarray(), 99)
        assert plan.cache.bytes == 0
    finally:
        plan.close()


@pytest.mark.parametrize(
    'kwargs',
    [
        {'mode': 'typo'},
        {'max_size_mb': -1},
        {'max_size_mb': float('inf')},
        {'read_workers': 0},
        {'read_workers': 9},
        {'read_workers': 2.5},
        {'read_workers': True},
    ],
)
def test_invalid_cache_policy_rejected(kwargs):
    with pytest.raises(ValueError, match='image_cache'):
        ImageCacheSettings(**kwargs)


def test_huge_logical_hdf5_scan_is_never_materialized(tmp_path):
    path = tmp_path / 'huge.h5'
    count, shape = 8192, (1024, 1024)  # 16 GiB of logical image data
    omega = np.column_stack((np.arange(count), np.arange(1, count + 1))) / count * 360
    with h5py.File(path, 'w') as f:
        group = f.create_group('entry', track_order=True)
        group.create_dataset(
            'images',
            (count, *shape),
            dtype='uint16',
            chunks=(1, 128, 128),
            fillvalue=81,
        )
        group.attrs['omega'] = omega
    source = OmegaImageSeries(
        _AdapterImageSeries('panel', str(path), 'hdf5', {'path': 'entry'})
    )
    plan = ImagePlan()
    try:
        series, _ = plan.prepare({'panel': source}, ImageCacheSettings('auto', 5), 1)
        assert plan.lazy
        for i in range(64):
            np.testing.assert_array_equal(
                series['panel'][i, np.array([0, 1000]), np.array([0, 1000])], 81
            )
            assert plan.cache.bytes <= 5 * 1024**2
            assert len(plan.cache.frames) <= 2
        assert plan.cache.misses == 64
        assert path.stat().st_size < 1024**2
        assert len(pickle.dumps(series)) < 1024**2
        assert not source._adapter._images
    finally:
        plan.close()


@pytest.mark.parametrize('fail', [False, True])
def test_fitting_closes_lazy_files_on_success_and_failure(
    sources, monkeypatch, tmp_path, fail
):
    from types import SimpleNamespace
    from hexrd.hedm import fitgrains
    from hexrd.hedm import _image_cache

    _, path, *_ = sources
    plans = []

    class RecordingPlan(ImagePlan):
        def __init__(self):
            super().__init__()
            plans.append(self)

    monkeypatch.setattr(_image_cache, 'ImagePlan', RecordingPlan)
    experiment = SimpleNamespace(
        image_series={'panel': OmegaImageSeries(ImageSeries('panel', str(path)))},
        instrument=None,
        max_workers=1,
        analysis_dir=tmp_path,
        find_orientations=SimpleNamespace(eta=SimpleNamespace(range=[[-np.pi, np.pi]])),
        fit_grains=SimpleNamespace(
            image_cache=ImageCacheSettings('lazy', 1),
            threshold=75,
            npdiv=2,
            refit=None,
            tolerance=SimpleNamespace(tth=[0.1], eta=[1.0], omega=[1.0]),
        ),
        get_active_material=lambda: SimpleNamespace(
            _material=SimpleNamespace(planeData=None)
        ),
    )

    def fit_one(grain_id):
        fitgrains.paramMP['imgser_dict']['panel'][0]
        assert plans[0].cache.files
        if fail:
            raise RuntimeError('simulated optimizer failure')
        return (grain_id, 1.0, 0.1, np.zeros(12))

    monkeypatch.setattr(fitgrains, 'fit_grain_FF_reduced', fit_one)
    if fail:
        with pytest.raises(RuntimeError, match='simulated optimizer failure'):
            fitgrains.fit_grains(experiment, np.zeros((1, 21)), write_spots_files=False)
    else:
        fitgrains.fit_grains(experiment, np.zeros((1, 21)), write_spots_files=False)
    assert not plans[0].cache.files and not plans[0].cache.frames
    assert not hasattr(fitgrains, 'paramMP')


@pytest.mark.parametrize('name', ['dark', 'rectangle', 'custom'])
def test_custom_processing_is_not_silently_replaced(sources, name):
    _, path, *_ = sources
    source = imageseries.open(str(path), 'frame-cache')
    processed = ProcessedImageSeries(source, [(name, 5)])
    processed.addop(name, processed._add)
    plan = ImagePlan()
    try:
        with pytest.raises(ValueError, match='standard ProcessedImageSeries'):
            plan.prepare(
                {'panel': OmegaImageSeries(processed)}, ImageCacheSettings('lazy', 1), 1
            )
    finally:
        plan.close()


@pytest.mark.parametrize('workers,cancelable', [(2, False), (1, True)])
def test_spawn_never_serializes_an_already_resident_scan(
    sources, monkeypatch, workers, cancelable
):
    import multiprocessing
    from hexrd.core import constants

    data, *_ = sources
    omega = np.column_stack((np.arange(len(data)), np.arange(1, len(data) + 1)))
    source = OmegaImageSeries(
        imageseries.open(None, 'array', data=data, meta={'omega': omega})
    )
    monkeypatch.setattr(constants, 'mp_context', multiprocessing.get_context('spawn'))
    plan = ImagePlan()
    try:
        with pytest.raises(
            ValueError, match='resident arrays in a worker requires fork'
        ):
            plan.prepare(
                {'panel': source},
                ImageCacheSettings('lazy', 1),
                workers,
                force_process=cancelable,
            )
    finally:
        plan.close()
