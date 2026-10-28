"""Function-level tests for the typed experiment inputs
(:mod:`hexrd.core.experiment`, :mod:`hexrd.hedm.experiment`) and the
material bridge (:mod:`hexrd.core.material.material_data`).

Everything here runs on synthetic data written to tmp_path, except the
material-bridge tests, which read the NIST ruby materials file from the
example repo (gated on ``HEXRD_EXAMPLE_REPO_PATH`` like the other
example-data tests).
"""
import os

import h5py
import numpy as np
import pytest
import yaml
from scipy.sparse import csr_array

from hexrd.core.experiment import (
    Beam,
    Detector,
    Experiment,
    ImageSeries,
    Transform,
    _frame_csr,
    _merge_config,
    _NpzReader,
    _parse_multiprocessing,
)
from hexrd.hedm.experiment import (
    Clustering,
    ClusteringAlgorithm,
    DirectSearch,
    Eta,
    FindOrientations,
    HedmExperiment,
    Omega,
    OrientationMaps,
    SeedSearch,
    SeedSearchMethod,
    _hkl_array,
)


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------
def test_parse_multiprocessing():
    ncpus = os.cpu_count() or 1
    assert _parse_multiprocessing('all') == ncpus
    assert _parse_multiprocessing('half') == max(ncpus // 2, 1)
    assert _parse_multiprocessing(1) == 1
    assert _parse_multiprocessing(10**6) == ncpus          # clamped high
    assert _parse_multiprocessing(-1) == max(ncpus - 1, 1)
    assert _parse_multiprocessing(-10**6) == 1             # clamped low
    with pytest.raises(ValueError):
        _parse_multiprocessing('most')
    with pytest.raises(ValueError):
        _parse_multiprocessing(2.5)


def test_merge_config_nested_overlay():
    base = {'a': 1, 'b': {'x': 1, 'y': 2}, 'c': [1, 2]}
    overlay = {'b': {'y': 20, 'z': 30}, 'c': [3], 'd': 4}
    merged = _merge_config(base, overlay)
    assert merged == {'a': 1, 'b': {'x': 1, 'y': 20, 'z': 30}, 'c': [3], 'd': 4}
    # the inputs are not mutated
    assert base['b'] == {'x': 1, 'y': 2} and overlay['b'] == {'y': 20, 'z': 30}


def test_beam_from_dict():
    beam = Beam.from_dict({
        'energy': 80.725,
        'vector': {'azimuth': 90.0, 'polar_angle': 90.0},
    })
    assert beam.energy == 80.725
    from hexrd.core.instrument import calc_beam_vec

    np.testing.assert_array_equal(beam.vector, calc_beam_vec(90., 90.))
    assert np.isinf(beam.source_distance)

    beams = {'XRS1': {'energy': 10.25, 'source_distance': 31.0}}
    assert Beam.from_dict(beams).source_distance == 31.0


def test_frame_csr_matches_scipy_constructor():
    shape = (4, 6)
    # canonical (row-major, unique) triplets: the fast path
    row = np.array([0, 0, 2, 3])
    col = np.array([1, 4, 0, 5])
    data = np.array([1.0, 2.0, 3.0, 4.0])
    fast = _frame_csr(data, row, col, shape, np.float64)
    ref = csr_array((data, (row, col)), shape=shape, dtype=np.float64)
    assert np.array_equal(fast.todense(), ref.todense())

    # unsorted triplets: must fall back and still be correct
    order = np.array([2, 0, 3, 1])
    slow = _frame_csr(data[order], row[order], col[order], shape, np.float64)
    assert np.array_equal(slow.todense(), ref.todense())

    # empty frame
    empty = _frame_csr(np.array([]), np.array([], dtype=int),
                       np.array([], dtype=int), shape, np.float64)
    assert empty.nnz == 0 and empty.shape == shape


@pytest.mark.parametrize('compressed', [True, False])
def test_npz_reader_matches_np_load(tmp_path, compressed):
    arrays = {
        'ints': np.arange(12, dtype=np.uint16).reshape(3, 4),
        'floats': np.linspace(0, 1, 7),
        'scalar': np.array(3),
        'text': np.array(b'uint16'),
    }
    path = tmp_path / 'arrays.npz'
    (np.savez_compressed if compressed else np.savez)(path, **arrays)

    reader = _NpzReader(str(path))
    with np.load(path) as expected:
        for name in arrays:
            assert np.array_equal(reader[name], expected[name])


# ---------------------------------------------------------------------------
# instrument pieces
# ---------------------------------------------------------------------------
def test_transform_rotation_matrix():
    assert np.allclose(
        Transform(np.zeros(3), np.zeros(3)).rotation_matrix, np.eye(3))
    # a 90 degree rotation about z
    rmat = Transform(np.array([0.0, 0.0, np.pi / 2]), np.zeros(3)).rotation_matrix
    assert np.allclose(rmat @ [1, 0, 0], [0, 1, 0])


@pytest.mark.parametrize('userblock', [0, 512, 4096])
def test_load_instrument_hdf5(tmp_path, userblock):
    """A .hexrd HDF5 instrument archive loads like its yaml equivalent."""
    from hexrd.core.experiment import _load_instrument

    spec = {
        'beam': {'energy': 65.35, 'vector': {'azimuth': 90.0, 'polar_angle': 90.0}},
        'oscillation_stage': {'chi': 0.0, 'translation': [0.0, 0.0, 0.0]},
        'detectors': {
            'GE2': {
                'pixels': {'rows': 2048, 'columns': 2048, 'size': [0.2, 0.2]},
                'transform': {'tilt': [0.001, -0.002, 0.0],
                              'translation': [10.0, -5.0, -900.0]},
                'distortion': {'function_name': 'GE_41RT',
                               'parameters': [1e-5, -2e-4, -1e-4, 2.0, 2.0, 2.0]},
            },
        },
    }

    def write_group(group, d):
        for key, value in d.items():
            if isinstance(value, dict):
                write_group(group.create_group(key), value)
            else:
                group[key] = value

    path = tmp_path / 'instrument.hexrd'
    with h5py.File(path, 'w', userblock_size=userblock) as f:
        write_group(f.create_group('instrument'), spec)
        # scalars often come back from hexrd archives as 1-element datasets
        del f['instrument/beam/energy']
        f['instrument/beam/energy'] = np.array([65.35])

    loaded = _load_instrument(str(path))
    assert loaded['beam']['energy'] == 65.35
    ge2 = loaded['detectors']['GE2']
    assert ge2['pixels']['rows'] == 2048
    assert ge2['transform']['translation'] == [10.0, -5.0, -900.0]
    assert ge2['distortion']['function_name'] == 'GE_41RT'

    # and the yaml path still works
    yml = tmp_path / 'instrument.yml'
    with open(yml, 'w') as f:
        yaml.safe_dump(spec, f)
    assert _load_instrument(str(yml)) == spec


def test_detector_pixel_coordinates():
    detector = Detector.from_dict('d', {
        'pixels': {'rows': 2, 'columns': 3, 'size': [0.5, 0.2]},
        'transform': {},
    })
    grid_i, grid_j = detector.pixel_coordinates
    assert grid_i.shape == grid_j.shape == (2, 3)
    # rows run top-down (+y first), columns left-right (-x first), origin centered
    assert np.allclose(grid_i[:, 0], [0.25, -0.25])
    assert np.allclose(grid_j[0, :], [-0.2, 0.0, 0.2])
    assert detector.buffer is None
    assert detector.distortion.function_name == ''

    buffered = Detector.from_dict('d', {'buffer': [1.0, 2.0]})
    assert np.array_equal(buffered.buffer, [1.0, 2.0])


# ---------------------------------------------------------------------------
# frame-cache ImageSeries
# ---------------------------------------------------------------------------
def _write_frame_cache(path, frames, omega_deg):
    """A hexrd-style sparse frame-cache npz from a list of dense frames."""
    arrays = {
        'shape': np.array(frames[0].shape),
        'nframes': np.array(len(frames)),
        'dtype': np.array(str(frames[0].dtype).encode()),
        'omega': np.asarray(omega_deg, dtype=float),
    }
    for i, frame in enumerate(frames):
        row, col = np.nonzero(frame)
        arrays[f'{i}_data'] = frame[row, col]
        arrays[f'{i}_row'] = row
        arrays[f'{i}_col'] = col
    np.savez_compressed(path, **arrays)


def test_image_series_lazy_frames(tmp_path):
    rng = np.random.default_rng(7)
    frames = [(rng.random((5, 8)) < 0.3).astype(np.uint16) * (i + 1)
              for i in range(3)]
    omega = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    path = tmp_path / 'panel_cache.npz'
    _write_frame_cache(path, frames, omega)

    ims = ImageSeries('p0', str(path))
    assert ims.panel == 'p0'
    assert len(ims) == 3
    assert np.allclose(ims.omega, np.radians(omega))
    assert ims._images is None                     # nothing decompressed yet

    images = ims.images
    assert len(images) == 3
    for image, frame in zip(images, frames):
        assert image.dtype == np.uint16
        assert np.array_equal(np.asarray(image.todense()), frame)
    assert ims.images is images                    # cached, not re-read


def test_roi_image_series_slices_parent(tmp_path):
    from hexrd.core.experiment import RoiImageSeries

    rng = np.random.default_rng(13)
    frames = [(rng.random((6, 8)) < 0.4).astype(np.uint16) * (i + 1)
              for i in range(2)]
    path = tmp_path / 'whole_panel.npz'
    _write_frame_cache(path, frames, [[0.0, 1.0], [1.0, 2.0]])
    parent = ImageSeries(None, str(path))

    top = RoiImageSeries('p_top', parent, roi=[0, 0], shape=(3, 8))
    bottom = RoiImageSeries('p_bottom', parent, roi=[3, 2], shape=(3, 6))
    assert len(top) == len(bottom) == 2
    assert np.array_equal(top.omega, parent.omega)
    for i, frame in enumerate(frames):
        assert np.array_equal(np.asarray(top.images[i].todense()), frame[:3, :])
        assert np.array_equal(np.asarray(bottom.images[i].todense()),
                              frame[3:6, 2:8])
    # the parent's frames were decompressed once and shared
    assert top._parent is bottom._parent


# ---------------------------------------------------------------------------
# find-orientations config sections
# ---------------------------------------------------------------------------
def test_hkl_array_forms():
    assert _hkl_array(None, 'k') is None
    assert np.array_equal(_hkl_array([[1, 0, 0], [1, 1, 0]], 'k'),
                          [[1, 0, 0], [1, 1, 0]])
    # a flat list of ints is the legacy master-hkl-ID spelling
    assert np.array_equal(_hkl_array([0, 3, 5], 'k'), [0, 3, 5])
    assert _hkl_array('all', 'k') is None
    assert np.array_equal(_hkl_array(3, 'k'), [3])
    for bad in ([[1, 0]], [[1.5, 0.0, 0.0]], [0.5, 1.5], 'invalid'):
        with pytest.raises(ValueError):
            _hkl_array(bad, 'k')


def test_orientation_maps_from_dict():
    om = OrientationMaps.from_dict({'threshold': 25})
    assert om.threshold == 25.0
    assert om.active_hkls is None and om.file is None
    assert om.eta_step == 0.25
    assert om.filter_maps is False and om.filter_fwhm is None

    # a null threshold (as in the DAC example configs) means no thresholding
    assert OrientationMaps.from_dict({'threshold': None}).threshold == 0.0

    om = OrientationMaps.from_dict({
        'threshold': 1, 'active_hkls': [[1, 0, 0]], 'eta_step': 1.0,
        'file': 'maps.npz', 'filter_maps': True, 'filter_fwhm': 2.0})
    assert om.filter_maps is True and om.filter_fwhm == 2.0

    numeric = OrientationMaps.from_dict({'filter_maps': 2.0})
    assert numeric.filter_maps is True and numeric.filter_fwhm == 2.0
    assert OrientationMaps.from_dict({'filter_maps': None}).filter_maps is False
    with pytest.raises(ValueError, match='filter_maps: true'):
        OrientationMaps.from_dict({'filter_fwhm': 2.0})


def test_seed_search_from_dict():
    ss = SeedSearch.from_dict({'hkl_seeds': [0, 1], 'fiber_step': 0.5,
                               'method': {'label': {'threshold': 5}}}, 0.25)
    assert ss.method is SeedSearchMethod.LABEL
    assert ss.method_kwargs == {'threshold': 5}
    assert np.array_equal(ss.hkl_seeds, [0, 1])
    assert np.isclose(ss.fiber_step, np.radians(0.5))
    assert ss.fiber_ndiv == 720

    # fiber_step falls back to the omega tolerance; method to plain label
    ss = SeedSearch.from_dict({'hkl_seeds': [0]}, 0.25)
    assert np.isclose(ss.fiber_step, np.radians(0.25))
    assert ss.method is SeedSearchMethod.LABEL and ss.method_kwargs == {}

    with pytest.raises(ValueError, match='hkl_seeds'):
        SeedSearch.from_dict({'hkl_seeds': [[0, 1]]}, 0.25)
    with pytest.raises(ValueError):
        SeedSearch.from_dict({'method': {'watershed': {}}}, 0.25)


def test_eta_range_masks_rotation_axis():
    eta = Eta.from_dict({})
    assert np.isclose(eta.tolerance, np.radians(0.5))
    assert np.isclose(eta.mask, np.radians(5))
    lo, hi = np.degrees(eta.range)
    assert np.allclose(lo, [-85, 85]) and np.allclose(hi, [95, 265])

    unmasked = Eta.from_dict({'mask': None})
    assert unmasked.mask is None
    assert np.allclose(unmasked.range, [[-np.pi, np.pi]])


def test_clustering_from_dict():
    cl = Clustering.from_dict({'radius': 1.0, 'completeness': 0.85,
                               'algorithm': 'sph-dbscan'})
    assert cl.algorithm is ClusteringAlgorithm.SPH_DBSCAN
    assert Clustering.from_dict(
        {'radius': 1, 'completeness': 0.5}).algorithm is ClusteringAlgorithm.DBSCAN
    with pytest.raises(ValueError, match="completeness.*radius|radius.*completeness"):
        Clustering.from_dict({})
    with pytest.raises(ValueError):
        Clustering.from_dict({'radius': 1, 'completeness': 0.5,
                              'algorithm': 'kmeans'})


def test_find_orientations_from_dict():
    fo = FindOrientations.from_dict({
        'orientation_maps': {'threshold': 25},
        'seed_search': {'hkl_seeds': [0]},
        'clustering': {'radius': 1.0, 'completeness': 0.85},
        'omega': {'tolerance': 1.0},
    })
    assert fo.orientation_maps.threshold == 25.0
    assert np.isclose(fo.omega.tolerance, np.radians(1.0))
    # seed_search fiber_step defaults to the omega tolerance
    assert np.isclose(fo.seed_search.fiber_step, np.radians(1.0))
    assert fo.threshold == 1.0
    assert fo.use_quaternion_grid is None
    assert fo.direct_search is None
    assert isinstance(fo.eta, Eta) and isinstance(fo.omega, Omega)


def test_direct_search_from_dict():
    config = {'direct_search': {'average_angular_spacing': 5.0},
              'clustering': {'radius': 1.0, 'completeness': 0.85}}
    direct = FindOrientations.from_dict(config).direct_search
    assert direct == DirectSearch(5.0)
    with pytest.raises(ValueError, match='requires average_angular_spacing'):
        DirectSearch.from_dict({})
    assert DirectSearch.from_dict(
        {'average_angular_spacing': 3.0, 'point_group_number': 32}).point_group_number == 32
    with pytest.raises(ValueError, match='mapping'):
        DirectSearch.from_dict(True)
    for spacing in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match='average_angular_spacing'):
            DirectSearch(spacing)
    for group in (0, 33, 1.5, True):
        with pytest.raises(ValueError, match='point_group_number'):
            DirectSearch(3.0, point_group_number=group)
    for extra in ({'seed_search': {'hkl_seeds': [0]}},
                  {'use_quaternion_grid': 'grid.npy'}):
        with pytest.raises(ValueError, match='choose'):
            FindOrientations.from_dict(config | extra)


# ---------------------------------------------------------------------------
# Experiment / HedmExperiment on a synthetic config
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_config(tmp_path):
    """A tiny but complete experiment: one 4x6-pixel panel, three frames."""
    instrument = {
        'beam': {'energy': 80.0,
                 'vector': {'azimuth': 90.0, 'polar_angle': 90.0}},
        'oscillation_stage': {'chi': 0.01, 'translation': [0.0, 0.0, 0.0]},
        'detectors': {
            'p0': {
                'pixels': {'rows': 4, 'columns': 6, 'size': [0.2, 0.2]},
                'transform': {'tilt': [0.0, 0.0, 0.0],
                              'translation': [0.0, 0.0, -100.0]},
            },
        },
    }
    with open(tmp_path / 'instrument.yml', 'w') as f:
        yaml.safe_dump(instrument, f)

    frames = [np.zeros((4, 6), dtype=np.uint16) for _ in range(3)]
    frames[0][1, 2] = 40000
    frames[1][1, 2] = 40000
    frames[2][3, 5] = 7
    _write_frame_cache(tmp_path / 'p0_cache.npz', frames,
                       [[0.0, 120.0], [120.0, 240.0], [240.0, 360.0]])

    config = {
        'analysis_name': 'analysis',
        'multiprocessing': 1,
        'material': {'definitions': 'materials.h5', 'active': 'ruby',
                     'dmin': 1.0, 'tth_width': 0.2},
        'instrument': 'instrument.yml',
        'image_series': {'format': 'frame-cache',
                         'data': [{'file': 'p0_cache.npz', 'panel': 'p0'}]},
        'find_orientations': {
            'orientation_maps': {'threshold': 1, 'eta_step': 5.0},
            'seed_search': {'hkl_seeds': [0],
                            'method': {'label': {'threshold': 1}}},
            'clustering': {'radius': 1.0, 'completeness': 0.5},
        },
    }
    study = {'analysis_name': 'study1',
             'find_orientations': {'orientation_maps': {'eta_step': 10.0}}}
    path = tmp_path / 'config.yml'
    with open(path, 'w') as f:
        yaml.safe_dump_all([config, study], f, sort_keys=False)
    return path


def test_experiment_parses_synthetic_config(synthetic_config):
    exp = HedmExperiment(str(synthetic_config))
    assert list(exp.detectors) == ['p0']
    assert exp.beam_energy == 80.0
    assert exp.oscillation_stage.chi == 0.01
    assert exp.max_workers == 1
    assert exp.analysis_name == 'analysis'
    assert exp.analysis_dir == str(synthetic_config.parent / 'analysis')
    assert exp.analysis_id == 'analysis_ruby'
    assert exp.active_material.two_theta_width == 0.2
    assert exp.active_material.reset_exclusions is True

    (ims,) = exp.image_series_list
    assert len(ims) == 3
    assert np.isclose(ims.omega[0, 1], np.radians(120.0))

    fo = exp.find_orientations
    assert fo.orientation_maps.eta_step == 5.0
    assert fo.clustering.completeness == 0.5


def test_experiment_study_overlay(synthetic_config):
    base = HedmExperiment(str(synthetic_config))
    assert len(base.studies) == 1
    study = HedmExperiment(str(synthetic_config), study=0)
    # overlaid values change; everything else survives from the base document
    assert study.analysis_name == 'study1'
    assert study.find_orientations.orientation_maps.eta_step == 10.0
    assert study.find_orientations.clustering.radius == 1.0


@pytest.mark.parametrize('style', ['npz', 'fch5'])
def test_experiment_from_objects(synthetic_config, ruby_materials_file, monkeypatch, style):
    from copy import deepcopy
    from hexrd.core.material.material_data import Material
    from hexrd.hedm.find_orientations import build_eta_omega_maps

    source = HedmExperiment(str(synthetic_config))
    instrument, series = source.instrument, source.image_series
    if style == 'fch5':
        from hexrd.core import imageseries
        from hexrd.core.imageseries.omega import OmegaImageSeries

        filename = synthetic_config.parent / 'frames.fch5'
        frames = imageseries.open(str(synthetic_config.parent / 'p0_cache.npz'), 'frame-cache')
        imageseries.write(frames, str(filename), 'frame-cache', style='fch5', threshold=0)
        frames = imageseries.open(str(filename), 'frame-cache', style='fch5')
        series = {'p0': OmegaImageSeries(frames)}

        def no_bulk_load():
            pytest.fail('in-memory series must retain streaming FCH5 access')

        monkeypatch.setattr(frames._adapter, '_load_framelist', no_bulk_load)
    pd = Material('ruby', ruby_materials_file, dmin=1.6, beam_energy=80.)._material.planeData
    pd.exclusions = np.arange(pd.getNhklRef()) % 3 == 0
    exclusions = pd.exclusions.copy()
    config = deepcopy(source.config)
    config.pop('instrument')
    config.pop('image_series')
    config['material'].pop('definitions')
    config['working_dir'] = str(synthetic_config.parent)

    experiment = HedmExperiment.from_objects(instrument, series, pd, config)
    assert experiment.instrument is instrument
    assert experiment.image_series['p0'] is series['p0']
    assert experiment.get_active_material()._material.planeData is pd
    assert experiment.get_materials() == [experiment.get_active_material()]
    assert experiment.get_active_material().name == 'ruby'
    np.testing.assert_array_equal(pd.exclusions, exclusions)
    assert 'definitions' not in config['material']
    assert experiment.analysis_dir == source.analysis_dir
    np.testing.assert_array_equal(
        build_eta_omega_maps(experiment, experiment.get_active_material().plane_data).ring_maps,
        build_eta_omega_maps(source, experiment.get_active_material().plane_data).ring_maps)
    assert source.image_series_list[0]._images is None


@pytest.mark.parametrize('new_file_placement', [False, True])
def test_cli_workflow_paths(synthetic_config, tmp_path, monkeypatch, new_file_placement):
    from hexrd.hedm.cli import fit_grains
    from types import SimpleNamespace

    cwd = tmp_path / 'run'
    cwd.mkdir()
    documents = list(yaml.safe_load_all(synthetic_config.read_text()))
    documents[0]['new_file_placement'] = new_file_placement
    synthetic_config.write_text(yaml.safe_dump_all(documents))
    monkeypatch.chdir(cwd)
    # Input files are relative to cwd, just as in master's CLI.
    for name in ('instrument.yml', 'p0_cache.npz'):
        (cwd / name).symlink_to(synthetic_config.parent / name)
    def check_paths(*args, **kwargs):
        experiment = HedmExperiment(*args, **kwargs)
        assert experiment.analysis_dir == str(cwd / 'analysis')
        accepted = (cwd / 'analysis' / 'accepted-orientations-ruby.dat'
                    if new_file_placement else cwd / 'accepted_orientations_analysis_ruby.dat')
        assert experiment.accepted_orientations_file == str(accepted)
        raise KeyboardInterrupt

    monkeypatch.setattr(fit_grains, 'HedmExperiment', check_paths)
    with pytest.raises(KeyboardInterrupt):
        fit_grains.execute(SimpleNamespace(yml=str(synthetic_config), force=False, clean=False), None)


@pytest.mark.parametrize('workers', [1, 2])
def test_map_cancellation_stops_inside_large_frame(synthetic_config, monkeypatch, workers):
    from types import SimpleNamespace
    from hexrd.hedm import find_orientations as fo
    from hexrd.hedm._progress import Dashboard, stage

    experiment = HedmExperiment(str(synthetic_config))
    experiment.max_workers = workers
    series, = experiment.image_series_list
    n = 262145
    chunk = (np.zeros(n, dtype=np.int64), np.ones(n), np.array([0, n, n, n]), np.arange(3))
    monkeypatch.setattr(series, 'pixel_chunks', lambda *args, **kwargs: iter([chunk]))
    monkeypatch.setattr(fo, '_iter_panel_bins', lambda *args: iter([
        (series, (np.zeros(24, dtype=np.int32), np.ones((1, 24), dtype=np.int64),
                  np.ones((1, 72), dtype=bool)))]))
    plane = SimpleNamespace(unexcluded_hkls=np.array([[1, 1, 1]]),
                            two_thetas=np.array([0.01]), exclusions=np.array([False]))
    dashboard = Dashboard()
    calls = []
    bin_pixels = fo._bin_pixels

    def cancel_after_first_block(*args):
        calls.append(len(args[1]))
        bin_pixels(*args)
        dashboard.cancelled.set()

    monkeypatch.setattr(fo, '_bin_pixels', cancel_after_first_block)
    dashboard.start()
    try:
        with pytest.raises(KeyboardInterrupt), stage('Eta/omega maps'):
            fo.build_eta_omega_maps(experiment, plane)
    finally:
        dashboard.close()
    assert calls == [262144]
    assert dashboard.stages['Eta/omega maps'].status == 'failed'


@pytest.mark.parametrize('repeated_rows', [False, True])
@pytest.mark.parametrize('style', ['npz', 'fch5'])
def test_parallel_maps_preserve_serial_bytes(synthetic_config, monkeypatch, repeated_rows, style):
    import threading
    from types import SimpleNamespace
    from hexrd.hedm import find_orientations as fo

    frames = np.zeros((3, 4, 6))
    frames[:, 1, 2] = [1e16, 1., 1.]
    _write_frame_cache(synthetic_config.parent / 'p0_cache.npz', frames,
                       [[0., 120.], [120., 240.], [240., 360.]])
    experiment = HedmExperiment(str(synthetic_config))
    if style == 'fch5':
        from hexrd.core import imageseries
        from hexrd.hedm.experiment import _AdapterImageSeries

        source = imageseries.open(str(synthetic_config.parent / 'p0_cache.npz'), 'frame-cache')
        cache = synthetic_config.parent / 'frames.fch5'
        imageseries.write(source, str(cache), 'frame-cache', style='fch5', threshold=0)
        experiment.image_series_list = [_AdapterImageSeries('p0', str(cache), 'frame-cache', {'style': 'fch5'})]
        monkeypatch.setattr('hexrd.hedm.experiment._FCH5_PARALLEL_MIN_PIXELS', 0)
    plane = SimpleNamespace(unexcluded_hkls=np.tile([1, 1, 1], (65, 1)),
                            two_thetas=np.full(65, 0.002), exclusions=np.zeros(65, bool))
    if repeated_rows:
        omega_grid = fo._omega_grid
        monkeypatch.setattr(fo, '_omega_grid', lambda series:
                            (np.zeros(3, dtype=int), *omega_grid(series)[1:]))
    expected = fo.build_eta_omega_maps(experiment, plane)
    threads, last_started = set(), threading.Event()
    bin_pixels = fo._bin_pixels

    def bin_concurrently(*args):
        threads.add(threading.get_ident())
        if not repeated_rows:
            if args[3][0] == 0:
                assert last_started.wait(5), 'completed frames must free worker slots'
            elif args[3][0] == 2:
                last_started.set()
        return bin_pixels(*args)

    monkeypatch.setattr(fo, '_bin_pixels', bin_concurrently)
    experiment.max_workers = 2
    if style == 'fch5':
        experiment.image_series_list[0]._max_workers = 2
    actual = fo.build_eta_omega_maps(experiment, plane)
    assert actual.ring_maps.tobytes() == expected.ring_maps.tobytes()
    assert np.nanmax(actual.ring_maps) == 1e16
    if repeated_rows:
        assert threads == {threading.get_ident()}
    else:
        assert len(threads) == 2


@pytest.mark.parametrize('style', ['hdf5', 'fch5'])
def test_registered_image_series_and_metadata_panel(synthetic_config, style, monkeypatch):
    with open(synthetic_config) as stream:
        config = next(yaml.safe_load_all(stream))
    frames = np.zeros((3, 4, 6), dtype=np.uint16)
    frames[:, 1, 2] = 40_000
    omega = np.array([[0., 120.], [120., 240.], [240., 360.]])
    filename = synthetic_config.parent / 'images.h5'
    with h5py.File(filename, 'w') as stream:
        group = stream.create_group('entry')
        group['images'] = frames
        group.attrs['omega'] = omega
        group.attrs['panel'] = 'p0'
    config['image_series'] = {
        'format': 'hdf5',
        'data': [{'file': 'images.h5', 'args': {'path': 'entry'}}],
    }
    if style == 'fch5':
        from hexrd.core import imageseries

        source = imageseries.open(str(filename), 'hdf5', path='entry')
        cache = synthetic_config.parent / 'images.fch5'
        imageseries.write(source, str(cache), 'frame-cache', style='fch5', threshold=0)
        config['image_series'] = {
            'format': 'frame-cache',
            'data': [{'file': cache.name, 'args': {'style': 'fch5'}}],
        }
    with open(synthetic_config, 'w') as stream:
        yaml.safe_dump(config, stream)
    experiment = HedmExperiment(str(synthetic_config))
    (series,) = experiment.image_series_list
    assert series.panel == 'p0'
    assert len(series) == 3
    assert np.array_equal(series.omega, np.radians(omega))
    with monkeypatch.context() as patch:
        if style == 'fch5':
            def no_bulk_load():
                pytest.fail('streaming must not load the complete frame cache')
            patch.setattr(series._series._adapter, '_load_framelist', no_bulk_load)
        chunks = list(series.pixel_chunks(np.arange(3), max_pixels=2))
    assert series._images is None
    if style == 'fch5':
        assert not series._series._adapter._framelist_was_loaded
        assert series._series._adapter._framelist == []
        monkeypatch.setattr('hexrd.hedm.experiment._FCH5_PARALLEL_MIN_PIXELS', 0)
        series._max_workers = 2
        parallel = list(series.pixel_chunks(np.arange(3), threshold=1.))
        assert sorted(int(chunk[3][0]) for chunk in parallel) == [0, 1, 2]
        for ids, values, offsets, rows in parallel:
            np.testing.assert_array_equal(ids, [8])
            np.testing.assert_array_equal(values, [40_000.])
            np.testing.assert_array_equal(offsets, [0, 1])
        import multiprocessing as mp

        children = set()

        def cancel_after_spawn():
            children.update(child.pid for child in mp.active_children())
            if children:
                raise KeyboardInterrupt

        with monkeypatch.context() as patch:
            patch.setattr('hexrd.hedm._progress.check_cancelled', cancel_after_spawn)
            with pytest.raises(KeyboardInterrupt):
                list(series.pixel_chunks(np.arange(3)))
        assert children and not children.intersection(child.pid for child in mp.active_children())
    assert len(chunks) == 2
    for ids, values, offsets, rows in chunks:
        np.testing.assert_array_equal(ids, np.full(len(rows), 8))
        np.testing.assert_array_equal(values, np.full(len(rows), 40_000.))
        np.testing.assert_array_equal(offsets, np.arange(len(rows) + 1))
    np.testing.assert_array_equal(np.concatenate([chunk[3] for chunk in chunks]), np.arange(3))
    assert np.array_equal(series[0].toarray(), frames[0])
    assert series[0, 1:2, 2:3].item() == 40_000

    # Explicit null means a shared series, even when metadata names a panel.
    config['image_series']['data'][0]['panel'] = None
    with open(synthetic_config, 'w') as stream:
        yaml.safe_dump(config, stream)
    with pytest.raises(ValueError, match='pixels: roi'):
        HedmExperiment(str(synthetic_config))


@pytest.mark.parametrize('dtype', [np.uint16, np.float64])
@pytest.mark.parametrize('index_dtype', [np.uint16, np.float64])
def test_fch5_filter_coalesces_before_thresholding(tmp_path, dtype, index_dtype):
    from hexrd.hedm.experiment import _fch5_pixels

    indices = np.array([[0, 1], [0, 0], [0, 1], [0, 1]], dtype=index_dtype)
    data = np.array([40000, 5, 40000, 5] if dtype == np.uint16 else [np.nan, 0.5, 1., 1.], dtype=dtype)
    reference = csr_array((data, (indices[:, 0], indices[:, 1])), shape=(1, 2)).tocoo()
    with h5py.File(tmp_path / 'pixels.h5', 'w') as group:
        group['indices'], group['data'] = indices, data[:, None]
        for threshold in (None, 10000.):
            ids, values = _fch5_pixels(group, 0, 4, (1, 2), np.dtype(dtype), threshold)
            keep = np.ones(reference.nnz, bool) if threshold is None else ~(reference.data.astype(float) < threshold)
            np.testing.assert_array_equal(ids, reference.col[keep])
            np.testing.assert_array_equal(values, reference.data[keep])


def test_grouped_panel_roi_config(synthetic_config):
    with open(synthetic_config) as stream:
        config = next(yaml.safe_load_all(stream))
    instrument_path = synthetic_config.parent / 'instrument.yml'
    with open(instrument_path) as stream:
        instrument = yaml.safe_load(stream)
    original = instrument['detectors'].pop('p0')
    instrument['detectors'] = {
        'left': {**original, 'group': 'p0',
                 'pixels': {'rows': 4, 'columns': 3, 'size': [0.2, 0.2], 'roi': [0, 0]}},
        'right': {**original, 'group': 'p0',
                  'pixels': {'rows': 4, 'columns': 3, 'size': [0.2, 0.2], 'roi': [0, 3]}},
    }
    with open(instrument_path, 'w') as stream:
        yaml.safe_dump(instrument, stream)
    experiment = HedmExperiment(str(synthetic_config))
    left, right = experiment.image_series_list
    assert [left.panel, right.panel] == ['left', 'right']
    assert left._parent is right._parent
    assert left.images[0].toarray()[1, 2] == 40_000
    assert not right.images[0].nnz


@pytest.mark.parametrize('panel', [None, 'physical-panel'])
def test_shared_image_series_requires_explicit_roi(synthetic_config, panel):
    with open(synthetic_config) as stream:
        config = next(yaml.safe_load_all(stream))
    config['image_series']['data'][0]['panel'] = panel
    with open(synthetic_config, 'w') as stream:
        yaml.safe_dump(config, stream)
    instrument_path = synthetic_config.parent / 'instrument.yml'
    with open(instrument_path) as stream:
        instrument = yaml.safe_load(stream)
    instrument['detectors']['p0']['group'] = 'physical-panel'
    with open(instrument_path, 'w') as stream:
        yaml.safe_dump(instrument, stream)
    with pytest.raises(ValueError, match='pixels: roi'):
        HedmExperiment(str(synthetic_config))
    instrument['detectors']['p0']['pixels']['roi'] = [0, 0]
    with open(instrument_path, 'w') as stream:
        yaml.safe_dump(instrument, stream)
    (series,) = HedmExperiment(str(synthetic_config)).image_series_list
    assert series[0, 1:2, 2:3].item() == 40_000


def test_eta_ome_maps_file_paths(synthetic_config, tmp_path):
    exp = HedmExperiment(str(synthetic_config))
    assert exp.eta_ome_maps_file == os.path.join(
        str(tmp_path), f'{exp.analysis_name}_ruby_eta-ome_maps.npz')

    exp.config['find_orientations']['orientation_maps']['file'] = 'my_maps.npz'
    relative = HedmExperiment.__new__(HedmExperiment)
    relative.__dict__.update(exp.__dict__)
    relative.find_orientations = FindOrientations.from_dict(
        exp.config['find_orientations'])
    assert relative.eta_ome_maps_file == str(tmp_path / 'my_maps.npz')

    exp.config['find_orientations']['orientation_maps']['file'] = '/abs/maps.npz'
    absolute = HedmExperiment.__new__(HedmExperiment)
    absolute.__dict__.update(exp.__dict__)
    absolute.find_orientations = FindOrientations.from_dict(
        exp.config['find_orientations'])
    assert absolute.eta_ome_maps_file == '/abs/maps.npz'


def test_quaternion_grid_file_paths(synthetic_config, tmp_path):
    exp = HedmExperiment(str(synthetic_config))
    assert exp.quaternion_grid_file is None

    exp.config['find_orientations']['use_quaternion_grid'] = 'grid.npy'
    relative = HedmExperiment.__new__(HedmExperiment)
    relative.__dict__.update(exp.__dict__)
    relative.find_orientations = FindOrientations.from_dict(
        exp.config['find_orientations'])
    assert relative.quaternion_grid_file == str(tmp_path / 'grid.npy')


# ---------------------------------------------------------------------------
# the material bridge, on the ruby materials file
# ---------------------------------------------------------------------------
@pytest.fixture
def ruby_materials_file(example_repo_path):
    return str(example_repo_path / 'NIST_ruby' / 'single_GE' / 'include'
               / 'materials.h5')


def test_load_materials(ruby_materials_file):
    from hexrd.core.material.material_data import Material, load_materials

    materials = load_materials(ruby_materials_file, dmin=1.6, beam_energy=80.725)
    assert 'ruby' in materials
    ruby = materials['ruby']
    assert isinstance(ruby, Material)
    assert ruby.plane_data.laue_group == 'd3d'
    assert ruby.plane_data.point_group_number == 20

    # an open h5py group works the same as a path
    with h5py.File(ruby_materials_file, 'r') as f:
        from_group = load_materials(f, dmin=1.6, beam_energy=80.725)
    assert list(from_group) == list(materials)
    assert np.array_equal(from_group['ruby'].plane_data.hkls,
                          ruby.plane_data.hkls)


def test_material_bridge_parameters(ruby_materials_file):
    from hexrd.core.material.material_data import Material

    loose = Material('ruby', ruby_materials_file, dmin=1.6, beam_energy=80.725)
    tight = Material('ruby', ruby_materials_file, dmin=1.0, beam_energy=80.725)
    # a smaller dmin admits more reflections
    assert len(tight.plane_data.unexcluded_hkls) > len(loose.plane_data.unexcluded_hkls)

    softer = Material('ruby', ruby_materials_file, dmin=1.6, beam_energy=40.0)
    # wavelength scales inversely with the beam energy
    assert np.isclose(softer.plane_data.wavelength,
                      loose.plane_data.wavelength * 80.725 / 40.0, rtol=1e-6)

    # per-reflection symmetry data is consistent
    pd = loose.plane_data
    assert len(pd.symm_hkls) == len(pd.unexcluded_hkls)
    for symm, hkl in zip(pd.symm_hkls, pd.unexcluded_hkls):
        assert symm.shape[0] == 3
        assert any(np.array_equal(col, hkl) for col in symm.T)


def test_experiment_preserves_saved_material_selection(
        synthetic_config, ruby_materials_file):
    from hexrd.core.material.material_data import Material

    saved = Material('ruby', ruby_materials_file, dmin=1.6,
                     beam_energy=80.0)._material
    saved.planeData.exclusions = np.arange(saved.planeData.getNhklRef()) % 3 == 0
    saved.dump_material(str(synthetic_config.parent/'materials.h5'))
    documents = list(yaml.safe_load_all(synthetic_config.read_text()))
    documents[0]['material'].update(reset_exclusions=False, dmin=0.35,
                                     min_sfac_ratio=0.99)
    synthetic_config.write_text(yaml.safe_dump_all(documents))

    experiment = HedmExperiment(str(synthetic_config))
    actual = experiment.get_active_material().plane_data
    assert experiment.active_material.reset_exclusions is False
    assert len(actual.hkls) == saved.planeData.getNhklRef()
    np.testing.assert_array_equal(actual.exclusions, saved.planeData.exclusions)
    np.testing.assert_array_equal(actual.unexcluded_hkls, saved.planeData.hkls.T)

    # Recover automatically from maps built while saved exclusions were ignored.
    from hexrd.hedm.find_orientations import (
        EtaOmegaMaps, load_or_build_eta_omega_maps,
    )
    os.makedirs(experiment.analysis_dir, exist_ok=True)
    edges = np.array([-np.pi, np.pi])
    stale = EtaOmegaMaps(np.zeros((1, 1, 1)), edges, edges, np.array([0.]),
                         edges, np.array([[0., 1.]]),
                         np.array([len(actual.unexcluded_hkls)]), 2*np.pi)
    np.savez(experiment.eta_ome_maps_file, ring_maps=stale.ring_maps,
             eta_edges=edges, omega_edges=edges, omegas=stale.omegas,
             omega_period=edges, two_theta_ranges=stale.two_theta_ranges,
             ring_ids=stale.ring_ids, eta_step=stale.eta_step)
    maps = load_or_build_eta_omega_maps(experiment, actual)
    np.testing.assert_array_equal(maps.ring_ids, np.arange(len(actual.unexcluded_hkls)))
