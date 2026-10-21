"""Regression coverage for the find-orientations launch-config workflow."""
from dataclasses import replace
import io
import os
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numba
import numpy as np
import pytest
import yaml

from hexrd.core.experiment import Detector, ImageSeries
from hexrd.core.extensions import transforms as native
from hexrd.core.material.material_data import PlaneData, load_plane_data_cached
from hexrd.core.utils.npz import _NpzReader, _read_array
from hexrd.hedm.experiment import HedmExperiment, SeedSearch
from hexrd.hedm.find_orientations import (
    _cached_panel_bin_data, _hit_status_maps, _load_or_generate_fibers,
    find_orientations,
)


@pytest.mark.skipif(
    'HEXRD_EXAMPLE_REPO_PATH' not in os.environ,
    reason='requires the NIST ruby example data',
)
def test_ruby_pipeline(example_repo_path, test_data_dir, tmp_path, monkeypatch):
    import hexrd.hedm.find_orientations as fo

    score = fo.score_orientations
    scored = []

    def check_scoring(*args, **kwargs):
        scored.append(1)
        return score(*args, **kwargs)

    monkeypatch.setattr(fo, 'score_orientations', check_scoring)
    include = example_repo_path / 'NIST_ruby/single_GE/include'
    with open(include / 'cstudy.yml') as stream:
        config = next(yaml.safe_load_all(stream))
    config['analysis_name'] = str(tmp_path / 'results')
    config['multiprocessing'] = 1
    filename = tmp_path / 'config.yml'
    filename.write_text(yaml.safe_dump(config))

    reference = test_data_dir / 'find_orientations_pipeline'
    expected_scores = np.load(reference / 'ruby_completeness.npy')
    expected_qbar = np.load(reference / 'ruby_qbar.npy')
    previous_threads = numba.get_num_threads()
    try:
        for clean in (True, False):
            experiment = HedmExperiment(str(filename), working_dir=str(include))
            load_material = experiment.get_active_material

            def check_workers():
                assert numba.get_num_threads() == 1
                return load_material()

            monkeypatch.setattr(experiment, 'get_active_material', check_workers)
            numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
            result = find_orientations(experiment, clean=clean)
            np.testing.assert_array_equal(result.completeness, expected_scores)
            np.testing.assert_allclose(result.grain_orientations, expected_qbar,
                                       rtol=0, atol=1e-12)
            if not clean:
                assert experiment.image_series_list[0]._images is None
    finally:
        numba.set_num_threads(previous_threads)
    assert len(scored) == 2
    # A fresh cached CLI/API process needs neither JIT initialization nor
    # compilation. Scoring still executes on every invocation.
    script = '''
import sys
from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.find_orientations import find_orientations
from hexrd.core.utils.npz import _NpzReader
assert 'numba' not in sys.modules
read = _NpzReader.__getitem__
def metadata_only(self, name):
    assert name not in ('ring_maps', 'dataStore'), 'cached scoring must not decompress maps'
    return read(self, name)
_NpzReader.__getitem__ = metadata_only
experiment = HedmExperiment(sys.argv[1], working_dir=sys.argv[2])
result = find_orientations(experiment)
assert len(result.completeness) == 13680
assert 'numba' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', script, str(filename), str(include)], check=True)


def _reference_hit(eta, ome, de, do, ring, threshold):
    window = ring[max(0, ome-do):ome+do+1, max(0, eta-de):eta+de+1].ravel()
    events = window[(window > threshold) | np.isnan(window)]
    return (1 if events[0] > threshold else -1) if len(events) else 0


def test_legacy_geometry_entry_points_use_cpp():
    from hexrd.core.extensions import transforms_c_api

    for name in ('oscillAnglesOfHKLs', 'quat_distance', 'detectorXYToGvec',
                 'makeEtaFrameRotMat', 'rotate_vecs_about_axis'):
        assert getattr(transforms_c_api, name) is getattr(native, name)
    assert transforms_c_api.gvecToDetectorXY is native.projectGvecs
    assert transforms_c_api.gvecToDetectorXYArray is native.projectGvecsArray


def test_cpp_predictor_analytic_bragg_solutions():
    angle = np.arcsin(0.05)
    first, second = native.oscillAnglesOfHKLs(
        np.array([[1., 0, 0]]), 0., np.eye(3), np.eye(3)*0.1, 1.,
        np.array([1., 1, 1, 0, 0, 0]), np.array([0., 0, -1]), np.array([1., 0, 0]))
    np.testing.assert_allclose(first, [[2*angle, 0., -angle]], atol=1e-15, rtol=0)
    np.testing.assert_allclose(second, [[2*angle, np.pi, np.pi+angle]], atol=1e-15, rtol=0)


def test_cpp_scoring_handles_no_reflections():
    args = _scoring_inputs(5, [0., 0., -1.])
    args[1] = np.empty((0, 3))
    args[11] = np.empty(0, dtype=np.intp)
    np.testing.assert_array_equal(native.scoreOrientations(*args), np.zeros(5))


def test_lazy_kernels_preserve_dispatcher_features():
    from hexrd.core.rotations import _rotmatofquat

    quaternion = np.array([[1.], [0.], [0.], [0.]])
    np.testing.assert_array_equal(_rotmatofquat.py_func(quaternion), [np.eye(3)])
    np.testing.assert_array_equal(_rotmatofquat(quaternion), [np.eye(3)])
    assert _rotmatofquat.signatures


def _reference_scores(angles, args):
    """Independent scalar oracle, using the established diffraction predictor."""
    scores = []
    for solutions in angles:
        hits = []
        for solution in solutions:
            for h, (tth, eta, ome) in enumerate(solution):
                if np.isnan(tth):
                    continue
                eta = (eta + np.pi) % (2*np.pi) - np.pi
                ome = (ome - args[9]) % (2*np.pi) + args[9]
                if any(not (spans[0] <= value < spans[-1] and
                            np.searchsorted(spans, value, side='right') % 2)
                       for value, spans in ((eta, args[7]), (ome, args[8]))):
                    continue
                e = np.searchsorted(args[5], eta, side='right') - 1
                o = np.searchsorted(args[6], ome, side='right') - 1
                if 0 <= e < args[10].shape[2] and 0 <= o < args[10].shape[1]:
                    hit = _reference_hit(e, o, args[12], args[13],
                                         args[10][args[11][h]], args[14])
                    if hit >= 0:
                        hits.append(hit)
        scores.append(sum(hits) / len(hits) if hits else 0.)
    return np.asarray(scores)


@pytest.mark.parametrize('dpix_eta,dpix_ome', [(0, 0), (0, 3), (2, 0), (1, 2), (40, 20)])
@pytest.mark.parametrize('shape', [(3, 13, 17), (3, 1, 17), (3, 13, 1), (3, 1, 1)])
def test_hit_status_preserves_nan_precedence(dpix_eta, dpix_ome, shape):
    rng = np.random.default_rng(4)
    maps = rng.random(shape)
    maps[rng.random(maps.shape) < 0.15] = np.nan
    maps[0].fill(0)
    maps[1].flat[:2] = [np.nan, 1.0][:min(2, maps[1].size)]
    maps[2].flat[:2] = [1.0, np.nan][:min(2, maps[2].size)]
    status = _hit_status_maps(maps, dpix_eta, dpix_ome, 0.7)
    for r in range(len(maps)):
        for o, e in np.ndindex(maps.shape[1:]):
            assert status[r, o, e] == _reference_hit(
                e, o, dpix_eta, dpix_ome, maps[r], 0.7)


def _scoring_inputs(count, beam):
    rng = np.random.default_rng(52)
    quats = rng.normal(size=(count, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    rmats = native.quaternionRotations(quats)
    hkls = np.array([[1., 0, 0], [0., 1, 0], [1., 1, 0], [1., 0, 1], [1., 1, 1]])
    maps = rng.random((2, 21, 32))
    maps[rng.random(maps.shape) < 0.1] = np.nan
    return [rmats, hkls, np.diag([0.12, 0.14, 0.15]), 0.15,
            np.asarray(beam, dtype=float), np.linspace(-np.pi, np.pi, 33),
            np.linspace(-np.pi, np.pi, 22), np.array([-np.pi, -0.4, 0.1, np.pi]),
            np.array([-np.pi, 0.9]), -np.pi, maps,
            np.array([0, 0, 1, 1, 1]), 1, 2, 0.65]


@pytest.mark.parametrize('count', [0, 1, 97])
@pytest.mark.parametrize('beam', [[0., 0, -1], [0.2, -0.1, -1.]])
def test_native_scoring_matches_reference(count, beam):
    args = _scoring_inputs(count, beam)
    rmats, hkls, bmat, wavelength, beam = args[:5]
    angles = np.empty((count, 2, len(hkls), 3))
    for i, rmat in enumerate(rmats):
        angles[i] = native.oscillAnglesOfHKLs(
            hkls, 0.0, rmat, bmat, wavelength,
            np.array([1., 1, 1, 0, 0, 0]), beam, np.array([1., 0, 0]))
    expected = _reference_scores(angles, args)
    np.testing.assert_array_equal(native.scoreOrientations(*args), expected)
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    np.testing.assert_array_equal(native.scoreOrientations(*args, status), expected)
    bins = np.empty((count, 2, len(hkls)), dtype=np.int32)
    np.testing.assert_array_equal(native.scoreOrientations(*args, status, bins), expected)
    np.testing.assert_array_equal(native._scoreOrientationBins(bins, status), expected)
    args[14] = 0.9
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    np.testing.assert_array_equal(native._scoreOrientationBins(bins, status),
                                  _reference_scores(angles, args))


@pytest.mark.parametrize('index,value', [
    (0, np.empty((2, 2, 3))), (1, np.empty((2, 2))), (2, np.empty((2, 3))),
    (4, np.empty(2)), (5, np.empty(2)), (7, np.empty(0)),
    (7, np.array([-np.pi, 0., np.pi])),
    (4, np.array([0., 0., 0.])), (4, np.array([1., 0., 0.])),
    (4, np.array([np.nan, 0., -1.])),
    (11, np.array([0, 0, 1, 1, 2])), (11, np.array([0])), (12, -1),
])
def test_native_scoring_rejects_invalid_shapes(index, value):
    args = _scoring_inputs(1, [0., 0, -1])
    args[index] = value
    with pytest.raises(ValueError):
        native.scoreOrientations(*args)


@pytest.mark.parametrize('symm_hkls', [[np.eye(3)], []])
def test_material_cache_invalidates_inputs_and_recovers(tmp_path, symm_hkls):
    definitions = tmp_path / 'material.h5'
    definitions.write_bytes(b'first material contents')
    cache = tmp_path / 'cache' / 'planes.npz'
    pd = PlaneData(np.array([[1, 0, 0]]), np.array([False]),
                   np.array([[1, 0, 0]]), np.array([0.1]), symm_hkls,
                   'Oh', np.array([[1.], [0], [0], [0]]), np.eye(3), 0.15)
    calls = []

    def load():
        calls.append(1)
        return SimpleNamespace(plane_data=pd)

    settings = {'dmin': 1.6, 'energy': 80.0}
    assert load_plane_data_cached(definitions, cache, settings, load) is pd
    restored = load_plane_data_cached(definitions, cache, settings, load)
    assert len(calls) == 1
    for name in ('hkls', 'exclusions', 'unexcluded_hkls', 'two_thetas', 'q_sym', 'B'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(pd, name))
    for actual, expected in zip(restored.symm_hkls, pd.symm_hkls, strict=True):
        np.testing.assert_array_equal(actual, expected)
    definitions.write_bytes(b'other material contents')
    load_plane_data_cached(definitions, cache, settings, load)
    assert len(calls) == 2
    load_plane_data_cached(definitions, cache, dict(settings, energy=90.0), load)
    assert len(calls) == 3
    cache.write_bytes(b'truncated archive')
    load_plane_data_cached(definitions, cache, settings, load)
    assert len(calls) == 4
    blocked = tmp_path / 'not_a_directory'
    blocked.write_bytes(b'file')
    assert load_plane_data_cached(definitions, blocked / 'cache.npz', settings, load) is pd


def test_unavailable_fingerprint_disables_cache(tmp_path):
    from hexrd.core.utils.npz import fingerprint, load_array_cache, save_array_cache

    key = fingerprint({}, files=[tmp_path / 'missing-source.py'])
    assert key is None
    path = tmp_path / 'cache.npz'
    save_array_cache(path, key, value=np.ones(1))
    assert not path.exists()
    assert load_array_cache(path, key) is None


def test_geometry_cache_invalidates_inputs(tmp_path, monkeypatch):
    import hexrd.hedm.find_orientations as fo

    detector = Detector.from_dict('p', {
        'pixels': {'rows': 3, 'columns': 4, 'size': [0.2, 0.2]},
        'transform': {'translation': [0., 0, -100.]}})
    experiment = SimpleNamespace(detectors={'p': detector}, analysis_dir=str(tmp_path),
                                 beam=SimpleNamespace(vector=np.array([0., 0, -1.])))
    calls = []

    def build(*args):
        calls.append(1)
        return np.arange(12), np.ones((1, 12), dtype=np.int64), np.ones((1, 4), bool)

    monkeypatch.setattr(fo, '_panel_bin_data', build)
    ranges, edges = np.array([[0., 0.1]]), np.linspace(-np.pi, np.pi, 5)
    first = _cached_panel_bin_data(experiment, 'p', ranges, edges)
    second = _cached_panel_bin_data(experiment, 'p', ranges, edges)
    assert len(calls) == 1
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    detector.transform.translation[2] = -101
    _cached_panel_bin_data(experiment, 'p', ranges, edges)
    assert len(calls) == 2
    _cached_panel_bin_data(experiment, 'p', ranges + 0.01, edges)
    assert len(calls) == 3


def test_streamed_pixels_match_frames_without_loading_images(tmp_path):
    arrays = dict(shape=np.array([3, 4]), dtype=np.array(b'uint16'),
                  nframes=np.array(4), omega=np.array([[i, i + 1] for i in range(4)]))
    for i in range(4):
        arrays.update({f'{i}_row': np.array([0, 1, 2]),
                       f'{i}_col': np.array([2, 1, 3]),
                       f'{i}_data': np.array([i, 40000, 40000], dtype=np.uint16)})
    file = tmp_path / 'frames.npz'
    np.savez_compressed(file, **arrays)
    series = ImageSeries('p', str(file), max_workers=2)
    chunks = list(series.pixel_chunks(np.arange(4), max_pixels=4))
    assert series._images is None
    assert len(chunks) == 4
    for i, (ids, values, offsets, rows) in enumerate(chunks):
        np.testing.assert_array_equal(ids, [2, 5, 11])
        np.testing.assert_array_equal(values, [i, 40000, 40000])
        np.testing.assert_array_equal(offsets, [0, 3])
        np.testing.assert_array_equal(rows, [i])
        assert values.dtype == np.float64


def test_streaming_preserves_duplicate_uint16_weights(tmp_path):
    from hexrd.hedm.find_orientations import _bin_pixels

    path = tmp_path / 'duplicates.npz'
    np.savez_compressed(
        path, shape=np.array([1, 1]), dtype=np.array(b'uint16'), nframes=np.array(1),
        omega=np.array([[0., 1.]]), **{
            '0_data': np.array([40000, 40000], dtype=np.uint16),
            '0_row': np.array([0, 0]), '0_col': np.array([0, 0])})
    series = ImageSeries('p', str(path), max_workers=1)
    result = np.zeros((1, 1, 1))
    for ids, values, offsets, rows in series.pixel_chunks(np.array([0])):
        _bin_pixels(offsets, ids, values, rows, np.array([1]), np.array([0]), 0., result)
    assert result[0, 0, 0] == 80000
    assert series._images is None


def test_package_import_and_legacy_aliases_are_lazy():
    script = '''
import importlib, sys
import hexrd
assert 'numba' not in sys.modules
assert 'scipy' not in sys.modules
for name, target in hexrd.module_aliases.items():
    assert importlib.import_module(name) is importlib.import_module(target)
'''
    subprocess.run([sys.executable, '-c', script], check=True)


def test_fiber_cache_invalidates_search_and_maps(tmp_path, monkeypatch):
    import hexrd.hedm.find_orientations as fo

    seed = SeedSearch.from_dict({'hkl_seeds': [0]}, 1.0)
    experiment = SimpleNamespace(
        find_orientations=SimpleNamespace(seed_search=seed, threshold=1),
        oscillation_stage=SimpleNamespace(chi=0.0))
    pd = SimpleNamespace(laue_group='Oh', B=np.eye(3), q_sym=np.ones((4, 1)),
                         unexcluded_hkls=np.array([[1, 0, 0]]))
    maps = SimpleNamespace(ring_maps=np.ones((1, 3, 4)), eta_step=0.5,
                           eta_edges=np.arange(5.), omega_edges=np.arange(4.),
                           omegas=np.arange(3.), two_theta_ranges=np.array([[0., 0.1]]),
                           ring_ids=np.array([0]))
    calls = []

    def generate(*args):
        calls.append(1)
        return np.array([[1.], [0.], [0.], [0.]])

    monkeypatch.setattr(fo, 'generate_orientation_fibers', generate)
    cache = tmp_path / 'fibers.npz'
    first = _load_or_generate_fibers(experiment, pd, maps, cache)
    experiment.find_orientations.threshold = 2
    second = _load_or_generate_fibers(experiment, pd, maps, cache)
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1
    maps.ring_maps[0, 0, 0] += 1
    _load_or_generate_fibers(experiment, pd, maps, cache)
    assert len(calls) == 2
    experiment.find_orientations.seed_search = replace(seed, fiber_step=0.1)
    _load_or_generate_fibers(experiment, pd, maps, cache)
    assert len(calls) == 3
    pd.B[0, 0] = 2
    _load_or_generate_fibers(experiment, pd, maps, cache)
    assert len(calls) == 4


@pytest.mark.parametrize('codec', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                                   zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_npz_reader_preserves_codecs_and_checksums(tmp_path, codec):
    stream = io.BytesIO()
    expected = np.arange(15).reshape(3, 5)
    np.save(stream, expected)
    path = tmp_path / 'archive.npz'
    with zipfile.ZipFile(path, 'w', compression=codec) as archive:
        archive.writestr('data.npy', stream.getvalue())
    with _NpzReader(path) as reader:
        restored = reader['data']
    np.testing.assert_array_equal(restored, expected)
    if codec == zipfile.ZIP_STORED:
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo('data.npy')
        corrupt = bytearray(path.read_bytes())
        last_byte = member.header_offset + 30 + len(member.filename) + member.compress_size - 1
        corrupt[last_byte] ^= 1
        path.write_bytes(corrupt)
        with _NpzReader(path) as reader, pytest.raises(zipfile.BadZipFile):
            reader['data']


@pytest.mark.parametrize('array', [
    np.array(3), np.array('text'), np.array(True), np.empty((0, 3)),
    np.arange(6, dtype='>i4').reshape(2, 3),
    np.asfortranarray(np.arange(6).reshape(2, 3)),
    np.zeros(2, dtype=[('x', 'f8'), ('y', 'i4')]),
])
@pytest.mark.parametrize('version', [(1, 0), (2, 0), (3, 0)])
def test_numeric_header_fast_path_preserves_numpy_format(array, version):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=version)
    restored = _read_array(stream.getvalue())
    np.testing.assert_array_equal(restored, array)
    assert restored.dtype == array.dtype
    assert restored.flags.writeable


def test_npz_numeric_reader_never_unpickles():
    stream = io.BytesIO()
    np.save(stream, np.array([{'unsafe': 'object'}], dtype=object))
    with pytest.raises(ValueError, match='allow_pickle=False'):
        _read_array(stream.getvalue())


def test_numeric_header_fast_path_rejects_malformed_shapes():
    stream = io.BytesIO()
    np.save(stream, np.arange(6).reshape(2, 3))
    malformed = stream.getvalue().replace(b'(2, 3)', b'(2,,3)')
    with pytest.raises(ValueError):
        _read_array(malformed)


def test_scoring_caches_invalidate_maps_threshold_and_geometry(tmp_path, monkeypatch):
    import hexrd.hedm.find_orientations as fo

    args = _scoring_inputs(97, [0., 0., -1.])
    args[8] = np.array([-np.pi, np.pi])
    quats = np.random.default_rng(52).normal(size=(97, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    maps = SimpleNamespace(ring_maps=args[10], ring_ids=np.arange(2),
                           eta_edges=args[5], omega_edges=args[6])
    plane = SimpleNamespace(symm_hkls=[args[1][:2].T, args[1][2:].T],
                            B=args[2], wavelength=args[3])
    settings = SimpleNamespace(threshold=args[14],
        eta=SimpleNamespace(range=args[7].reshape(-1, 2), tolerance=2*np.pi/32),
        omega=SimpleNamespace(period=np.array([-np.pi, np.pi]), tolerance=4*np.pi/21))
    experiment = SimpleNamespace(find_orientations=settings, max_workers=1,
        beam=SimpleNamespace(vector=args[4]), analysis_dir=str(tmp_path))
    predict = native.scoreOrientations
    calls = []
    def record(*a):
        calls.append(1)
        return predict(*a)
    monkeypatch.setattr(native, 'scoreOrientations', record)
    for change in ('initial', 'cached', 'threshold', 'map', 'geometry', 'damaged'):
        if change == 'threshold':
            settings.threshold = args[14] = 0.95
        elif change == 'map':
            args[10][:] = 1.
        elif change == 'geometry':
            plane.B[0, 0] *= 1.02
        elif change == 'damaged':
            (tmp_path/'.hexrd/reflection-bins.npz').write_bytes(b'broken cache')
        np.testing.assert_array_equal(fo.score_orientations(experiment, plane, maps, quats.T),
                                      predict(*args))
        assert len(calls) == {'initial': 1, 'cached': 1, 'threshold': 1,
                             'map': 1, 'geometry': 2, 'damaged': 3}[change]


def test_cpp_ring_masks_preserve_multiple_blocks():
    tth = np.array([np.nan, 0., .1, .5, 1.])
    eta = np.array([-1, 0, 1, 2, 0])
    ranges = np.tile([0., .5], (130, 1))
    ranges[::3] = [.1, 1.]
    expected = np.zeros((3, len(tth)), dtype=np.int64)
    coverage = np.zeros((130, 3), dtype=bool)
    for ring, (lo, hi) in enumerate(ranges):
        for pixel, value in enumerate(tth):
            if eta[pixel] >= 0 and lo <= value <= hi:
                expected[ring//63, pixel] |= 1 << (ring%63)
                coverage[ring, eta[pixel]] = True
    masks, covered = native._ringMasks(tth, eta, ranges, 3)
    np.testing.assert_array_equal(masks, expected)
    np.testing.assert_array_equal(covered, coverage)


@pytest.mark.parametrize('extension', ['unicode', 'prefix', 'zip64', 'comment'])
def test_npz_index_falls_back_for_extended_archives(tmp_path, monkeypatch, extension):
    name = 'dátá' if extension == 'unicode' else 'data'
    expected = np.arange(8)
    stream = io.BytesIO()
    np.save(stream, expected)
    if extension == 'zip64':
        monkeypatch.setattr(zipfile, 'ZIP64_LIMIT', 20)
    path = tmp_path/'extended.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(name+'.npy', stream.getvalue())
        if extension == 'comment':
            archive.comment = b'EOCD signature in a comment: PK\x05\x06'
    if extension == 'prefix':
        path.write_bytes(b'self-extracting-prefix'+path.read_bytes())
    with _NpzReader(path) as reader:
        np.testing.assert_array_equal(reader[name], expected)


def test_npz_metadata_index_expands_on_demand(tmp_path):
    from hexrd.core.utils.npz import savez_compressed

    path = tmp_path/'frames.npz'
    savez_compressed(path, shape=np.array([2, 3]), dtype=np.array(b'uint16'),
                      nframes=np.array(1), omega=np.array([[0., 1.]]),
                      **{'0_data': np.array([40000], dtype=np.uint16)})
    with _NpzReader(path, metadata_only=True) as reader:
        assert reader._metadata_only
        assert len(reader._members) == 4
        np.testing.assert_array_equal(reader['omega'], [[0., 1.]])
        np.testing.assert_array_equal(reader['0_data'], [40000])
        assert not reader._metadata_only
    with np.load(path) as reader:
        np.testing.assert_array_equal(reader['0_data'], [40000])


def test_scoring_cache_views_outlive_reader_and_support_unaligned_bins(tmp_path):
    from hexrd.core.utils.npz import load_array_cache, save_array_cache

    path = tmp_path/'cache.npz'
    expected = np.array([[[0, -1], [1, 2]]], dtype=np.int32)
    save_array_cache(path, 'key', bins=expected)
    data = load_array_cache(path, 'key', mmap=True)
    assert not data['bins'].flags.writeable
    np.testing.assert_array_equal(data['bins'], expected)
    # NPY data inside ZIP members need not satisfy native int32 alignment.
    raw = bytearray(1+expected.nbytes)
    bins = np.ndarray(expected.shape, dtype=np.int32, buffer=raw, offset=1)
    bins[:] = expected
    assert not bins.flags.aligned
    status = np.array([[[1, -1, 0]]], dtype=np.int8)
    np.testing.assert_array_equal(native._scoreOrientationBins(bins, status), [.5])
    np.testing.assert_array_equal(native._scoreOrientationBins(data['bins'], status), [.5])
