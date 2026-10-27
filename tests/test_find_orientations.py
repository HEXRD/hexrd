"""Regression coverage for the find-orientations launch-config workflow."""
from dataclasses import replace
import io
import os
from pathlib import Path
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
    _cached_panel_bin_data, _hit_status_maps, _load_or_generate_fibers, _pack_hit_status,
    _resolve_active_rings,
    find_orientations,
)

LAUE_GROUPS = ('Ci', 'C2h', 'D2h', 'C4h', 'D4h', 'C3i', 'D3d', 'C6h', 'D6h', 'Th', 'Oh')


@pytest.fixture(autouse=True)
def isolate_workflow_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


def test_direct_grid_matches_fundamental_zone_sampler():
    from hexrd.hedm.experiment import DirectSearch
    from hexrd.hedm.find_orientations import generate_orientation_grid
    from hexrd.hedm.sampleOrientations import sampleRFZ

    previous_threads = numba.get_num_threads()
    try:
        for group in (1, 20, 32):
            experiment = SimpleNamespace(max_workers=1, find_orientations=SimpleNamespace(
                direct_search=DirectSearch(30.0)))
            plane = SimpleNamespace(point_group_number=group)
            actual = generate_orientation_grid(experiment, plane)
            expected = sampleRFZ(group, average_angular_spacing=30.0).orientations.T
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(np.linalg.norm(actual, axis=0), 1., atol=1e-14)
            assert actual.shape[0] == 4 and actual.flags.f_contiguous
            assert numba.get_num_threads() == 1
        experiment.find_orientations.direct_search = DirectSearch(30.0, 20)
        np.testing.assert_array_equal(
            generate_orientation_grid(experiment, plane),
            sampleRFZ(20, average_angular_spacing=30.0).orientations.T)
    finally:
        numba.set_num_threads(previous_threads)


@pytest.mark.skipif('HEXRD_EXAMPLE_REPO_PATH' not in os.environ,
                    reason='requires the NIST ruby example data')
def test_direct_search_matches_saved_grid(example_repo_path, tmp_path, monkeypatch, caplog):
    import hexrd.hedm.find_orientations as fo

    def no_seeds(*args, **kwargs):
        pytest.fail('direct search must not generate fibers or estimate seed neighborhoods')

    monkeypatch.setattr(fo, '_load_or_generate_fibers', no_seeds)
    monkeypatch.setattr(fo, '_cached_min_samples', no_seeds)
    include = example_repo_path / 'NIST_ruby/single_GE/include'
    config = next(yaml.safe_load_all((include / 'cstudy.yml').read_text()))
    config['analysis_name'] = str(tmp_path / 'results')
    config['multiprocessing'] = 1
    config['find_orientations'].pop('seed_search')
    config['find_orientations']['direct_search'] = {'average_angular_spacing': 10.0}
    filename = tmp_path / 'config.yml'
    filename.write_text(yaml.safe_dump(config))
    previous_threads = numba.get_num_threads()
    try:
        experiment = HedmExperiment(str(filename), working_dir=str(include))
        direct = fo.find_orientations(experiment, clean=True)
        assert 'coarser than the eta/omega tolerances' in caplog.text
        assert direct.test_orientations.shape[1] > 0
        assert np.any(direct.completeness > 0)
        fo.write_results(direct, experiment)
        outputs = {p.name: p.read_bytes() for p in Path(experiment.analysis_dir).iterdir()}
        monkeypatch.setattr(experiment, 'get_active_material', no_seeds)
        cached = fo.find_orientations(experiment)
        np.testing.assert_array_equal(cached.test_orientations, direct.test_orientations)
        np.testing.assert_array_equal(cached.completeness, direct.completeness)
        grid = tmp_path / 'grid.npy'
        np.save(grid, direct.test_orientations)
        config['find_orientations'].pop('direct_search')
        config['find_orientations']['use_quaternion_grid'] = str(grid)
        config['find_orientations']['seed_search'] = {'hkl_seeds': [999]}
        filename.write_text(yaml.safe_dump(config))
        experiment = HedmExperiment(str(filename), working_dir=str(include))
        saved = fo.find_orientations(experiment)
        np.testing.assert_array_equal(saved.test_orientations, direct.test_orientations)
        np.testing.assert_array_equal(saved.completeness, direct.completeness)
        np.testing.assert_array_equal(saved.grain_orientations, direct.grain_orientations)
        fo.write_results(saved, experiment)
        assert {p.name: p.read_bytes() for p in Path(experiment.analysis_dir).iterdir()} == outputs
    finally:
        numba.set_num_threads(previous_threads)


@pytest.mark.parametrize('step', [0.1, 0.25, 1.0])
def test_eta_grid_preserves_master_arithmetic(step):
    from hexrd.core.instrument import PlanarDetector
    from hexrd.hedm.find_orientations import _eta_bin_edges

    expected = PlanarDetector().make_powder_rings(
        [5.], delta_tth=0.25, delta_eta=step, full_output=True)[-1]
    np.testing.assert_array_equal(_eta_bin_edges(step), expected)


@pytest.mark.parametrize('symmetry,divisions,b_matrix', [
    *((symmetry, 360, np.eye(3)) for symmetry in LAUE_GROUPS),
    ('Oh', 1440, np.array([[.2, .03, -.07], [0., .3, .04], [0., 0., .4]])),
])
def test_prepared_fibers_match_discrete_fiber(symmetry, divisions, b_matrix):
    from hexrd.core import rotations, matrixutil
    from hexrd.core.transforms import xfcapi
    from hexrd.hedm.find_orientations import _prepare_fiber, _fiber

    operators = rotations.quatProductMatrix(rotations.quatOfLaueGroup(symmetry), 'right')
    for angles in np.random.default_rng(0).uniform(-1., 1., size=(3, 3)):
        sample = xfcapi.angles_to_gvec(np.atleast_2d(angles), chi=0.2).T
        for hkl in (np.ones(3), sample.ravel(), -sample.ravel()):
            expected = rotations.discreteFiber(hkl, sample, B=b_matrix, ndiv=divisions, csym=symmetry)[0]
            prepared = _prepare_fiber(hkl, b_matrix, divisions)
            np.testing.assert_array_equal(_fiber(*prepared, *angles, 0.2, operators),
                                          matrixutil.uniqueVectors(expected))


@pytest.mark.parametrize('n', [0, 1, 3, 5, 8])
def test_native_fiber_reduction_tails_and_views(n):
    from hexrd.core import rotations, matrixutil

    csym = rotations.quatOfLaueGroup('Oh')
    operators = rotations.quatProductMatrix(csym, 'right')
    fibers = np.random.default_rng(81).normal(size=(4, n))
    if n:
        fibers[:, 0] = [1., 0., 0., 0.]
    views = ((fibers, np.asfortranarray(fibers), fibers[:, ::-1], fibers[::-1])
             if n == 5 else (fibers,))
    for view in views:
        # Generated fibers and the native dot kernel use contiguous components.
        expected = (rotations.toFundamentalRegion(np.asfortranarray(view), csym)
                    if n else np.empty((4, 0)))
        if n > 1:
            expected = matrixutil.uniqueVectors(expected)
        np.testing.assert_array_equal(native._reduceFiber(view, operators), expected)


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
    saved_outputs = None
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
            fo.write_results(result, experiment)
            directory = Path(experiment.analysis_dir)
            assert {p.name for p in directory.iterdir()} == {
                Path(experiment.eta_ome_maps_file).name,
                Path(experiment.scored_orientations_file).name,
                Path(experiment.accepted_orientations_file).name, 'grains.out'}
            assert result.test_orientations.flags.f_contiguous
            with zipfile.ZipFile(experiment.scored_orientations_file) as archive:
                assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in archive.infolist())
            with zipfile.ZipFile(experiment.eta_ome_maps_file) as archive:
                assert archive.namelist() == [name + '.npy' for name in (
                    'dataStore', 'etas', 'etaEdges', 'iHKLList', 'omegas', 'omeEdges',
                    'planeData_args', 'planeData_hkls', 'planeData_excl')]
            outputs = {p.name: p.read_bytes() for p in directory.iterdir()}
            if not clean:
                assert outputs == saved_outputs
                assert experiment.image_series_list[0]._images is None
            saved_outputs = outputs
    finally:
        numba.set_num_threads(previous_threads)
    assert len(scored) == 2
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


def _reference_scores(angles, args, return_bins=False):
    """Independent scalar oracle, using the established diffraction predictor."""
    scores = []
    bins = np.full(angles.shape[:3], -1, dtype=np.int32)
    for iq, solutions in enumerate(angles):
        hits = []
        for branch, solution in enumerate(solutions):
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
                    bins[iq, branch, h] = (args[11][h]*args[10].shape[1]+o)*args[10].shape[2]+e
                    hit = _reference_hit(e, o, args[12], args[13],
                                         args[10][args[11][h]], args[14])
                    if hit >= 0:
                        hits.append(hit)
        scores.append(sum(hits) / len(hits) if hits else 0.)
    return (np.asarray(scores), bins) if return_bins else np.asarray(scores)


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


@pytest.mark.parametrize('count,beam,grid,drift,wavelength,paired', [
    (0, [0., 0., -1.], 'regular', 0., .15, False),
    (1, [0., 0., -1.], 'regular', 0., .15, False),
    (97, [0., 0., -1.], 'regular', 0., .15, False),
    (97, [0., 0., -1.], 'paired_regular', 0., .15, True),
    (97, [0., 0., -1.], 'shifted_regular', 0., .15, True),
    (97, [0., 0., -1.], 'shifted_period', 0., .15, True),
    (97, [0., 0., -1.], 'positive_maps', 0., .15, True),
    (97, [0., 0., -1.], 'masked', 0., .15, True),
    (97, [0., 0., -1.], 'nonuniform', 0., .15, True),
    (97, [-6e-17, -6e-17, -1.], 'on_edge', 0., .15, True),
    (97, [-6e-17, -6e-17, -1.], 'below_edge', 0., .15, True),
    (97, [-6e-17, -6e-17, -1.], 'above_edge', 0., .15, True),
    (97, [.2, -.1, -1.], 'regular', 0., .15, True),
    (97, [0., 0., -1.], 'paired_regular', 1e-15, .15, True),
    (97, [0., 0., -1.], 'paired_regular', 1e-12, .15, True),
    (97, [0., 0., -1.], 'regular', 0., 8., False),
    (97, [0., 0., -1.], 'masked', 0., 8., True),
    (97, [.2, -.1, -1.], 'nonuniform', 0., 8., True),
])
def test_native_scoring_matches_reference(count, beam, grid, drift, wavelength, paired):
    args = _scoring_inputs(count, beam)
    if grid in ('paired_regular', 'shifted_regular', 'shifted_period', 'positive_maps'):
        args[10] = args[10][:, :20].copy()
        args[6] = np.linspace(-np.pi, np.pi, 21)
        if grid.startswith('shifted'):
            args[9] = 0.6
        if grid == 'shifted_regular':
            args[6] = np.linspace(args[9], args[9]+2*np.pi, 21)
    if grid == 'positive_maps':
        args[6] = np.linspace(0., 2*np.pi, 21)
        args[8] = np.array([-np.pi, np.pi])
    if grid == 'masked':
        sine = np.abs(np.sin((args[5][:-1]+args[5][1:])/2))
        args[10][:, :, (sine < 0.35) | (sine > 0.85)] = np.nan
    if paired:
        args[1] = np.vstack([args[1], -args[1]])
        args[11] = np.tile(args[11], 2)
    args[3] = wavelength
    rmats, hkls, bmat, wavelength, beam = args[:5]
    rmats *= 1. + drift
    angles = np.empty((count, 2, len(hkls), 3))
    for i, rmat in enumerate(rmats):
        angles[i] = native.oscillAnglesOfHKLs(
            hkls, 0.0, rmat, bmat, wavelength,
            np.array([1., 1, 1, 0, 0, 0]), beam, np.array([1., 0, 0]))
    if grid == 'nonuniform':
        rng = np.random.default_rng(18)
        for index in (5, 6):
            args[index][1:-1] = np.sort(rng.uniform(-np.pi, np.pi, len(args[index])-2))
    elif grid.endswith('edge') and count:
        for index, coordinate in ((5, 1), (6, 2)):
            values = (angles[0, :, :, coordinate].ravel() + np.pi) % (2*np.pi) - np.pi
            values = values[np.isfinite(values)]
            if grid != 'on_edge':
                values = np.nextafter(values, -np.inf if grid == 'below_edge' else np.inf)
            args[index][1:1+len(values)] = values
            args[index].sort()
    expected, expected_bins = _reference_scores(angles, args, return_bins=True)
    np.testing.assert_array_equal(native.scoreOrientations(*args), expected)
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    np.testing.assert_array_equal(native.scoreOrientations(*args, status), expected)
    packed = _pack_hit_status(status)
    np.testing.assert_array_equal(native.scoreOrientations(*args, packed), expected)
    bins = np.empty((count, 2, len(hkls)), dtype=np.int32)
    np.testing.assert_array_equal(native.scoreOrientations(*args, status, bins), expected)
    np.testing.assert_array_equal(bins, expected_bins)
    np.testing.assert_array_equal(native.scoreOrientations(*args, packed, bins), expected)
    np.testing.assert_array_equal(bins, expected_bins)
    np.testing.assert_array_equal(native._scoreOrientationBins(bins, status), expected)
    args[14] = 0.9
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    np.testing.assert_array_equal(native._scoreOrientationBins(bins, status),
                                  _reference_scores(angles, args))


@pytest.mark.parametrize('size,wavelength', [(1, .15), (2, .15), (4, .15), (4, 8.)])
def test_scoring_unpaired_opposite_duplicate_and_zero_hkls(size, wavelength):
    args = _scoring_inputs(31, [0., 0., -1.])
    rng = np.random.default_rng(41)
    hkls = rng.integers(-3, 4, size=(17, 3)).astype(float)
    hkls[:4] = [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]]
    args[1] = hkls[:size]
    args[11] = np.arange(size, dtype=np.intp) % 2
    args[3] = wavelength
    args[6] = np.linspace(0., 2*np.pi, 22)
    angles = np.array([native.oscillAnglesOfHKLs(
        args[1], 0., r, args[2], wavelength, np.array([1., 1, 1, 0, 0, 0]),
        args[4], np.array([1., 0, 0])) for r in args[0]])
    expected, expected_bins = _reference_scores(angles, args, return_bins=True)
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    for lookup in (status, _pack_hit_status(status)):
        bins = np.empty_like(expected_bins)
        np.testing.assert_array_equal(native.scoreOrientations(*args, lookup, bins), expected)
        np.testing.assert_array_equal(bins, expected_bins)


@pytest.mark.parametrize('count,layout', [
    (0, 'C'), (1, 'C'), (97, 'C'), (97, 'F'), (97, 'slice'),
    (97, 'reverse'), (97, 'reverse_components'),
])
def test_scoring_quaternion_views_match_matrices(count, layout):
    q = np.random.default_rng(52).normal(size=(count, 4)).T.copy()
    q /= np.linalg.norm(q, axis=0)
    if layout == 'F':
        q = np.asfortranarray(q)
    elif layout == 'slice':
        storage = np.empty((8, 2*count))
        storage[::2, ::2] = q
        q = storage[::2, ::2]
    elif layout == 'reverse':
        q = q[:, ::-1]
    elif layout == 'reverse_components':
        q = q[::-1]
    args = _scoring_inputs(count, [0., 0., -1.])
    args[0] = native.quaternionRotations(np.ascontiguousarray(q.T))
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    expected_bins = np.empty((count, 2, len(args[1])), dtype=np.int32)
    expected = native.scoreOrientations(*args, status, expected_bins)
    args[0] = q
    bins = np.empty_like(expected_bins)
    for lookup in (status, _pack_hit_status(status)):
        np.testing.assert_array_equal(native.scoreOrientations(*args, lookup), expected)
        np.testing.assert_array_equal(native.scoreOrientations(*args, lookup, bins), expected)
        np.testing.assert_array_equal(bins, expected_bins)


@pytest.mark.parametrize('count,positive_maps,paired', [
    (3, False, False), (4, False, False), (5, False, True),
    (7, True, True), (8, True, False), (1001, True, True),
])
def test_four_orientation_scoring_matches_matrix_path(count, positive_maps, paired):
    args = _scoring_inputs(count, [0., 0., -1.])
    q = np.random.default_rng(52).normal(size=(count, 4)).T.copy()
    q /= np.linalg.norm(q, axis=0)
    if count >= 4:
        q[:, :2] = np.array([[1.], [0.], [0.], [0.]])
    args[0] = native.quaternionRotations(np.ascontiguousarray(q.T))
    if paired:
        args[1] = np.vstack((args[1], -args[1]))
        args[11] = np.tile(args[11], 2)
    args[10] = np.random.default_rng(14).random((2, 8, 8))
    args[10][:, ::3, ::2] = np.nan
    args[12] = args[13] = 0
    args[5] = np.linspace(-np.pi, np.pi, 9)
    args[6] = np.linspace(0., 2*np.pi, 9) if positive_maps else args[5].copy()
    args[7] = args[8] = np.array([-np.pi, np.pi])
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    expected = native.scoreOrientations(*args, status)
    args[0] = q
    np.testing.assert_array_equal(native.scoreOrientations(*args, _pack_hit_status(status)), expected)
    q *= 1. + 1e-12
    args[0] = native.quaternionRotations(np.ascontiguousarray(q.T))
    expected = native.scoreOrientations(*args, status)
    args[0] = q
    np.testing.assert_array_equal(native.scoreOrientations(*args, _pack_hit_status(status)), expected)


@pytest.mark.parametrize('boundary', ['eta', 'omega', 'accepted_eta', 'accepted_omega'])
@pytest.mark.parametrize('positive_maps', [False, True])
def test_vector_scoring_recomputes_individual_boundaries(boundary, positive_maps):
    args = _scoring_inputs(97, [-6e-17, -6e-17, -1.])
    q = np.random.default_rng(52).normal(size=(97, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    args[1] = np.vstack((args[1], -args[1]))
    args[11] = np.tile(args[11], 2)
    if positive_maps:
        args[6] = np.linspace(0., 2*np.pi, 22)
    args[7] = args[8] = np.array([-np.pi, np.pi])
    args[12] = args[13] = 0
    angles = np.array([native.oscillAnglesOfHKLs(
        args[1], 0., r, args[2], args[3], np.array([1., 1, 1, 0, 0, 0]),
        args[4], np.array([1., 0, 0])) for r in args[0]])
    points = (angles[0].reshape(-1, 3)[:, 1:] + np.pi) % (2*np.pi) - np.pi
    point = points[points[:, 1] > 0][0]
    coordinate = 0 if boundary.endswith('eta') else 1
    value = point[coordinate]
    edges = args[5+coordinate].copy()
    step = edges[1]-edges[0]
    for edge in (np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf)):
        if boundary.startswith('accepted'):
            args[7+coordinate] = np.array([-np.pi, edge])
        else:
            args[5+coordinate] = edges + edge - (edges[0]+round((edge-edges[0])/step)*step)
        expected = _reference_scores(angles, args)
        args[0] = q.T
        status = _hit_status_maps(args[10], args[12], args[13], args[14])
        np.testing.assert_array_equal(native.scoreOrientations(*args, _pack_hit_status(status)),
                                      expected)


@pytest.mark.parametrize('offset', [-np.pi, 17.])
def test_native_scoring_nonorthogonal_lattice(offset):
    b_matrix = np.array([[.2, .03, -.07], [0., .3, .04], [0., 0., .4]])
    args = _scoring_inputs(65, [0., 0., -1.])
    args[2] = b_matrix
    # Exercise both SIMD scoring and its multi-period wrapping fallback.
    args[6] = np.linspace(offset, offset+2*np.pi, 22)
    args[8] = np.array([offset, offset+2*np.pi])
    args[9] = offset
    args[12] = args[13] = 0
    angles = np.array([native.oscillAnglesOfHKLs(
        args[1], 0., r, b_matrix, args[3], np.array([1., 1, 1, 0, 0, 0]),
        args[4], np.array([1., 0, 0])) for r in args[0]])
    expected = _reference_scores(angles, args)
    q = np.random.default_rng(52).normal(size=(65, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    args[0] = q.T
    status = _hit_status_maps(args[10], args[12], args[13], args[14])
    np.testing.assert_array_equal(native.scoreOrientations(*args, _pack_hit_status(status)), expected)


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
                   'Oh', np.array([[1.], [0], [0], [0]]), np.eye(3), 0.15,
                   master_hkl_ids=np.array([7]), lattice_parameters=np.array([1.]),
                   strain_magnitude=0.01, point_group_number=32)
    calls = []

    def load():
        calls.append(1)
        return SimpleNamespace(plane_data=pd)

    settings = {'dmin': 1.6, 'energy': 80.0}
    assert load_plane_data_cached(definitions, cache, settings, load) is pd
    restored = load_plane_data_cached(definitions, cache, settings, load)
    assert len(calls) == 1
    for name in ('hkls', 'exclusions', 'unexcluded_hkls', 'two_thetas', 'q_sym', 'B', 'master_hkl_ids', 'lattice_parameters'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(pd, name))
    assert restored.strain_magnitude == pd.strain_magnitude
    assert restored.point_group_number == 32 and isinstance(restored.point_group_number, int)
    np.testing.assert_array_equal(_resolve_active_rings(np.array([7]), restored), [0])
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


@pytest.mark.parametrize('unsorted', [False, True])
def test_streamed_pixels_match_frames_without_loading_images(tmp_path, unsorted):
    arrays = dict(shape=np.array([3, 4]), dtype=np.array(b'uint16'),
                  nframes=np.array(4), omega=np.array([[i, i + 1] for i in range(4)]))
    for i in range(4):
        order = [2, 0, 1] if unsorted else [0, 1, 2]
        arrays.update({f'{i}_row': np.array([0, 1, 2])[order],
                       f'{i}_col': np.array([2, 1, 3])[order],
                       f'{i}_data': np.array([i, 40000, 40000], dtype=np.uint16)[order]})
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


@pytest.mark.parametrize('value,threshold,expected', [
    (40000, 0., 80000), (40000, 70000., 80000), (5, 8., 10),
])
def test_streaming_preserves_duplicate_uint16_weights(tmp_path, value, threshold, expected):
    from hexrd.hedm.find_orientations import _bin_pixels

    path = tmp_path / 'duplicates.npz'
    np.savez_compressed(
        path, shape=np.array([1, 1]), dtype=np.array(b'uint16'), nframes=np.array(1),
        omega=np.array([[0., 1.]]), **{
            '0_data': np.array([value, value], dtype=np.uint16),
            '0_row': np.array([0, 0]), '0_col': np.array([0, 0])})
    series = ImageSeries('p', str(path), max_workers=1)
    result = np.zeros((1, 1, 1))
    for ids, values, offsets, rows in series.pixel_chunks(np.array([0])):
        _bin_pixels(offsets, ids, values, rows, np.array([1]), np.array([0]), threshold, result)
    assert result[0, 0, 0] == expected
    assert series._images is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.bool_])
def test_streamed_duplicate_pixels_preserve_csr_dtype_semantics(tmp_path, dtype):
    from scipy.sparse import csr_array
    from hexrd.hedm.find_orientations import _bin_pixels

    data = np.array([.1, .2], dtype=dtype)
    indices = np.array([0, 0])
    pixel = float(csr_array((data, (indices, indices)), shape=(1, 1)).data[0])
    path = tmp_path / 'duplicates.npz'
    np.savez(path, shape=[1, 1], dtype=np.array(data.dtype.name.encode()), nframes=1,
             omega=[[0., 1.]], **{'0_data': data, '0_row': indices, '0_col': indices})
    series = ImageSeries('p', str(path), max_workers=1)
    result = np.zeros((1, 1, 1))
    for ids, values, offsets, rows in series.pixel_chunks(np.array([0])):
        _bin_pixels(offsets, ids, values, rows, np.array([1]), np.array([0]),
                    np.nextafter(pixel, -np.inf), result)
    assert result.item() == pixel
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


@pytest.mark.parametrize('workers', [1, 2])
def test_terminal_scoring_progress_preserves_results(tmp_path, monkeypatch, workers):
    import hexrd.hedm.find_orientations as fo

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    args = _scoring_inputs(0, [0., 0., -1.])
    quats = np.random.default_rng(52).normal(size=(4, 524291))
    quats /= np.linalg.norm(quats, axis=0)
    maps = SimpleNamespace(ring_maps=args[10], ring_ids=np.arange(2),
                           eta_edges=args[5], omega_edges=args[6])
    plane = SimpleNamespace(symm_hkls=[args[1][:2].T, args[1][2:].T],
                            B=args[2], wavelength=args[3])
    settings = SimpleNamespace(threshold=args[14],
        eta=SimpleNamespace(range=args[7].reshape(-1, 2), tolerance=2*np.pi/32),
        omega=SimpleNamespace(period=np.array([-np.pi, np.pi]), tolerance=4*np.pi/21))
    experiment = SimpleNamespace(find_orientations=settings, max_workers=workers,
        beam=SimpleNamespace(vector=args[4]), analysis_dir=str(tmp_path / 'quiet'))
    quiet = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', quiet)
    expected = fo.score_orientations(experiment, plane, maps, quats)
    assert quiet.getvalue() == ''
    terminal = Terminal()
    monkeypatch.setattr(sys, 'stderr', terminal)
    experiment.analysis_dir = str(tmp_path / 'terminal')
    actual = fo.score_orientations(experiment, plane, maps, quats)
    np.testing.assert_array_equal(actual, expected)
    assert 'Scoring orientations' in terminal.getvalue()
    assert '100%' in terminal.getvalue()


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
    for change in ('initial', 'cached', 'threshold', 'map', 'geometry', 'damaged', 'lazy'):
        if change == 'threshold':
            settings.threshold = args[14] = 0.95
        elif change == 'map':
            args[10][:] = 1.
        elif change == 'geometry':
            plane.B[0, 0] *= 1.02
        elif change == 'damaged':
            (Path(fo._cache_dir(experiment))/'reflection-bins.npz').write_bytes(b'broken cache')
        elif change == 'lazy':
            (Path(fo._cache_dir(experiment))/'reflection-bins.npz').unlink()
            lazy = object.__new__(fo._LazyEtaOmegaMaps)
            lazy.__dict__.update(vars(maps))
            lazy._content_key = fo._map_key(maps)
            lazy.ring_maps = lambda: pytest.fail('cached scoring loaded dense maps')
            maps = lazy
        np.testing.assert_array_equal(fo.score_orientations(experiment, plane, maps, quats.T),
                                      predict(*args))
        assert len(calls) == {'initial': 1, 'cached': 1, 'threshold': 1,
                             'map': 1, 'geometry': 2, 'damaged': 3, 'lazy': 4}[change]


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
