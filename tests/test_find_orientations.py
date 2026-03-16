import os
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import coloredlogs


from hexrd.hedm.findorientations import (
    SeedReflection,
    SeedPeak,
    _candidate_quaternions_from_pairwise_intersections,
    _match_predicted_seed_peaks,
    merge_orientations_by_misorientation,
    _pair_friedel_seed_peaks,
    _predict_friedel_pair_angles,
    find_orientations,
    generate_eta_ome_maps,
)
from hexrd.hedm import config

# TODO: Check that this test is still sensible after PlaneData change.
from hexrd.core.material.crystallography import PlaneData
from hexrd.core import rotations as rot

import find_orientations_testing as test_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = coloredlogs.ColoredFormatter(
    '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@pytest.fixture
def example_repo_include_path(example_repo_path):
    return example_repo_path / 'NIST_ruby' / 'multiruby_dexelas' / 'include'


@pytest.fixture
def example_repo_results_path(example_repo_path):
    return example_repo_path / 'NIST_ruby' / 'multiruby_dexelas' / 'results'


@pytest.fixture
def example_repo_config_path(example_repo_include_path):
    return example_repo_include_path / 'mruby_config_composite.yml'


@pytest.fixture
def test_config(example_repo_config_path, example_repo_include_path):
    conf = config.open(example_repo_config_path)[0]
    conf.working_dir = str(example_repo_include_path)

    return conf


@pytest.fixture
def reference_eta_ome_maps(example_repo_results_path):
    filename = 'results_mruby_composite_hexrd06_py27_ruby_eta-ome_maps.npz'
    return example_repo_results_path / filename


@pytest.fixture
def example_repo_config_with_eta_ome_maps(test_config, reference_eta_ome_maps):
    # Set eta omega maps file
    cfg = test_config._cfg.copy()

    results_path = str(Path('../results') / reference_eta_ome_maps.name)
    cfg['find_orientations']['orientation_maps']['file'] = results_path

    patch_config = config.root.RootConfig(cfg)

    return patch_config


@pytest.fixture
def reference_orientations_path(example_repo_results_path):
    filename = (
        'accepted_orientations_results_mruby_composite_hexrd06_py27_ruby.dat'
    )
    return example_repo_results_path / filename


@pytest.fixture
def reference_orientations(reference_orientations_path):
    return np.loadtxt(reference_orientations_path, ndmin=2)


def plane_data(plane_data):
    args = plane_data.getParams()[:4]
    hkls = plane_data.hkls

    return PlaneData(hkls, *args)


def to_eomap(eta_ome_maps):
    return test_utils.EOMap(
        np.array(eta_ome_maps.dataStore),
        eta_ome_maps.etas,
        eta_ome_maps.etaEdges,
        eta_ome_maps.omegas,
        eta_ome_maps.omeEdges,
        eta_ome_maps.iHKLList,
        plane_data(eta_ome_maps.planeData),
    )


def test_generate_eta_ome_maps(
    example_repo_include_path, test_config, reference_eta_ome_maps
):
    os.chdir(example_repo_include_path)
    eta_ome_maps = generate_eta_ome_maps(test_config, save=False)
    eta_ome_maps = to_eomap(eta_ome_maps)

    expected = test_utils.load(reference_eta_ome_maps)
    comparison = test_utils.Comparison(expected, eta_ome_maps)
    assert comparison.compare()


def test_find_orientations(
    example_repo_include_path,
    example_repo_config_with_eta_ome_maps,
    reference_orientations,
):

    os.chdir(example_repo_include_path)
    results = find_orientations(example_repo_config_with_eta_ome_maps)

    orientations = results['qbar']

    try:
        test_utils.compare_quaternion_lists(
            orientations.T, reference_orientations
        )
    except RuntimeError as err:
        pytest.fail(str(err))


def test_pairwise_intersection_candidates():
    identity_qsym = np.array([[1.0], [0.0], [0.0], [0.0]])

    c1 = np.array([1.0, 0.0, 0.0])
    c2 = np.array([0.0, 1.0, 0.0])
    rmat = rot.rotMatOfExpMap(np.radians(np.array([12.0, -8.0, 17.0])))
    s1 = np.dot(rmat, c1)
    s2 = np.dot(rmat, c2)

    reflections = [
        SeedReflection(
            seed_index=0,
            hkl_id=0,
            hkl=np.array([1.0, 0.0, 0.0]),
            tth=0.0,
            eta=0.0,
            ome=0.0,
            gvec_s=s1,
        ),
        SeedReflection(
            seed_index=1,
            hkl_id=1,
            hkl=np.array([0.0, 1.0, 0.0]),
            tth=0.0,
            eta=0.0,
            ome=0.0,
            gvec_s=s2,
        ),
    ]
    seed_crystal_dirs = [c1.reshape(3, 1), c2.reshape(3, 1)]

    candidates, raw_count, counts = _candidate_quaternions_from_pairwise_intersections(
        reflections,
        seed_crystal_dirs,
        identity_qsym,
        identity_qsym,
        1.0e-6,
        10,
    )

    assert raw_count == 1
    assert counts.tolist() == [1]
    assert candidates.shape == (4, 1)
    np.testing.assert_allclose(rot.rotMatOfQuat(candidates[:, 0]), rmat, atol=1.0e-8)


def test_pair_friedel_seed_peaks():
    tth = np.radians(35.0)
    eta = np.radians(22.0)
    ome = np.radians(-47.0)
    chi = 0.0

    partner_ome, partner_eta = _predict_friedel_pair_angles(
        tth,
        np.array([eta]),
        np.array([ome]),
        chi=chi,
    )

    peaks = [
        SeedPeak(eta=eta, ome=ome, intensity=10.0),
        SeedPeak(
            eta=float(partner_eta[0]),
            ome=float(partner_ome[0]),
            intensity=6.0,
        ),
        SeedPeak(
            eta=np.radians(-90.0),
            ome=np.radians(10.0),
            intensity=2.0,
        ),
    ]

    reduced = _pair_friedel_seed_peaks(
        peaks,
        tth=tth,
        chi=chi,
        eta_tol=np.radians(0.25),
        ome_tol=np.radians(0.25),
    )

    assert len(reduced) == 2
    supports = sorted(peak.support for peak in reduced)
    assert supports == [1, 2]
    paired_peak = next(peak for peak in reduced if peak.support == 2)
    assert paired_peak.intensity == pytest.approx(16.0)
    assert paired_peak.eta == pytest.approx(eta)
    assert paired_peak.ome == pytest.approx(ome)


def test_match_predicted_seed_peaks():
    reflections = [
        SeedReflection(
            seed_index=0,
            hkl_id=11,
            hkl=np.array([1.0, 0.0, 0.0]),
            tth=0.0,
            eta=np.radians(10.0),
            ome=np.radians(-20.0),
            gvec_s=np.array([1.0, 0.0, 0.0]),
            support=2,
        ),
        SeedReflection(
            seed_index=1,
            hkl_id=13,
            hkl=np.array([0.0, 1.0, 0.0]),
            tth=0.0,
            eta=np.radians(-35.0),
            ome=np.radians(45.0),
            gvec_s=np.array([0.0, 1.0, 0.0]),
            support=1,
        ),
    ]
    predicted_by_hkl = {
        11: [SeedPeak(eta=np.radians(10.1), ome=np.radians(-20.1), intensity=1.0)],
        13: [SeedPeak(eta=np.radians(-34.9), ome=np.radians(45.1), intensity=1.0)],
    }
    observed_by_hkl = {11: [0], 13: [1]}

    support_mask, predicted_total, matched_total, matched_support, seed_support = (
        _match_predicted_seed_peaks(
            reflections,
            np.array([True, True]),
            predicted_by_hkl,
            observed_by_hkl,
            eta_tol=np.radians(0.5),
            ome_tol=np.radians(0.5),
        )
    )

    assert support_mask.tolist() == [True, True]
    assert predicted_total == 2
    assert matched_total == 2
    assert matched_support == 3
    assert seed_support == 2


def test_merge_orientations_by_misorientation():
    identity_qsym = np.array([[1.0], [0.0], [0.0], [0.0]])

    q0 = rot.quatOfExpMap(np.radians(np.array([0.0, 0.0, 0.0]))).flatten()
    q1 = rot.quatOfExpMap(np.radians(np.array([0.0, 0.0, 0.35]))).flatten()
    q2 = rot.quatOfExpMap(np.radians(np.array([0.0, 0.0, 4.0]))).flatten()

    qfib = np.column_stack([q0, q1, q2])
    completeness = np.array([0.96, 0.93, 0.91])

    qbar, labels = merge_orientations_by_misorientation(
        completeness,
        qfib,
        identity_qsym,
        compl_thresh=0.9,
        radius=1.0,
    )

    assert qbar.shape == (4, 2)
    assert labels.tolist() == [1, 1, 2]
    np.testing.assert_allclose(
        rot.rotMatOfQuat(qbar[:, 1]),
        rot.rotMatOfQuat(q2),
        atol=1.0e-8,
    )
