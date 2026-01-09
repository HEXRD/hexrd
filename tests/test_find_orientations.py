import os
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import coloredlogs


from hexrd.hedm.findorientations import (
    find_orientations,
    generate_eta_ome_maps,
)
from hexrd.hedm import config

# TODO: Check that this test is still sensible after PlaneData change.
from hexrd.core.material.crystallography import PlaneData

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
