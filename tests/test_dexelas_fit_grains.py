import os
import logging
import sys

import numpy as np
import pytest
import coloredlogs


from hexrd import config
from hexrd.fitgrains import fit_grains


from fit_grains_check import compare_grain_fits


root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = coloredlogs.ColoredFormatter(
    '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


@pytest.fixture
def dexelas_path(example_repo_path):
    return example_repo_path / 'NIST_ruby/multiruby_dexelas'


@pytest.fixture
def dexelas_include_path(dexelas_path):
    return dexelas_path / 'include'


@pytest.fixture
def dexelas_results_path(dexelas_path):
    return dexelas_path / 'results'


@pytest.fixture
def dexelas_config_path(dexelas_include_path):
    return dexelas_include_path / 'mruby_config_composite_roi.yml'


@pytest.fixture
def grains_find_orientations_file_path(dexelas_results_path):
    return dexelas_results_path / 'composite_roi_find_orientations_grains.out'


@pytest.fixture
def grains_reference_file_path(dexelas_results_path):
    return dexelas_results_path / 'composite_roi_fit_grains_grains.out'


@pytest.fixture
def test_config(dexelas_config_path, dexelas_include_path):
    conf = config.open(dexelas_config_path)[0]
    conf.working_dir = dexelas_include_path

    return conf


@pytest.fixture
def spots_data_file_path(example_repo_path):
    return example_repo_path / 'spot_finding/spots_file.h5'


def test_dexelas_fit_grains(dexelas_include_path, test_config,
                            grains_find_orientations_file_path,
                            grains_reference_file_path):
    os.chdir(str(dexelas_include_path))

    input_grains_table = np.loadtxt(
        grains_find_orientations_file_path,
        ndmin=2,
    )
    ref_grains_table = np.loadtxt(grains_reference_file_path, ndmin=2)
    ref_grain_params = ref_grains_table[:, 3:15]

    gresults = fit_grains(test_config,
                          input_grains_table,
                          show_progress=False,
                          ids_to_refine=None,
                          write_spots_files=False)

    output_grain_params = np.vstack([i[-1] for i in gresults])
    assert compare_grain_fits(
        output_grain_params, ref_grain_params,
        # Maybe we should tighten these? Well, at least this
        # verifies that the results aren't way off.
        mtol=1.e-2, ctol=1.e-1, vtol=1.e-3
    )


def test_dexelas_fit_grains_from_spots_data(
    dexelas_include_path, test_config, grains_find_orientations_file_path,
    grains_reference_file_path, spots_data_file_path
):
    os.chdir(str(dexelas_include_path))

    input_grains_table = np.loadtxt(
        grains_find_orientations_file_path,
        ndmin=2,
    )
    ref_grains_table = np.loadtxt(grains_reference_file_path, ndmin=2)
    ref_grain_params = ref_grains_table[:, 3:15]

    # Set the spots data file and run
    # This will skip `pull_spots()`
    # FIXME: verify it is much faster
    test_config.fit_grains.spots_data_file = spots_data_file_path

    gresults = fit_grains(test_config,
                          input_grains_table,
                          show_progress=False,
                          ids_to_refine=None,
                          write_spots_files=False)

    output_grain_params = np.vstack([i[-1] for i in gresults])
    assert compare_grain_fits(
        output_grain_params, ref_grain_params,
        # Maybe we should tighten these? Well, at least this
        # verifies that the results aren't way off.
        mtol=1.e-3, ctol=1.e-2, vtol=1.e-3
    )
