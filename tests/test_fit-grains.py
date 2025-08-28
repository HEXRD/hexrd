import os
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import coloredlogs


from hexrd.hedm import config
from hexrd.hedm.fitgrains import fit_grains


from fit_grains_check import compare_grain_fits


root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = coloredlogs.ColoredFormatter(
    '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
root.addHandler(handler)


@pytest.fixture
def single_ge_path(example_repo_path):
    return example_repo_path / 'NIST_ruby' / 'single_GE'


@pytest.fixture
def single_ge_include_path(single_ge_path):
    return single_ge_path / 'include'


@pytest.fixture
def single_ge_results_path(single_ge_path):
    return single_ge_path / 'results'


@pytest.fixture
def single_ge_config_path(single_ge_include_path):
    return single_ge_include_path / 'ruby_config.yml'


@pytest.fixture
def analysis_path(tmpdir):
    analysis_path = Path(tmpdir) / 'analysis'

    analysis_path.mkdir()

    return analysis_path


@pytest.fixture
def grains_file_path(analysis_path):
    return analysis_path / 'grains.out'


@pytest.fixture
def grains_reference_file_path(single_ge_results_path):
    return single_ge_results_path / 'ruby-b035e' / 'scan-0' / 'grains.out'


@pytest.fixture
def test_config(single_ge_config_path, single_ge_include_path):
    conf = config.open(single_ge_config_path)[0]
    conf.working_dir = single_ge_include_path

    return conf


def test_fit_grains(
    single_ge_include_path,
    test_config,
    grains_file_path,
    grains_reference_file_path,
):
    os.chdir(str(single_ge_include_path))

    grains_table = np.loadtxt(grains_reference_file_path, ndmin=2)
    ref_grain_params = grains_table[:, 3:15]
    gresults = fit_grains(
        test_config,
        grains_table,
        show_progress=False,
        ids_to_refine=None,
        write_spots_files=False,
    )

    cresult = compare_grain_fits(
        np.vstack([i[-1] for i in gresults]),
        ref_grain_params,
        mtol=1.0e-4,
        ctol=1.0e-3,
        vtol=1.0e-4,
    )

    assert cresult
