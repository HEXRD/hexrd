import os
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import coloredlogs

from hexrd.core import matrixutil as mutil
from hexrd.core import rotations as rot
from hexrd.hedm import config
from hexrd.hedm.fitgrains import fit_grains


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = coloredlogs.ColoredFormatter(
    '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


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

def compare_grain_fits(
    fit_grain_params, ref_grain_params, mtol=1.0e-4, ctol=1.0e-3, vtol=1.0e-4
):
    """
    Executes comparison between reference and fit grain parameters for ff-HEDM
    for the same initial parameters.

    Parameters
    ----------
    fit_grain_params : array_like, (n, 12)
        The fit grain parameters to be tested.
    ref_grain_params : array_like, (n, 12)
        The reference grain parameters (see Notes below).

    Returns
    -------
    bool
        True is successful comparison

    Notes
    -----
    The fitgrains action currently returns
        grain_id, completeness, chisq, grain_params.
    We will have to assume that the grain_ids are in the *same order* as the
    reference, which can be enforces by running the comparison using the
    reference orientation list.
    """
    fit_grain_params = np.atleast_2d(fit_grain_params)
    ref_grain_params = np.atleast_2d(ref_grain_params)
    cresult = False
    ii = 0
    for fg, rg in zip(fit_grain_params, ref_grain_params):
        # test_orientation
        quats = rot.quatOfExpMap(np.vstack([fg[:3], rg[:3]]).T)
        ang, mis = rot.misorientation(
            quats[:, 0].reshape(4, 1), quats[:, 1].reshape(4, 1)
        )
        if ang <= mtol:
            cresult = True
        else:
            logger.warning(f"orientations for grain {ii} do not agree.")
            return cresult

        # test position
        if np.linalg.norm(fg[3:6] - rg[3:6]) > ctol:
            logger.warning(f"centroidal coordinates for grain {ii} do not agree.")
            return False

        # test strain
        vmat_fit = mutil.symmToVecMV(
            np.linalg.inv(mutil.vecMVToSymm(fg[6:])), scale=False
        )
        vmat_ref = mutil.symmToVecMV(
            np.linalg.inv(mutil.vecMVToSymm(rg[6:])), scale=False
        )
        if np.linalg.norm(vmat_fit - vmat_ref, ord=1) > vtol:
            logger.warning(f"stretch components for grain {ii} do not agree.")
            return False

        # index grain id
        ii += 1
    return cresult

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


def test_fit_grains_return_pull_spots_data(
    single_ge_include_path: Path,
    test_config: config.root.RootConfig,
    grains_reference_file_path: Path,
) -> None:
    os.chdir(str(single_ge_include_path))

    grains_table: np.ndarray = np.loadtxt(grains_reference_file_path, ndmin=2)

    result = fit_grains(
        test_config,
        grains_table,
        show_progress=False,
        ids_to_refine=None,
        write_spots_files=False,
        return_pull_spots_data=True,
    )

    # Should return a (fit_results, spots_data) tuple
    assert isinstance(result, tuple)
    assert len(result) == 2

    fit_results, spots_data = result

    # fit_results should be a list of 4-element tuples
    assert isinstance(fit_results, list)
    assert len(fit_results) > 0
    for grain_result in fit_results:
        assert len(grain_result) == 4
        grain_id, completeness, chisq, grain_params = grain_result
        assert isinstance(grain_id, (int, np.integer))
        assert isinstance(completeness, float)
        assert isinstance(grain_params, np.ndarray)
        assert grain_params.shape == (12,)

    # spots_data should be a dict keyed by grain_id
    assert isinstance(spots_data, dict)
    assert len(spots_data) == len(fit_results)

    for grain_id, (complvec, results) in spots_data.items():
        # complvec is a list of booleans
        assert isinstance(complvec, list)

        # results is a dict keyed by detector name
        assert isinstance(results, dict)
        assert len(results) > 0

        for det_key, det_results in results.items():
            assert isinstance(det_key, str)
            assert isinstance(det_results, list)
            assert len(det_results) > 0

            for spot in det_results:
                # Each spot should have 9 elements (including pred_xy)
                assert len(spot) == 9, (
                    f'Expected 9 elements per spot, got {len(spot)}'
                )

                peak_id = spot[0]
                hkl = spot[2]
                pred_angs = spot[5]
                meas_angs = spot[6]
                meas_xy = spot[7]
                pred_xy = spot[8]

                assert isinstance(peak_id, (int, np.integer))
                assert isinstance(hkl, np.ndarray)
                assert hkl.shape == (3,)
                assert isinstance(pred_angs, np.ndarray)
                assert pred_angs.shape == (3,)

                # meas_angs/meas_xy may be None for invalid spots
                if peak_id >= 0:
                    assert isinstance(meas_angs, np.ndarray)
                    assert meas_angs.shape == (3,)
                    assert isinstance(meas_xy, np.ndarray)
                    assert meas_xy.shape == (2,)

                assert isinstance(pred_xy, np.ndarray)
                assert pred_xy.shape == (2,)
