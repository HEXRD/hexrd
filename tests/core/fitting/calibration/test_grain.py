import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from hexrd.core.fitting.calibration.grain import (
    GrainCalibrator,
    sxcal_obj_func,
)

# --- Fixtures ---


@pytest.fixture
def mock_instr():
    instr = MagicMock()
    instr.beam_wavelength = 0.5
    instr.beam_vector = np.array([0, 0, 1])
    instr.chi = 15.0
    instr.tvec = np.zeros(3)
    instr.energy_correction = 0.0

    det1 = MagicMock()
    det1.rmat = np.eye(3)
    det1.tvec = np.zeros(3)
    det1.distortion = None
    instr.detectors = {'det1': det1}
    return instr


@pytest.fixture
def mock_material():
    mat = MagicMock()
    mat.name = "CeO2"
    mat.planeData.latVecOps = {'B': np.eye(3)}
    return mat


@pytest.fixture
def grain_params():
    return np.array(
        [0.1, 0.2, 0.3, 10.0, 20.0, 30.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )


@pytest.fixture
def calibrator(mock_instr, mock_material, grain_params):
    return GrainCalibrator(
        instr=mock_instr,
        material=mock_material,
        grain_params=grain_params,
        ome_period=[0, 360],
        index=1,
        calibration_picks=None,
    )


# --- GrainCalibrator Tests ---


def test_initialization(calibrator, mock_material):
    assert calibrator.ome_period == [0, 360]
    assert calibrator.index == 1
    assert calibrator.type == 'grain'
    assert calibrator.name == f"{mock_material.name}_1"


def test_autopick_points_not_implemented(calibrator):
    with pytest.raises(NotImplementedError):
        calibrator.autopick_points()


def test_evaluate(calibrator):
    calibrator.data_dict = {
        'pick_xys': {'det1': [[10, 10, 0.1], [np.nan, np.nan, np.nan]]},
        'hkls': {'det1': [[1, 1, 1], [3, 1, 1]]},
    }

    hkls_dict, xys_dict = calibrator._evaluate()

    assert len(hkls_dict['det1'][0]) == 1
    assert len(xys_dict['det1'][0]) == 1
    np.testing.assert_array_equal(xys_dict['det1'][0][0], [10, 10, 0.1])


@patch('hexrd.core.fitting.calibration.grain.sxcal_obj_func')
def test_residual(mock_obj, calibrator):
    calibrator.data_dict = {
        'pick_xys': {'det1': [[10, 10, 0]]},
        'hkls': {'det1': [[1, 1, 1]]},
    }
    calibrator.residual()

    mock_obj.assert_called_once()
    assert mock_obj.call_args[1].get('sim_only', False) is False


@patch('hexrd.core.fitting.calibration.grain.sxcal_obj_func')
def test_model(mock_obj, calibrator):
    calibrator.data_dict = {
        'pick_xys': {'det1': [[10, 10, 0]]},
        'hkls': {'det1': [[1, 1, 1]]},
    }
    calibrator.model()

    mock_obj.assert_called_once()
    assert mock_obj.call_args[1].get('sim_only') is True


# --- Objective Function Tests ---


@patch('hexrd.core.fitting.calibration.grain.angularDifference')
@patch('hexrd.core.fitting.calibration.grain.xfcapi')
@patch('hexrd.core.fitting.calibration.grain.grainutil')
@patch('hexrd.core.fitting.calibration.grain.xrdutil')
@patch('hexrd.core.fitting.calibration.grain.mutil')
def test_sxcal_obj_func_residual_mode(
    mock_mutil, mock_xrd, mock_gutil, mock_xfc, mock_ang, mock_instr
):
    grain_params = [np.zeros(12)]
    xyo_det = {'det1': [np.array([[10, 10, 0.1], [20, 20, 0.2]])]}
    hkls_idx = {'det1': [np.array([[1, 1, 1], [2, 0, 0]])]}

    mock_xfc.make_rmat_of_expmap.return_value = np.eye(3)
    mock_mutil.vecMVToSymm.return_value = np.eye(3)
    mock_mutil.unitVector.side_effect = lambda x: x
    mock_xrd.apply_correction_to_wavelength.return_value = 0.5

    mock_gutil.matchOmegas.return_value = (
        [0, 1],
        np.array([0.15, 0.25]),
    )  # d_ome = 0.05
    mock_xfc.make_sample_rmat.return_value = np.zeros((2, 3, 3))
    mock_xfc.gvec_to_xy.return_value = np.array(
        [[11, 11], [21, 21]]
    )  # dx, dy = 1.0
    mock_ang.return_value = np.array([0.05, 0.05])

    result = sxcal_obj_func(
        grain_params,
        mock_instr,
        xyo_det,
        hkls_idx,
        np.eye(3),
        [0, 360],
        sim_only=False,
    )

    assert result.shape == (6,)
    np.testing.assert_allclose(result[0::3], 1.0)
    np.testing.assert_allclose(result[1::3], 1.0)
    np.testing.assert_allclose(result[2::3], 0.05)


@patch('hexrd.core.fitting.calibration.grain.xfcapi')
@patch('hexrd.core.fitting.calibration.grain.grainutil')
@patch('hexrd.core.fitting.calibration.grain.xrdutil')
@patch('hexrd.core.fitting.calibration.grain.mutil')
def test_sxcal_obj_func_sim_mode(
    mock_mutil, mock_xrd, mock_gutil, mock_xfc, mock_instr
):
    grain_params = [np.zeros(12)]
    xyo_det = {'det1': [np.array([[10, 10, 0.1]])]}
    hkls_idx = {'det1': [np.array([[1, 1, 1]])]}

    mock_xfc.make_rmat_of_expmap.return_value = np.eye(3)
    mock_mutil.vecMVToSymm.return_value = np.eye(3)
    mock_mutil.unitVector.side_effect = lambda x: x
    mock_xrd.apply_correction_to_wavelength.return_value = 0.5

    mock_gutil.matchOmegas.return_value = ([0], np.array([0.5]))
    mock_xfc.gvec_to_xy.return_value = np.array([[100, 100]])

    retval = sxcal_obj_func(
        grain_params,
        mock_instr,
        xyo_det,
        hkls_idx,
        np.eye(3),
        [0, 360],
        sim_only=True,
    )

    assert np.array_equal(retval['det1'][0][0], [100, 100, 0.5])


@patch('hexrd.core.fitting.calibration.grain.logger')
@patch('hexrd.core.fitting.calibration.grain.xfcapi')
@patch('hexrd.core.fitting.calibration.grain.grainutil')
@patch('hexrd.core.fitting.calibration.grain.xrdutil')
@patch('hexrd.core.fitting.calibration.grain.mutil')
def test_sxcal_obj_func_infeasible_warning(
    mock_mutil, mock_xrd, mock_gutil, mock_xfc, mock_log, mock_instr
):
    mock_xfc.make_rmat_of_expmap.return_value = np.eye(3)
    mock_mutil.vecMVToSymm.return_value = np.eye(3)
    mock_mutil.unitVector.side_effect = lambda x: x
    mock_gutil.matchOmegas.return_value = ([0], np.array([0.5]))
    mock_xfc.gvec_to_xy.return_value = np.array([[np.nan, np.nan]])

    xyo = {'det1': [np.array([[10, 10, 0.1]])]}
    hkl = {'det1': [np.array([[1, 1, 1]])]}

    sxcal_obj_func(
        [np.zeros(12)],
        mock_instr,
        xyo,
        hkl,
        np.eye(3),
        [0, 360],
        sim_only=True,
    )

    mock_log.warning.assert_called()
    assert "infeasible parameters" in mock_log.warning.call_args[0][0]


@patch('hexrd.core.fitting.calibration.grain.xfcapi')
@patch('hexrd.core.fitting.calibration.grain.grainutil')
@patch('hexrd.core.fitting.calibration.grain.xrdutil')
@patch('hexrd.core.fitting.calibration.grain.mutil')
def test_sxcal_obj_func_with_distortion(
    mock_mutil, mock_xrd, mock_gutil, mock_xfc, mock_instr
):
    distortion_mock = MagicMock()
    distortion_mock.apply.side_effect = lambda xy: xy + 1.0
    mock_instr.detectors['det1'].distortion = distortion_mock

    mock_xfc.make_rmat_of_expmap.return_value = np.eye(3)
    mock_mutil.vecMVToSymm.return_value = np.eye(3)
    mock_mutil.unitVector.side_effect = lambda x: x
    mock_gutil.matchOmegas.return_value = ([0], np.array([0.5]))
    mock_xfc.gvec_to_xy.return_value = np.array([[100, 100]])

    xyo = {'det1': [np.array([[10, 10, 0.1]])]}
    hkl = {'det1': [np.array([[1, 1, 1]])]}

    with patch(
        'hexrd.core.fitting.calibration.grain.angularDifference',
        return_value=np.array([0]),
    ):
        sxcal_obj_func(
            [np.zeros(12)],
            mock_instr,
            xyo,
            hkl,
            np.eye(3),
            [0, 360],
            sim_only=False,
        )

    distortion_mock.apply.assert_called()
    np.testing.assert_array_equal(
        distortion_mock.apply.call_args[0][0], [[10, 10]]
    )
