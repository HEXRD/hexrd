import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import lmfit

from hexrd.core.fitting.calibration.structureless import (
    StructurelessCalibrator,
    RelativeConstraintsType,
)

# --- Fixtures ---


@pytest.fixture
def mock_instr():
    instr = MagicMock()
    instr.tvec = np.zeros(3)
    instr.chi = 0.0

    instr.beam_names = ['beam1']
    instr.has_multi_beam = False
    beam_info = {
        'vector': np.array([0, 0, 1]),
        'energy': 50.0,
        'energy_correction': {'intercept': 0, 'slope': 0},
    }
    instr.beam_dict = {'beam1': beam_info}

    det1 = MagicMock()
    det1.tvec = np.zeros(3)
    det1.tilt = np.zeros(3)
    det1.rmat = np.eye(3)
    det1.distortion = None
    det1.detector_type = 'flat'

    def mock_cart_to_angles(xy, **kwargs):
        if xy.size == 0:
            return np.empty((0, 2)), None
        xy_2d = np.atleast_2d(xy)
        return np.column_stack([xy_2d[:, 0] * 0.1, xy_2d[:, 1] * 0.1]), None

    det1.cart_to_angles.side_effect = mock_cart_to_angles

    instr.detectors = {'det1': det1}
    instr.detector_groups = {'group1': ['det1']}

    return instr


@pytest.fixture
def mock_data():
    xy = np.array([[10.0, 10.0]])
    return {'beam1': [{'det1': xy}]}


@pytest.fixture
def calibrator(mock_instr, mock_data):
    with patch(
        'hexrd.core.fitting.calibration.structureless.create_instr_params',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.create_tth_parameters',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.add_engineering_constraints'
    ):

        cal = StructurelessCalibrator(mock_instr, mock_data)
        return cal


# --- Initialization & Properties ---


def test_init_defaults(calibrator, mock_instr):
    assert calibrator.instr is mock_instr
    assert calibrator.engineering_constraints is None
    assert calibrator.relative_constraints_type == RelativeConstraintsType.none
    assert calibrator.tth_distortion is None


def test_setters(calibrator, mock_data):
    new_instr = MagicMock()
    new_det = MagicMock()
    new_det.cart_to_angles.return_value = (np.array([[0.1, 0.1]]), None)
    new_instr.detectors = {'det1': new_det}
    new_instr.beam_dict = {'beam1': {}}

    with patch(
        'hexrd.core.fitting.calibration.structureless.create_instr_params',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.create_tth_parameters',
        return_value=[],
    ):
        calibrator.instr = new_instr

    assert calibrator.instr is new_instr

    new_data = {'beam1': []}
    with patch(
        'hexrd.core.fitting.calibration.structureless.create_instr_params',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.create_tth_parameters',
        return_value=[],
    ):
        calibrator.data = new_data

    assert calibrator.data == new_data

    new_instr.has_multi_beam = True
    assert calibrator.two_XRS is True


# --- TTh Distortion Logic ---


def test_tth_distortion_property(calibrator):
    mock_dist = MagicMock()
    dist_dict = {'det1': mock_dist}

    calibrator.tth_distortion = dist_dict

    stored = calibrator.tth_distortion['det1']
    assert stored is not mock_dist
    assert stored.panel == calibrator.instr.detectors['det1']


def test_tth_correction_calculation(calibrator):
    mock_dist = MagicMock()
    mock_dist.apply.return_value = np.array([[0.05, 0.0]])
    calibrator.tth_distortion = {'det1': mock_dist}

    corr = calibrator.tth_correction

    assert 'beam1' in corr
    assert corr['beam1'][0]['det1'][0] == 0.05


# --- Constraints ---


def test_relative_constraints_setter(calibrator):
    with patch(
        'hexrd.core.fitting.calibration.structureless.create_instr_params',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.create_tth_parameters',
        return_value=[],
    ):

        calibrator.relative_constraints_type = RelativeConstraintsType.group
        assert (
            calibrator.relative_constraints.type
            == RelativeConstraintsType.group
        )

        calibrator.relative_constraints_type = None
        assert (
            calibrator.relative_constraints.type
            == RelativeConstraintsType.none
        )


def test_engineering_constraints_setter(calibrator):
    with patch(
        'hexrd.core.fitting.calibration.structureless.create_instr_params',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.create_tth_parameters',
        return_value=[],
    ), patch(
        'hexrd.core.fitting.calibration.structureless.add_engineering_constraints'
    ) as mock_add:

        calibrator.engineering_constraints = 'TARDIS'
        assert calibrator.engineering_constraints == 'TARDIS'
        mock_add.assert_called()

        calibrator.engineering_constraints = 'TARDIS'

        with pytest.raises((Exception, TypeError)):
            calibrator.engineering_constraints = 'GARBAGE'


# --- Residual Calculation ---


@patch(
    'hexrd.core.fitting.calibration.structureless.update_instrument_from_params'
)
def test_calc_residual(mock_update, calibrator):
    params = lmfit.Parameters()
    params.add('DS_ring_0', value=10.0)

    res = calibrator.calc_residual(params)
    mock_update.assert_called()

    expected = 1.0 - np.radians(10.0)
    assert np.isclose(res[0], expected)


@patch(
    'hexrd.core.fitting.calibration.structureless.update_instrument_from_params'
)
def test_calc_residual_with_distortion(mock_update, calibrator):
    params = lmfit.Parameters()
    params.add('DS_ring_0', value=0.0)

    mock_dist = MagicMock()
    mock_dist.apply.return_value = np.array([[0.1, 0.0]])
    calibrator.tth_distortion = {'det1': mock_dist}

    res = calibrator.calc_residual(params)

    assert np.isclose(res[0], 0.9)


@patch(
    'hexrd.core.fitting.calibration.structureless.update_instrument_from_params'
)
def test_calc_residual_empty_ring(mock_update, calibrator):
    det1 = calibrator.instr.detectors['det1']
    det2 = MagicMock()
    det2.cart_to_angles.side_effect = lambda xy, **kwargs: (
        np.column_stack([xy[:, 0] * 0.1, xy[:, 1] * 0.1]),
        None,
    )
    calibrator.instr.detectors = {'det1': det1, 'det2': det2}

    calibrator.data['beam1'][0]['det1'] = np.array([])
    calibrator.data['beam1'][0]['det2'] = np.array([[10.0, 10.0]])

    params = lmfit.Parameters()
    params.add('DS_ring_0', 0.0)

    res = calibrator.calc_residual(params)

    assert len(res) == 1


# --- Running Calibration ---


def test_run_calibration_least_squares(calibrator):
    calibrator.fitter = MagicMock()
    calibrator.fitter.least_squares.return_value = MagicMock(params={})

    calibrator.run_calibration(method='least_squares')
    calibrator.fitter.least_squares.assert_called()


def test_run_calibration_scalar_minimize(calibrator):
    calibrator.fitter = MagicMock()
    calibrator.fitter.scalar_minimize.return_value = MagicMock(params={})

    calibrator.run_calibration(method='nelder-mead')
    calibrator.fitter.scalar_minimize.assert_called()


# --- Properties coverage ---


@patch(
    'hexrd.core.fitting.calibration.structureless.update_instrument_from_params'
)
def test_residual_property(mock_update, calibrator):
    calibrator.params = lmfit.Parameters()
    calibrator.params.add('DS_ring_0', value=0.0)

    res = calibrator.residual
    assert isinstance(res, np.ndarray)
