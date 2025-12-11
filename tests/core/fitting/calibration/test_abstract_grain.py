import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import lmfit

from hexrd.core.fitting.calibration.abstract_grain import (
    AbstractGrainCalibrator,
    force_param_value
)

class ConcreteGrainCalibrator(AbstractGrainCalibrator):
    @property
    def name(self): return "TestGrain"
    @property
    def type(self): return "grain"
    def autopick_points(self): return "autopicked"
    def residual(self): return 0.0
    def model(self): return "modeled"

@pytest.fixture
def calibrator():
    """Returns a fully set-up ConcreteGrainCalibrator with mocks."""
    instr = MagicMock()
    instr.detectors = {'det1': MagicMock(), 'det2': MagicMock()}
    
    mat = MagicMock()
    mat.planeData.latVecOps = {'B': np.eye(3)}
    
    params = np.array([0.1, 0.2, 0.3, 10.0, 20.0, 30.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    
    return ConcreteGrainCalibrator(
        instr=instr,
        material=mat,
        grain_params=params,
        euler_convention={'axes': 'xyz', 'extrinsic': True}
    )

@pytest.mark.parametrize("start_val, force_val, min_val, max_val, expected_min, expected_max", [
    (5.0, 6.0, 0.0, 10.0, 0.0, 10.0),
    (5.0, 2.0, 4.0, 10.0, 1.9999, 10.0),
    (5.0, 8.0, 0.0, 6.0, 0.0, 8.0001),
])
def test_force_param_value(start_val, force_val, min_val, max_val, expected_min, expected_max):
    p = lmfit.Parameter(name='test', value=start_val, min=min_val, max=max_val)
    force_param_value(p, force_val)
    assert p.value == force_val
    assert np.isclose(p.min, expected_min)
    assert np.isclose(p.max, expected_max)


def test_initialization(calibrator):
    assert calibrator.name == "TestGrain"
    assert calibrator.type == "grain"
    assert calibrator.data_dict is None
    assert calibrator.param_names == []
    # Test property delegations
    assert calibrator.plane_data == calibrator.material.planeData
    np.testing.assert_array_equal(calibrator.bmatx, np.eye(3))

@patch('hexrd.core.fitting.calibration.abstract_grain.create_grain_params')
@patch('hexrd.core.fitting.calibration.abstract_grain.rename_to_avoid_collision')
def test_create_lmfit_params(mock_rename, mock_create, calibrator):
    mock_list = [('p1', 'obj')]
    mock_create.return_value = mock_list
    mock_rename.return_value = (mock_list, {})
    
    with patch.object(ConcreteGrainCalibrator, 'grain_params_euler', new_callable=PropertyMock) as mock_gp:
        mock_gp.return_value = np.zeros(12)
        res = calibrator.create_lmfit_params(lmfit.Parameters())
    
    assert res == mock_list
    assert calibrator.param_names == ['p1']

def test_update_from_lmfit_params(calibrator):
    calibrator.param_names = ['p0']
    params = lmfit.Parameters()
    params.add('p0', value=99)
    
    with patch.object(ConcreteGrainCalibrator, 'grain_params_euler', new_callable=PropertyMock) as mock_prop:
        calibrator.update_from_lmfit_params(params)
        set_args = mock_prop.call_args[0][0]
        np.testing.assert_array_equal(set_args, np.array([99.0]))

@pytest.mark.parametrize("method_name, param_indices, expected_vals", [
    ("fix_strain_to_identity", range(6, 12), [1, 1, 1, 0, 0, 0]),
    ("fix_translation_to_origin", range(3, 6), [0, 0, 0]),
    ("fix_y_to_zero", [4], [0]),
])
def test_fix_params_methods(calibrator, method_name, param_indices, expected_vals):
    names = [f'p{i}' for i in range(12)]
    calibrator.param_names = names
    params = lmfit.Parameters()
    for n in names: params.add(n, value=999, vary=True)
    
    with patch('hexrd.core.fitting.calibration.abstract_grain.cnst') as mock_cnst:
        mock_cnst.identity_6x1 = np.array([1, 1, 1, 0, 0, 0])
        mock_cnst.zeros_3 = np.array([0, 0, 0])
        
        getattr(calibrator, method_name)(params)
        
        for idx, val in zip(param_indices, expected_vals):
            name = names[idx]
            assert params[name].value == val
            assert params[name].vary is False

def test_param_name_slices(calibrator):
    calibrator.param_names = [str(i) for i in range(12)]
    assert calibrator.orientation_param_names == ['0', '1', '2']
    assert calibrator.translation_param_names == ['3', '4', '5']
    assert calibrator.strain_param_names == ['6', '7', '8', '9', '10', '11']


@patch('hexrd.core.fitting.calibration.abstract_grain.xfcapi')
@patch('hexrd.core.fitting.calibration.abstract_grain.RotMatEuler')
def test_grain_params_euler_io(mock_rme, mock_xfcapi, calibrator):
    mock_rme.return_value.angles = np.array([np.pi, 0, 0]) # 180 deg
    res = calibrator.grain_params_euler
    np.testing.assert_allclose(res[:3], [180, 0, 0])
    mock_xfcapi.make_rmat_of_expmap.assert_called()

    with patch('hexrd.core.fitting.calibration.abstract_grain.angleAxisOfRotMat') as mock_aa:
        mock_aa.return_value = (np.pi, np.array([[0], [0], [1]])) # Rotation around Z
        
        input_params = np.zeros(12)
        input_params[:3] = [180, 0, 0]
        calibrator.grain_params_euler = input_params
        
        np.testing.assert_allclose(mock_rme.return_value.angles, [np.pi, 0, 0])
        np.testing.assert_allclose(calibrator.grain_params[:3], [0, 0, np.pi])

def test_grain_params_euler_no_convention(calibrator):
    calibrator.euler_convention = None
    np.testing.assert_array_equal(calibrator.grain_params_euler, calibrator.grain_params)


@patch('hexrd.core.fitting.calibration.abstract_grain.hkl_to_str', side_effect=lambda x: f"{x[0]}")
@patch('hexrd.core.fitting.calibration.abstract_grain.str_to_hkl', side_effect=lambda x: [int(x)])
def test_calibration_picks_io(mock_s2h, mock_h2s, calibrator):
    input_picks = {'det1': {'1': [10, 10]}}
    calibrator.calibration_picks = input_picks
    
    assert calibrator.data_dict['hkls']['det1'] == [[1]]
    assert calibrator.data_dict['pick_xys']['det1'] == [[10, 10]]

    calibrator.data_dict['pick_xys']['det2'] = []
    calibrator.data_dict['hkls']['det2'] = []
    
    picks = calibrator.calibration_picks
    assert picks['det1']['1'] == [10, 10]
    assert picks['det2'] == {}

def test_init_with_picks(calibrator):
    c = ConcreteGrainCalibrator(
        instr=calibrator.instr, 
        material=calibrator.material, 
        grain_params=calibrator.grain_params,
        calibration_picks={'det1': {}}
    )
    assert c.data_dict is not None


def test_abstract_enforcement():
    class Incomplete(AbstractGrainCalibrator): pass
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Incomplete(None, None, None)
