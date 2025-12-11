import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import lmfit

from hexrd.core.fitting.calibration import lmfit_param_handling as lph
from hexrd.core.fitting.calibration.relative_constraints import RelativeConstraintsType

# --- Fixtures ---

@pytest.fixture
def mock_rotations():
    with patch('hexrd.core.fitting.calibration.lmfit_param_handling.calc_angles_from_beam_vec', return_value=(0.1, 0.2)), \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.calc_beam_vec', return_value=np.array([0, 0, 1])), \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.angleAxisOfRotMat', return_value=(0.5, np.array([1, 0, 0]))), \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.expMapOfQuat', return_value=np.array([0.1, 0.1, 0.1])), \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.make_rmat_euler', return_value=np.eye(3)) as m_mre, \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.quatOfRotMat', return_value=np.array([1, 0, 0, 0])), \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.rotMatOfExpMap', return_value=np.eye(3)) as m_rme, \
         patch('hexrd.core.fitting.calibration.lmfit_param_handling.RotMatEuler') as m_class_rme:
        
        mock_instance = MagicMock()
        mock_instance.angles = np.array([0.1, 0.2, 0.3])
        m_class_rme.return_value = mock_instance
        yield {'mre': m_mre, 'rme': m_rme, 'class_rme': m_class_rme}

@pytest.fixture
def mock_instrument():
    instr = MagicMock()
    instr.chi, instr.tvec = 0.5, np.array([1.0, 2.0, 3.0])
    instr.beam_dict = {'beam1': {'vector': np.array([0, 0, 1]), 'energy': 50.0, 
                                 'energy_correction': {'intercept': 0.0, 'slope': 0.0}}}
    instr.beam_names, instr.has_multi_beam = list(instr.beam_dict.keys()), False
    
    det1 = MagicMock(tvec=np.array([10.0, 0.0, 0.0]), tilt=np.zeros(3), rmat=np.eye(3), 
                     distortion=None, detector_type='flat')
    
    class StrictDistortion:
        def __init__(self): self._params = np.array([0.1, 0.2])
        @property
        def params(self): return self._params
        @params.setter
        def params(self, val):
            if len(val) != 2: raise AssertionError("Size mismatch")
            self._params = val

    det2 = MagicMock(tvec=np.array([-10.0, 0.0, 0.0]), tilt=np.zeros(3), rmat=np.eye(3),
                     detector_type='cylindrical', radius=500.0, distortion=StrictDistortion())
    
    instr.detectors = {'det-1': det1, 'det-2': det2}
    instr.detector_groups = {'group-1': ['det-1', 'det-2']}
    instr.detectors_in_group.return_value = instr.detectors
    return instr

@pytest.fixture
def mock_material():
    mat = MagicMock()
    mat.name = "CeO2"
    mat.lparms = np.array([0.54, 0.54, 0.54, 90.0, 90.0, 90.0])
    mat.unitcell.is_editable.return_value = True
    return mat

# --- Euler & Utilities ---

@pytest.mark.parametrize("inp, expected", [
    (None, None),
    (('zxz', False), ('zxz', False)),
    ({'axes_order': 'xyz', 'extrinsic': True}, ('xyz', True))
])
def test_normalize_euler(inp, expected):
    assert lph.normalize_euler_convention(inp) == expected

@pytest.mark.parametrize("conv, expected_suffix", [
    (None, "expmap_x"),
    (('xyz', True), "euler_x"),
    (('zxz', False), "euler_z")
])
def test_param_names_euler(conv, expected_suffix):
    names = lph.param_names_euler_convention("d", conv)
    assert names[0] == f"d_{expected_suffix}"

def test_tilt_to_rmat_branches(mock_rotations):
    lph._tilt_to_rmat(np.zeros(3), None)
    mock_rotations['rme'].assert_called() # ExpMap path
    lph._tilt_to_rmat(np.zeros(3), ('xyz', True))
    mock_rotations['mre'].assert_called() # Euler path

# --- Parameter Creation ---

@pytest.mark.parametrize("ctype, convention, checks, fails", [
    (None, None, ['instr_chi', 'det_1_tvec_x', 'det_2_radius', 'det_1_expmap_x'], False),
    (RelativeConstraintsType.group, None, ['group_1_tvec_x', 'group_1_expmap_x'], False),
    (RelativeConstraintsType.system, None, ['system_tvec_x', 'system_expmap_x'], False),
    (RelativeConstraintsType.system, ('xyz', True), ['system_euler_x'], False),
    ("invalid", None, [], True),
])
def test_create_instr_params(mock_instrument, mock_rotations, ctype, convention, checks, fails):
    con = None
    if ctype and ctype != "invalid":
        con = MagicMock(type=ctype, params={'group-1': {'translation': np.zeros(3), 'tilt': np.zeros(3)},
                                            'translation': np.zeros(3), 'tilt': np.zeros(3)})
    elif ctype == "invalid":
        con = MagicMock(type="invalid")

    if fails:
        with pytest.raises(NotImplementedError):
            lph.create_instr_params(mock_instrument, relative_constraints=con)
        return

    params = lph.create_instr_params(mock_instrument, relative_constraints=con, euler_convention=convention)
    names = [p[0] for p in params]
    
    for check in checks:
        assert check in names
    
    if ctype == RelativeConstraintsType.group:
        assert 'det_1_tvec_x' not in names
    if convention:
        assert 'system_expmap_x' not in names

# --- Update Logic ---

def test_update_instrument_input_validation(mock_instrument):
    with pytest.raises(NotImplementedError):
        lph.update_instrument_from_params(mock_instrument, {}, None)

def test_update_instrument_scenarios(mock_instrument, mock_rotations):
    params = lmfit.Parameters()
    base_vars = ['beam_energy', 'beam_azimuth', 'beam_polar', 'beam_energy_correction_slope', 
                 'beam_energy_correction_intercept', 'instr_chi', 'instr_tvec_x', 
                 'instr_tvec_y', 'instr_tvec_z',
                 'det_1_tvec_x', 'det_1_tvec_y', 'det_1_tvec_z', 
                 'det_1_expmap_x', 'det_1_expmap_y', 'det_1_expmap_z',
                 'det_2_tvec_x', 'det_2_tvec_y', 'det_2_tvec_z', 
                 'det_2_expmap_x', 'det_2_expmap_y', 'det_2_expmap_z',
                 'det_2_radius', 'det_2_distortion_param_0', 'det_2_distortion_param_1',
                 'group_1_tvec_x', 'group_1_tvec_y', 'group_1_tvec_z',
                 'group_1_expmap_x', 'group_1_expmap_y', 'group_1_expmap_z',
                 'system_tvec_x', 'system_tvec_y', 'system_tvec_z',
                 'system_expmap_x', 'system_expmap_y', 'system_expmap_z']
    
    for k in base_vars: params.add(k, 0.0)
    
    params['det_1_tvec_x'].value = 15.0
    params['det_2_radius'].value = 600.0
    lph.update_instrument_from_params(mock_instrument, params, euler_convention=None)
    assert mock_instrument.detectors['det-1'].tvec[0] == 15.0
    assert mock_instrument.detectors['det-2'].radius == 600.0

    con = MagicMock(type=RelativeConstraintsType.group, 
                    params={'group-1': {'translation': np.zeros(3), 'tilt': np.zeros(3)}})
    con.center_of_rotation.return_value = np.zeros(3)
    
    mock_instrument.detectors['det-1'].tvec[0] = 10.0 
    
    params['group_1_tvec_x'].value = 10.0 # Shift
    params['group_1_tvec_x'].vary = True
    lph.update_instrument_from_params(mock_instrument, params, relative_constraints=con, euler_convention=None)
    
    assert mock_instrument.detectors['det-1'].tvec[0] == 20.0
    assert con.params['group-1']['translation'][0] == 10.0

    con.type = RelativeConstraintsType.system
    con.params = {'translation': np.zeros(3), 'tilt': np.zeros(3)}
    
    mock_instrument.detectors['det-1'].tvec[0] = 10.0
    params['system_tvec_x'].value = 5.0
    params['system_tvec_x'].vary = True
    lph.update_instrument_from_params(mock_instrument, params, relative_constraints=con, euler_convention=None)
    assert mock_instrument.detectors['det-1'].tvec[0] == 15.0

    con.type = "garbage"
    with pytest.raises(NotImplementedError):
        lph.update_instrument_from_params(mock_instrument, params, relative_constraints=con)

def test_update_distortion_mismatch(mock_instrument):
    det2 = mock_instrument.detectors['det-2']
    mock_instrument.detectors = {'det-2': det2}
    params = lmfit.Parameters()
    for k in ['det_2_expmap_x', 'det_2_expmap_y', 'det_2_expmap_z', 
              'det_2_tvec_x', 'det_2_tvec_y', 'det_2_tvec_z', 'det_2_radius']:
        params.add(k, 0.0)
    params.add('det_2_distortion_param_0', 0.0)
    
    with pytest.raises(RuntimeError, match="expects 2 params"):
        lph.update_unconstrained_detector_parameters(mock_instrument, params, None)

# --- Other Constraints & Utils ---

@pytest.mark.parametrize("ctype, prefixes", [
    (RelativeConstraintsType.none, ['det_1', 'det_2']),
    (RelativeConstraintsType.group, ['group_1']),
    (RelativeConstraintsType.system, ['system']),
    (None, ['det_1', 'det_2'])
])
def test_fix_detector_y(ctype, prefixes, mock_instrument):
    params = lmfit.Parameters()
    for p in ['det_1', 'det_2', 'group_1', 'system']: params.add(f'{p}_tvec_y', 0.0, vary=True)
    con = MagicMock(type=ctype) if ctype else None
    lph.fix_detector_y(mock_instrument, params, relative_constraints=con)
    for p in prefixes: assert params[f'{p}_tvec_y'].vary is False

@pytest.mark.parametrize("y2, y4, expect", [
    (15.0, -15.0, 23.43), # Too far (30) -> Cap Max
    (10.0, -10.0, 22.83), # Too close (20) -> Cap Min
])
def test_engineering_constraints_tardis(y2, y4, expect):
    params = lmfit.Parameters()
    params.add('IMAGE_PLATE_2_tvec_y', y2, vary=True)
    params.add('IMAGE_PLATE_4_tvec_y', y4, vary=True)
    lph.add_engineering_constraints(params, 'TARDIS')
    assert np.isclose(params['tardis_distance_between_plates'].value, expect)
    
    p2 = lmfit.Parameters()
    lph.add_engineering_constraints(p2, 'Unknown') # No-op
    assert len(p2) == 0

def test_material_grain_and_utils(mock_material, mock_instrument):
    p = lph.create_material_params(mock_material)
    assert p[0][0] == 'CeO2_a' and p[0][1] == 5.4
    
    params = lmfit.Parameters()
    params.add_many(('CeO2_a', 5.5), ('beam_energy', 100.0))
    lph.update_material_from_params(params, mock_material)
    assert mock_material.lparms[0] == 0.55
    
    assert lph.create_grain_params("g1", np.zeros(12))[0][0] == 'g1_grain_param_0'
    
    lph.validate_params_list([('a', 1), ('b', 2)])
    with pytest.raises(lph.LmfitValidationException): lph.validate_params_list([('a', 1), ('a', 2)])
    
    res, _ = lph.rename_to_avoid_collision([('a', 2)], [('a', 1)])
    assert res[0][0] == '1_a'

    mock_instrument.beam_names = ['beam1']
    meas = {'beam1': [{0: np.array([[np.radians(10.0)]])}]}
    assert lph.create_tth_parameters(mock_instrument, meas)[0][0] == 'DS_ring_0'

# --- Detector Angles Get/Set ---

def test_detector_angles_get_set(mock_instrument, mock_rotations):
    det = mock_instrument.detectors['det-1']
    
    np.testing.assert_array_equal(lph.detector_angles_euler(det, None), det.tilt)
    res = lph.detector_angles_euler(det, ('xyz', True))
    np.testing.assert_allclose(res, np.degrees([0.1, 0.2, 0.3])) # from mock
    
    params = lmfit.Parameters()
    params.add_many(('det_1_expmap_x', 1), ('det_1_expmap_y', 2), ('det_1_expmap_z', 3))
    lph.set_detector_angles_euler(det, "det_1", params, None)
    np.testing.assert_array_equal(det.tilt, [1, 2, 3])
    
    with patch('hexrd.core.fitting.calibration.lmfit_param_handling.expMapOfQuat', return_value=np.array([5, 5, 5])):
        params.add_many(('det_1_euler_x', 1), ('det_1_euler_y', 2), ('det_1_euler_z', 3))
        lph.set_detector_angles_euler(det, "det_1", params, ('xyz', True))
        np.testing.assert_array_equal(det.tilt, [5, 5, 5])