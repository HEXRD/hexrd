import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from hexrd.core.fitting.grains import fitGrain, objFuncFitGrain, matchOmegas

# --- Fixtures ---

@pytest.fixture
def env():
    """Consolidated environment fixture."""
    instr = MagicMock()
    instr.beam_vector, instr.eta_vector = np.array([0, 0, 1]), np.array([1, 0, 0])
    det1 = MagicMock(distortion=None)
    instr.detectors = {'det1': det1}
    instr.detector_parameters = {'det1': MagicMock()}
    
    # Standard grain data
    return {
        'inst': instr,
        'g': np.zeros(12),
        'B': np.eye(3),
        'wl': 0.5,
        'hkl': np.array([[1], [1], [1]]),
        # Mocking input data structure: list of objects with indexing
        'refl_list': {'det1': [MagicMock(__getitem__=lambda s, i: np.array([1,1,1]) if i==2 else (np.array([10., 10.]) if i==7 else np.array([0,0,0]))) ]},
        'refl_arr': {'det1': np.zeros((1, 20))}
    }

# --- fitGrain Tests ---

@patch('hexrd.core.fitting.grains.optimize.leastsq')
def test_fitGrain(mock_lsq, env):
    # 1. Success
    mock_lsq.return_value = (np.ones(12), None)
    res = fitGrain(env['g'], env['inst'], {}, env['B'], env['wl'])
    np.testing.assert_array_equal(res, np.ones(12))
    
    # 2. Error (omePeriod)
    with pytest.raises(RuntimeError):
        fitGrain(env['g'], env['inst'], {}, env['B'], env['wl'], omePeriod=[0, 360])

# --- matchOmegas Tests ---

@pytest.mark.parametrize("case", ["success", "period", "error"])
@patch('hexrd.core.fitting.grains.xfcapi.oscill_angles_of_hkls')
@patch('hexrd.core.fitting.grains.rotations.mapAngle', return_value=np.array([0]))
def test_matchOmegas(mock_map, mock_os, case, env):
    xyo = np.array([[0, 0, 0]])
    
    if case == "error":
        mock_os.return_value = (np.array([[np.nan]*3]), np.array([[np.nan]*3]))
        with pytest.raises(RuntimeError, match="Infeasible"):
            matchOmegas(xyo, env['hkl'], 0.0, np.eye(3), env['B'], 0.5)
        return

    # Success / Period cases
    mock_os.return_value = (np.array([[0, 0, 0.1]]), np.array([[0, 0, 1.0]]))
    ome_p = [0, 360] if case == "period" else None
    
    _, res = matchOmegas(xyo, env['hkl'], 0.0, np.eye(3), env['B'], 0.5, omePeriod=ome_p)
    
    if case == "period": assert mock_map.call_count >= 3
    else: assert np.isclose(res[0], 0.1)

# --- objFuncFitGrain Tests ---

@pytest.mark.parametrize("mode", [
    "list_resid",   # Input as list, return residual vector
    "arr_resid",    # Input as array, return residual vector
    "sim_arr",      # simOnly=True (default flag), returns array
    "sim_dict",     # simOnly=True, flag=2, returns dict
    "scalar_sum",   # flag=1, scalar sum abs
    "scalar_chi",   # flag=2, normalized chi sq
    "distortion",   # active distortion
])
@patch('hexrd.core.fitting.grains.matchOmegas', return_value=(None, np.array([0.])))
@patch('hexrd.core.fitting.grains.extract_detector_transformation')
@patch('hexrd.core.fitting.grains.xfcapi.gvec_to_xy', return_value=np.array([[1., 1.]]))
def test_objFunc_evaluation(mock_g2xy, mock_ext, mock_match, mode, env):
    # Setup
    mock_ext.return_value = (np.eye(3), np.zeros(3), 0.0, np.zeros(3))
    
    # Input Data Selection
    data = env['refl_arr'] if mode == "arr_resid" else env['refl_list']
    
    # Distortion Setup
    if mode == "distortion":
        env['inst'].detectors['det1'].distortion = MagicMock()
        env['inst'].detectors['det1'].distortion.apply.return_value = np.array([[5., 5.]])

    # Args Construction
    kwargs = {'simOnly': mode.startswith("sim")}
    if mode == "sim_dict" or mode == "scalar_chi": kwargs['return_value_flag'] = 2
    elif mode == "scalar_sum": kwargs['return_value_flag'] = 1
    
    # Run
    res = objFuncFitGrain(env['g'], env['g'], np.ones(12, dtype=bool), env['inst'], 
                          data, env['B'], env['wl'], None, **kwargs)

    # Assertions
    if mode.startswith("sim"):
        assert isinstance(res, dict if mode == "sim_dict" else np.ndarray)
    elif "scalar" in mode:
        assert isinstance(res, float)
    elif mode == "distortion":
        env['inst'].detectors['det1'].distortion.apply.assert_called()
    else:
        # Residual check (target 1.0, meas 0.0 or 10.0 depending on input mock)
        # Just check it returns array
        assert isinstance(res, np.ndarray)

@patch('hexrd.core.fitting.grains.matchOmegas')
@patch('hexrd.core.fitting.grains.extract_detector_transformation')
@patch('hexrd.core.fitting.grains.xfcapi.gvec_to_xy')
def test_objFunc_empty_data(mock_g2xy, mock_ext, mock_match, env):
    """
    Test skip logic for empty data.
    We supply two detectors: 'det1' is empty, 'det2' has data.
    This exercises the 'continue' logic for det1, but allows the function
    to complete successfully for det2 (avoiding vstack empty error).
    """
    # Create det2
    det2 = MagicMock()
    det2.distortion = None
    env['inst'].detectors['det2'] = det2
    env['inst'].detector_parameters['det2'] = MagicMock()
    
    # Reflections: det1 empty, det2 valid
    data = {'det1': [], 'det2': env['refl_arr']['det1']}
    
    # Set return values for mocks
    mock_ext.return_value = (np.eye(3), np.zeros(3), 0.0, np.zeros(3))
    mock_match.return_value = (None, np.array([0]))
    mock_g2xy.return_value = np.array([[0.0, 0.0]])
    
    res = objFuncFitGrain(
        env['g'], env['g'], np.ones(12, dtype=bool), env['inst'], 
        data, env['B'], env['wl'], None
    )
    
    # Should have results from det2
    assert len(res) > 0

@patch('hexrd.core.fitting.grains.matchOmegas')
@patch('hexrd.core.fitting.grains.extract_detector_transformation')
@patch('hexrd.core.fitting.grains.xfcapi.gvec_to_xy')
def test_objFunc_mixed_empty(mock_g2xy, mock_ext, mock_mat, env):
    # Test valid vstack when one det is empty and one is full
    mock_ext.return_value = (np.eye(3), np.zeros(3), 0., np.zeros(3))
    mock_g2xy.return_value = np.array([[0., 0.]])
    mock_mat.return_value = (None, np.array([0.]))
    
    env['inst'].detectors['det2'] = MagicMock(distortion=None)
    env['inst'].detector_parameters['det2'] = MagicMock()
    
    # det1 empty, det2 valid
    data = {'det1': [], 'det2': env['refl_arr']['det1']}
    
    res = objFuncFitGrain(env['g'], env['g'], np.ones(12, dtype=bool), env['inst'], 
                          data, env['B'], env['wl'], None)
    assert res.size == 3 # 1 pt (x,y,ome)

@patch('hexrd.core.fitting.grains.matchOmegas', return_value=(None, np.array([0])))
@patch('hexrd.core.fitting.grains.extract_detector_transformation')
@patch('hexrd.core.fitting.grains.xfcapi.gvec_to_xy', return_value=np.array([[np.nan, np.nan]]))
def test_objFunc_infeasible(mock_g2xy, mock_ext, mock_mat, env):
    mock_ext.return_value = (np.eye(3), np.zeros(3), 0., np.zeros(3))
    with pytest.raises(RuntimeError, match="infeasible"):
        objFuncFitGrain(env['g'], env['g'], np.ones(12, dtype=bool), env['inst'], 
                        env['refl_arr'], env['B'], env['wl'], None)