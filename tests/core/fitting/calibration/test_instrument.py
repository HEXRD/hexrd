import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import lmfit

from hexrd.core.fitting.calibration.instrument import (
    InstrumentCalibrator,
    _normalized_ssqr
)
from hexrd.core.fitting.calibration.relative_constraints import (
    RelativeConstraintsType,
    RelativeConstraints
)

# --- Fixtures ---

@pytest.fixture
def instrument_calibrator():
    """Returns a fully set-up InstrumentCalibrator with internal mocks."""
    mock_instr = MagicMock()
    mock_calib = MagicMock()
    mock_calib.instr = mock_instr
    mock_calib.create_lmfit_params.return_value = []
    mock_calib.residual.return_value = np.array([0.1, 0.2])

    with patch('hexrd.core.fitting.calibration.instrument.create_instr_params', return_value=[]), \
         patch('hexrd.core.fitting.calibration.instrument.create_relative_constraints') as mock_rel_create:
        
        mock_rel_create.return_value = MagicMock(spec=RelativeConstraints, type=RelativeConstraintsType.none)
        
        ic = InstrumentCalibrator(mock_calib)
        
        ic.mock_calib = mock_calib 
        ic.mock_instr = mock_instr
        return ic

# --- Basic Logic Tests ---

def test_normalized_ssqr():
    assert _normalized_ssqr(np.array([1, -1, 1, -1])) == 1.0

def test_init_validation():
    with pytest.raises(AssertionError, match="must have at least one"):
        InstrumentCalibrator()
    
    c1, c2 = MagicMock(), MagicMock()
    c1.instr, c2.instr = MagicMock(), MagicMock()
    with pytest.raises(AssertionError, match="same instrument"):
        InstrumentCalibrator(c1, c2)

def test_make_lmfit_params(instrument_calibrator):
    p_tuple = ('p1', 1.0, True, None, None, None)
    
    with patch('hexrd.core.fitting.calibration.instrument.create_instr_params', return_value=[p_tuple]) as mock_create, \
         patch('hexrd.core.fitting.calibration.instrument.validate_params_list') as mock_val, \
         patch('hexrd.core.fitting.calibration.instrument.add_engineering_constraints') as mock_add:
        
        instrument_calibrator.mock_calib.create_lmfit_params.return_value = [('p2', 2.0, True, None, None, None)]
        
        params = instrument_calibrator.make_lmfit_params()
        
        assert isinstance(params, lmfit.Parameters)
        assert 'p1' in params and 'p2' in params
        mock_create.assert_called()
        mock_val.assert_called()

def test_update_and_reset_logic(instrument_calibrator):
    with patch('hexrd.core.fitting.calibration.instrument.update_instrument_from_params') as mock_upd:
        params = lmfit.Parameters()
        instrument_calibrator.update_all_from_params(params)
        mock_upd.assert_called()
        instrument_calibrator.mock_calib.update_from_lmfit_params.assert_called()

    with patch.object(instrument_calibrator, 'make_lmfit_params', return_value="new_params"):
        instrument_calibrator.reset_lmfit_params()
        assert instrument_calibrator.params == "new_params"

    instrument_calibrator.reset_relative_constraint_params()
    instrument_calibrator.relative_constraints.reset.assert_called()

# --- Minimizer Logic Tests ---

@pytest.mark.parametrize("method, call_check", [
    ('least_squares', 'least_squares'),
    ('nelder-mead', 'scalar_minimize')
])
def test_minimize_dispatcher(instrument_calibrator, method, call_check):
    instrument_calibrator.fitter = MagicMock()
    instrument_calibrator.minimize(method=method)
    getattr(instrument_calibrator.fitter, call_check).assert_called()

def test_minimizer_function_and_residual(instrument_calibrator):
    np.testing.assert_array_equal(instrument_calibrator.residual(), [0.1, 0.2])
    
    with patch.object(instrument_calibrator, 'update_all_from_params') as mock_upd:
        res = instrument_calibrator.minimizer_function("params")
        mock_upd.assert_called_with("params")
        np.testing.assert_array_equal(res, [0.1, 0.2])

# --- Constraints Tests ---

def test_engineering_constraints_logic(instrument_calibrator):
    instrument_calibrator._engineering_constraints = 'TARDIS'
    with patch.object(instrument_calibrator, 'make_lmfit_params') as mock_make:
        instrument_calibrator.engineering_constraints = 'TARDIS'
        mock_make.assert_not_called()

    with patch.object(instrument_calibrator, 'make_lmfit_params') as mock_make:
        instrument_calibrator.engineering_constraints = 'None'
        assert instrument_calibrator.engineering_constraints == 'None'
        mock_make.assert_called()

    with patch('hexrd.core.fitting.calibration.instrument.map', create=True, return_value=['valid1']):
        with pytest.raises(Exception, match="Invalid engineering constraint"):
            instrument_calibrator.engineering_constraints = "INVALID"

def test_relative_constraints_logic(instrument_calibrator):
    instrument_calibrator._relative_constraints.type = RelativeConstraintsType.none
    assert instrument_calibrator.relative_constraints_type == RelativeConstraintsType.none

    new_cnst = MagicMock()
    with patch.object(instrument_calibrator, 'make_lmfit_params') as mock_make:
        instrument_calibrator.relative_constraints = new_cnst
        assert instrument_calibrator.relative_constraints is new_cnst
        mock_make.assert_called()

    class MockRelType: # Fake Enum
        none = 0
        instrument = 1
    
    with patch('hexrd.core.fitting.calibration.instrument.RelativeConstraintsType', MockRelType), \
         patch('hexrd.core.fitting.calibration.instrument.create_relative_constraints') as mock_create, \
         patch.object(instrument_calibrator, 'make_lmfit_params'): # prevent real rebuild call
        
        mock_create.return_value = MagicMock(type=MockRelType.instrument)
        instrument_calibrator.relative_constraints_type = MockRelType.instrument
        
        mock_create.assert_called()
        assert instrument_calibrator.relative_constraints.type == MockRelType.instrument

# --- Run Calibration (Parametrized) ---

@pytest.mark.parametrize("residuals, expected_log", [
    ([np.array([1.0]), np.array([0.707])], 'OPTIMIZATION SUCCESSFUL'), # Improved
    ([np.array([1.0]), np.array([1.0])], 'no improvement in residual') # No change
])
def test_run_calibration(instrument_calibrator, residuals, expected_log):
    mock_res_obj = MagicMock()
    mock_res_obj.params = "optimized_params"
    
    with patch('hexrd.core.fitting.calibration.instrument.logger') as mock_logger, \
         patch.object(instrument_calibrator, 'minimize', return_value=mock_res_obj), \
         patch.object(instrument_calibrator, 'update_all_from_params') as mock_upd, \
         patch.object(instrument_calibrator, 'residual', side_effect=residuals):
        
        instrument_calibrator.run_calibration(odict={})
        
        if expected_log == 'OPTIMIZATION SUCCESSFUL':
            mock_logger.info.assert_any_call(expected_log)
            mock_upd.assert_called_with("optimized_params")
        else:
            mock_logger.warning.assert_any_call(expected_log)