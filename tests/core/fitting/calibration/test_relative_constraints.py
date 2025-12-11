import pytest
from unittest.mock import MagicMock
import numpy as np

from hexrd.core.fitting.calibration.relative_constraints import (
    RelativeConstraintsType,
    RotationCenter,
    RelativeConstraintsNone,
    RelativeConstraintsGroup,
    RelativeConstraintsSystem,
    create_relative_constraints
)

# --- Fixtures ---

@pytest.fixture
def mock_instr():
    instr = MagicMock()
    instr.detector_groups = ['group1', 'group2']
    
    instr.mean_detector_center = np.array([10.0, 10.0, 10.0])
    
    def get_group_center(group_name):
        if group_name == 'group1':
            return np.array([1.0, 1.0, 1.0])
        return np.array([2.0, 2.0, 2.0])
    
    instr.mean_group_center.side_effect = get_group_center
    return instr

# --- Tests for RelativeConstraintsNone ---

def test_constraints_none(mock_instr):
    con = create_relative_constraints(RelativeConstraintsType.none, mock_instr)
    assert isinstance(con, RelativeConstraintsNone)
    assert con.type == RelativeConstraintsType.none
    
    assert con.params == {}
    assert con.rotation_center == RotationCenter.instrument_mean_center
    
    con.reset()
    con.reset_params()

# --- Tests for RelativeConstraintsGroup ---

def test_constraints_group_init_and_reset(mock_instr):
    con = create_relative_constraints(RelativeConstraintsType.group, mock_instr)
    assert isinstance(con, RelativeConstraintsGroup)
    assert con.type == RelativeConstraintsType.group
    
    # Verify Initial Params
    assert 'group1' in con.params
    assert 'group2' in con.params
    np.testing.assert_array_equal(con.params['group1']['tilt'], [0, 0, 0])
    
    # Verify Default Rotation Center
    assert con.rotation_center == RotationCenter.group_mean_center

def test_constraints_group_rotation_centers(mock_instr):
    con = RelativeConstraintsGroup(mock_instr)
    
    center = con.center_of_rotation(mock_instr, 'group1')
    np.testing.assert_array_equal(center, [1.0, 1.0, 1.0])
    
    con.rotation_center = RotationCenter.instrument_mean_center
    center = con.center_of_rotation(mock_instr, 'group1')
    np.testing.assert_array_equal(center, [10.0, 10.0, 10.0])
    
    con.rotation_center = RotationCenter.lab_origin
    center = con.center_of_rotation(mock_instr, 'group1')
    np.testing.assert_array_equal(center, [0.0, 0.0, 0.0])
    
    con.rotation_center = "Invalid_Center_Type"
    with pytest.raises(NotImplementedError):
        con.center_of_rotation(mock_instr, 'group1')

def test_constraints_group_reset_methods(mock_instr):
    con = RelativeConstraintsGroup(mock_instr)
    
    # Change state
    con.rotation_center = RotationCenter.lab_origin
    con.params['group1']['tilt'][0] = 5.0
    
    # Reset
    con.reset()
    
    assert con.rotation_center == RotationCenter.group_mean_center
    assert con.params['group1']['tilt'][0] == 0.0

# --- Tests for RelativeConstraintsSystem ---

def test_constraints_system_init_and_reset(mock_instr):
    con = create_relative_constraints(RelativeConstraintsType.system, mock_instr)
    assert isinstance(con, RelativeConstraintsSystem)
    assert con.type == RelativeConstraintsType.system
    
    assert 'tilt' in con.params
    np.testing.assert_array_equal(con.params['translation'], [0, 0, 0])
    
    assert con.rotation_center == RotationCenter.instrument_mean_center

def test_constraints_system_rotation_centers(mock_instr):
    con = RelativeConstraintsSystem()
    
    center = con.center_of_rotation(mock_instr)
    np.testing.assert_array_equal(center, [10.0, 10.0, 10.0])
    
    con.rotation_center = RotationCenter.lab_origin
    center = con.center_of_rotation(mock_instr)
    np.testing.assert_array_equal(center, [0.0, 0.0, 0.0])
    
    con.rotation_center = RotationCenter.group_mean_center
    with pytest.raises(NotImplementedError):
        con.center_of_rotation(mock_instr)

def test_constraints_system_reset_methods():
    con = RelativeConstraintsSystem()
    
    con.rotation_center = RotationCenter.lab_origin
    con.params['tilt'][0] = 5.0
    
    con.reset()
    
    assert con.rotation_center == RotationCenter.instrument_mean_center
    assert con.params['tilt'][0] == 0.0

# --- Enums and Factory Tests ---

def test_enum_types():
    assert RelativeConstraintsType.none.value == 'None'
    assert RelativeConstraintsType.group.value == 'Group'
    assert RelativeConstraintsType.system.value == 'System'

    assert RotationCenter.instrument_mean_center.value == 'InstrumentMeanCenter'
    assert RotationCenter.group_mean_center.value == 'GroupMeanCenter'
    assert RotationCenter.lab_origin.value == 'Origin'

def test_factory_types(mock_instr):
    c_none = create_relative_constraints(RelativeConstraintsType.none, mock_instr)
    assert isinstance(c_none, RelativeConstraintsNone)
    
    c_grp = create_relative_constraints(RelativeConstraintsType.group, mock_instr)
    assert isinstance(c_grp, RelativeConstraintsGroup)
    
    c_sys = create_relative_constraints(RelativeConstraintsType.system, mock_instr)
    assert isinstance(c_sys, RelativeConstraintsSystem)