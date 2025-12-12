import pytest
import math
from hexrd.core import valunits

# --- valWUnit Class Tests ---

def test_valWUnit_init():
    v = valunits.valWUnit("test", "length", 1.0, "m")
    assert v.name == "test"
    # 'length' is in uTDict, so it maps to "LENGTH"
    assert v.uT == "LENGTH"
    assert v.value == 1.0
    assert v.unit == "m"
    
    # Custom unit type (not in dict)
    v2 = valunits.valWUnit("test2", "custom", 2.0, "cu")
    assert v2.uT == "custom"

def test_valWUnit_str_repr():
    v = valunits.valWUnit("obj", "length", 5.5, "mm")
    
    # Check __str__
    s = str(v)
    assert 'item named "obj"' in s
    assert '5.5 mm' in s
    
    # Check __repr__
    r = repr(v)
    assert 'valWUnit("obj","LENGTH",5.5,"mm")' in r

def test_valWUnit_is_methods():
    v_len = valunits.valWUnit("L", "length", 1, "m")
    assert v_len.isLength()
    assert not v_len.isAngle()
    
    v_ang = valunits.valWUnit("A", "angle", 90, "degrees")
    assert v_ang.isAngle()
    assert not v_ang.isEnergy()
    
    v_en = valunits.valWUnit("E", "energy", 10, "keV")
    assert v_en.isEnergy()
    assert not v_en.isLength()

def test_valWUnit_arithmetic_mul():
    v = valunits.valWUnit("L", "length", 10.0, "m")
    
    # Scalar multiplication
    v2 = v * 2.0
    assert v2.value == 20.0
    assert v2.unit == "m"
    
    # valWUnit multiplication
    v3 = valunits.valWUnit("W", "length", 5.0, "m")
    v_prod = v * v3
    assert v_prod.value == 50.0
    assert v_prod.unit == "(m)*(m)"
    assert v_prod.name == "L_times_W"
    
    # Error
    with pytest.raises(RuntimeError):
        v * "string"

def test_valWUnit_arithmetic_add():
    v = valunits.valWUnit("L", "length", 10.0, "m")
    
    # Scalar add (assumes same unit implicitly for scalar)
    v2 = v + 5.0
    assert v2.value == 15.0
    assert v2.unit == "m"
    
    # valWUnit add (with conversion)
    # 10 m + 1000 mm (=1m) = 11 m
    v_mm = valunits.valWUnit("small", "length", 1000.0, "mm")
    v_sum = v + v_mm
    assert v_sum.value == 11.0
    assert v_sum.unit == "m"
    
    # Error
    with pytest.raises(RuntimeError):
        v + "string"

def test_valWUnit_arithmetic_sub():
    v = valunits.valWUnit("L", "length", 10.0, "m")
    
    # Scalar sub
    v2 = v - 2.0
    assert v2.value == 8.0
    
    # valWUnit sub (with conversion)
    # 10 m - 1000 mm (=1m) = 9 m
    v_mm = valunits.valWUnit("small", "length", 1000.0, "mm")
    v_diff = v - v_mm
    assert v_diff.value == 9.0
    
    # Error
    with pytest.raises(RuntimeError):
        v - "string"

# --- Conversion Tests ---

def test_getVal_convert():
    # Lengths
    v = valunits.valWUnit("L", "length", 1.0, "m")
    assert v.getVal("m") == 1.0
    assert v.getVal("mm") == 1000.0
    assert v.getVal("nm") == 1e9
    
    # Angles
    a = valunits.valWUnit("A", "angle", 180.0, "degrees")
    assert np.isclose(a.getVal("radians"), math.pi)
    
    a_rad = valunits.valWUnit("R", "angle", math.pi, "radians")
    assert np.isclose(a_rad.getVal("degrees"), 180.0)
    
    # Energy
    e = valunits.valWUnit("E", "energy", 1.0, "keV")
    # Just ensure it converts without error; specific constant is implementation detail
    assert e.getVal("J") > 0 

def test_convert_errors():
    v = valunits.valWUnit("L", "length", 1.0, "m")
    with pytest.raises(RuntimeError, match="not recognized"):
        v.getVal("invalid_unit")

def test_kevangstrom_conversion():
    # This specific conversion relies on hexrd.core.constants.keVToAngstrom
    # which is imported inside the _convert logic.
    # Case: keV to angstrom (not simple multiply, usually lambda = const / E)
    # The source code handles this via keVToAngstrom function call
    
    v = valunits.valWUnit("E", "energy", 10.0, "keV")
    
    # We trust keVToAngstrom works; just checking the dispatch
    res = v.getVal("angstrom")
    assert res > 0

# --- Helper Function Tests ---

import numpy as np

def test_toFloat():
    # Scalar input
    assert valunits.toFloat(1.5, "m") == 1.5
    
    # valWUnit input
    v = valunits.valWUnit("L", "length", 1000.0, "mm")
    assert valunits.toFloat(v, "m") == 1.0
    
    # List/Array input
    vals = [valunits.valWUnit("1", "len", 1, "m"), valunits.valWUnit("2", "len", 1000, "mm")]
    res = valunits.toFloat(vals, "m")
    assert res == [1.0, 1.0]

def test_valWithDflt():
    # None passed -> return default
    assert valunits.valWithDflt(None, 5.0) == 5.0
    
    # Value passed -> return value
    assert valunits.valWithDflt(10.0, 5.0) == 10.0
    
    # With unit conversion
    v = valunits.valWUnit("L", "length", 1000.0, "mm")
    # Should convert v to 'm' -> 1.0
    assert valunits.valWithDflt(v, 5.0, toUnit="m") == 1.0
    
    # None passed with unit conversion -> return default converted (if default is valWUnit)
    # Or if default is scalar, returns scalar.
    # toFloat(scalar, unit) returns scalar.
    assert valunits.valWithDflt(None, 10.0, toUnit="m") == 10.0