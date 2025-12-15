import pytest
from hexrd.core.fitting.calibration.calibrator import Calibrator


class ConcreteCalibrator(Calibrator):
    @property
    def type(self):
        return "test_type"

    def create_lmfit_params(self, current_params):
        return []

    def update_from_lmfit_params(self, params_dict):
        pass

    def residual(self, calibration_data=None):
        return 0.0

    @property
    def calibration_picks(self):
        return {}

    @calibration_picks.setter
    def calibration_picks(self, val):
        pass


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError) as excinfo:
        Calibrator()
    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_incomplete_subclass_enforcement():
    class Incomplete(Calibrator):
        pass

    with pytest.raises(TypeError) as excinfo:
        Incomplete()

    msg = str(excinfo.value)
    assert "type" in msg
    assert "create_lmfit_params" in msg
    assert "update_from_lmfit_params" in msg
    assert "residual" in msg
    assert "calibration_picks" in msg


def test_concrete_subclass_instantiation():
    """Ensure a properly implemented subclass works."""
    c = ConcreteCalibrator()
    assert c.type == "test_type"
    assert c.create_lmfit_params(None) == []
    assert c.residual() == 0.0
    assert c.calibration_picks == {}


def test_tth_distortion_default_behavior():
    """Test the default implementation of tth_distortion."""
    c = ConcreteCalibrator()

    assert c.tth_distortion is None

    c.tth_distortion = None

    with pytest.raises(NotImplementedError):
        c.tth_distortion = "some_value"


def test_abstract_methods_raise_error():
    with pytest.raises(NotImplementedError):
        Calibrator.type.fget(None)

    with pytest.raises(NotImplementedError):
        Calibrator.create_lmfit_params(None, None)

    with pytest.raises(NotImplementedError):
        Calibrator.update_from_lmfit_params(None, None)

    with pytest.raises(NotImplementedError):
        Calibrator.residual(None)

    with pytest.raises(NotImplementedError):
        Calibrator.calibration_picks.fget(None)

    with pytest.raises(NotImplementedError):
        Calibrator.calibration_picks.fset(None, None)
