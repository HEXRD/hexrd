import pytest
import numpy as np
from hexrd.core.distortion import distortionabc

class DummyDistortion(distortionabc.DistortionABC):
    """Implement abstract methods just to hit the base NotImplementedError."""
    def apply(self, xy_in):
        return super().apply(xy_in)

    def apply_inverse(self, xy_in):
        return super().apply_inverse(xy_in)

def test_distortionabc_instantiation_and_methods():
    with pytest.raises(TypeError):
        distortionabc.DistortionABC()

    dummy = DummyDistortion()
    xy_in = np.array([[0.0, 0.0]])

    with pytest.raises(NotImplementedError):
        dummy.apply(xy_in)

    with pytest.raises(NotImplementedError):
        dummy.apply_inverse(xy_in)
