"""Identity distortion class

Simple class that returns it's input.
"""

from .distortionabc import DistortionABC
from .registry import _RegisterDistortionClass


class Identity(DistortionABC, metaclass=_RegisterDistortionClass):

    maptype = "identity"

    def __init__(self, params, **kwargs):
        return

    def apply(self, xy_in):
        return xy_in

    def apply_inverse(self, xy_in):
        return xy_in
