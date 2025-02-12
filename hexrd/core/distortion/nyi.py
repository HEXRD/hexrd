"""Not Yet Implemented

To illustrate error when abstract method is not implemented
"""
from .distortionabc import DistortionABC
from .registry import _RegisterDistortionClass


class NYI(DistortionABC, metaclass=_RegisterDistortionClass):

    maptype = "nyi"

    def __init__(self, params, **kwargs):
        return
