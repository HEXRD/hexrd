import numpy as np
from hexrd.material import Material

class filterpack():
    """this class deals with intensity corrections
    related to absorption by the filter pack.
    """
    def __init__(self,
                 filtermaterial=None,
                 thickness=10):
    if isinstance(filtermaterial, Material):
        self.filtermaterial = filtermaterial


    @property
    def absorption_length(self):
        if hasattr(self, material):
            self.material.absorption_length
        else:
            return self._absorption_length

    @absorption_length.setter
    def absorption_length(self, val):
        """absorption length in microns
        """
        self._absorption_length = val