import numpy as np
from hexrd.material import Material

class layeredsamples():
    """this class deals with intensity corrections
    related to self-absorption by a layered physics
    package. the class uses information from a material
    and instrument class to determine these corrections.

    we will only consider absorption after the
    diffraction has occured i.e. absorption by the 
    sample and the window only. absorption by the 
    filterpacks etc. will be in a separate class

    The most important equation to use will be eqn. 42
    from Rygg et al., X-ray diffraction at the National 
    Ignition Facility, Rev. Sci. Instrum. 91, 043902 (2020)
    """
    def __init__(
                self,
                sample=None,
                window=None):
