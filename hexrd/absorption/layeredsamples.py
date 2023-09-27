import numpy as np
from hexrd.material import Material

class layeredsamples():
    """this class deals with intensity corrections
    related to self-absorption by a layered physics
    package. the class uses information from a material
    and instrument class to determine these corrections.

    """
