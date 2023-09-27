import numpy as np
from hexrd.material import Material

class filterpack():
    """this class deals with intensity corrections
    related to absorption by the filter pack.
    """
    def __init__(self,
                 material)