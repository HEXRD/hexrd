import numpy as np
from hexrd.constants import FZtypeArray, FZorderArray
from hexrd.utils.decorators import numba_njit_if_available
from hexrd import constants

if constants.USE_NUMBA:
    from numba import prange
else:
    prange = range

@numba_njit_if_available(cache=True, nogil=True)
def insideFZ(ro, pgnum):
    return True