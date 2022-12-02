import numpy as np
from hexrd.constants import FZtypeArray, FZorderArray
from hexrd.utils.decorators import numba_njit_if_available
from hexrd import constants

if constants.USE_NUMBA:
    from numba import prange
else:
    prange = range

@numba_njit_if_available(cache=True, nogil=True)
def getFZtypeandOrder(pgnum):
    FZtype = FZtypeArray[pgnum-1]
    FZorder = FZorderArray[pgnum-1]
    return np.array([FZtype, FZorder])

@numba_njit_if_available(cache=True, nogil=True)
def insideCyclicFZ(ro, FZtype, FZorder):
    return True

@numba_njit_if_available(cache=True, nogil=True)
def insideDihedralFZ(ro, FZorder):
    return True

@numba_njit_if_available(cache=True, nogil=True)
def insideCubicFZ(ro, kwrd):
    return True

@numba_njit_if_available(cache=True, nogil=True)
def insideFZ(ro, pgnum):
    res = getFZtypeandOrder(pgnum)
    FZtype = res[0] 
    FZorder = res[1]

    if FZtype == 0:
        return True
    elif FZtype == 1:
        return insideCyclicFZ(ro, FZtype, FZorder)
    elif FZtype == 2:
        return insideDihedralFZ(ro, FZorder)
    elif FZtype == 3:
        return insideCubicFZ(ro, 'tet')
    elif FZtype == 4:
        return insideCubicFZ(ro, 'oct')

