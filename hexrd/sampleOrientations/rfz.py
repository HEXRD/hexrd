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
def insideCyclicFZ(ro, FZorder):
    res = False
    if ro[3] == np.inf:
        if FZorder == 2:
            if ro[1] == 0.0:
                res = True
        else:
            if ro[2] == 0.0:
                res = True
    else:
        if FZorder == 2:
            res = np.abs(ro[1]*ro[3]) <= constants.BP[FZorder-1]
        else:
            res = np.abs(ro[2]*ro[3]) <= constants.BP[FZorder-1]

    return res

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
        return insideCyclicFZ(ro, FZorder)
    elif FZtype == 2:
        if ro[3] == np.inf:
            return False
        else:
            return insideDihedralFZ(ro, FZorder)
    elif FZtype == 3:
        if ro[3] == np.inf:
            return False
        else:
            return insideCubicFZ(ro, 'tet')
    elif FZtype == 4:
        if ro[3] == np.inf:
            return False
        else:
            return insideCubicFZ(ro, 'oct')

