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
    if np.abs(ro[3]) >= np.sqrt(3.0):
        return False
    else:
        rod = ro[0:3] * ro[3]

    c1 = (np.abs(rod[2]) <= constants.BP[FZorder-1])

    if c1:
        if   FZorder == 2:
            c2 = np.logical_and(np.abs(rod[0]) <= 1.0, 
                                np.abs(rod[1]) <= 1.0)

        elif FZorder == 3:
            srt = np.sqrt(3.0)/2.0
            c2 = np.abs(srt*rod[0] + 0.5*rod[1]) <= 1.0
            c3 = np.abs(srt*rod[0] - 0.5*rod[1]) <= 1.0
            c4 = np.abs(rod[1]) <= 1.0
            return np.logical_and(c2, 
                   np.logical_and(c3, c4))

        elif FZorder == 4:
            r22 = 1.0/np.sqrt(2.0)
            c2 = np.logical_and(np.abs(rod[0]) <= 1.0,
                                np.abs(rod[1]) <= 1.0)
            c3 = np.logical_and(r22*np.abs(rod[0]+rod[1]) <= 1.0,
                                r22*np.abs(rod[0]-rod[1]) <= 1.0)
            return np.logical_and(c2, c3)

        elif FZorder == 6:
            srt = np.sqrt(3.0)/2.0
            c2 = np.abs(0.5*rod[0] + srt*rod[1]) < 1.0
            c2 = np.logical_and(c2,
                 np.abs(srt*rod[0] + 0.5*rod[1]) < 1.0)
            c2 = np.logical_and(c2,
                 np.abs(0.5*rod[0] - srt*rod[1]) < 1.0)
            c2 = np.logical_and(c2,
                 np.abs(srt*rod[0] - 0.5*rod[1]) < 1.0)
            c2 = np.logical_and(c2, 
                 np.logical_and(np.abs(rod[0]) <= 1.0,
                                np.abs(rod[1]) <= 1.0))
            return c2
    else:
        return False

@numba_njit_if_available(cache=True, nogil=True)
def insideCubicFZ(ro, kwrd):
    rod = np.abs(ro[0:3] * ro[3])

    if kwrd == 'oct':
        c1 = (np.max(rod) - constants.BP[3]) <= 1E-8
    else:
        c1 = True

    c2 = (rod[0]+rod[1]+rod[2] - 1.0) <= 1E-8
    res = np.logical_and(c1, c2)
    return res

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

