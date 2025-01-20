import numpy as np
from numba import njit
from hexrd.core import constants

ap_2 = constants.cuA_2
sc = constants.sc


@njit(cache=True, nogil=True)
def getPyramid(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    if (np.abs(x) <= z) and (np.abs(y) <= z):
        return 1

    elif (np.abs(x) <= -z) and (np.abs(y) <= -z):
        return 2

    elif (np.abs(z) <= x) and (np.abs(y) <= x):
        return 3

    elif (np.abs(z) <= -x) and (np.abs(y) <= -x):
        return 4

    elif (np.abs(x) <= y) and (np.abs(z) <= y):
        return 5

    elif (np.abs(x) <= -y) and (np.abs(z) <= -y):
        return 6


@njit(cache=True, nogil=True)
def cu2ro(cu):
    ho = cu2ho(cu)
    return ho2ro(ho)


@njit(cache=True, nogil=True)
def cu2ho(cu):
    ma = np.max(np.abs(cu))
    assert ma <= ap_2, "point outside cubochoric grid"
    pyd = getPyramid(cu)

    if pyd == 1 or pyd == 2:
        sXYZ = cu
    elif pyd == 3 or pyd == 4:
        sXYZ = np.array([cu[1], cu[2], cu[0]])
    elif pyd == 5 or pyd == 6:
        sXYZ = np.array([cu[2], cu[0], cu[1]])

    xyz = sXYZ * sc
    ma = np.max(np.abs(xyz))
    if ma < 1E-8:
        return np.array([0.0, 0.0, 0.0])

    ma2 = np.max(np.abs(xyz[0:2]))
    if ma2 < 1E-8:
        LamXYZ = np.array([0.0, 0.0, constants.pref * xyz[2]])

    else:
        if np.abs(xyz[1]) <= np.abs(xyz[0]):
            q = (np.pi/12.0) * xyz[1]/xyz[0]
            c = np.cos(q)
            s = np.sin(q)
            q = constants.prek * xyz[0] / np.sqrt(np.sqrt(2.0)-c)
            T1 = (np.sqrt(2.0) * c - 1.0) * q
            T2 = np.sqrt(2.0) * s * q
        else:
            q = (np.pi/12.0) * xyz[0]/xyz[1]
            c = np.cos(q)
            s = np.sin(q)
            q = constants.prek * xyz[1] / np.sqrt(np.sqrt(2.0)-c)
            T1 = np.sqrt(2.0) * s * q
            T2 = (np.sqrt(2.0) * c - 1.0) * q

        c = T1**2 + T2**2
        s = np.pi * c / (24.0 * xyz[2]**2)
        c = np.sqrt(np.pi) * c / np.sqrt(24.0) / xyz[2]
        q = np.sqrt( 1.0 - s )
        LamXYZ = np.array([T1 * q, T2 * q, constants.pref * xyz[2] - c])

    if pyd == 1 or pyd == 2:
        return LamXYZ
    elif pyd == 3 or pyd == 4:
        return np.array([LamXYZ[2], LamXYZ[0], LamXYZ[1]])
    elif pyd == 5 or pyd == 6:
        return np.array([LamXYZ[1], LamXYZ[2], LamXYZ[0]])


@njit(cache=True, nogil=True)
def ho2ro(ho):
    ax = ho2ax(ho)
    return ax2ro(ax)


@njit(cache=True, nogil=True)
def ho2ax(ho):
    hmag = np.linalg.norm(ho[:])**2
    if hmag < 1E-8:
        return np.array([0.0, 0.0, 1.0, 0.0])
    hm = hmag
    hn = ho/np.sqrt(hmag)
    s = constants.tfit[0] + constants.tfit[1] * hmag
    for ii in range(2, 21):
        hm = hm*hmag
        s = s + constants.tfit[ii] * hm
    s = 2.0 * np.arccos(s)
    diff = np.abs(s - np.pi)
    if diff < 1E-8:
        return np.array([hn[0], hn[1], hn[2], np.pi])
    else:
        return np.array([hn[0], hn[1], hn[2], s])


@njit(cache=True, nogil=True)
def ax2ro(ax):
    if np.abs(ax[3]) < 1E-8:
        return np.array([0.0, 0.0, 1.0, 0.0])

    elif np.abs(ax[3] - np.pi) < 1E-8:
        return np.array([ax[0], ax[1], ax[2], np.inf])

    else:
        return np.array([ax[0], ax[1], ax[2], np.tan(ax[3]*0.5)])


@njit(cache=True, nogil=True)
def ro2qu(ro):
    ax = ro2ax(ro)
    return ax2qu(ax)


@njit(cache=True, nogil=True)
def ro2ax(ro):
    if np.abs(ro[3]) < 1E-8:
        return np.array([0.0, 0.0, 1.0, 0.0])
    elif ro[3] == np.inf:
        return np.array([ro[0], ro[1], ro[2], np.pi])
    else:
        ang = 2.0*np.arctan(ro[3])
        mag = 1.0/np.linalg.norm(ro[0:3])
        return np.array([ro[0]*mag, ro[1]*mag, ro[2]*mag, ang])


@njit(cache=True, nogil=True)
def ax2qu(ro):
    if np.abs(ro[3]) < 1E-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        c = np.cos(ro[3]*0.5)
        s = np.sin(ro[3]*0.5)
        return np.array([c, ro[0]*s, ro[1]*s, ro[2]*s])
