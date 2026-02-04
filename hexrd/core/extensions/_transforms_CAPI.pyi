from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def anglesToGVec(
    angs: NDArray[np.float64],  # (n, 3)
    bHat_l: NDArray[np.float64],  # (3,)
    eHat_l: NDArray[np.float64],  # (3,)
    chi: float,
    rMat_c: NDArray[np.float64],  # (3, 3)
) -> NDArray[np.float64]:  # (n, 3)
    ...

def anglesToDVec(
    angs: NDArray[np.float64],  # (n, 3)
    bHat_l: NDArray[np.float64],  # (3,)
    eHat_l: NDArray[np.float64],  # (3,)
    chi: float,
    rMat_c: NDArray[np.float64],  # (3, 3)
) -> NDArray[np.float64]:  # (n, 3)
    ...

def gvecToDetectorXY(
    gVec_c: NDArray[np.float64],  # (n, 3)
    rMat_d: NDArray[np.float64],  # (3, 3)
    rMat_s: NDArray[np.float64],  # (3, 3)
    rMat_c: NDArray[np.float64],  # (3, 3)
    tVec_d: NDArray[np.float64],  # (3,)
    tVec_s: NDArray[np.float64],  # (3,)
    tVec_c: NDArray[np.float64],  # (3,)
    beamVec: NDArray[np.float64],  # (3,)
) -> NDArray[np.float64]:  # (n, 2)
    ...

def gvecToDetectorXYArray(
    gVec_c: NDArray[np.float64],  # (n, 3)
    rMat_d: NDArray[np.float64],  # (3, 3)
    rMat_s: NDArray[np.float64],  # (n, 3, 3)
    rMat_c: NDArray[np.float64],  # (3, 3)
    tVec_d: NDArray[np.float64],  # (3,)
    tVec_s: NDArray[np.float64],  # (3,)
    tVec_c: NDArray[np.float64],  # (3,)
    beamVec: NDArray[np.float64],  # (3,)
) -> NDArray[np.float64]:  # (n, 2)
    ...

def detectorXYToGvec(
    xy_det: NDArray[np.float64],  # (n, 2)
    rMat_d: NDArray[np.float64],  # (3, 3)
    rMat_s: NDArray[np.float64],  # (3, 3)
    tVec_d: NDArray[np.float64],  # (3,)
    tVec_s: NDArray[np.float64],  # (3,)
    tVec_c: NDArray[np.float64],  # (3,)
    beamVec: NDArray[np.float64],  # (3,)
    etaVec: NDArray[np.float64],  # (3,)
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Returns:
      ((tTh, eta), gVec_l)
    where:
      tTh: (n,) float64
      eta: (n,) float64
      gVec_l: (n, 3) float64
    """
    ...

def detectorXYToGvecArray(
    xy_det: NDArray[np.float64],  # (n, 2)
    rMat_d: NDArray[np.float64],  # (3, 3)
    rMat_s: NDArray[np.float64],  # (n, 3, 3)
    tVec_d: NDArray[np.float64],  # (3,)
    tVec_s: NDArray[np.float64],  # (3,)
    tVec_c: NDArray[np.float64],  # (3,)
    beamVec: NDArray[np.float64],  # (3,)
    etaVec: NDArray[np.float64],  # (3,)
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Returns:
      ((tTh, eta), gVec_l)
    where:
      tTh: (n,) float64
      eta: (n,) float64
      gVec_l: (n, 3) float64
    """
    ...

def oscillAnglesOfHKLs(
    hkls: NDArray[np.float64],  # (n, 3)
    chi: float,
    rMat_c: NDArray[np.float64],  # (3, 3)
    bMat: NDArray[np.float64],  # (3, 3)
    wavelength: float,
    vInv_s: NDArray[np.float64],  # (6,)
    beamVec: NDArray[np.float64],  # (3,)
    etaVec: NDArray[np.float64],  # (3,)
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns:
      (oangs0, oangs1) each (n, 3) float64
    """
    ...

def unitRowVector(
    vecIn: NDArray[np.float64],  # (n,)
) -> NDArray[np.float64]:  # (n,)
    ...

def unitRowVectors(
    vecIn: NDArray[np.float64],  # (m, n)
) -> NDArray[np.float64]:  # (m, n)
    ...

def makeDetectorRotMat(
    tiltAngles: NDArray[np.float64],  # (3,)
) -> NDArray[np.float64]:  # (3, 3)
    ...

def makeOscillRotMat(
    oscillAngles: NDArray[np.float64],  # (2,)
) -> NDArray[np.float64]:  # (3, 3)
    ...

def makeOscillRotMatArray(
    chi: float,
    omeArray: NDArray[np.float64],  # (n,)
) -> NDArray[np.float64]:  # (n, 3, 3)
    ...

def makeRotMatOfExpMap(
    expMap: NDArray[np.float64],  # (3,)
) -> NDArray[np.float64]:  # (3, 3)
    ...

def makeRotMatOfQuat(
    quat: NDArray[np.float64],  # (n, 4)
) -> NDArray[np.float64]:  # (n, 3, 3)
    ...

def makeBinaryRotMat(
    axis: NDArray[np.float64],  # (3,)
) -> NDArray[np.float64]:  # (3, 3)
    ...

def makeEtaFrameRotMat(
    bHat: NDArray[np.float64],  # (3,)
    eHat: NDArray[np.float64],  # (3,)
) -> NDArray[np.float64]:  # (3, 3)
    ...

def validateAngleRanges(
    angList: NDArray[np.float64],  # (na,)
    angMin: NDArray[np.float64],  # (nmin,)
    angMax: NDArray[np.float64],  # (nmin,)
    ccw: object,  # expects bool-ish
) -> NDArray[np.bool_]:  # (na,)
    ...

def rotate_vecs_about_axis(
    angles: NDArray[np.float64],  # (1,) or (n,)
    axes: NDArray[np.float64],  # (1,3) or (n,3)
    vecs: NDArray[np.float64],  # (n,3)
) -> NDArray[np.float64]:  # (n,3)
    ...

def quat_distance(
    q1: NDArray[np.float64],  # (4,)
    q2: NDArray[np.float64],  # (4,)
    qsym: NDArray[np.float64],  # (4, nsym)
) -> float: ...
def homochoricOfQuat(
    quat: NDArray[np.float64],  # (n, 4)
) -> NDArray[np.float64]:  # (n, 3)
    ...
