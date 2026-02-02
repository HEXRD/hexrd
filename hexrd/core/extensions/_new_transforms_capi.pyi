from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def anglesToGVec(
    angs: NDArray[np.float64],
    beam_vec: NDArray[np.float64],
    eta_vec: NDArray[np.float64],
    chi: float,
    rmat_c: NDArray[np.float64],
) -> NDArray[np.float64]: ...

def anglesToDVec(
    angs: NDArray[np.float64],
    beam_vec: NDArray[np.float64],
    eta_vec: NDArray[np.float64],
    chi: float,
    rmat_c: NDArray[np.float64],
) -> NDArray[np.float64]: ...

def gvecToDetectorXY(
    gvec_c: NDArray[np.float64],
    rmat_d: NDArray[np.float64],
    rmat_s: NDArray[np.float64],
    rmat_c: NDArray[np.float64],
    tvec_d: NDArray[np.float64],
    tvec_s: NDArray[np.float64],
    tvec_c: NDArray[np.float64],
    beam_vec: NDArray[np.float64] | None = ...,
    /,
) -> NDArray[np.float64]: ...

def gvecToDetectorXYArray(
    gVec_c: NDArray[np.float64],
    rMat_d: NDArray[np.float64],
    rMat_s: NDArray[np.float64],
    rMat_c: NDArray[np.float64],
    tVec_d: NDArray[np.float64],
    tVec_s: NDArray[np.float64],
    tVec_c: NDArray[np.float64],
    beamVec: NDArray[np.float64] | None = ...,
    /,
) -> NDArray[np.float64]: ...

def detectorXYToGvec(
    xy_det: NDArray[np.float64],
    rmat_d: NDArray[np.float64],
    rmat_s: NDArray[np.float64],
    tvec_d: NDArray[np.float64],
    tvec_s: NDArray[np.float64],
    tvec_c: NDArray[np.float64],
    beam_vec: NDArray[np.float64],
    eta_vec: NDArray[np.float64],
    /,
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]: ...

def oscillAnglesOfHKLs(
    hkls: NDArray[np.float64],
    chi: float,
    rMat_c: NDArray[np.float64],
    bMat: NDArray[np.float64],
    wavelen: float,
    vInv: NDArray[np.float64],
    beamVec: NDArray[np.float64],
    etaVec: NDArray[np.float64],
    /,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def unitRowVector(
    vecIn: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def unitRowVectors(
    vecIn: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def makeOscillRotMat(
    chi: float,
    ome: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def makeRotMatOfExpMap(
    exp_map: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def makeDetectorRotMat(
    tiltAngles: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def makeBinaryRotMat(
    axis: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def makeEtaFrameRotMat(
    bvec_l: NDArray[np.float64],
    evec_l: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def validateAngleRanges(
    ang_list: NDArray[np.float64],
    start_ang: NDArray[np.float64],
    stop_ang: NDArray[np.float64],
    ccw: bool = ...,
    /,
) -> NDArray[np.bool_]: ...

def rotate_vecs_about_axis(
    angles: NDArray[np.float64],
    axis: NDArray[np.float64],
    vecs: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def quat_distance(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
    qsym: NDArray[np.float64],
    /,
) -> float: ...
