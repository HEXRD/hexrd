from __future__ import annotations

from typing import Final, TypeAlias
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

def _npzIndex(buffer: object, metadata_only: bool) -> dict[str, tuple[int, int, int, int]] | None: ...
def _csrPixelIds(offsets: NDArray, columns: NDArray, width: int, output: NDArray[np.int64]) -> None: ...
def _ringMasks(tth: FloatArray, eta: NDArray, ranges: FloatArray, n_eta: int) -> tuple[NDArray[np.int64], NDArray[np.bool_]]: ...
def _binPixels(offsets: NDArray, ids: NDArray, values: FloatArray, rows: NDArray, masks: NDArray, eta: NDArray, threshold: float, output: FloatArray) -> None: ...
def _reduceFiber(fiber: FloatArray, symmetry: FloatArray, /) -> FloatArray | None: ...
def _scoreOrientationBins(bins: NDArray[np.int32], status: NDArray[np.int8]) -> FloatArray: ...

Vec3: TypeAlias = FloatArray
Mat3: TypeAlias = FloatArray

RotMatStack: TypeAlias = FloatArray
Angles: TypeAlias = FloatArray
DetectorXY: TypeAlias = FloatArray

def make_binary_rot_mat(a: Vec3, /) -> Mat3:
    """
    Compute a 3x3 rotation matrix from a binary vector `a`.

    Parameters
    ----------
    a
        3-vector.

    Returns
    -------
    3x3 rotation matrix.
    """

def make_rot_mat_of_exp_map(e: Vec3, /) -> Mat3:
    """
    Compute a 3x3 rotation matrix from an exponential map vector `e`.

    Parameters
    ----------
    e
        3-vector (axis * angle).

    Returns
    -------
    3x3 rotation matrix.
    """

def makeOscillRotMat(chi: float, ome: FloatArray, /) -> RotMatStack:
    """
    Generate a stack of oscillation rotation matrices.

    The underlying C++ returns an array with shape (3*N, 3), where each
    consecutive 3-row block is a 3x3 matrix for the corresponding omega.

    Parameters
    ----------
    chi
        Oscillation axis tilt angle (radians).
    ome
        Omega angles, shape (N,).

    Returns
    -------
    Array of shape (3*N, 3) containing N rotation matrices (3x3 blocks).
    """

def anglesToGVec(
    angs: Angles,
    bHat_l: Vec3,
    eHat_l: Vec3,
    chi: float,
    rMat_c: Mat3,
    /,
) -> FloatArray:
    """
    Convert angles to g-vectors in the crystal frame.

    Parameters
    ----------
    angs
        Angles array, shape (N, 3).
    bHat_l
        Beam direction (lab), shape (3,).
    eHat_l
        Reference direction (lab), shape (3,).
    chi
        Oscillation axis tilt angle (radians).
    rMat_c
        Crystal rotation matrix, shape (3, 3).

    Returns
    -------
    g-vectors, shape (N, 3).
    """

def anglesToDVec(
    angs: Angles,
    bHat_l: Vec3,
    eHat_l: Vec3,
    chi: float,
    rMat_c: Mat3,
    /,
) -> FloatArray:
    """
    Convert angles to d-vectors in the crystal frame.

    Returns an array of shape (N, 3).
    """

def gvecToDetectorXY(
    gVec_c: FloatArray,
    rMat_d: Mat3,
    rMat_s: FloatArray,
    rMat_c: Mat3,
    tVec_d: Vec3,
    tVec_s: Vec3,
    tVec_c: Vec3,
    beamVec: Vec3,
    /,
) -> DetectorXY:
    """
    Convert g-vectors (crystal frame) to detector XY.

    Parameters
    ----------
    gVec_c
        g-vectors, shape (N, 3).
    rMat_d
        Detector rotation matrix, shape (3, 3).
    rMat_s
        Stack of sample rotation matrices in the "3-row blocks" layout:
        shape (3*N, 3).
    rMat_c
        Crystal rotation matrix, shape (3, 3).
    tVec_d, tVec_s, tVec_c
        Translation vectors, each shape (3,).
    beamVec
        Beam direction vector, shape (3,).

    Returns
    -------
    Detector XY, shape (N, 2). May contain NaNs for invalid intersections.
    """

def gvec_to_detector_xy_one(
    gVec_c: Vec3,
    rMat_d: Mat3,
    rMat_sc: Mat3,
    tVec_d: Vec3,
    bHat_l: Vec3,
    nVec_l: Vec3,
    num: float,
    P0_l: Vec3,
    /,
) -> FloatArray:
    """
    Convert a single g-vector to detector XY.

    Returns
    -------
    XY vector, shape (2,). Returns NaNs if invalid.
    """

def gvecToDetectorXYFromAngles(
    chi: float,
    omes: FloatArray,
    gVec_c: FloatArray,
    rMat_d: Mat3,
    rMat_c: Mat3,
    tVec_d: Vec3,
    tVec_s: Vec3,
    tVec_c: Vec3,
    beamVec: Vec3,
    /,
) -> DetectorXY:
    """
    Convert g-vectors to detector XY using chi and omega angles.

    Parameters
    ----------
    chi
        Oscillation axis tilt angle.
    omes
        Omega array, shape (N,). (Matches the C++ signature's VectorXd usage.)
    gVec_c
        g-vectors, shape (N, 3).

    Returns
    -------
    Detector XY, shape (N, 2).
    """

def anglesToDetectorXY(
    chi: float,
    omes: Angles,
    rMat_d: Mat3,
    rMat_c: Mat3,
    tVec_d: Vec3,
    tVec_s: Vec3,
    tVec_c: Vec3,
    beamVec: Vec3,
    /,
) -> DetectorXY:
    """
    Convert angles directly to detector XY.

    Parameters
    ----------
    chi
        Oscillation axis tilt angle.
    omes
        Angles array, shape (N, 3). (Despite the name, this is the full angles array.)
    rMat_d, rMat_c
        Rotation matrices, shape (3, 3).
    tVec_d, tVec_s, tVec_c
        Translation vectors, shape (3,).
    beamVec
        Beam direction vector, shape (3,).

    Returns
    -------
    Detector XY, shape (N, 2). May contain NaNs for invalid intersections.
    """


def quaternionRotations(quaternions: NDArray[np.float64], /) -> NDArray[np.float64]: ...
def orientationHitStatus(
    ring_maps: NDArray[np.float64], dpix_eta: int, dpix_omega: int, threshold: float, /,
) -> NDArray[np.int8]: ...
def scoreOrientations(
    rotation_matrices: NDArray[np.float64],
    hkls: NDArray[np.float64],
    bmat: NDArray[np.float64],
    wavelength: float,
    beam: NDArray[np.float64],
    eta_edges: NDArray[np.float64],
    omega_edges: NDArray[np.float64],
    valid_eta: NDArray[np.float64],
    valid_omega: NDArray[np.float64],
    omega_offset: float,
    ring_maps: NDArray[np.float64],
    ring_for_hkl: NDArray[np.intp],
    dpix_eta: int,
    dpix_omega: int,
    threshold: float,
    hit_status: NDArray[np.int8] | None = ...,
    bin_indices: NDArray[np.int32] | None = ...,
    /,
) -> NDArray[np.float64]: ...
def oscillAnglesOfHKLs(
    hkls: FloatArray, chi: float, rmat_c: Mat3, bmat: Mat3, wavelength: float,
    v_inv: FloatArray, beam_vec: Vec3, eta_vec: Vec3, /,
) -> tuple[FloatArray, FloatArray]: ...

def quat_distance(q1: FloatArray, q2: FloatArray, qsym: FloatArray, /) -> float: ...

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
) -> tuple[NDArray[np.float64], ...]: ...

def makeEtaFrameRotMat(
    bvec_l: NDArray[np.float64],
    evec_l: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def rotate_vecs_about_axis(
    angles: NDArray[np.float64],
    axis: NDArray[np.float64],
    vecs: NDArray[np.float64],
    /,
) -> NDArray[np.float64]: ...

def projectGvecs(
    vectors: FloatArray, rmat_d: Mat3, rmat_s: FloatArray, rmat_c: Mat3,
    tvec_d: Vec3, tvec_s: Vec3, tvec_c: Vec3, beam: Vec3 | None = ..., /,
) -> DetectorXY: ...

def projectGvecsArray(
    vectors: FloatArray, rmat_d: Mat3, rmat_s: FloatArray, rmat_c: Mat3,
    tvec_d: Vec3, tvec_s: Vec3, tvec_c: Vec3, beam: Vec3 | None = ..., /,
) -> DetectorXY: ...
