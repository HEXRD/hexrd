# -*- coding: utf-8 -*-
"""
Transforms module implementation using a support C extension module.

Currently, this implementation contains code for the following functions:

 - angles_to_gvec
 - angles_to_dvec
 - gvec_to_xy
 - xy_to_gvec (partial)

 - unit_vector
 - make_rmat_of_exp_map
 - make_binary_rmat
 - make_beam_rmat
 - validate_angle_ranges
 - rotate_vecs_about_axis
 - quat_distance

There are also some functions that maybe would be needed in the transforms
module:
 - makeGVector
 - makeRotMatOfQuat
 - homochoricOfQuat
"""
from typing import Optional, Tuple, Union
import numpy as np

from hexrd.extensions import _new_transforms_capi as _impl
from hexrd.distortion.distortionabc import DistortionABC
from hexrd import constants as cnst


def angles_to_gvec(
    angs: np.ndarray,
    beam_vec: Optional[np.ndarray] = None,
    eta_vec: Optional[np.ndarray] = None,
    chi: Optional[float] = None,
    rmat_c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """

    Takes triplets of angles in the beam frame (2*theta, eta[, omega])
    to components of unit G-vectors in the LAB frame.  If the omega
    values are not trivial (i.e. angs[:, 2] = 0.), then the components
    are in the SAMPLE frame.  If the crystal rmat is specified and
    is not the identity, then the components are in the CRYSTAL frame.

    G vectors here referes to the reciprocal lattice vectors.

    Parameters
    ----------
    angs : ndarray
        The euler angles of diffraction. The last dimension must be 2 or 3. In
        (2*theta, eta[, omega]) format.
    beam_vec : ndarray, optional
        Unit vector pointing towards the X-ray source in the lab frame.
        Defaults to [0,0,-1]
    eta_vec : ndarray, optional
        Vector defining eta=0 in the lab frame.  Defaults to [1,0,0]
    chi : float, optional
        The inclination angle of the sample frame about the lab frame X.
    rmat_c : ndarray, optional
        The change of basis matrix from the reciprocal frame to the crystal
        frame.  Defaults to the identity.

    Returns
    -------
    ndarray
        (3,n) array of unit reciprocal lattice vectors, frame depends on the
        input parameters
    """

    orig_ndim = angs.ndim

    # if only a pair is provided... convert to a triplet with omegas == 0
    # so that behavior is preserved.
    if angs.shape[-1] == 2:
        angs = np.hstack((angs, np.zeros(angs.shape[:-1] + (1,))))

    angs = np.ascontiguousarray(np.atleast_2d(angs))
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    beam_vec = np.ascontiguousarray(beam_vec.flatten())
    eta_vec = eta_vec if eta_vec is not None else cnst.eta_vec
    eta_vec = np.ascontiguousarray(eta_vec.flatten())
    chi = 0.0 if chi is None else float(chi)
    rmat_c = (
        cnst.identity_3x3 if rmat_c is None else np.ascontiguousarray(rmat_c)
    )

    result = _impl.anglesToGVec(angs, beam_vec, eta_vec, chi, rmat_c)

    return result[0] if orig_ndim == 1 else result


def angles_to_dvec(
    angs: np.ndarray,
    beam_vec: Optional[np.ndarray] = None,
    eta_vec: Optional[np.ndarray] = None,
    chi: Optional[float] = None,
    rmat_c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """

    Takes triplets of angles in the beam frame (2*theta, eta[, omega])
    to components of unit diffraction vectors in the LAB frame.  If the omega
    values are not trivial (i.e. angs[:, 2] = 0.), then the components
    are in the SAMPLE frame.  If the crystal rmat is specified and
    is not the identity, then the components are in the CRYSTAL frame.


    Parameters
    ----------
    angs : ndarray
        The euler angles of diffraction. The last dimension must be 2 or 3. In
        (2*theta, eta[, omega]) format.
    beam_vec : ndarray, optional
        Unit vector pointing towards the X-ray source in the lab frame.
        Defaults to [0,0,-1]
    eta_vec : ndarray, optional
        Vector defining eta=0 in the lab frame.  Defaults to [1,0,0]
    chi : float, optional
        The inclination angle of the sample frame about the lab frame X.
    rmat_c : ndarray, optional
        The change of basis matrix from the reciprocal frame to the crystal
        frame.  Defaults to the identity.

    Returns
    -------
    ndarray
        (3,n) array of unit diffraction vectors, frame depends on the input
        parameters
    """
    # TODO: Improve capi to avoid multiplications when rmat_c is None
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.eta_vec

    # if only a pair is provided... convert to a triplet with omegas == 0
    # so that behavior is preserved.
    if angs.shape[-1] == 2:
        angs = np.hstack((angs, np.zeros(angs.shape[:-1] + (1,))))

    angs = np.ascontiguousarray(np.atleast_2d(angs))

    beam_vec = np.ascontiguousarray(beam_vec.flatten())
    eta_vec = np.ascontiguousarray(eta_vec.flatten())
    rmat_c = (
        np.ascontiguousarray(rmat_c)
        if rmat_c is not None
        else cnst.identity_3x3
    )
    chi = 0.0 if chi is None else float(chi)

    return _impl.anglesToDVec(angs, beam_vec, eta_vec, chi, rmat_c)


def makeGVector(hkl: np.ndarray, bMat: np.ndarray) -> np.ndarray:
    """
    Take a crystal relative b matrix onto a list of hkls to output unit
    reciprocal latice vectors (a.k.a. lattice plane normals)


    Parameters
    ----------
    hkl : ndarray
        (3,n) ndarray of n hstacked reciprocal lattice vector component
        triplets
    bMat : ndarray
        (3,3) ndarray of the change of basis matrix from the reciprocal
        lattice to the crystal reference frame

    Returns
    -------
    ndarray
        (3,n) ndarray of n unit reciprocal lattice vectors (a.k.a. lattice
        plane normals)

    """
    assert hkl.shape[0] == 3, 'hkl input must be (3, n)'
    return unit_vector(np.dot(bMat, hkl))


def gvec_to_xy(
    gvec_c: np.ndarray,
    rmat_d: np.ndarray,
    rmat_s: np.ndarray,
    rmat_c: np.ndarray,
    tvec_d: np.ndarray,
    tvec_s: np.ndarray,
    tvec_c: np.ndarray,
    beam_vec: Optional[np.ndarray] = None,
    vmat_inv: Optional[np.ndarray] = None,
    bmat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Takes a concatenated list of reciprocal lattice vectors components in the
    CRYSTAL FRAME to the specified detector-relative frame, subject to the
    following:

        1) it must be able to satisfy a bragg condition
        2) the associated diffracted beam must intersect the detector plane

    Parameters
    ----------
    gvec_c : array_like
        ([N,] 3) G-vector components in the CRYSTAL FRAME.
    rmat_d : array_like
        The (3, 3) COB matrix taking components in the
        DETECTOR FRAME to the LAB FRAME
    rmat_s : array_like
        The ([N,] 3, 3) COB matrix taking components in the SAMPLE FRAME to the
        LAB FRAME. It may be a single (3, 3) rotation matrix to use for all
        gvec_c, or just one rotation matrix per gvec.
    rmat_c : array_like
        The (3, 3) COB matrix taking components in the
        CRYSTAL FRAME to the SAMPLE FRAME
    tvec_d : array_like
        The (3, ) translation vector connecting LAB FRAME to DETECTOR FRAME
    tvec_s : array_like
        The (3, ) translation vector connecting LAB FRAME to SAMPLE FRAME
    tvec_c : array_like
        The ([M,] 3, ) translation vector(s) connecting SAMPLE FRAME to
        CRYSTAL FRAME
    beam_vec : array_like, optional
        The (3, ) incident beam propagation vector components in the LAB FRAME;
        the default is [0, 0, -1], which is the standard setting.
    vmat_inv : array_like, optional
        The (3, 3) matrix of inverse stretch tensor components in the
        SAMPLE FRAME.  The default is None, which implies a strain-free state
        (i.e. V = I).
    bmat : array_like, optional
        The (3, 3) COB matrix taking components in the
        RECIPROCAL LATTICE FRAME to the CRYSTAL FRAME; if supplied, it is
        assumed that the input `gvecs` are G-vector components in the
        RECIPROCL LATTICE FRAME (the default is None, which implies components
        in the CRYSTAL FRAME)

    Returns
    -------
    array_like
        The ([M, ][N, ] 2) array of [x, y] diffracted beam intersections for
        each of the N input G-vectors in the DETECTOR FRAME (all Z_d
        coordinates are 0 and excluded) and for each of the M candidate
        positions. For each input G-vector that cannot satisfy a Bragg
        condition or intersect the detector plane, [NaN, Nan] is returned.

    Notes
    -----
        Previously only a single candidate position was allowed. This is in
        fact a vectored version of the previous API function. It is backwards
        compatible, as passing single tvec_c is supported and has the same
        result.

    """
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec

    orig_ndim = gvec_c.ndim
    gvec_c = np.ascontiguousarray(np.atleast_2d(gvec_c))
    rmat_d = np.ascontiguousarray(rmat_d)
    rmat_s = np.ascontiguousarray(rmat_s)
    rmat_c = np.ascontiguousarray(rmat_c)
    tvec_d = np.ascontiguousarray(tvec_d.flatten())
    tvec_s = np.ascontiguousarray(tvec_s.flatten())
    tvec_c = np.ascontiguousarray(tvec_c.flatten())
    beam_vec = np.ascontiguousarray(beam_vec.flatten())

    # depending on the number of dimensions of rmat_s use either the array
    # version or the "scalar" (over rmat_s) version. Note that rmat_s is either
    # a 3x3 matrix (ndim 2) or an nx3x4 array of matrices (ndim 3)
    if rmat_s.ndim > 2:
        result = _impl.gvecToDetectorXYArray(
            gvec_c, rmat_d, rmat_s, rmat_c, tvec_d, tvec_s, tvec_c, beam_vec
        )
    else:
        result = _impl.gvecToDetectorXY(
            gvec_c, rmat_d, rmat_s, rmat_c, tvec_d, tvec_s, tvec_c, beam_vec
        )
    return result[0] if orig_ndim == 1 else result


def xy_to_gvec(
    xy_d: np.ndarray,
    rmat_d: np.ndarray,
    rmat_s: np.ndarray,
    tvec_d: np.ndarray,
    tvec_s: np.ndarray,
    tvec_c: np.ndarray,
    rmat_b: Optional[np.ndarray] = None,
    distortion: Optional[DistortionABC] = None,
    output_ref: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a list cartesian (x, y) pairs in the DETECTOR FRAME and calculates
    the associated reciprocal lattice (G) vectors and (bragg angle, azimuth)
    pairs with respect to the specified beam and azimth (eta) reference
    directions.

    Parameters
    ----------
    xy_d : array_like
        (n, 2) array of n (x, y) coordinates in DETECTOR FRAME
    rmat_d : array_like
        (3, 3) COB matrix taking components in the
        DETECTOR FRAME to the LAB FRAME
    rmat_s : array_like
        (3, 3) COB matrix taking components in the
        SAMPLE FRAME to the LAB FRAME
    tvec_d : array_like
        (3, ) translation vector connecting LAB FRAME to DETECTOR FRAME
    tvec_s : array_like
        (3, ) translation vector connecting LAB FRAME to SAMPLE FRAME
    tvec_c : array_like
        (3, ) translation vector connecting SAMPLE FRAME to CRYSTAL FRAME
    rmat_b : array_like, optional
        (3, 3) COB matrix taking components in the BEAM FRAME to the LAB FRAME;
        defaults to None, which implies the standard setting of identity.
    distortion : distortion class, optional
        Default is None
    output_ref : bool, optional
        If True, prepends the apparent bragg angle and azimuth with respect to
        the SAMPLE FRAME (ignoring effect of non-zero tvec_c)

    Returns
    -------
    array_like
        (n, 2) ndarray containing the (tth, eta) pairs associated with each
        (x, y) associated with gVecs
    array_like
        (n, 3) ndarray containing the associated G vector directions in the
        LAB FRAME
    array_like, optional
        if output_ref is True
    """
    # It seems that the output_ref version is not present as the argument
    # gets ignored

    rmat_b = rmat_b if rmat_b is not None else cnst.identity_3x3

    # the code seems to ignore this argument, assume output_ref == True not
    # implemented
    assert not output_ref, 'output_ref not implemented'

    if distortion is not None:
        xy_d = distortion.apply_inverse(xy_d)

    xy_d = np.ascontiguousarray(np.atleast_2d(xy_d))
    rmat_d = np.ascontiguousarray(rmat_d)
    rmat_s = np.ascontiguousarray(rmat_s)
    tvec_d = np.ascontiguousarray(tvec_d.flatten())
    tvec_s = np.ascontiguousarray(tvec_s.flatten())
    tvec_c = np.ascontiguousarray(tvec_c.flatten())
    beam_vec = np.ascontiguousarray((-rmat_b[:, 2]).flatten())
    eta_vec = np.ascontiguousarray(rmat_b[:, 0].flatten())
    return _impl.detectorXYToGvec(
        xy_d, rmat_d, rmat_s, tvec_d, tvec_s, tvec_c, beam_vec, eta_vec
    )


def oscill_angles_of_hkls(
    hkls: np.ndarray,
    chi: float,
    rmat_c: np.ndarray,
    bmat: np.ndarray,
    wavelength: float,
    v_inv: Optional[np.ndarray] = None,
    beam_vec: np.ndarray = cnst.beam_vec,
    eta_vec: np.ndarray = cnst.eta_vec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a list of unit reciprocal lattice vectors in crystal frame to the
    specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Parameters
    ----------
    hkls       -- (n, 3) ndarray of n reciprocal lattice vectors
                  in the CRYSTAL FRAME
    chi        -- float representing the inclination angle of the
                  oscillation axis (std coords)
    rmat_c     -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components
                  to SAMPLE FRAME
    bmat       -- (3, 3) ndarray, the COB taking RECIPROCAL LATTICE components
                  to CRYSTAL FRAME
    wavelength -- float representing the x-ray wavelength in Angstroms

    Optional Keyword Arguments:
    v_inv      -- (6, ) ndarray, vec representation inverse stretch tensor
                  components in the SAMPLE FRAME. The default is None, which
                  implies a strain-free state (i.e. V = I).
    beam_vec -- (3, 1) mdarray containing the incident beam direction
               components in the LAB FRAME
    eta_vec  -- (3, 1) mdarray containing the reference azimuth direction
               components in the LAB FRAME

    Outputs:
    ome0 -- (n, 3) ndarray containing the feasible (tTh, eta, ome) triplets
            for each input hkl (first solution)
    ome1 -- (n, 3) ndarray containing the feasible (tTh, eta, ome) triplets
            for each input hkl (second solution)

    Notes:
    ------------------------------------------------------------------------
    The reciprocal lattice vector, G, will satisfy the the Bragg condition
    when:

        b.T * G / ||G|| = -sin(theta)

    where b is the incident beam direction (k_i) and theta is the Bragg
    angle consistent with G and the specified wavelength. The components of
    G in the lab frame in this case are obtained using the crystal
    orientation, Rc, and the single-parameter oscillation matrix, Rs(ome):

        Rs(ome) * Rc * G / ||G||

    The equation above can be rearranged to yield an expression of the form:

        a*sin(ome) + b*cos(ome) = c

    which is solved using the relation:

        a*sin(x) + b*cos(x) = sqrt(a**2 + b**2) * sin(x + alpha)

        --> sin(x + alpha) = c / sqrt(a**2 + b**2)

    where:

        alpha = atan2(b, a)

     The solutions are:

                /
                |       arcsin(c / sqrt(a**2 + b**2)) - alpha
            x = <
                |  pi - arcsin(c / sqrt(a**2 + b**2)) - alpha
                \

    There is a double root in the case the reflection is tangent to the
    Debye-Scherrer cone (c**2 = a**2 + b**2), and no solution if the
    Laue condition cannot be satisfied (filled with NaNs in the results
    array here)
    """
    # this was oscillAnglesOfHKLs
    hkls = np.array(hkls, dtype=float, order='C')
    if v_inv is None:
        v_inv = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    else:
        v_inv = np.ascontiguousarray(v_inv.flatten())
    rmat_c = np.ascontiguousarray(rmat_c)
    beam_vec = np.ascontiguousarray(beam_vec.flatten())
    eta_vec = np.ascontiguousarray(eta_vec.flatten())
    bmat = np.ascontiguousarray(bmat)
    return _impl.oscillAnglesOfHKLs(
        hkls, chi, rmat_c, bmat, wavelength, v_inv, beam_vec, eta_vec
    )


def unit_vector(vec_in: np.ndarray) -> np.ndarray:
    """
    Normalize the input vector(s) to unit length.

    Parameters
    ----------
    vec_in : ndarray
        The input vector(s) (n,3) to normalize.

    Returns
    -------
    ndarray
        The normalized vector(s) of the same shape as the input.

    Raises
    ------
    ValueError
        If the input vector(s) do not have the correct shape.
    """
    vec_in = np.ascontiguousarray(vec_in)
    if vec_in.ndim == 1:
        return _impl.unitRowVector(vec_in)
    elif vec_in.ndim == 2:
        return _impl.unitRowVectors(vec_in)
    else:
        raise ValueError(
            "incorrect arg shape; must be 1-d or 2-d, yours is %d-d"
            % (vec_in.ndim)
        )


def make_detector_rot_mat(tilt_angles: np.ndarray) -> np.ndarray:
    """
    Form the (3, 3) tilt rotations from the tilt angle list:

    tilt_angles = [gamma_Xl, gamma_Yl, gamma_Zl] in radians
    Returns the (3, 3) rotation matrix from the detector frame to the lab frame
    """
    t_angles = np.ascontiguousarray(np.r_[tilt_angles].flatten())
    return _impl.makeDetectorRotMat(t_angles)


# make_sample_rmat in CAPI is split between makeOscillRotMat
# and makeOscillRotMatArray...


def make_oscill_rot_mat(oscillAngles):
    chi, ome = oscillAngles
    ome = np.atleast_1d(ome)
    result = _impl.makeOscillRotMat(chi, ome)
    return result.reshape((3, 3))


def make_oscill_rot_mat_array(chi, omeArray):
    arg = np.ascontiguousarray(omeArray)
    return _impl.makeOscillRotMat(chi, arg)


def make_sample_rmat(chi: float, ome: Union[float, np.ndarray]) -> np.ndarray:
    # TODO: Check this docstring
    """
    Make a rotation matrix representing the COB from sample frame to the lab
    frame.

    Parameters
    ----------
    chi : float
        The inclination angle of the sample frame about the lab frame Y.
    ome : float or ndarray
        The oscillation angle(s) about the sample frame Y.

    Returns
    -------
    ndarray
        A 3x3 rotation matrix representing the input oscillation angles.

    """
    ome_array = np.atleast_1d(ome)
    if ome is ome_array:
        ome_array = np.ascontiguousarray(ome_array)
        result = _impl.makeOscillRotMat(chi, ome_array)
    else:
        # converted to 1d array of 1 element, no need
        # to call ascontiguousarray, but need to remove
        # the outer dimension from the result
        result = _impl.makeOscillRotMat(chi, ome_array)
        result = result.reshape(3, 3)

    return result


def make_rmat_of_expmap(exp_map: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation matrix of an exponential map.

    Parameters
    ----------
    exp_map : array_like
        A 3-element sequence representing the exponential map n*phi.

    Returns
    -------
    ndarray
        A 3x3 rotation matrix representing the input exponential map
    """
    arg = np.ascontiguousarray(exp_map.flatten())
    return _impl.makeRotMatOfExpMap(arg)


def make_binary_rmat(axis: np.ndarray) -> np.ndarray:
    """
    Create a 180 degree rotation matrix about the input axis.

    Parameters
    ----------
    axis : ndarray
        3-element sequence representing the rotation axis, in radians.

    Returns
    -------
    ndarray
        A 3x3 rotation matrix representing a 180 degree rotation about the
        input axis.
    """

    arg = np.ascontiguousarray(axis.flatten())
    return _impl.makeBinaryRotMat(arg)


def make_beam_rmat(bvec_l: np.ndarray, evec_l: np.ndarray) -> np.ndarray:
    """
    Creates a COB matrix from the beam frame to the lab frame
    Note: beam and eta vectors must not be colinear

    Parameters
    ----------
    bvec_l : ndarray
        The beam vector in the lab frame, The (3, ) incident beam propagation
        vector components in the lab frame.
        the default is [0, 0, -1], which is the standard setting.
    evec_l : ndarray
        Vector defining eta=0 in the lab frame.  Defaults to [1,0,0]
    """
    arg1 = np.ascontiguousarray(bvec_l.flatten())
    arg2 = np.ascontiguousarray(evec_l.flatten())
    return _impl.makeEtaFrameRotMat(arg1, arg2)


def validate_angle_ranges(
    ang_list: Union[float, np.ndarray],
    start_angs: Union[float, np.ndarray],
    stop_angs: Union[float, np.ndarray],
    ccw: bool = True,
) -> np.ndarray[bool]:
    """
    Find out if angles are in the CCW or CW range from start to stop

    Parameters
    ----------
    ang_list : ndarray
        The list of angles to validate
    start_angs : ndarray
        The starting angles
    stop_angs : ndarray
        The stopping angles
    ccw : bool, optional
        If True, the angles are in the CCW range from start to stop.
        Defaults to True.

    Returns
    -------
    ndarray
        List of bools indicating if the angles are in the correct range

    """
    ang_list = np.atleast_1d(ang_list).astype(np.double, order='C')
    start_angs = np.atleast_1d(start_angs).astype(np.double, order='C')
    stop_angs = np.atleast_1d(stop_angs).astype(np.double, order='C')

    return _impl.validateAngleRanges(ang_list, start_angs, stop_angs, ccw)


def rotate_vecs_about_axis(
    angle: np.ndarray, axis: np.ndarray, vecs: np.ndarray
) -> np.ndarray:
    """
    Rotate vectors about an axis

    Parameters
    ----------
    angle : array_like
        Array of angles (len==1 or n)
    axis : ndarray
        Array of unit vectors (shape == (3,1) or (3,n))
    vecs : ndarray
        Array of vectors to rotate (shape == (3,n))

    Returns
    -------
    ndarray
        Array of rotated vectors (shape == (3,n))
    """
    angle = np.asarray(angle)
    axis = np.ascontiguousarray(axis.T)
    vecs = np.ascontiguousarray(vecs.T)
    result = _impl.rotate_vecs_about_axis(angle, axis, vecs)
    return result.T


def quat_distance(
    q1: np.ndarray, q2: np.ndarray, qsym: np.ndarray
) -> np.ndarray:
    """
    Distance between two quaternions, taking quaternions of symmetry into
    account.

    Parameters
    ----------
    q1 : arary_like
        First quaternion.
    q2 : arary_like
        Second quaternion.
    qsym : ndarray
        List of symmetry quaternions.

    Returns
    -------
    float
        The distance between the two quaternions.
    """
    q1 = np.ascontiguousarray(q1.flatten())
    q2 = np.ascontiguousarray(q2.flatten())
    # C module expects quaternions in row major, numpy code in column major.
    qsym = np.ascontiguousarray(qsym.T)
    return _impl.quat_distance(q1, q2, qsym)
