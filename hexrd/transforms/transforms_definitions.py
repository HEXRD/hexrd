"""This module provides the definitions for the transforms API. It will also
provide a decorator to add to any implementation of the API. This module will
contain the reference comment that will be added to any function that implements
an API function, as well as a means to add pre and post conditions as an
additional way to document the implementations.

Pre and Post conditions will be in the form of code, there will be means to
execute the scripts forcing those conditions to be evaluated and raise errors if
they are not met. This should always be optional and incur on no overhead unless
enabled, only to be used for debugging and validation purposes.

Checking of signature definitions has been added. This happens if CHECK_API is
enabled (via the HEXRD_XF_CHECK environment variable). This is implemented via
the signature method in inspect (Python 3.3+). If not available, it falls back
to the backport in funcsigs. If unavailable, CHECK_API is disabled.

"""
from __future__ import absolute_import, print_function

import os
import functools

# Just a list of the API functions...
# Note this can be kind of redundant with the definition classes, but it also
# allows for some coherence checks.
API = (
    "angles_to_gvec",
    "angles_to_dvec",
    "gvec_to_xy",
    "xy_to_gvec",
    "solve_omega",

    "angular_difference",
    "map_angle",
    "row_norm",
    "unit_vector",
    "make_sample_rmat",
    "make_rmat_of_expmap",
    "make_binary_rmat",
    "make_beam_rmat",
    "angles_in_range",
    "validate_angle_ranges",
    "rotate_vecs_about_axis",
    "quat_product_matrix",
    "quat_distance"
)

CHECK_API = os.getenv("XRD_TRANSFORMS_CHECK")
try:
    from inspect import signature as get_signature
except ImportError:
    try:
        from funcsigs import signature as get_signature
    except:
        import warnings

        warnings.warn("Failed to import from inspect/funcsigs."
                      "Transforms API signature checking disabled.")
        get_signature = None


class DEF_Func(object):
    """Documentation to use for the function"""

    def _signature():
        """The signature of this method defines the one for the API
        including default values."""
        pass

    @classmethod
    def _PRECOND(cls, *args, **kwargs):
        print("PRECOND (", cls.__class__.__name__,")")
        pass

    @classmethod
    def _POSTCOND(cls, results, *args, **kwargs):
        print("PRECOND (", cls.__class__.__name__,")")
        pass


# ==============================================================================
# API
# ==============================================================================

class DEF_angles_to_gvec(DEF_Func):
    """
    Takes triplets of angles in the beam frame (2*theta, eta, omega)
    to components of unit G-vectors in the LAB frame.  If the omega
    values are not trivial (i.e. angs[:, 2] = 0.), then the components
    are in the SAMPLE frame.  If the crystal rmat is specified and
    is not the identity, then the components are in the CRYSTAL frame.

    default beam_vec is defined in hexrd.constants.beam_vec
    default eta_vec is defined in hexrd.constants.eta_vec
    """
    def _signature(angs,
                   beam_vec=None,
                   eta_vec=None,
                   chi=None,
                   rmat_c=None):
        pass


class DEF_angles_to_dvec(DEF_Func):
    """
    Takes triplets of angles in the beam frame (2*theta, eta, omega)
    to components of unit diffraction vectors in the LAB frame.  If the
    omega values are not trivial (i.e. angs[:, 2] = 0.), then the
    components are in the SAMPLE frame.  If the crystal rmat is specified
    and is not the identity, then the components are in the CRYSTAL frame.

    default beam_vec is defined in hexrd.constants.beam_vec
    default eta_vec is defined in hexrd.constants.eta_vec
    """
    def _signature(angs,
                   beam_vec=None,
                   eta_vec=None,
                   chi=None,
                   rmat_c=None):
        pass


class DEF_gvec_to_xy(DEF_Func):
    """
    Takes a concatenated list of reciprocal lattice vectors components in the
    CRYSTAL FRAME to the specified detector-relative frame, subject to the
    following:

        1) it must be able to satisfy a bragg condition
        2) the associated diffracted beam must intersect the detector plane

    Parameters
    ----------
    gvec_c : array_like
        Concatenated triplets of G-vector components in the CRYSTAL FRAME.
    rmat_d : array_like
        The (3, 3) COB matrix taking components in the
        DETECTOR FRAME to the LAB FRAME
    rmat_s : array_like
        The (3, 3) COB matrix taking components in the
        SAMPLE FRAME to the LAB FRAME
    rmat_c : array_like
        The (3, 3) COB matrix taking components in the
        CRYSTAL FRAME to the SAMPLE FRAME
    tvec_d : array_like
        The (3, ) translation vector connecting LAB FRAME to DETECTOR FRAME
    tvec_s : array_like
        The (3, ) translation vector connecting LAB FRAME to SAMPLE FRAME
    tvec_c : array_like
        The (3, ) translation vector connecting SAMPLE FRAME to CRYSTAL FRAME
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
        The (n, 2) array of [x, y] diffracted beam intersections for each of
        the n input G-vectors in the DETECTOR FRAME (all Z_d coordinates are 0
        and excluded).  For each input G-vector that cannot satisfy a Bragg
        condition or intersect the detector plane, [NaN, Nan] is returned.

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    Notes
    -----

    """
    def _signature(gvec_c,
                   rmat_d, rmat_s, rmat_c,
                   tvec_d, tvec_s, tvec_c,
                   beam_vec=None,
                   vmat_inv=None,
                   bmat=None):
        pass


class DEF_xy_to_gvec(DEF_Func):
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

    Notes
    -----
    ???: is there a need to flatten the tvec inputs?
    ???: include optional wavelength input for returning G with magnitude?
    ???: is there a need to check that rmat_b is orthogonal if spec'd?
    """
    def _signature(xy_d,
                   rmat_d, rmat_s,
                   tvec_d, tvec_s, tvec_c,
                   rmat_b=None,
                   distortion=None,
                   output_ref=False):
        pass


class DEF_solve_omega(DEF_Func):
    """
    For the monochromatic rotation method.

    Solve the for the rotation angle pairs that satisfy the bragg conditions
    for an input list of G-vector components.

    Parameters
    ----------
    gvecs : array_like
        Concatenated triplets of G-vector components in either the
        CRYSTAL FRAME or RECIPROCAL FRAME (see optional kwarg `bmat` below).
        The shape when cast as a 2-d ndarray is (n, 3), representing n vectors.
    chi : float
        The inclination angle of the goniometer axis (standard coords)
    rmat_c : array_like
        (3, 3) COB matrix taking components in the
        CRYSTAL FRAME to the SAMPLE FRAME
    wavelength : float
        The X-ray wavelength in Angstroms
    bmat : array_like, optional
        The (3, 3) COB matrix taking components in the
        RECIPROCAL LATTICE FRAME to the CRYSTAL FRAME; if supplied, it is
        assumed that the input `gvecs` are G-vector components in the
        RECIPROCL LATTICE FRAME (the default is None, which implies components
        in the CRYSTAL FRAME)
    vmat_inv : array_like, optional
        The (3, 3) matrix of inverse stretch tensor components in the
        SAMPLE FRAME.  The default is None, which implies a strain-free state
        (i.e. V = I).
    rmat_b : array_like, optional
        (3, 3) COB matrix taking components in the BEAM FRAME to the LAB FRAME;
        defaults to None, which implies the standard setting of identity.

    Returns
    -------
    ome0 : array_like
        The (n, 3) ndarray containing the feasible (tth, eta, ome) triplets for
        each input hkl (first solution)
    ome1 : array_like
        The (n, 3) ndarray containing the feasible (tth, eta, ome) triplets for
        each input hkl (second solution)

    Notes
    -----
    The reciprocal lattice vector, G, will satisfy the the Bragg condition
    when:

        b.T * G / ||G|| = -sin(theta)

    where b is the incident beam direction (k_i) and theta is the Bragg
    angle consistent with G and the specified wavelength. The components of
    G in the lab frame in this case are obtained using the crystal
    orientation, Rc, and the single-parameter oscillation matrix, Rs(ome):

        Rs(ome) * Rc * G / ||G||

    The equation above can be rearranged to yeild an expression of the form:

        a*sin(ome) + b*cos(ome) = c

    which is solved using the relation:

        a*sin(x) + b*cos(x) = sqrt(a**2 + b**2) * sin(x + alpha)

        --> sin(x + alpha) = c / sqrt(a**2 + b**2)

    where:

        alpha = arctan2(b, a)

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
    def _signature(gvecs, chi, rmat_c, wavelength,
                   bmat=None, vmat_inv=None, rmat_b=None):
        pass


# ==============================================================================
# UTILITY FUNCTIONS API
# ==============================================================================

class DEF_angular_difference(DEF_Func):
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range is [-pi, pi]
    """
    def _signature(ang_list0, ang_list1, units=None):
        pass


class DEF_map_angle(DEF_Func):
    """
    Utility routine to map an angle into a specified period

    actual function is map_angle(ang[, range], units=None).
    range is optional and defaults to the appropriate angle for the unit
    centered on 0.

    accepted units are: 'radians' and 'degrees'
    """

    def _signature(ang, range=None, units=None):
        pass


class DEF_row_norm(DEF_Func):
    """
    Compute the norm of row vectors.

    guaranteed to work for 1d and 2d arrays.

    note: numpy.linalg.norm could be used instead as this is equivalent to
          numpy.linalg.norm(vec_in, axis=1)
    """
    def _signature(vec_in):
        pass


class DEF_unit_vector(DEF_Func):
    """
    Normalize an array of row vectors (vstacked, axis=0)
    For vectors with (very close to) zero norm, the original
    vector is returned.

    guaranteed to work for 1d and 2d arrays.
    """
    def _signature(vec_in):
        pass


class DEF_make_sample_rmat(DEF_Func):
    """
    Make SAMPLE frame rotation matrices as composition of
    rotation of ome about the axis

    [0., cos(chi), sin(chi)]

    in the lab frame
    """
    def _signature(chi, ome):
        pass


class DEF_make_rmat_of_expmap(DEF_Func):
    """
    Calculates the rotation matrix from an exponential map
    """
    def _signature(exp_map):
        pass


class DEF_make_binary_rmat(DEF_Func):
    """
    make a binary rotation matrix about the specified axis

    Used to compute the refracted vector wrt the axis.
    """
    def _signature(axis):
        pass


class DEF_make_beam_rmat(DEF_Func):
    """
    make eta basis COB matrix with beam antiparallel with Z

    takes components from BEAM frame to LAB
    """
    def _signature(bvec_l, evec_l):
        pass


class DEF_angles_in_range(DEF_Func):
    """Determine whether angles lie in or out of specified ranges

    *angles* - a list/array of angles
    *starts* - a list of range starts
    *stops* - a list of range stops

    OPTIONAL ARGS:
    *degrees* - [True] angles & ranges in degrees (or radians)
    """
    def _signature(angles, starts, stops, degrees=True):
        pass


class DEF_validate_angle_ranges(DEF_Func):
    """
    A better way to go.  find out if an angle is in the range
    CCW or CW from start to stop
    There is, of course, an ambigutiy if the start and stop angle are
    the same; we treat them as implying 2*pi having been mapped
    """
    def _signature(ang_list, start_angs, stop_angs, ccw=True):
        pass


class DEF_rotate_vecs_about_axis(DEF_Func):
    """
    Rotate vectors about an axis

    INPUTS
    *angle* - array of angles (len == 1 or n)
    *axis*  - array of unit vectors (shape == (3, 1) or (3, n))
    *vec*   - array of vectors to be rotated (shape = (3, n))

    Quaternion formula:
    if we split v into parallel and perpedicular components w.r.t. the
    axis of quaternion q,

        v = a + n

    then the action of rotating the vector dot(R(q), v) becomes

        v_rot = (q0**2 - |q|**2)(a + n) + 2*dot(q, a)*q + 2*q0*cross(q, n)

    """
    def _signature(angle, axis, vecs):
        pass


class DEF_quat_product_matrix(DEF_Func):
    """
    Form 4 x 4 array to perform the quaternion product

    USAGE
        qmat = quatProductMatrix(q, mult='right')

    INPUTS
        1) quats is (4,), an iterable representing a unit quaternion
           horizontally concatenated
        2) mult is a keyword arg, either 'left' or 'right', denoting
           the sense of the multiplication:

                       / quatProductMatrix(h, mult='right') * q
           q * h  --> <
                       \ quatProductMatrix(q, mult='left') * h

    OUTPUTS
        1) qmat is (4, 4), the left or right quaternion product
           operator

    NOTES
       *) This function is intended to replace a cross-product based
          routine for products of quaternions with large arrays of
          quaternions (e.g. applying symmetries to a large set of
          orientations).
    """
    def _signature(q, mult='right'):
        pass


class DEF_quat_distance(DEF_Func):
    """
    find the distance between two unit quaternions under symmetry group
    """
    def _signature(q1, q2, qsym):
        pass


# ==============================================================================
# Decorator to mark implementations of the API. Names must match.
# ==============================================================================

def xf_api(f, name=None):
    """decorator to apply to the entry points of the transforms module"""
    api_call = name if name is not None else f.__name__

    if not api_call in API:
        raise RuntimeError("'%s' is not part of the transforms API.")
    
    try:
        fn_def = globals()['DEF_'+api_call]
    except KeyError:
        # This happens if there is no definition for the decorated function
        raise RuntimeError("'%s' definition not found." % api_call)

    try:
        # python 2
        _string_type = basestring
    except NameError:
        # This will happen on python 3
        _string_type = str
        
    try:
        if not (isinstance(fn_def.__doc__, _string_type) and
                callable(fn_def._PRECOND) and
                callable(fn_def._POSTCOND) and
                callable(fn_def._signature)):
            raise Exception()
    except Exception:
        # A valid definition requires a string doc, and callable _PRECOND, 
        # _POSTCOND and _signature.
        #
        # __doc__ will become the decorated function's documentation.
        # _PRECOND will be run on every call with args and kwargs
        # _POSTCOND will be run on every call with result, args and kwargs
        # _signature will be used to enforce a signature on implementations.
        #
        # _PRECOND and _POSTCOND will only be called if CHECK_API is enabled,
        # as they will slow down execution.
        raise RuntimeError("'{0}' definition error.".format(api_call))

    # Sanity check: make sure the decorated function has the expected signature.
    if get_signature is not None:
        # Check that the function has the right signature
        if get_signature(fn_def._signature) != get_signature(f):
            raise RuntimeError("'{0}' signature mismatch.".format(api_call))

    # At this point use a wrapper that calls pre and post conditions if checking
    # is enabled, otherwise leave the function "as is".
    if CHECK_API:
        @functools.wraps(f, assigned={"__doc__": fn_def.__doc__})
        def wrapper(*args, **kwargs):
            fn_def._PRECOND(*args, **kwargs)
            result = f(*args, **kwargs)
            fn_def._POSTCOND(result, *args, **kwargs)
            return result

        return wrapper
    else:
        # just try to put the right documentation on the function
        try:
            f.__doc__ = fn_def.__doc__
        except Exception:
            pass
        return f
