# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================
#
# Module containing functions relevant to rotations
#
import sys

import numpy as np
from numba import njit
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R

from hexrd.core.deprecation import deprecated
from hexrd.core import constants as cnst
from hexrd.core.matrixutil import (
    columnNorm,
    unitVector,
    findDuplicateVectors,
    multMatArray,
    nullSpace,
)
from hexrd.core.utils.warnings import ignore_warnings


# =============================================================================
# Module Data
# =============================================================================

angularUnits = 'radians'  # module-level angle units
periodDict = {'degrees': 360.0, 'radians': 2 * np.pi}
conversion_to_dict = {'degrees': cnst.r2d, 'radians': cnst.d2r}

I3 = cnst.identity_3x3  # (3, 3) identity matrix

# axes orders, all permutations
axes_orders = [
    'xyz',
    'zyx',
    'zxy',
    'yxz',
    'yzx',
    'xzy',
    'xyx',
    'xzx',
    'yxy',
    'yzy',
    'zxz',
    'zyz',
]

sq3by2 = np.sqrt(3.0) / 2.0
piby2 = np.pi / 2.0
piby3 = np.pi / 3.0
piby4 = np.pi / 4.0
piby6 = np.pi / 6.0

# =============================================================================
# Functions
# =============================================================================


def arccosSafe(cosines):
    """
    Protect against numbers slightly larger than 1 in magnitude
    due to round-off
    """
    cosines = np.atleast_1d(cosines)
    if (np.abs(cosines) > 1.00001).any():
        print("attempt to take arccos of %s" % cosines, file=sys.stderr)
        raise RuntimeError("unrecoverable error")
    return np.arccos(np.clip(cosines, -1.0, 1.0))


#
#  ==================== Quaternions
#


def _quat_to_scipy_rotation(q: np.ndarray) -> R:
    """
    Scipy has quaternions in a differnt order, this method converts them
    q must be a 2d array of shape (4, n).
    """
    return R.from_quat(np.roll(q.T, -1, axis=1))


def _scipy_rotation_to_quat(r: R) -> np.ndarray:
    quat = np.roll(np.atleast_2d(r.as_quat()), 1, axis=1).T
    # Fix quat would work, but it does too much.  Only need to check positive
    quat *= np.sign(quat[0, :])
    return quat


def fixQuat(q):
    """
    flip to positive q0 and normalize
    """
    qdims = q.ndim
    if qdims == 3:
        l, m, n = q.shape
        assert m == 4, 'your 3-d quaternion array isn\'t the right shape'
        q = q.transpose(0, 2, 1).reshape(l * n, 4).T

    qfix = unitVector(q)

    q0negative = qfix[0,] < 0
    qfix[:, q0negative] = -1 * qfix[:, q0negative]

    if qdims == 3:
        qfix = qfix.T.reshape(l, n, 4).transpose(0, 2, 1)

    return qfix


def invertQuat(q):
    """
    silly little routine for inverting a quaternion
    """
    numq = q.shape[1]

    imat = np.tile(np.vstack([-1, 1, 1, 1]), (1, numq))

    qinv = imat * q

    return fixQuat(qinv)


def misorientation(q1, q2, symmetries=None):
    """
    PARAMETERS
    ----------
    q1: array(4, 1)
        a single quaternion
    q2: array(4, n)
        array of quaternions
    symmetries: tuple, optional
        1- or 2-tuple with symmetries (quaternion arrays);
        for crystal symmetry only, use a 1-tuple;
        with both crystal and sample symmetry use a 2-tuple
        Default is no symmetries.

    RETURNS
    -------
    angle: array(n)
        the misorientation angle between `q1` and each quaternion in `q2`
    mis: array(4, n)
        the quaternion of the smallest misorientation angle
    """
    if not isinstance(q1, np.ndarray) or not isinstance(q2, np.ndarray):
        raise RuntimeError("quaternion args are not of type `numpy ndarray'")

    if q1.ndim != 2 or q2.ndim != 2:
        raise RuntimeError(
            "quaternion args are the wrong shape; must be 2-d (columns)"
        )

    if q1.shape[1] != 1:
        raise RuntimeError(
            "first argument should be a single quaternion, got shape %s"
            % (q1.shape,)
        )

    if symmetries is None:
        # no symmetries; use identity
        symmetries = (np.c_[1.0, 0, 0, 0].T, np.c_[1.0, 0, 0, 0].T)
    else:
        # check symmetry argument
        if len(symmetries) == 1:
            if not isinstance(symmetries[0], np.ndarray):
                raise RuntimeError("symmetry argument is not an numpy array")
            else:
                # add triclinic sample symmetry (identity)
                symmetries += (np.c_[1.0, 0, 0, 0].T,)
        elif len(symmetries) == 2:
            if not isinstance(symmetries[0], np.ndarray) or not isinstance(
                symmetries[1], np.ndarray
            ):
                raise RuntimeError(
                    "symmetry arguments are not an numpy arrays"
                )
        elif len(symmetries) > 2:
            raise RuntimeError(
                "symmetry argument has %d entries; should be 1 or 2"
                % (len(symmetries))
            )

    # set some lengths
    n = q2.shape[1]  # length of misorientation list
    m = symmetries[0].shape[1]  # crystal (right)
    p = symmetries[1].shape[1]  # sample  (left)

    # tile q1 inverse
    q1i = quatProductMatrix(invertQuat(q1), mult='right').squeeze()

    # convert symmetries to (4, 4) qprod matrices
    rsym = quatProductMatrix(symmetries[0], mult='right')
    lsym = quatProductMatrix(symmetries[1], mult='left')

    # Do R * Gc, store as
    # [q2[:, 0] * Gc[:, 0:m], ..., q2[:, n-1] * Gc[:, 0:m]]
    q2 = np.dot(rsym, q2).transpose(2, 0, 1).reshape(m * n, 4).T

    # Do Gs * (R * Gc), store as
    # [Gs[:, 0:p]*q[:,   0]*Gc[:, 0], ..., Gs[:, 0:p]*q[:,   0]*Gc[:, m-1], ...
    #  Gs[:, 0:p]*q[:, n-1]*Gc[:, 0], ..., Gs[:, 0:p]*q[:, n-1]*Gc[:, m-1]]
    q2 = np.dot(lsym, q2).transpose(2, 0, 1).reshape(p * m * n, 4).T

    # Calculate the class misorientations for full symmetrically equivalent
    # classes for q1 and q2.  Note the use of the fact that the application
    # of the symmetry groups is an isometry.
    eqvMis = fixQuat(np.dot(q1i, q2))

    # Reshape scalar comp columnwise by point in q2 (and q1, if applicable)
    sclEqvMis = eqvMis[0, :].reshape(n, p * m).T

    # Find misorientation closest to origin for each n equivalence classes
    #   - fixed quats so garaunteed that sclEqvMis is nonnegative
    qmax = sclEqvMis.max(0)

    # remap indices to use in eqvMis
    qmaxInd = (sclEqvMis == qmax).nonzero()
    qmaxInd = np.c_[qmaxInd[0], qmaxInd[1]]

    eqvMisColInd = np.sort(qmaxInd[:, 0] + qmaxInd[:, 1] * p * m)

    # store Rmin in q
    mis = eqvMis[np.ix_(list(range(4)), eqvMisColInd)]

    angle = 2 * arccosSafe(qmax)

    return angle, mis


def quatProduct(q1, q2):
    """
    Product of two unit quaternions.

    qp = quatProduct(q2, q1)

    q2, q1 are 4 x n, arrays whose columns are
           quaternion parameters

    qp is 4 x n, an array whose columns are the
       quaternion parameters of the product; the
       first component of qp is nonnegative

    If R(q) is the rotation corresponding to the
    quaternion parameters q, then

    R(qp) = R(q2) R(q1)
    """
    rot_1 = _quat_to_scipy_rotation(q1)
    rot_2 = _quat_to_scipy_rotation(q2)
    rot_p = rot_2 * rot_1
    return _scipy_rotation_to_quat(rot_p)


def quatProductMatrix(quats, mult='right'):
    """
    Form 4 x 4 arrays to perform the quaternion product

    USAGE
        qmats = quatProductMatrix(quats, mult='right')

    INPUTS
        1) quats is (4, n), a numpy ndarray array of n quaternions
           horizontally concatenated
        2) mult is a keyword arg, either 'left' or 'right', denoting
           the sense of the multiplication:

                       | quatProductMatrix(h, mult='right') * q
           q * h  --> <
                       | quatProductMatrix(q, mult='left') * h

    OUTPUTS
        1) qmats is (n, 4, 4), the left or right quaternion product
           operator

    NOTES
       *) This function is intended to replace a cross-product based
          routine for products of quaternions with large arrays of
          quaternions (e.g. applying symmetries to a large set of
          orientations).
    """

    if quats.shape[0] != 4:
        raise RuntimeError("input is the wrong size along the 0-axis")

    nq = quats.shape[1]
    q0 = quats[0, :].copy()
    q1 = quats[1, :].copy()
    q2 = quats[2, :].copy()
    q3 = quats[3, :].copy()
    if mult == 'right':
        qmats = np.array(
            [
                [q0],
                [q1],
                [q2],
                [q3],
                [-q1],
                [q0],
                [-q3],
                [q2],
                [-q2],
                [q3],
                [q0],
                [-q1],
                [-q3],
                [-q2],
                [q1],
                [q0],
            ]
        )
    elif mult == 'left':
        qmats = np.array(
            [
                [q0],
                [q1],
                [q2],
                [q3],
                [-q1],
                [q0],
                [q3],
                [-q2],
                [-q2],
                [-q3],
                [q0],
                [q1],
                [-q3],
                [q2],
                [-q1],
                [q0],
            ]
        )
    # some fancy reshuffling...
    qmats = qmats.T.reshape((nq, 4, 4)).transpose(0, 2, 1)
    return qmats


def quatOfAngleAxis(angle, rotaxis):
    """
    make an hstacked array of quaternions from arrays of angle/axis pairs
    """
    angle = np.atleast_1d(angle)
    n = len(angle)

    if rotaxis.shape[1] == 1:
        rotaxis = np.tile(rotaxis, (1, n))
    elif rotaxis.shape[1] != n:
        raise RuntimeError("rotation axes argument has incompatible shape")

    # Normalize the axes
    rotaxis = unitVector(rotaxis)
    rot = R.from_rotvec((angle * rotaxis).T)
    return _scipy_rotation_to_quat(rot)


def quatOfExpMap(expMaps):
    """
    Returns the unit quaternions associated with exponential map parameters.

    Parameters
    ----------
    expMaps : array_like
        The (3,) or (3, n) list of hstacked exponential map parameters to
        convert.

    Returns
    -------
    quats : array_like
        The (4,) or (4, n) array of unit quaternions.

    Notes
    -----
    1) be aware that the output will always have non-negative q0; recall the
       antipodal symmetry of unit quaternions

    """
    cdim = 3  # critical dimension of input
    expMaps = np.atleast_2d(expMaps)
    if len(expMaps) == 1:
        assert expMaps.shape[1] == cdim, (
            "your input quaternion must have %d elements" % cdim
        )
        expMaps = np.reshape(expMaps, (cdim, 1))
    else:
        assert len(expMaps) == cdim, (
            "your input quaternions must have shape (%d, n) for n > 1" % cdim
        )

    return _scipy_rotation_to_quat(R.from_rotvec(expMaps.T)).squeeze()


def quatOfRotMat(r_mat):
    """
    Generate quaternions from rotation matrices
    """
    return _scipy_rotation_to_quat(R.from_matrix(r_mat))


def quatAverageCluster(q_in, qsym):
    """ """
    assert q_in.ndim == 2, 'input must be 2-s hstacked quats'

    # renormalize
    q_in = unitVector(q_in)

    # check to see num of quats is > 1
    if q_in.shape[1] < 3:
        if q_in.shape[1] == 1:
            q_bar = q_in
        else:
            ma, mq = misorientation(
                q_in[:, 0].reshape(4, 1), q_in[:, 1].reshape(4, 1), (qsym,)
            )

            q_bar = quatProduct(
                q_in[:, 0].reshape(4, 1),
                quatOfExpMap(0.5 * ma * unitVector(mq[1:])).reshape(4, 1),
            )
    else:
        # first drag to origin using first quat (arb!)
        q0 = q_in[:, 0].reshape(4, 1)
        qrot = np.dot(quatProductMatrix(invertQuat(q0), mult='left'), q_in)

        # second, re-cast to FR
        qrot = toFundamentalRegion(qrot.squeeze(), crysSym=qsym)

        # compute arithmetic average
        q_bar = unitVector(np.average(qrot, axis=1).reshape(4, 1))

        # unrotate!
        q_bar = np.dot(quatProductMatrix(q0, mult='left'), q_bar)

        # re-map
        q_bar = toFundamentalRegion(q_bar, crysSym=qsym)
    return q_bar


def quatAverage(q_in, qsym):
    """ """
    assert q_in.ndim == 2, 'input must be 2-s hstacked quats'

    # renormalize
    q_in = unitVector(q_in)

    # check to see num of quats is > 1
    if q_in.shape[1] < 3:
        if q_in.shape[1] == 1:
            q_bar = q_in
        else:
            ma, mq = misorientation(
                q_in[:, 0].reshape(4, 1), q_in[:, 1].reshape(4, 1), (qsym,)
            )
            q_bar = quatProduct(
                q_in[:, 0].reshape(4, 1),
                quatOfExpMap(0.5 * ma * unitVector(mq[1:].reshape(3, 1))),
            )
    else:
        # use first quat as initial guess
        phi = 2.0 * np.arccos(q_in[0, 0])
        if phi <= np.finfo(float).eps:
            x0 = np.zeros(3)
        else:
            n = unitVector(q_in[1:, 0].reshape(3, 1))
            x0 = phi * n.flatten()
        results = leastsq(quatAverage_obj, x0, args=(q_in, qsym))
        phi = np.sqrt(sum(results[0] * results[0]))
        if phi <= np.finfo(float).eps:
            q_bar = np.c_[1.0, 0.0, 0.0, 0.0].T
        else:
            n = results[0] / phi
            q_bar = np.hstack(
                [np.cos(0.5 * phi), np.sin(0.5 * phi) * n]
            ).reshape(4, 1)
    return q_bar


def quatAverage_obj(xi_in, quats, qsym):
    phi = np.sqrt(sum(xi_in.flatten() * xi_in.flatten()))
    if phi <= np.finfo(float).eps:
        q0 = np.c_[1.0, 0.0, 0.0, 0.0].T
    else:
        n = xi_in.flatten() / phi
        q0 = np.hstack([np.cos(0.5 * phi), np.sin(0.5 * phi) * n])
    resd = misorientation(q0.reshape(4, 1), quats, (qsym,))[0]
    return resd


def expMapOfQuat(quats):
    """
    Return the exponential map parameters for an array of unit quaternions

    Parameters
    ----------
    quats : array_like
        The (4, ) or (4, n) array of hstacked unit quaternions.  The convention
        is [q0, q] where q0 is the scalar part and q is the vector part.

    Returns
    -------
    expmaps : array_like
        The (3, ) or (3, n) array of exponential map parameters associated
        with the input quaternions.

    """
    cdim = 4  # critical dimension of input
    quats = np.atleast_2d(quats)
    if len(quats) == 1:
        assert quats.shape[1] == cdim, (
            "your input quaternion must have %d elements" % cdim
        )
        quats = np.reshape(quats, (cdim, 1))
    else:
        assert len(quats) == cdim, (
            "your input quaternions must have shape (%d, n) for n > 1" % cdim
        )

    return _quat_to_scipy_rotation(quats).as_rotvec().T.squeeze()


def rotMatOfExpMap(expMap):
    """
    Make a rotation matrix from an expmap
    """
    if expMap.ndim == 1:
        expMap = expMap.reshape(3, 1)

    return R.from_rotvec(expMap.T).as_matrix().squeeze()

@deprecated(new_func="Use `rotMatOfExpMap` instead", removal_date="2025-07-01")
def rotMatOfExpMap_orig(expMap): # pragma: no cover
    return rotMatOfExpMap(expMap)


@deprecated(new_func="Use `rotMatOfExpMap` instead", removal_date="2025-07-01")
def rotMatOfExpMap_opt(expMap): # pragma: no cover
    return rotMatOfExpMap(expMap)


@njit(cache=True, nogil=True)
def _rotmatofquat(quat):
    n = quat.shape[1]
    # FIXME: maybe preallocate for speed?
    # R = np.zeros(n*3*3, dtype='float64')

    a = np.ascontiguousarray(quat[0, :]).reshape(n, 1)
    b = np.ascontiguousarray(quat[1, :]).reshape(n, 1)
    c = np.ascontiguousarray(quat[2, :]).reshape(n, 1)
    d = np.ascontiguousarray(quat[3, :]).reshape(n, 1)

    R = np.hstack(
        (
            a**2 + b**2 - c**2 - d**2,
            2 * b * c - 2 * a * d,
            2 * a * c + 2 * b * d,
            2 * a * d + 2 * b * c,
            a**2 - b**2 + c**2 - d**2,
            2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c,
            2 * a * b + 2 * c * d,
            a**2 - b**2 - c**2 + d**2,
        )
    )

    return R.reshape(n, 3, 3)


def rotMatOfQuat(quat):
    """
    Convert quaternions to rotation matrices.

    Take an array of n quats (numpy ndarray, 4 x n) and generate an
    array of rotation matrices (n x 3 x 3)

    Parameters
    ----------
    quat : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    rmat : TYPE
        DESCRIPTION.

    Notes
    -----
    Uses the truncated series expansion for the exponential map;
    didvide-by-zero is checked using the global 'cnst.epsf'
    """
    if quat.ndim == 1:
        if len(quat) != 4:
            raise RuntimeError("input is the wrong shape")
        else:
            quat = quat.reshape(4, 1)
    else:
        if quat.shape[0] != 4:
            raise RuntimeError("input is the wrong shape")

    rmat = _rotmatofquat(quat)

    return np.squeeze(rmat)


def angleAxisOfRotMat(rot_mat):
    """
    Extracts angle and axis invariants from rotation matrices.

    Parameters
    ----------
    R : numpy.ndarray
        The (3, 3) or (n, 3, 3) array of rotation matrices.
        Note that these are assumed to be proper orthogonal.

    Raises
    ------
    RuntimeError
        If `R` is not an shape is not (3, 3) or (n, 3, 3).

    Returns
    -------
    phi : numpy.ndarray
        The (n, ) array of rotation angles for the n input
        rotation matrices.
    n : numpy.ndarray
        The (3, n) array of unit rotation axes for the n
        input rotation matrices.

    """
    if not isinstance(rot_mat, np.ndarray):
        raise RuntimeError('Input must be a 2 or 3-d ndarray')
    else:
        rdim = rot_mat.ndim
        if rdim == 2:
            rot_mat = np.tile(rot_mat, (1, 1, 1))
        elif rdim == 3:
            pass
        else:
            raise RuntimeError(
                "rot_mat array must be (3, 3) or (n, 3, 3); "
                "input has dimension %d" % (rdim)
            )

    rot_vec = R.from_matrix(rot_mat).as_rotvec()
    angs = np.linalg.norm(rot_vec, axis=1)
    axes = unitVector(rot_vec.T)
    return angs, axes


def _check_axes_order(x):
    if not isinstance(x, str):
        raise RuntimeError("argument must be str")
    axo = x.lower()
    if axo not in axes_orders:
        raise RuntimeError("order '%s' is not a valid choice" % x)
    return axo


def _check_is_rmat(x):
    x = np.asarray(x)
    if x.shape != (3, 3):
        raise RuntimeError("shape of input must be (3, 3)")
    chk1 = np.linalg.det(x)
    chk2 = np.sum(np.abs(np.eye(3) - np.dot(x, x.T)))
    if 1.0 - np.abs(chk1) < cnst.sqrt_epsf and chk2 < cnst.sqrt_epsf:
        return x
    else:
        raise RuntimeError("input is not an orthogonal matrix")


def make_rmat_euler(tilt_angles, axes_order, extrinsic=True):
    """
    Generate rotation matrix from Euler angles.

    Parameters
    ----------
    tilt_angles : array_like
        The (3, ) list of Euler angles in RADIANS.
    axes_order : str
        The axes order specification (case-insensitive).  This must be one
        of the following: 'xyz', 'zyx'
                          'zxy', 'yxz'
                          'yzx', 'xzy'
                          'xyx', 'xzx'
                          'yxy', 'yzy'
                          'zxz', 'zyz'
    extrinsic : bool, optional
        Flag denoting the convention.  If True, the convention is
        extrinsic (passive); if False, the convention is
        instrinsic (active). The default is True.

    Returns
    -------
    numpy.ndarray
        The (3, 3) rotation matrix corresponding to the input specification.

    TODO: add kwarg for unit selection for `tilt_angles`
    TODO: input checks
    """
    axo = _check_axes_order(axes_order)
    if not extrinsic:
        axo = axo.upper()

    return R.from_euler(axo, tilt_angles).as_matrix()


def angles_from_rmat_xyz(rmat):
    """
    Calculate passive x-y-z Euler angles from a rotation matrix.

    Parameters
    ----------
    rmat : TYPE
        DESCRIPTION.

    Returns
    -------
    rx : TYPE
        DESCRIPTION.
    ry : TYPE
        DESCRIPTION.
    rz : TYPE
        DESCRIPTION.

    """
    rmat = _check_is_rmat(rmat)

    # Ignore Gimbal Lock warning. It is okay.
    with ignore_warnings(UserWarning):
        return R.from_matrix(rmat).as_euler('xyz')


def angles_from_rmat_zxz(rmat):
    """
    Calculate active z-x-z Euler angles from a rotation matrix.

    Parameters
    ----------
    rmat : TYPE
        DESCRIPTION.

    Returns
    -------
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    """
    rmat = _check_is_rmat(rmat)

    # Ignore Gimbal Lock warning. It is okay.
    with ignore_warnings(UserWarning):
        return R.from_matrix(rmat).as_euler('ZXZ')


class RotMatEuler(object):
    def __init__(self, angles, axes_order, extrinsic=True, units=angularUnits):
        """
        Abstraction of a rotation matrix defined by Euler angles.

        Parameters
        ----------
        angles : array_like
            The (3, ) list of Euler angles in RADIANS.
        axes_order : str
            The axes order specification (case-insensitive).  This must be one
            of the following:

                'xyz', 'zyx'
                'zxy', 'yxz'
                'yzx', 'xzy'
                'xyx', 'xzx'
                'yxy', 'yzy'
                'zxz', 'zyz'

        extrinsic : bool, optional
            Flag denoting the convention.  If True, the convention is
            extrinsic (passive); if False, the convention is
            instrinsic (active). The default is True.

        Returns
        -------
        None.

        TODO: add check that angle input is array-like, len() = 3?
        TODO: add check on extrinsic as bool
        """
        self._axes = np.eye(3)
        self._axes_dict = dict(x=0, y=1, z=2)

        # these will be properties
        self._angles = angles
        self._axes_order = _check_axes_order(axes_order)
        self._extrinsic = extrinsic
        if units.lower() not in periodDict.keys():
            raise RuntimeError("angular units '%s' not understood" % units)
        self._units = units

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, x):
        x = np.atleast_1d(x).flatten()
        if len(x) == 3:
            self._angles = x
        else:
            raise RuntimeError("input must be array-like with __len__ = 3")

    @property
    def axes_order(self):
        return self._axes_order

    @axes_order.setter
    def axes_order(self, x):
        axo = _check_axes_order(x)
        self._axes_order = axo

    @property
    def extrinsic(self):
        return self._extrinsic

    @extrinsic.setter
    def extrinsic(self, x):
        if isinstance(x, bool):
            self._extrinsic = x
        else:
            raise RuntimeError("input must be a bool")

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, x):
        if isinstance(x, str) and x in periodDict.keys():
            if self._units != x:
                # !!! we are changing units; update self.angles
                self.angles = conversion_to_dict[x] * np.asarray(self.angles)
            self._units = x
        else:
            raise RuntimeError("input must be 'degrees' or 'radians'")

    @property
    def rmat(self):
        """
        Return the rotation matrix.

        As calculated from angles, axes_order, and convention.

        Returns
        -------
        numpy.ndarray
            The (3, 3) proper orthogonal matrix according to the specification.

        """
        angs_in = self.angles
        if self.units == 'degrees':
            angs_in = conversion_to_dict['radians'] * angs_in
        self._rmat = make_rmat_euler(angs_in, self.axes_order, self.extrinsic)
        return self._rmat

    @rmat.setter
    def rmat(self, x):
        """
        Update class via input rotation matrix.

        Parameters
        ----------
        x : array_like
            A (3, 3) array to be interpreted as a rotation matrix.

        Returns
        -------
        None
        """
        rmat = _check_is_rmat(x)
        self._rmat = rmat

        axo = self.axes_order
        if not self.extrinsic:
            axo = axo.upper()

        # Ignore Gimbal Lock warning. It is okay.
        with ignore_warnings(UserWarning):
            self._angles = R.from_matrix(rmat).as_euler(
                axo, self.units == 'degrees'
            )

    @property
    def exponential_map(self):
        """
        The matrix invariants of self.rmat as exponential map parameters

        Returns
        -------
        np.ndarray
            The (3, ) array representing the exponential map parameters of
            the encoded rotation (self.rmat).

        """
        phi, n = angleAxisOfRotMat(self.rmat)
        return phi * n.flatten()

    @exponential_map.setter
    def exponential_map(self, x):
        """
        Updates encoded rotation via exponential map parameters

        Parameters
        ----------
        x : array_like
            The (3, ) vector representing exponential map parameters of a
            rotation.

        Returns
        -------
        None.

        Notes
        -----
        Updates the encoded rotation from expoential map parameters via
        self.rmat property
        """
        x = np.atleast_1d(x).flatten()
        assert len(x) == 3, "input must have exactly 3 elements"
        self.rmat = rotMatOfExpMap(x.reshape(3, 1))  # use local func


#
#  ==================== Fiber
#


def distanceToFiber(c, s, q, qsym, centrosymmetry=False, bmatrix=I3):
    """
    Calculate symmetrically reduced distance to orientation fiber.

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    qsym : TYPE
        DESCRIPTION.
    centrosymmetry : bool, optional
        If True, apply centrosymmetry to c. The default is False.
    bmatrix : np.ndarray, optional
        (3,3) b matrix. Default is the identity

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    if len(c) != 3 or len(s) != 3:
        raise RuntimeError('c and/or s are not 3-vectors')

    c = unitVector(np.dot(bmatrix, np.asarray(c)))
    s = unitVector(np.asarray(s).reshape(3, 1))

    nq = q.shape[1]  # number of quaternions
    rmats = rotMatOfQuat(q)  # (nq, 3, 3)

    csym = applySym(c, qsym, centrosymmetry)  # (3, m)
    m = csym.shape[1]  # multiplicity

    if nq == 1:
        rc = np.dot(rmats, csym)  # apply q's to c's

        sdotrc = np.dot(s.T, rc).max()
    else:
        rc = multMatArray(rmats, np.tile(csym, (nq, 1, 1)))  # apply q's to c's

        sdotrc = (
            np.dot(s.T, rc.swapaxes(1, 2).reshape(nq * m, 3).T)
            .reshape(nq, m)
            .max(1)
        )

    d = arccosSafe(np.array(sdotrc))

    return d


def discreteFiber(c, s, B=I3, ndiv=120, invert=False, csym=None, ssym=None):
    """
    Generate symmetrically reduced discrete orientation fiber.

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    B : TYPE, optional
        DESCRIPTION. The default is I3.
    ndiv : TYPE, optional
        DESCRIPTION. The default is 120.
    invert : TYPE, optional
        DESCRIPTION. The default is False.
    csym : TYPE, optional
        DESCRIPTION. The default is None.
    ssym : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    retval : TYPE
        DESCRIPTION.

    """

    ztol = cnst.sqrt_epsf

    c = np.asarray(c).reshape((3, 1))
    s = np.asarray(s).reshape((3, 1))

    nptc = c.shape[1]
    npts = s.shape[1]

    c = unitVector(np.dot(B, c))  # turn c hkls into unit vector in crys frame
    s = unitVector(s)  # convert s to unit vector in samp frame

    retval = []
    for i_c in range(nptc):
        dupl_c = np.tile(c[:, i_c], (npts, 1)).T

        ax = s + dupl_c
        anrm = columnNorm(ax).squeeze()  # should be 1-d

        okay = anrm > ztol
        nokay = okay.sum()
        if nokay == npts:
            ax = ax / np.tile(anrm, (3, 1))
        else:
            nspace = nullSpace(c[:, i_c].reshape(3, 1))
            hperp = nspace[:, 0].reshape(3, 1)
            if nokay == 0:
                ax = np.tile(hperp, (1, npts))
            else:
                ax[:, okay] = ax[:, okay] / np.tile(anrm[okay], (3, 1))
                ax[:, not okay] = np.tile(hperp, (1, npts - nokay))

        q0 = np.vstack([np.zeros(npts), ax])

        # find rotations
        # note: the following line fixes bug with use of arange
        # with float increments
        phi = np.arange(0, ndiv) * (2 * np.pi / float(ndiv))
        qh = quatOfAngleAxis(phi, np.tile(c[:, i_c], (ndiv, 1)).T)

        # the fibers, arraged as (npts, 4, ndiv)
        qfib = np.dot(quatProductMatrix(qh, mult='right'), q0).transpose(
            2, 1, 0
        )
        if csym is not None:
            retval.append(
                toFundamentalRegion(qfib.squeeze(), crysSym=csym, sampSym=ssym)
            )
        else:
            retval.append(fixQuat(qfib).squeeze())
    return retval


#
#  ==================== Utility Functions
#


def mapAngle(ang, ang_range=None, units=angularUnits):
    """
    Utility routine to map an angle into a specified period
    """
    if units.lower() == 'degrees':
        period = 360.0
    elif units.lower() == 'radians':
        period = 2.0 * np.pi
    else:
        raise RuntimeError("unknown angular units: " + units)

    ang = np.nan_to_num(np.atleast_1d(np.float64(ang)))

    min_val = -period / 2
    max_val = period / 2

    # if we have a specified angular range, use that
    if ang_range is not None:
        ang_range = np.atleast_1d(np.float64(ang_range))

        min_val = ang_range.min()
        max_val = ang_range.max()

        if not np.allclose(max_val - min_val, period):
            raise RuntimeError('range is incomplete!')

    val = np.mod(ang - min_val, max_val - min_val) + min_val
    # To match old implementation, map to closer value on the boundary
    # Not doing this breaks hedm_instrument's _extract_polar_maps
    val[np.logical_and(val == min_val, ang > min_val)] = max_val
    return val


def angularDifference_orig(angList0, angList1, units=angularUnits): # pragma: no cover
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range in the code is [-pi, pi]
    *) ... maybe more efficient not to vectorize?
    """
    if units == 'radians':
        period = 2 * np.pi
    elif units == 'degrees':
        period = 360.0
    else:
        raise RuntimeError(
            "'%s' is an unrecognized option for angular units!" % (units)
        )

    # take difference as arrays
    diffAngles = np.asarray(angList0) - np.asarray(angList1)

    return np.abs(np.mod(diffAngles + 0.5 * period, period) - 0.5 * period)


def angularDifference_opt(angList0, angList1, units=angularUnits):
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range in the code is [-pi, pi]
    """
    period = periodDict[units]
    d = np.abs(angList1 - angList0)
    return np.minimum(d, period - d)


angularDifference = angularDifference_opt


def applySym(vec, qsym, csFlag=False, cullPM=False, tol=cnst.sqrt_epsf):
    """
    Apply symmetry group to a single 3-vector (columnar) argument.

    csFlag : centrosymmetry flag
    cullPM : cull +/- flag
    """
    nsym = qsym.shape[1]
    Rsym = rotMatOfQuat(qsym)
    if nsym == 1:
        Rsym = np.array(
            [
                Rsym,
            ]
        )
    allhkl = (
        multMatArray(Rsym, np.tile(vec, (nsym, 1, 1)))
        .swapaxes(1, 2)
        .reshape(nsym, 3)
        .T
    )

    if csFlag:
        allhkl = np.hstack([allhkl, -1 * allhkl])
    _, uid = findDuplicateVectors(allhkl, tol=tol, equivPM=cullPM)

    return allhkl[np.ix_(list(range(3)), uid)]


# =============================================================================
# Symmetry functions
# =============================================================================


def toFundamentalRegion(q, crysSym='Oh', sampSym=None):
    """
    Map quaternions to fundamental region.

    Parameters
    ----------
    q : TYPE
        DESCRIPTION.
    crysSym : TYPE, optional
        DESCRIPTION. The default is 'Oh'.
    sampSym : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    qr : TYPE
        DESCRIPTION.
    """
    qdims = q.ndim
    if qdims == 3:
        l3, m3, n3 = q.shape
        assert m3 == 4, 'your 3-d quaternion array isn\'t the right shape'
        q = q.transpose(0, 2, 1).reshape(l3 * n3, 4).T
    if isinstance(crysSym, str):
        qsym_c = quatProductMatrix(
            quatOfLaueGroup(crysSym), 'right'
        )  # crystal symmetry operator
    else:
        qsym_c = quatProductMatrix(crysSym, 'right')

    n = q.shape[1]  # total number of quats
    m = qsym_c.shape[0]  # number of symmetry operations

    #
    # MAKE EQUIVALENCE CLASS
    #
    # Do R * Gc, store as
    # [q[:, 0] * Gc[:, 0:m], ..., 2[:, n-1] * Gc[:, 0:m]]
    qeqv = np.dot(qsym_c, q).transpose(2, 0, 1).reshape(m * n, 4).T

    if sampSym is None:
        # need to fix quats to sort
        qeqv = fixQuat(qeqv)

        # Reshape scalar comp columnwise by point in qeqv
        q0 = qeqv[0, :].reshape(n, m).T

        # Find q0 closest to origin for each n equivalence classes
        q0maxColInd = np.argmax(q0, 0) + [x * m for x in range(n)]

        # store representatives in qr
        qr = qeqv[:, q0maxColInd]
    else:
        if isinstance(sampSym, str):
            qsym_s = quatProductMatrix(
                quatOfLaueGroup(sampSym), 'left'
            )  # sample symmetry operator
        else:
            qsym_s = quatProductMatrix(sampSym, 'left')

        p = qsym_s.shape[0]  # number of sample symmetry operations

        # Do Gs * (R * Gc), store as
        # [Gs[:, 0:p]*q[:,   0]*Gc[:, 0], ..., Gs[:, 0:p]*q[:,   0]*Gc[:, m-1],
        #  ...,
        #  Gs[:, 0:p]*q[:, n-1]*Gc[:, 0], ..., Gs[:, 0:p]*q[:, n-1]*Gc[:, m-1]]
        qeqv = fixQuat(
            np.dot(qsym_s, qeqv).transpose(2, 0, 1).reshape(p * m * n, 4).T
        )

        raise NotImplementedError

    # debug
    assert qr.shape[1] == n, 'oops, something wrong here with your reshaping'

    if qdims == 3:
        qr = qr.T.reshape(l3, n3, 4).transpose(0, 2, 1)

    return qr


def ltypeOfLaueGroup(tag):
    """
    Yield lattice type of input tag.

    Parameters
    ----------
    tag : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    ltype : TYPE
        DESCRIPTION.

    """
    if not isinstance(tag, str):
        raise RuntimeError("entered flag is not a string!")

    if tag.lower() == 'ci' or tag.lower() == 's2':
        ltype = 'triclinic'
    elif tag.lower() == 'c2h':
        ltype = 'monoclinic'
    elif tag.lower() == 'd2h' or tag.lower() == 'vh':
        ltype = 'orthorhombic'
    elif tag.lower() == 'c4h' or tag.lower() == 'd4h':
        ltype = 'tetragonal'
    elif tag.lower() == 'c3i' or tag.lower() == 's6' or tag.lower() == 'd3d':
        ltype = 'trigonal'
    elif tag.lower() == 'c6h' or tag.lower() == 'd6h':
        ltype = 'hexagonal'
    elif tag.lower() == 'th' or tag.lower() == 'oh':
        ltype = 'cubic'
    else:
        raise RuntimeError(
            "unrecognized symmetry group.  "
            + "See ''help(quatOfLaueGroup)'' for a list of valid options.  "
            + "Oh, and have a great day ;-)"
        )

    return ltype


def quatOfLaueGroup(tag):
    """
    Return quaternion representation of requested symmetry group.

    Parameters
    ----------
    tag : str
        A case-insensitive string representing the Schoenflies symbol for the
        desired Laue group.  The 14 available choices are:

              Class           Symbol      N
             -------------------------------
              Triclinic       Ci (S2)     1
              Monoclinic      C2h         2
              Orthorhombic    D2h (Vh)    4
              Tetragonal      C4h         4
                              D4h         8
              Trigonal        C3i (S6)    3
                              D3d         6
              Hexagonal       C6h         6
                              D6h         12
              Cubic           Th          12
                              Oh          24

    Raises
    ------
    RuntimeError
        For invalid symmetry group tag.

    Returns
    -------
    qsym : (4, N) ndarray
        the quaterions associated with each element of the chosen symmetry
        group having n elements (dep. on group -- see INPUTS list above).

    Notes
    -----
    The conventions used for assigning a RHON basis, {x1, x2, x3}, to each
    point group are consistent with those published in Appendix B of [1]_.

    References
    ----------
    [1] Nye, J. F., ``Physical Properties of Crystals: Their
    Representation by Tensors and Matrices'', Oxford University Press,
    1985. ISBN 0198511655
    """
    if not isinstance(tag, str):
        raise RuntimeError("entered flag is not a string!")

    if tag.lower() == 'ci' or tag.lower() == 's2':
        # TRICLINIC
        angleAxis = np.vstack([0.0, 1.0, 0.0, 0.0])  # identity
    elif tag.lower() == 'c2h':
        # MONOCLINIC
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [np.pi, 0, 1, 0],  # twofold about 010 (x2)
        ]
    elif tag.lower() == 'd2h' or tag.lower() == 'vh':
        # ORTHORHOMBIC
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [np.pi, 1, 0, 0],  # twofold about 100
            [np.pi, 0, 1, 0],  # twofold about 010
            [np.pi, 0, 0, 1],  # twofold about 001
        ]
    elif tag.lower() == 'c4h':
        # TETRAGONAL (LOW)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby2, 0, 0, 1],  # fourfold about 001 (x3)
            [np.pi, 0, 0, 1],  #
            [piby2 * 3, 0, 0, 1],  #
        ]
    elif tag.lower() == 'd4h':
        # TETRAGONAL (HIGH)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby2, 0, 0, 1],  # fourfold about 0  0  1 (x3)
            [np.pi, 0, 0, 1],  #
            [piby2 * 3, 0, 0, 1],  #
            [np.pi, 1, 0, 0],  # twofold about  1  0  0 (x1)
            [np.pi, 0, 1, 0],  # twofold about  0  1  0 (x2)
            [np.pi, 1, 1, 0],  # twofold about  1  1  0
            [np.pi, -1, 1, 0],  # twofold about -1  1  0
        ]
    elif tag.lower() == 'c3i' or tag.lower() == 's6':
        # TRIGONAL (LOW)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3 * 2, 0, 0, 1],  # threefold about 0001 (x3,c)
            [piby3 * 4, 0, 0, 1],  #
        ]
    elif tag.lower() == 'd3d':
        # TRIGONAL (HIGH)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3 * 2, 0, 0, 1],  # threefold about 0001 (x3,c)
            [piby3 * 4, 0, 0, 1],  #
            [np.pi, 1, 0, 0],  # twofold about  2 -1 -1  0 (x1,a1)
            [np.pi, -0.5, sq3by2, 0],  # twofold about -1  2 -1  0 (a2)
            [np.pi, -0.5, -sq3by2, 0],  # twofold about -1 -1  2  0 (a3)
        ]
    elif tag.lower() == 'c6h':
        # HEXAGONAL (LOW)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3, 0, 0, 1],  # sixfold about 0001 (x3,c)
            [piby3 * 2, 0, 0, 1],  #
            [np.pi, 0, 0, 1],  #
            [piby3 * 4, 0, 0, 1],  #
            [piby3 * 5, 0, 0, 1],  #
        ]
    elif tag.lower() == 'd6h':
        # HEXAGONAL (HIGH)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby3, 0, 0, 1],  # sixfold about  0  0  1 (x3,c)
            [piby3 * 2, 0, 0, 1],  #
            [np.pi, 0, 0, 1],  #
            [piby3 * 4, 0, 0, 1],  #
            [piby3 * 5, 0, 0, 1],  #
            [np.pi, 1, 0, 0],  # twofold about  2 -1  0 (x1,a1)
            [np.pi, -0.5, sq3by2, 0],  # twofold about -1  2  0 (a2)
            [np.pi, -0.5, -sq3by2, 0],  # twofold about -1 -1  0 (a3)
            [np.pi, sq3by2, 0.5, 0],  # twofold about  1  0  0
            [np.pi, 0, 1, 0],  # twofold about -1  1  0 (x2)
            [np.pi, -sq3by2, 0.5, 0],  # twofold about  0 -1  0
        ]
    elif tag.lower() == 'th':
        # CUBIC (LOW)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [np.pi, 1, 0, 0],  # twofold about    1  0  0 (x1)
            [np.pi, 0, 1, 0],  # twofold about    0  1  0 (x2)
            [np.pi, 0, 0, 1],  # twofold about    0  0  1 (x3)
            [piby3 * 2, 1, 1, 1],  # threefold about  1  1  1
            [piby3 * 4, 1, 1, 1],  #
            [piby3 * 2, -1, 1, 1],  # threefold about -1  1  1
            [piby3 * 4, -1, 1, 1],  #
            [piby3 * 2, -1, -1, 1],  # threefold about -1 -1  1
            [piby3 * 4, -1, -1, 1],  #
            [piby3 * 2, 1, -1, 1],  # threefold about  1 -1  1
            [piby3 * 4, 1, -1, 1],  #
        ]
    elif tag.lower() == 'oh':
        # CUBIC (HIGH)
        angleAxis = np.c_[
            [0.0, 1, 0, 0],  # identity
            [piby2, 1, 0, 0],  # fourfold about   1  0  0 (x1)
            [np.pi, 1, 0, 0],  #
            [piby2 * 3, 1, 0, 0],  #
            [piby2, 0, 1, 0],  # fourfold about   0  1  0 (x2)
            [np.pi, 0, 1, 0],  #
            [piby2 * 3, 0, 1, 0],  #
            [piby2, 0, 0, 1],  # fourfold about   0  0  1 (x3)
            [np.pi, 0, 0, 1],  #
            [piby2 * 3, 0, 0, 1],  #
            [piby3 * 2, 1, 1, 1],  # threefold about  1  1  1
            [piby3 * 4, 1, 1, 1],  #
            [piby3 * 2, -1, 1, 1],  # threefold about -1  1  1
            [piby3 * 4, -1, 1, 1],  #
            [piby3 * 2, -1, -1, 1],  # threefold about -1 -1  1
            [piby3 * 4, -1, -1, 1],  #
            [piby3 * 2, 1, -1, 1],  # threefold about  1 -1  1
            [piby3 * 4, 1, -1, 1],  #
            [np.pi, 1, 1, 0],  # twofold about    1  1  0
            [np.pi, -1, 1, 0],  # twofold about   -1  1  0
            [np.pi, 1, 0, 1],  # twofold about    1  0  1
            [np.pi, 0, 1, 1],  # twofold about    0  1  1
            [np.pi, -1, 0, 1],  # twofold about   -1  0  1
            [np.pi, 0, -1, 1],  # twofold about    0 -1  1
        ]
    else:
        raise RuntimeError(
            "unrecognized symmetry group.  "
            + "See ``help(quatOfLaueGroup)'' for a list of valid options.  "
            + "Oh, and have a great day ;-)"
        )

    angle = angleAxis[0,]
    axis = angleAxis[1:,]

    #  Note: Axis does not need to be normalized in call to quatOfAngleAxis
    #  05/01/2014 JVB -- made output a contiguous C-ordered array
    qsym = np.array(quatOfAngleAxis(angle, axis).T, order='C').T

    return qsym
