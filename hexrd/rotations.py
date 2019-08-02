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
import time
import numpy as np

from numpy import \
     arange, arctan2, array, asarray, atleast_1d, average, \
     ndarray, diag, zeros, \
     cross, dot, pi, arccos, arcsin, cos, sin, sqrt, \
     sort, tile, vstack, hstack, c_, ix_, \
     abs, mod, sign, \
     finfo, isscalar

from numpy import float_ as nFloat
from numpy import int_ as nInt

from scipy.optimize import leastsq

from hexrd import constants as cnst

from hexrd.matrixutil import \
    columnNorm, unitVector, \
    skewMatrixOfVector, findDuplicateVectors, \
    multMatArray, nullSpace


# =============================================================================
# Module Data
# =============================================================================


angularUnits = 'radians'  # module-level angle units
periodDict = {'degrees': 360.0, 'radians': 2*np.pi}
I3 = cnst.identity_3x3    # (3, 3) identity matrix

# axes orders, all permutations
axes_orders = [
    'xyz', 'zyx',
    'zxy', 'yxz',
    'yzx', 'xzy',
    'xyx', 'xzx',
    'yxy', 'yzy',
    'zxz', 'zyz',
]

# =============================================================================
# Functions
# =============================================================================


def arccosSafe(temp):
    """
    Protect against numbers slightly larger than 1 in magnitude
    due to round-off
    """
    temp = atleast_1d(temp)
    if (abs(temp) > 1.00001).any():
        print("attempt to take arccos of %s" % temp, file=sys.stderr)
        raise RuntimeError("unrecoverable error")

    gte1 = temp >= 1.
    lte1 = temp <= -1.

    temp[gte1] = 1
    temp[lte1] = -1

    ang = arccos(temp)

    return ang


#
#  ==================== Quaternions
#


def fixQuat(q):
    """
    flip to positive q0 and normalize
    """
    qdims = q.ndim
    if qdims == 3:
        l, m, n = q.shape
        assert m == 4, 'your 3-d quaternion array isn\'t the right shape'
        q = q.transpose(0, 2, 1).reshape(l*n, 4).T

    qfix = unitVector(q)

    q0negative = qfix[0, ] < 0
    qfix[:, q0negative] = -1*qfix[:, q0negative]

    if qdims == 3:
        qfix = qfix.T.reshape(l, n, 4).transpose(0, 2, 1)

    return qfix


def invertQuat(q):
    """
    silly little routine for inverting a quaternion
    """
    numq = q.shape[1]

    imat = tile(vstack([-1, 1, 1, 1]), (1, numq))

    qinv = imat * q

    return fixQuat(qinv)


def misorientation(q1, q2, *args):
    """
    sym is a tuple (crystal_symmetry, *sample_symmetry)
    generally coded.

    !!! may split up special cases for no symmetry or crystal/sample only...
    """
    if not isinstance(q1, ndarray) or not isinstance(q2, ndarray):
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

    if len(args) == 0:
        # no symmetries; use identity
        sym = (c_[1., 0, 0, 0].T, c_[1., 0, 0, 0].T)
    else:
        sym = args[0]
        if len(sym) == 1:
            if not isinstance(sym[0], ndarray):
                raise RuntimeError("symmetry argument is not an numpy array")
            else:
                # add triclinic sample symmetry (identity)
                sym += (c_[1., 0, 0, 0].T,)
        elif len(sym) == 2:
            if not isinstance(sym[0], ndarray) \
              or not isinstance(sym[1], ndarray):
                raise RuntimeError(
                    "symmetry arguments are not an numpy arrays"
                )
        elif len(sym) > 2:
            raise RuntimeError(
                "symmetry argument has %d entries; should be 1 or 2"
                % (len(sym))
            )

    # set some lengths
    n = q2.shape[1]             # length of misorientation list
    m = sym[0].shape[1]         # crystal (right)
    p = sym[1].shape[1]         # sample  (left)

    # tile q1 inverse
    q1i = quatProductMatrix(invertQuat(q1), mult='right').squeeze()

    # convert symmetries to (4, 4) qprod matrices
    rsym = quatProductMatrix(sym[0], mult='right')
    lsym = quatProductMatrix(sym[1], mult='left')

    # Do R * Gc, store as
    # [q2[:, 0] * Gc[:, 0:m], ..., q2[:, n-1] * Gc[:, 0:m]]
    q2 = dot(rsym, q2).transpose(2, 0, 1).reshape(m*n, 4).T

    # Do Gs * (R * Gc), store as
    # [Gs[:, 0:p] * q[:,   0] * Gc[:, 0], ... , Gs[:, 0:p] * q[:,   0] * Gc[:, m-1], ...
    #  Gs[:, 0:p] * q[:, n-1] * Gc[:, 0], ... , Gs[:, 0:p] * q[:, n-1] * Gc[:, m-1]]
    q2 = dot(lsym, q2).transpose(2, 0, 1).reshape(p*m*n, 4).T

    # Calculate the class misorientations for full symmetrically equivalent
    # classes for q1 and q2.  Note the use of the fact that the application
    # of the symmetry groups is an isometry.
    eqvMis = fixQuat(dot(q1i, q2))

    # Reshape scalar comp columnwise by point in q2 (and q1, if applicable)
    sclEqvMis = eqvMis[0, :].reshape(n, p*m).T

    # Find misorientation closest to origin for each n equivalence classes
    #   - fixed quats so garaunteed that sclEqvMis is nonnegative
    qmax = sclEqvMis.max(0)

    # remap indices to use in eqvMis
    qmaxInd = (sclEqvMis == qmax).nonzero()
    qmaxInd = c_[qmaxInd[0], qmaxInd[1]]

    eqvMisColInd = sort(qmaxInd[:, 0] + qmaxInd[:, 1]*p*m)

    # store Rmin in q
    mis = eqvMis[ix_(list(range(4)), eqvMisColInd)]

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
    n1 = q1.shape[1]
    n2 = q2.shape[1]

    nq = 1
    if n1 == 1 or n2 == 1:
        if n2 > n1:
            q1 = tile(q1, (1, n2))
            nq = n2
        else:
            q2 = tile(q2, (1, n1))
            nq = n1

    a = q2[0, ]
    a3 = tile(a, (3, 1))
    b = q1[0, ]
    b3 = tile(b, (3, 1))

    avec = q2[1:, ]
    bvec = q1[1:, ]

    axb = zeros((3, nq))
    for i in range(nq):
        axb[:, i] = cross(avec[:, i], bvec[:, i])

    qp = vstack([a*b - diag(dot(avec.T, bvec)),
                 a3*bvec + b3*avec + axb])

    return fixQuat(qp)


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

                       / quatProductMatrix(h, mult='right') * q
           q * h  --> <
                       \ quatProductMatrix(q, mult='left') * h

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
        qmats = array([[ q0], [ q1], [ q2], [ q3],
                       [-q1], [ q0], [-q3], [ q2],
                       [-q2], [ q3], [ q0], [-q1],
                       [-q3], [-q2], [ q1], [ q0]])
    elif mult == 'left':
        qmats = array([[ q0], [ q1], [ q2], [ q3],
                       [-q1], [ q0], [ q3], [-q2],
                       [-q2], [-q3], [ q0], [ q1],
                       [-q3], [ q2], [-q1], [ q0]])

    # some fancy reshuffling...
    qmats = qmats.T.reshape(nq, 4, 4).transpose(0, 2, 1)

    return qmats


def quatOfAngleAxis(angle, rotaxis):
    """
    make an hstacked array of quaternions from arrays of angle/axis pairs
    """
    if isinstance(angle, list):
        n = len(angle)
        angle = asarray(angle)
    elif isinstance(angle, float) or isinstance(angle, int):
        n = 1
    elif isinstance(angle, ndarray):
        n = angle.shape[0]
    else:
        raise RuntimeError(
            "angle argument is of incorrect type; "
            + "must be a list, int, float, or ndarray."
        )

    if rotaxis.shape[1] == 1:
        rotaxis = tile(rotaxis, (1, n))
    else:
        if rotaxis.shape[1] != n:
            raise RuntimeError("rotation axes argument has incompatible shape")

    halfangle = 0.5*angle
    cphiby2 = cos(halfangle)
    sphiby2 = sin(halfangle)

    quat = vstack([cphiby2, tile(sphiby2, (3, 1)) * unitVector(rotaxis)])

    return fixQuat(quat)


def quatOfExpMap(expMap):
    """
    """
    angles = columnNorm(expMap)
    axes = unitVector(expMap)

    quats = quatOfAngleAxis(angles, axes)
    return quats


def quatOfRotMat(R):
    """
    """
    angs, axxs = angleAxisOfRotMat(R)
    quats = vstack(
        [cos(0.5 * angs),
         tile(sin(0.5 * angs), (3, 1)) * axxs]
    )
    return quats


def quatAverageCluster(q_in, qsym):
    """
    """
    from symmetry import toFundamentalRegion

    assert q_in.ndim == 2, 'input must be 2-s hstacked quats'

    # renormalize
    q_in = unitVector(q_in)

    # check to see num of quats is > 1
    if q_in.shape[1] < 3:
        if q_in.shape[1] == 1:
            q_bar = q_in
        else:
            ma, mq = misorientation(q_in[:, 0].reshape(4, 1),
                                    q_in[:, 1].reshape(4, 1), (qsym,))
            q_bar = quatProduct(
                q_in[:, 0].reshape(4, 1),
                quatOfExpMap(0.5*ma*unitVector(mq[1:].reshape(3, 1)))
            )
    else:
        # first drag to origin using first quat (arb!)
        q0 = q_in[:, 0].reshape(4, 1)
        qrot = dot(
            quatProductMatrix(invertQuat(q0), mult='left'),
            q_in)

        # second, re-cast to FR
        qrot = toFundamentalRegion(qrot.squeeze(), crysSym=qsym)

        # compute arithmetic average
        q_bar = unitVector(average(qrot, axis=1).reshape(4, 1))

        # unrotate!
        q_bar = dot(
            quatProductMatrix(q0, mult='left'),
            q_bar)

        # re-map
        q_bar = toFundamentalRegion(q_bar, crysSym=qsym)
    return q_bar


def quatAverage(q_in, qsym):
    """
    """
    assert q_in.ndim == 2, 'input must be 2-s hstacked quats'

    # renormalize
    q_in = unitVector(q_in)

    # check to see num of quats is > 1
    if q_in.shape[1] < 3:
        if q_in.shape[1] == 1:
            q_bar = q_in
        else:
            ma, mq = misorientation(q_in[:, 0].reshape(4, 1),
                                    q_in[:, 1].reshape(4, 1), (qsym,))
            q_bar = quatProduct(
                q_in[:, 0].reshape(4, 1),
                quatOfExpMap(0.5*ma*unitVector(mq[1:].reshape(3, 1)))
            )
    else:
        # use first quat as initial guess
        phi = 2. * arccos(q_in[0, 0])
        if phi <= finfo(float).eps:
            x0 = zeros(3)
        else:
            n = unitVector(q_in[1:, 0].reshape(3, 1))
            x0 = phi*n.flatten()
        results = leastsq(quatAverage_obj, x0, args=(q_in, qsym))
        phi = sqrt(sum(results[0]*results[0]))
        if phi <= finfo(float).eps:
            q_bar = c_[1., 0., 0., 0.].T
        else:
            n = results[0] / phi
            q_bar = hstack([cos(0.5*phi), sin(0.5*phi)*n]).reshape(4, 1)
    return q_bar


def quatAverage_obj(xi_in, quats, qsym):
    phi = sqrt(sum(xi_in.flatten()*xi_in.flatten()))
    if phi <= finfo(float).eps:
        q0 = c_[1., 0., 0., 0.].T
    else:
        n = xi_in.flatten() / phi
        q0 = hstack([cos(0.5*phi), sin(0.5*phi)*n])
    resd = misorientation(q0.reshape(4, 1), quats, (qsym, ))[0]
    return resd


def rotMatOfExpMap_opt(expMap):
    """Optimized version of rotMatOfExpMap
    """
    if expMap.ndim == 1:
        expMap = expMap.reshape(3, 1)

    # angles of rotation from exponential maps
    phi = atleast_1d(columnNorm(expMap))

    # skew matrices of exponential maps
    W = skewMatrixOfVector(expMap)

    # Find tiny angles to avoid divide-by-zero and apply limits in expressions
    zeroIndex = phi < cnst.epsf
    phi[zeroIndex] = 1

    # first term
    C1 = sin(phi) / phi
    C1[zeroIndex] = 1  # is this right?  might be OK since C1 multiplies W

    # second term
    C2 = (1 - cos(phi)) / phi**2
    C2[zeroIndex] = 0.5  # won't matter because W^2 is small

    numObjs = expMap.shape[1]
    if numObjs == 1:  # case of single point
        W = np.reshape(W, [1, 3, 3])
        pass

    C1 = np.tile(
        np.reshape(C1, [numObjs, 1]),
        [1, 9]).reshape([numObjs, 3, 3])
    C2 = np.tile(
        np.reshape(C2, [numObjs, 1]),
        [1, 9]).reshape([numObjs, 3, 3])

    W2 = np.zeros([numObjs, 3, 3])

    for i in range(3):
        for j in range(3):
            W2[:, i, j] = np.sum(W[:, i, :]*W[:, :, j], 1)
            pass
        pass

    rmat = C1*W + C2 * W2
    rmat[:, 0, 0] += 1.
    rmat[:, 1, 1] += 1.
    rmat[:, 2, 2] += 1.

    return rmat.squeeze()


def rotMatOfExpMap_orig(expMap):
    """
    Original rotMatOfExpMap, used for comparison to optimized version
    """
    if isinstance(expMap, ndarray):
        if expMap.ndim != 2:
            if expMap.ndim == 1 and len(expMap) == 3:
                numObjs = 1
                expMap = expMap.reshape(3, 1)
            else:
                raise RuntimeError("input is the wrong dimensionality")
        elif expMap.shape[0] != 3:
            raise RuntimeError(
                "input is the wrong shape along the 0-axis; "
                + "Yours is %d when is should be 3"
                % (expMap.shape[0])
            )
        else:
            numObjs = expMap.shape[1]
    elif isinstance(expMap, list) or isinstance(expMap, tuple):
        if len(expMap) != 3:
            raise RuntimeError(
                "for list/tuple input only one exponential map "
                + "vector is allowed"
            )
        else:
            if not isscalar(expMap[0]) or not isscalar(expMap[1]) \
              or not isscalar(expMap[2]):
                raise RuntimeError(
                    "for list/tuple input only one exponential map "
                    + "vector is allowed"
                )
            else:
                numObjs = 1
                expMap = asarray(expMap).reshape(3, 1)

    phi = columnNorm(expMap)  # angles of rotation from exponential maps
    W = skewMatrixOfVector(expMap)  # skew matrices of exponential maps

    # Find tiny angles to avoid divide-by-zero and apply limits in expressions
    zeroIndex = phi < cnst.epsf
    phi[zeroIndex] = 1

    # first term
    C1 = sin(phi) / phi
    C1[zeroIndex] = 1

    # second term
    C2 = (1 - cos(phi)) / phi**2
    C2[zeroIndex] = 1

    if numObjs == 1:
        rmat = I3 + C1 * W + C2 * dot(W, W)
    else:
        rmat = zeros((numObjs, 3, 3))
        for i in range(numObjs):
            rmat[i, :, :] = \
                I3 + C1[i] * W[i, :, :] + C2[i] * dot(W[i, :, :], W[i, :, :])

    return rmat


# Donald Boyce's
rotMatOfExpMap = rotMatOfExpMap_opt


def rotMatOfQuat(quat):
    """
    Take an array of n quats (numpy ndarray, 4 x n) and generate an
    array of rotation matrices (n x 3 x 3)

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

    n = quat.shape[1]
    rmat = zeros((n, 3, 3), dtype='float64')

    a = quat[0, :].reshape(n, 1)
    b = quat[1, :].reshape(n, 1)
    c = quat[2, :].reshape(n, 1)
    d = quat[3, :].reshape(n, 1)

    R = c_[a**2 + b**2 - c**2 - d**2,
           2*b*c - 2*a*d,
           2*a*c + 2*b*d,
           2*a*d + 2*b*c,
           a**2 - b**2 + c**2 - d**2,
           2*c*d - 2*a*b,
           2*b*d - 2*a*c,
           2*a*b + 2*c*d,
           a**2 - b**2 - c**2 + d**2]

    if n > 1:
        rmat = R.reshape(n, 3, 3)
    else:
        rmat = R.reshape(3, 3)

    # for i in range(n):
    #     theta = 2. * arccosSafe(quat[0, i])
    #
    #     # find axial vector
    #     if (theta > cnst.epsf):
    #         a = sin(theta) / theta
    #         b = (1. - cos(theta)) / theta**2
    #         w = (theta / sin(0.5 * theta)) * quat[1:4, i]
    #
    #         wskew = array([[   0., -w[2],  w[1]],
    #                        [ w[2],    0., -w[0]],
    #                        [-w[1],  w[0],    0.]])
    #
    #         rmat[i, :, :] = I3 + a * wskew + b * dot(wskew, wskew)
    #     else:
    #         rmat[i, :, :] = I3

    return rmat


def angleAxisOfRotMat(R):
    """
    """
    if not isinstance(R, ndarray):
        raise RuntimeError('Input must be a 2 or 3-d ndarray')
    else:
        rdim = R.ndim
        if rdim == 2:
            nrot = 1
            R = tile(R, (1, 1, 1))
        elif rdim == 3:
            nrot = R.shape[0]
        else:
            raise RuntimeError(
                "R array must be (3, 3) or (n, 3, 3); input has dimension %d"
                % (rdim)
            )

    #
    #  Find angle of rotation.
    #
    ca = 0.5*(R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1)

    angle = arccosSafe(ca)

    #
    #  Three cases for the angle:
    #
    #  *   near zero -- matrix is effectively the identity
    #  *   near pi   -- binary rotation; need to find axis
    #  *   neither   -- general case; can use skew part
    #
    tol = 1.0e-4

    anear0 = angle < tol

    angle[anear0] = 0

    raxis = vstack(
        [R[:, 2, 1] - R[:, 1, 2],
         R[:, 0, 2] - R[:, 2, 0],
         R[:, 1, 0] - R[:, 0, 1]]
    )
    raxis[:, anear0] = 1

    special = angle > pi - tol
    nspec = special.sum()
    if nspec > 0:
        tmp = R[special, :, :] + tile(I3, (nspec, 1, 1))
        tmpr = tmp.transpose(0, 2, 1).reshape(nspec*3, 3).T

        tmpnrm = (tmpr*tmpr).sum(0).reshape(3, nspec)
        mx = tmpnrm.max(0)

        # remap indices
        maxInd = (tmpnrm == mx).nonzero()
        maxInd = c_[maxInd[0], maxInd[1]]

        tmprColInd = sort(maxInd[:, 0] + maxInd[:, 1]*nspec)

        saxis = tmpr[:, tmprColInd]

        raxis[:, special] = saxis

    return angle, unitVector(raxis)


def _check_axes_order(x):
    if not isinstance(x, str):
        raise RuntimeError("argument must be str")
    axo = x.lower()
    if axo not in axes_orders:
        raise RuntimeError(
            "order '%s' is not a valid choice"
            % x
        )
    return axo


def _check_is_rmat(x):
    x = np.asarray(x)
    if x.shape != (3, 3):
        raise RuntimeError("shape of input must be (3, 3)")
    chk1 = np.linalg.det(x)
    chk2 = np.sum(abs(np.eye(3) - np.dot(x, x.T)))
    if 1. - abs(chk1) < cnst.sqrt_epsf and chk2 < cnst.sqrt_epsf:
        return x
    else:
        raise RuntimeError("input is not an orthogonal matrix")


def make_rmat_euler(tilt_angles, axes_order, extrinsic=True):
    """
    extrinsic (PASSIVE) or intrinsic (ACTIVE) by kw
    tilt_angles are in RADIANS
    """
    axes = np.eye(3)
    axes_dict = dict(x=0, y=1, z=2)

    axo = _check_axes_order(axes_order)

    if extrinsic:
        rmats = np.zeros((3, 3, 3))
        for i, ax in enumerate(axo):
            rmats[i] = rotMatOfExpMap(
                tilt_angles[i]*axes[axes_dict[ax]]
            )
        return np.dot(rmats[2], np.dot(rmats[1], rmats[0]))
    else:
        rm0 = rotMatOfExpMap(
            tilt_angles[0]*axes[axes_dict[axo[0]]]
        )
        rm1 = rotMatOfExpMap(
            tilt_angles[1]*rm0[:, axes_dict[axo[1]]]
        )
        rm2 = rotMatOfExpMap(
            tilt_angles[2]*np.dot(rm1, rm0[:, axes_dict[axo[2]]])
        )
        return np.dot(rm2, np.dot(rm1, rm0))


def angles_from_rmat_xyz(rmat):
    """
    calculate x-y-z euler angles from a rotation matrix in
    the PASSIVE convention
    """
    rmat = _check_is_rmat(rmat)

    eps = sqrt(finfo('float').eps)
    ry = -arcsin(rmat[2, 0])
    sgny = sign(ry)
    if abs(ry) < 0.5*pi - eps:
        cosy = cos(ry)
        rz = arctan2(rmat[1, 0]/cosy, rmat[0, 0]/cosy)
        rx = arctan2(rmat[2, 1]/cosy, rmat[2, 2]/cosy)
    else:
        rz = 0.5*arctan2(sgny*rmat[1, 2], sgny*rmat[0, 2])
        if sgny > 0:
            rx = -rz
        else:
            rx = rz
    return rx, ry, rz

def angles_from_rmat_zxz(rmat):
    """
    calculate z-x-z euler angles from a rotation matrix in
    the ACTIVE convention

    alpha, beta, gamma
    """
    rmat = _check_is_rmat(rmat)

    if abs(rmat[2, 2]) > 1. - sqrt(finfo('float').eps):
        beta = 0.
        alpha = arctan2(rmat[1, 0], rmat[0, 0])
        gamma = 0.
    else:
        xnew = rmat[:, 0]
        znew = rmat[:, 2]
        alpha = arctan2(znew[0], -znew[1])
        rma = rotMatOfExpMap(alpha*c_[0., 0., 1.].T)
        znew1 = dot(rma.T, znew)
        beta = arctan2(-znew1[1], znew1[2])
        rmb = rotMatOfExpMap(beta*c_[cos(alpha), sin(alpha), 0.].T)
        xnew2 = dot(rma.T, dot(rmb.T, xnew))
        gamma = arctan2(xnew2[1], xnew2[0])
    return alpha, beta, gamma


class RotMatEuler(object):
    def __init__(self, angles, axes_order, extrinsic=True):
        """
        ??? add check that angle input is array-like, len() = 3?
        ??? add check on extrinsic as bool
        """
        self._axes = np.eye(3)
        self._axes_dict = dict(x=0, y=1, z=2)

        # these will be properties
        self._angles = angles
        self._axes_order = _check_axes_order(axes_order)
        self._extrinsic = extrinsic

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
    def rmat(self):
        self._rmat = make_rmat_euler(
            self.angles, self.axes_order, self.extrinsic)
        return self._rmat

    @rmat.setter
    def rmat(self, x):
        rmat = _check_is_rmat(x)
        self._rmat = rmat
        if self.axes_order == 'xyz':
            if self.extrinsic == True:
                angles = angles_from_rmat_xyz(rmat)
            else:
                raise NotImplementedError
        elif self.axes_order == 'zxz':
            if self.extrinsic == True:
                raise NotImplementedError
            else:
                angles = angles_from_rmat_zxz(rmat)
        else:
            raise NotImplementedError
        self._angles = angles


#
#  ==================== Fiber
#


def distanceToFiber(c, s, q, qsym, **kwargs):
    """
    """
    csymFlag = False
    B = I3

    arglen = len(kwargs)

    if len(c) != 3 or len(s) != 3:
        raise RuntimeError('c and/or s are not 3-vectors')

    # argument handling
    if arglen > 0:
        argkeys = list(kwargs.keys())
        for i in range(arglen):
            if argkeys[i] == 'centrosymmetry':
                csymFlag = kwargs[argkeys[i]]
            elif argkeys[i] == 'bmatrix':
                B = kwargs[argkeys[i]]
            else:
                raise RuntimeError("keyword arg \'%s\' is not recognized"
                                   % (argkeys[i]))

    c = unitVector(dot(B, asarray(c)))
    s = unitVector(asarray(s).reshape(3, 1))

    nq = q.shape[1]  # number of quaternions
    rmats = rotMatOfQuat(q)  # (nq, 3, 3)

    csym = applySym(c, qsym, csymFlag)  # (3, m)
    m = csym.shape[1]  # multiplicity

    if nq == 1:
        rc = dot(rmats, csym)  # apply q's to c's

        sdotrc = dot(s.T, rc).max()
    else:
        rc = multMatArray(
            rmats, tile(csym, (nq, 1, 1))
        )  # apply q's to c's

        sdotrc = dot(
            s.T,
            rc.swapaxes(1, 2).reshape(nq*m, 3).T
        ).reshape(nq, m).max(1)

    d = arccosSafe(array(sdotrc))

    return d


def discreteFiber(c, s, B=I3, ndiv=120, invert=False, csym=None, ssym=None):
    """
    """
    import symmetry as S

    ztol = 1.e-8

    # arg handling for c
    if hasattr(c, '__len__'):
        if hasattr(c, 'shape'):
            assert c.shape[0] == 3, \
                   'scattering vector must be 3-d; yours is %d-d' \
                   % (c.shape[0])
            if len(c.shape) == 1:
                c = c.reshape(3, 1)
            elif len(c.shape) > 2:
                raise RuntimeError(
                    'incorrect arg shape; must be 1-d or 2-d, yours is %d-d'
                    % (len(c.shape))
                )
        else:
            # convert list input to array and transpose
            if len(c) == 3 and isscalar(c[0]):
                c = asarray(c).reshape(3, 1)
            else:
                c = asarray(c).T
    else:
        raise RuntimeError('input must be array-like')

    # arg handling for s
    if hasattr(s, '__len__'):
        if hasattr(s, 'shape'):
            assert s.shape[0] == 3, \
                   'scattering vector must be 3-d; yours is %d-d' \
                   % (s.shape[0])
            if len(s.shape) == 1:
                s = s.reshape(3, 1)
            elif len(s.shape) > 2:
                raise RuntimeError(
                    'incorrect arg shape; must be 1-d or 2-d, yours is %d-d'
                    % (len(s.shape)))
        else:
            # convert list input to array and transpose
            if len(s) == 3 and isscalar(s[0]):
                s = asarray(s).reshape(3, 1)
            else:
                s = asarray(s).T
    else:
        raise RuntimeError('input must be array-like')

    nptc = c.shape[1]
    npts = s.shape[1]

    c = unitVector(dot(B, c))  # turn c hkls into unit vector in crys frame
    s = unitVector(s)  # convert s to unit vector in samp frame

    retval = []
    for i_c in range(nptc):
        dupl_c = tile(c[:, i_c], (npts, 1)).T

        ax = s + dupl_c
        anrm = columnNorm(ax).squeeze()  # should be 1-d

        okay = anrm > ztol
        nokay = okay.sum()
        if nokay == npts:
            ax = ax / tile(anrm, (3, 1))
        else:
            nspace = nullSpace(c[:, i_c].reshape(3, 1))
            hperp = nspace[:, 0].reshape(3, 1)
            if nokay == 0:
                ax = tile(hperp, (1, npts))
            else:
                ax[:,     okay] = ax[:, okay] / tile(anrm[okay], (3, 1))
                ax[:, not okay] = tile(hperp, (1, npts - nokay))

        q0 = vstack([zeros(npts), ax])

        # find rotations
        # note: the following line fixes bug with use of arange
        # with float increments
        phi = arange(0, ndiv) * (2*pi/float(ndiv))
        qh = quatOfAngleAxis(phi, tile(c[:, i_c], (ndiv, 1)).T)

        # the fibers, arraged as (npts, 4, ndiv)
        qfib = dot(
            quatProductMatrix(qh, mult='right'),
            q0
        ).transpose(2, 1, 0)
        if csym is not None:
            retval.append(S.toFundamentalRegion(qfib.squeeze(),
                                                crysSym=csym,
                                                sampSym=ssym))
        else:
            retval.append(fixQuat(qfib).squeeze())
    return retval


#
#  ==================== Utility Functions
#


def mapAngle(ang, *args, **kwargs):
    """
    Utility routine to map an angle into a specified period
    """
    period = 2.*pi        # radians
    units = angularUnits  # usually

    kwargKeys = list(kwargs.keys())
    for iArg in range(len(kwargKeys)):
        if kwargKeys[iArg] == 'units':
            units = kwargs[kwargKeys[iArg]]
        else:
            raise RuntimeError(
                "Unknown keyword argument: "
                + str(kwargKeys[iArg])
            )

    if units.lower() == 'degrees':
        period = 360.
    elif units.lower() != 'radians':
        raise RuntimeError(
            "unknown angular units: "
            + str(kwargs[kwargKeys[iArg]])
        )

    ang = atleast_1d(nFloat(ang))

    # if we have a specified angular range, use that
    if len(args) > 0:
        angRange = atleast_1d(nFloat(args[0]))

        # divide of multiples of period
        ang = ang - nInt(ang / period) * period

        lb = angRange.min()
        ub = angRange.max()

        if abs(ub - lb) != period:
            raise RuntimeError('range is incomplete!')

        lbi = ang < lb
        while lbi.sum() > 0:
            ang[lbi] = ang[lbi] + period
            lbi = ang < lb
            pass
        ubi = ang > ub
        while ubi.sum() > 0:
            ang[ubi] = ang[ubi] - period
            ubi = ang > ub
            pass
        retval = ang
    else:
        retval = mod(ang + 0.5*period, period) - 0.5*period
    return retval


def angularDifference_orig(angList0, angList1, units=angularUnits):
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range in the code is [-pi, pi]
    *) ... maybe more efficient not to vectorize?
    """
    if units == 'radians':
        period = 2*pi
    elif units == 'degrees':
        period = 360.
    else:
        raise RuntimeError(
            "'%s' is an unrecognized option for angular units!"
            % (units)
        )

    # take difference as arrays
    diffAngles = asarray(angList0) - asarray(angList1)

    return abs(mod(diffAngles + 0.5*period, period) - 0.5*period)


def angularDifference_opt(angList0, angList1, units=angularUnits):
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range in the code is [-pi, pi]
    """
    period = periodDict[units]
    d = abs(angList1 - angList0)
    return np.minimum(d, period - d)


angularDifference = angularDifference_opt


def applySym(vec, qsym, csFlag=False, cullPM=False, tol=1e-8):
    """
    apply symmetry group to a single 3-vector (columnar) argument

    csFlag : centrosymmetry flag
    cullPM : cull +/- flag
    """
    nsym = qsym.shape[1]
    Rsym = rotMatOfQuat(qsym)
    if nsym == 1:
        Rsym = array([Rsym, ])
    allhkl = multMatArray(
        Rsym, tile(vec, (nsym, 1, 1))
    ).swapaxes(1, 2).reshape(nsym, 3).T

    if csFlag:
        allhkl = hstack([allhkl, -1*allhkl])
    eqv, uid = findDuplicateVectors(allhkl, tol=tol, equivPM=cullPM)

    return allhkl[ix_(list(range(3)), uid)]


# =============================================================================
# Tests
# =============================================================================


def printTestName(num, name):
    print('==================== Test %d:  %s' % (num, name))


def testRotMatOfExpMap(numpts):
    """Test rotation matrix from axial vector"""

    print('* checking case of 1D vector input')
    map = np.zeros(3)
    rmat_1 = rotMatOfExpMap_orig(map)
    rmat_2 = rotMatOfExpMap_opt(map)
    print('resulting shapes:  ', rmat_1.shape, rmat_2.shape)
    #
    #
    map = np.random.rand(3, numPts)
    map = np.zeros([3, numPts])
    map[0, :] = np.linspace(0, np.pi, numPts)
    #
    print('* testing rotMatOfExpMap with %d random points' % numPts)
    #
    t0 = time.clock()
    rmat_1 = rotMatOfExpMap_orig(map)
    et1 = time.clock() - t0
    #
    t0 = time.clock()
    rmat_2 = rotMatOfExpMap_opt(map)
    et2 = time.clock() - t0
    #
    print('   timings:\n   ... original ', et1)
    print('   ... optimized', et2)
    #
    drmat = np.absolute(rmat_2 - rmat_1)
    print('maximum difference between results')
    print(np.amax(drmat, 0))

    return


# FIXME: DeprecationWarning on time.clock
if __name__ == '__main__':
    #
    #  Simple tests.
    #
    #  1. Exponential map.
    #
    printTestName(1, 'rotMatOfExpMap')
    numPts = 10000
    testRotMatOfExpMap(numPts)
    #
    #  2.  Angular difference
    #
    num = 2
    name = 'angularDifference'
    printTestName(num, name)
    units = 'radians'
    numPts = 1000000
    a1 = 2*np.pi * np.random.rand(3, numPts) - np.pi
    a2 = 2*np.pi * np.random.rand(3, numPts) - np.pi
    print('* testing %s with %d random points' % (name, numPts))
    #
    t0 = time.clock()
    d1 = angularDifference_orig(a1, a2)
    et1 = time.clock() - t0
    #
    t0 = time.clock()
    d2 = angularDifference_opt(a1, a2)
    et2 = time.clock() - t0
    #
    print('   timings:\n   ... original ', et1)
    print('   ... optimized', et2)
    #
    dd = np.absolute(d2 - d1)
    print('maximum difference between results')
    print(np.max(dd, 0).max())

    pass
