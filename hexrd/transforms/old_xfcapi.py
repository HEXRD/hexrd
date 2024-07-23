#! /usr/bin/env python
# ============================================================
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
# ============================================================

import numpy as np
import sys

from hexrd.extensions import _transforms_CAPI

from numpy import float_ as nFloat
from numpy import int_ as nInt

# ######################################################################
# Module Data
epsf = np.finfo(float).eps      # ~2.2e-16
ten_epsf = 10 * epsf                # ~2.2e-15
sqrt_epsf = np.sqrt(epsf)            # ~1.5e-8

periodDict = {'degrees': 360.0, 'radians': 2*np.pi}
angularUnits = 'radians'        # module-level angle units

# basis vectors
I3 = np.eye(3)                                        # (3, 3) identity
Xl = np.ascontiguousarray(I3[:, 0].reshape(3, 1))     # X in the lab frame
Yl = np.ascontiguousarray(I3[:, 1].reshape(3, 1))     # Y in the lab frame
Zl = np.ascontiguousarray(I3[:, 2].reshape(3, 1))     # Z in the lab frame

# reference stretch
vInv_ref = np.array([[1., 1., 1., 0., 0., 0.]], order='C').T

# reference beam direction and eta=0 ref in LAB FRAME for standard geometry
bVec_ref = -Zl
eta_ref = Xl

# ######################################################################
# Funtions


def anglesToGVec(angs, bHat_l=bVec_ref, eHat_l=eta_ref, chi=0., rMat_c=I3):
    """
    from 'eta' frame out to lab (with handy kwargs to go to crystal or sample)

    * setting omega to zero in ang imput and omitting rMat_c leaves
      in the lab frame in accordance with beam frame specs.
    """
    angs = np.ascontiguousarray(np.atleast_2d(angs))
    bHat_l = np.ascontiguousarray(bHat_l.flatten())
    eHat_l = np.ascontiguousarray(eHat_l.flatten())
    rMat_c = np.ascontiguousarray(rMat_c)
    chi = float(chi)
    return _transforms_CAPI.anglesToGVec(angs,
                                         bHat_l, eHat_l,
                                         chi, rMat_c)


def anglesToDVec(angs, bHat_l=bVec_ref, eHat_l=eta_ref, chi=0., rMat_c=I3):
    """
    from 'eta' frame out to lab (with handy kwargs to go to crystal or sample)

    * setting omega to zero in ang imput and omitting rMat_c leaves
      in the lab frame in accordance with beam frame specs.
    """
    angs = np.ascontiguousarray(np.atleast_2d(angs))
    bHat_l = np.ascontiguousarray(bHat_l.flatten())
    eHat_l = np.ascontiguousarray(eHat_l.flatten())
    rMat_c = np.ascontiguousarray(rMat_c)
    chi = float(chi)
    return _transforms_CAPI.anglesToDVec(angs,
                                         bHat_l, eHat_l,
                                         chi, rMat_c)


def makeGVector(hkl, bMat):
    """
    take a CRYSTAL RELATIVE B matrix onto a list of hkls to output unit
    reciprocal lattice vectors (a.k.a. lattice plane normals)

    Required Arguments:
    hkls -- (3, n) ndarray of n hstacked reciprocal lattice vector component
            triplets
    bMat -- (3, 3) ndarray representing the matirix taking reciprocal lattice
            vectors to the crystal reference frame

    Output:
    gVecs -- (3, n) ndarray of n unit reciprocal lattice vectors
             (a.k.a. lattice plane normals)

    To Do:
    * might benefit from some assert statements to catch improperly shaped
      input.
    """
    assert hkl.shape[0] == 3, 'hkl input must be (3, n)'
    return unitRowVector(np.dot(bMat, hkl))


def gvecToDetectorXY(gVec_c,
                     rMat_d, rMat_s, rMat_c,
                     tVec_d, tVec_s, tVec_c,
                     beamVec=bVec_ref):
    """
    Takes a list of unit reciprocal lattice vectors in crystal frame to the
    specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Required Arguments:
    gVec_c -- (n, 3) ndarray of n reciprocal lattice vector components
              in the CRYSTAL FRAME
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components
              to LAB FRAME
    rMat_s -- (3, 3) ndarray, the COB taking SAMPLE FRAME components
              to LAB FRAME
    rMat_c -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components
              to SAMPLE FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting
              LAB to DETECTOR
    tVec_s -- (3, 1) ndarray, the translation vector connecting
              LAB to SAMPLE
    tVec_c -- (3, 1) ndarray, the translation vector connecting
              SAMPLE to CRYSTAL

    Outputs:
    (m, 2) ndarray containing the intersections of m <= n diffracted beams
    associated with gVecs
    """
    rMat_d = np.ascontiguousarray(rMat_d)
    rMat_s = np.ascontiguousarray(rMat_s)
    rMat_c = np.ascontiguousarray(rMat_c)
    gVec_c = np.ascontiguousarray(np.atleast_2d(gVec_c))
    tVec_d = np.ascontiguousarray(tVec_d.flatten())
    tVec_s = np.ascontiguousarray(tVec_s.flatten())
    tVec_c = np.ascontiguousarray(tVec_c.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    return _transforms_CAPI.gvecToDetectorXY(gVec_c,
                                             rMat_d, rMat_s, rMat_c,
                                             tVec_d, tVec_s, tVec_c,
                                             beamVec)


def gvecToDetectorXYArray(gVec_c,
                          rMat_d, rMat_s, rMat_c,
                          tVec_d, tVec_s, tVec_c,
                          beamVec=bVec_ref):
    """
    Takes a list of unit reciprocal lattice vectors in crystal frame to the
    specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Required Arguments:
    gVec_c -- (n, 3) ndarray of n reciprocal lattice vector components
              in the CRYSTAL FRAME
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components
              to LAB FRAME
    rMat_s -- (n, 3, 3) ndarray of n COB taking SAMPLE FRAME components
              to LAB FRAME
    rMat_c -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components
              to SAMPLE FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting
              LAB to DETECTOR in LAB
    tVec_s -- (3, 1) ndarray, the translation vector connecting
              LAB to SAMPLE in LAB
    tVec_c -- (3, 1) ndarray, the translation vector connecting
              SAMPLE to CRYSTAL in SAMPLE

    Outputs:
    (m, 2) ndarray containing the intersections of m <= n diffracted beams
    associated with gVecs
    """
    gVec_c = np.ascontiguousarray(gVec_c)
    rMat_d = np.ascontiguousarray(rMat_d)
    rMat_s = np.ascontiguousarray(rMat_s)
    rMat_c = np.ascontiguousarray(rMat_c)
    tVec_d = np.ascontiguousarray(tVec_d.flatten())
    tVec_s = np.ascontiguousarray(tVec_s.flatten())
    tVec_c = np.ascontiguousarray(tVec_c.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    return _transforms_CAPI.gvecToDetectorXYArray(gVec_c,
                                                  rMat_d, rMat_s, rMat_c,
                                                  tVec_d, tVec_s, tVec_c,
                                                  beamVec)


def detectorXYToGvec(xy_det,
                     rMat_d, rMat_s,
                     tVec_d, tVec_s, tVec_c,
                     beamVec=bVec_ref, etaVec=eta_ref):
    """
    Takes a list cartesian (x, y) pairs in the detector coordinates and
    calculates the associated reciprocal lattice (G) vectors and
    (bragg angle, azimuth) pairs with respect to the specified beam and
    azimth (eta) reference directions

    Required Arguments:
    xy_det -- (n, 2) ndarray or list-like input of n detector (x, y) points
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components
              to LAB FRAME
    rMat_s -- (3, 3) ndarray, the COB taking SAMPLE FRAME components
              to LAB FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting
              LAB to DETECTOR in LAB
    tVec_s -- (3, 1) ndarray, the translation vector connecting
              LAB to SAMPLE in LAB
    tVec_c -- (3, 1) ndarray, the translation vector connecting
              SAMPLE to CRYSTAL in SAMPLE

    Optional Keyword Arguments:
    beamVec -- (3, 1) mdarray containing the incident beam direction
               components in the LAB FRAME
    etaVec  -- (3, 1) mdarray containing the reference azimuth direction
               components in the LAB FRAME

    Outputs:
    (n, 2) ndarray containing the (tTh, eta) pairs associated with each (x, y)
    (n, 3) ndarray containing the associated G vector directions
    in the LAB FRAME associated with gVecs
    """
    xy_det = np.ascontiguousarray(np.atleast_2d(xy_det))
    rMat_d = np.ascontiguousarray(rMat_d)
    rMat_s = np.ascontiguousarray(rMat_s)
    tVec_d = np.ascontiguousarray(tVec_d.flatten())
    tVec_s = np.ascontiguousarray(tVec_s.flatten())
    tVec_c = np.ascontiguousarray(tVec_c.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    etaVec = np.ascontiguousarray(etaVec.flatten())
    return _transforms_CAPI.detectorXYToGvec(xy_det,
                                             rMat_d, rMat_s,
                                             tVec_d, tVec_s, tVec_c,
                                             beamVec, etaVec)


def detectorXYToGvecArray(xy_det,
                          rMat_d, rMat_s,
                          tVec_d, tVec_s, tVec_c,
                          beamVec=bVec_ref, etaVec=eta_ref):
    """
    Takes a list cartesian (x, y) pairs in the detector coordinates and
    calculates the associated reciprocal lattice (G) vectors and
    (bragg angle, azimuth) pairs with respect to the specified beam and azimth
    (eta) reference directions

    Required Arguments:
    xy_det -- (n, 2) ndarray or list-like input of n detector (x, y) points
    rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components
              to LAB FRAME
    rMat_s -- (n, 3, 3) ndarray, the COB taking SAMPLE FRAME components
              to LAB FRAME
    tVec_d -- (3, 1) ndarray, the translation vector connecting
              LAB to DETECTOR in LAB
    tVec_s -- (3, 1) ndarray, the translation vector connecting
              LAB to SAMPLE in LAB
    tVec_c -- (3, 1) ndarray, the translation vector connecting
              SAMPLE to CRYSTAL in SAMPLE

    Optional Keyword Arguments:
    beamVec -- (3, 1) mdarray containing the incident beam direction
               components in the LAB FRAME
    etaVec  -- (3, 1) mdarray containing the reference azimuth direction
               components in the LAB FRAME

    Outputs:
    (n, 2) ndarray containing the (tTh, eta) pairs associated with each (x, y)
    (n, 3) ndarray containing the associated G vector directions
    in the LAB FRAME associated with gVecs
    """
    xy_det = np.ascontiguousarray(np.atleast_2d(xy_det))
    rMat_d = np.ascontiguousarray(rMat_d)
    rMat_s = np.ascontiguousarray(rMat_s)
    tVec_d = np.ascontiguousarray(tVec_d.flatten())
    tVec_s = np.ascontiguousarray(tVec_s.flatten())
    tVec_c = np.ascontiguousarray(tVec_c.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    etaVec = np.ascontiguousarray(etaVec.flatten())
    return _transforms_CAPI.detectorXYToGvec(xy_det,
                                             rMat_d, rMat_s,
                                             tVec_d, tVec_s, tVec_c,
                                             beamVec, etaVec)


def oscillAnglesOfHKLs(hkls, chi, rMat_c, bMat, wavelength,
                       vInv=None, beamVec=bVec_ref, etaVec=eta_ref):
    """
    Takes a list of unit reciprocal lattice vectors in crystal frame to the
    specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Required Arguments:
    hkls       -- (n, 3) ndarray of n reciprocal lattice vectors
                  in the CRYSTAL FRAME
    chi        -- float representing the inclination angle of the
                  oscillation axis (std coords)
    rMat_c     -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components
                  to SAMPLE FRAME
    bMat       -- (3, 3) ndarray, the COB taking RECIPROCAL LATTICE components
                  to CRYSTAL FRAME
    wavelength -- float representing the x-ray wavelength in Angstroms

    Optional Keyword Arguments:
    beamVec -- (3, 1) mdarray containing the incident beam direction
               components in the LAB FRAME
    etaVec  -- (3, 1) mdarray containing the reference azimuth direction
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
    hkls = np.array(hkls, dtype=float, order='C')
    if vInv is None:
        vInv = np.ascontiguousarray(vInv_ref.flatten())
    else:
        vInv = np.ascontiguousarray(vInv.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    etaVec = np.ascontiguousarray(etaVec.flatten())
    bMat = np.ascontiguousarray(bMat)
    return _transforms_CAPI.oscillAnglesOfHKLs(
        hkls, chi, rMat_c, bMat, wavelength, vInv, beamVec, etaVec
    )


"""
#######################################################################
######                  Utility Functions                        ######
#######################################################################

"""


def arccosSafe(temp):
    """
    Protect against numbers slightly larger than 1 due to round-off
    """

    # Oh, the tricks we must play to make this overloaded and robust...
    if type(temp) is list:
        temp = np.asarray(temp)
    elif type(temp) is np.ndarray:
        if len(temp.shape) == 0:
            temp = temp.reshape(1)

    if (temp > 1.00001).any():
        print("attempt to take arccos of %s" % temp, file=sys.stderr)
        raise RuntimeError("unrecoverable error")
    elif (temp < -1.00001).any():
        print("attempt to take arccos of %s" % temp, file=sys.stderr)
        raise RuntimeError("unrecoverable error")

    gte1 = temp >= 1.
    lte1 = temp <= -1.

    temp[gte1] = 1
    temp[lte1] = -1

    ang = np.arccos(temp)

    return ang


def angularDifference(angList0, angList1, units=angularUnits):
    """
    Do the proper (acute) angular difference in the context of a branch cut.

    *) Default angular range is [-pi, pi]
    """
    period = periodDict[units]
    # take difference as arrays
    diffAngles = np.atleast_1d(angList0) - np.atleast_1d(angList1)

    return abs(np.remainder(diffAngles + 0.5*period, period) - 0.5*period)


def mapAngle(ang, *args, **kwargs):
    """
    Utility routine to map an angle into a specified period
    """
    units = angularUnits
    period = periodDict[units]

    kwargKeys = list(kwargs.keys())
    for iArg in range(len(kwargKeys)):
        if kwargKeys[iArg] == 'units':
            units = kwargs[kwargKeys[iArg]]
        else:
            raise RuntimeError(
                "Unknown keyword argument: " + str(kwargKeys[iArg]))

    try:
        period = periodDict[units.lower()]
    except(KeyError):
        raise RuntimeError("unknown angular units: " +
                           str(kwargs[kwargKeys[iArg]]))

    ang = np.atleast_1d(nFloat(ang))

    # if we have a specified angular range, use that
    if len(args) > 0:
        angRange = np.atleast_1d(nFloat(args[0]))

        # divide of multiples of period
        ang = ang - nInt(ang / period) * period

        lb = angRange.min()
        ub = angRange.max()

        if abs(abs(ub - lb) - period) > sqrt_epsf:
            raise RuntimeError('range is incomplete!')

        lbi = ang < lb
        while lbi.sum() > 0:
            ang[lbi] = ang[lbi] + period
            lbi = ang < lb
        ubi = ang > ub
        while ubi.sum() > 0:
            ang[ubi] = ang[ubi] - period
            ubi = ang > ub
        retval = ang
    else:
        retval = np.mod(ang + 0.5*period, period) - 0.5*period
    return retval


def columnNorm(a):
    """
    normalize array of column vectors (hstacked, axis = 0)
    """
    if len(a.shape) > 2:
        raise RuntimeError(
            "incorrect shape: arg must be 1-d or 2-d, yours is %d"
            % (len(a.shape))
        )

    cnrma = np.sqrt(np.sum(np.asarray(a)**2, 0))

    return cnrma


def rowNorm(a):
    """
    normalize array of row vectors (vstacked, axis = 1)
    """
    if len(a.shape) > 2:
        raise RuntimeError(
            "incorrect shape: arg must be 1-d or 2-d, yours is %d"
            % (len(a.shape))
        )

    cnrma = np.sqrt(np.sum(np.asarray(a)**2, 1))

    return cnrma


def unitRowVector(vecIn):
    vecIn = np.ascontiguousarray(vecIn)
    if vecIn.ndim == 1:
        return _transforms_CAPI.unitRowVector(vecIn)
    elif vecIn.ndim == 2:
        return _transforms_CAPI.unitRowVectors(vecIn)
    else:
        assert vecIn.ndim in [1, 2], \
            "arg shape must be 1-d or 2-d, yours is %d-d" % (vecIn.ndim)


def makeDetectorRotMat(tiltAngles):
    """
    Form the (3, 3) tilt rotations from the tilt angle list:

    tiltAngles = [gamma_Xl, gamma_Yl, gamma_Zl] in radians
    """
    arg = np.ascontiguousarray(np.r_[tiltAngles].flatten())
    return _transforms_CAPI.makeDetectorRotMat(arg)


def makeOscillRotMat(oscillAngles):
    """
    oscillAngles = [chi, ome]
    """
    arg = np.ascontiguousarray(np.r_[oscillAngles].flatten())
    return _transforms_CAPI.makeOscillRotMat(arg)


def makeOscillRotMatArray(chi, omeArray):
    """
    Applies makeOscillAngles multiple times, for one
    chi value and an array of omega values.
    """
    arg = np.ascontiguousarray(omeArray)
    return _transforms_CAPI.makeOscillRotMatArray(chi, arg)


def makeRotMatOfExpMap(expMap):
    """
    make a rotation matrix from an exponential map
    """
    arg = np.ascontiguousarray(expMap.flatten())
    return _transforms_CAPI.makeRotMatOfExpMap(arg)


def makeRotMatOfQuat(quats):
    """
    make rotation matrix from vstacked unit quaternions

    """
    arg = np.ascontiguousarray(quats)
    return _transforms_CAPI.makeRotMatOfQuat(arg)


def makeBinaryRotMat(axis):
    arg = np.ascontiguousarray(axis.flatten())
    return _transforms_CAPI.makeBinaryRotMat(arg)


def makeEtaFrameRotMat(bHat_l, eHat_l):
    arg1 = np.ascontiguousarray(bHat_l.flatten())
    arg2 = np.ascontiguousarray(eHat_l.flatten())
    return _transforms_CAPI.makeEtaFrameRotMat(arg1, arg2)


def validateAngleRanges(angList, angMin, angMax, ccw=True):
    # FIXME: broken
    angList = np.asarray(angList, dtype=np.double, order="C")
    angMin = np.asarray(angMin, dtype=np.double, order="C")
    angMax = np.asarray(angMax, dtype=np.double, order="C")
    return _transforms_CAPI.validateAngleRanges(angList, angMin, angMax, ccw)


def rotate_vecs_about_axis(angle, axis, vecs):
    return _transforms_CAPI.rotate_vecs_about_axis(angle, axis, vecs)


def quat_distance(q1, q2, qsym):
    """
    qsym coming from hexrd.crystallogray.PlaneData.getQSym() is C-contiguous
    """
    q1 = np.ascontiguousarray(q1.flatten())
    q2 = np.ascontiguousarray(q2.flatten())
    return _transforms_CAPI.quat_distance(q1, q2, qsym)


def homochoricOfQuat(quats):
    """
    Compute homochoric parameters of unit quaternions

    quats is (4, n)
    """
    q = np.ascontiguousarray(quats.T)
    return _transforms_CAPI.homochoricOfQuat(q)

# def rotateVecsAboutAxis(angle, axis, vecs):
#     return _transforms_CAPI.rotateVecsAboutAxis(angle, axis, vecs)
