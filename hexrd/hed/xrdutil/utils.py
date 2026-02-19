#! /usr/bin/env python3
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


# TODO: Resolve extra-workflow dependency
from hexrd.core.distortion.distortionabc import DistortionABC

from typing import Optional

import numpy as np

from hexrd.core import constants
from hexrd.core.material.crystallography import processWavelength, PlaneData
from hexrd.core.transforms import xfcapi


simlp = 'hexrd.core.instrument.hedm_instrument.HEDMInstrument.simulate_laue_pattern'

# =============================================================================
# PARAMETERS
# =============================================================================

distortion_key = 'distortion'

d2r = piby180 = constants.d2r
r2d = constants.r2d

epsf = constants.epsf  # ~2.2e-16
ten_epsf = 10 * epsf  # ~2.2e-15
sqrt_epsf = constants.sqrt_epsf  # ~1.5e-8

bHat_l_DFLT = constants.beam_vec.flatten()
eHat_l_DFLT = constants.eta_vec.flatten()

nans_1x2 = np.nan * np.ones((1, 2))

# =============================================================================
# FUNCTIONS
# =============================================================================

validateAngleRanges = xfcapi.validate_angle_ranges


def _project_on_detector_plane(
    allAngs: np.ndarray,
    rMat_d: np.ndarray,
    rMat_c: np.ndarray,
    chi: float,
    tVec_d: np.ndarray,
    tVec_c: np.ndarray,
    tVec_s: np.ndarray,
    distortion: Optional[DistortionABC] = None,
    beamVec: np.ndarray = constants.beam_vec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    utility routine for projecting a list of (tth, eta, ome) onto the
    detector plane parameterized by the args
    """
    gVec_cs = xfcapi.angles_to_gvec(allAngs, chi=chi, rmat_c=rMat_c, beam_vec=beamVec)

    rMat_ss = xfcapi.make_sample_rmat(chi, allAngs[:, 2])

    tmp_xys = xfcapi.gvec_to_xy(
        gVec_cs,
        rMat_d,
        rMat_ss,
        rMat_c,
        tVec_d,
        tVec_s,
        tVec_c,
        beam_vec=beamVec,
    )

    valid_mask = ~(np.isnan(tmp_xys[:, 0]) | np.isnan(tmp_xys[:, 1]))

    det_xy = np.atleast_2d(tmp_xys[valid_mask, :])

    # apply distortion if specified
    if distortion is not None:
        det_xy = distortion.apply_inverse(det_xy)

    return det_xy, rMat_ss, valid_mask


def _project_on_detector_cylinder(
    allAngs: np.ndarray,
    chi: float,
    tVec_d: np.ndarray,
    caxis: np.ndarray,
    paxis: np.ndarray,
    radius: float,
    physical_size: np.ndarray,
    angle_extent: float,
    distortion: Optional[DistortionABC] = None,
    beamVec: np.ndarray = constants.beam_vec,
    etaVec: np.ndarray = constants.eta_vec,
    tVec_s: np.ndarray = constants.zeros_3x1,
    rmat_s: np.ndarray = constants.identity_3x3,
    tVec_c: np.ndarray = constants.zeros_3x1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    utility routine for projecting a list of (tth, eta, ome) onto the
    detector plane parameterized by the args. this function does the
    computation for a cylindrical detector
    """
    dVec_cs = xfcapi.angles_to_dvec(
        allAngs, chi=chi, rmat_c=np.eye(3), beam_vec=beamVec, eta_vec=etaVec
    )

    rMat_ss = np.tile(rmat_s, [allAngs.shape[0], 1, 1])

    tmp_xys, valid_mask = _dvecToDetectorXYcylinder(
        dVec_cs,
        tVec_d,
        caxis,
        paxis,
        radius,
        physical_size,
        angle_extent,
        tVec_s=tVec_s,
        rmat_s=rmat_s,
        tVec_c=tVec_c,
    )

    det_xy = np.atleast_2d(tmp_xys[valid_mask, :])

    # apply distortion if specified
    if distortion is not None:
        det_xy = distortion.apply_inverse(det_xy)

    return det_xy, rMat_ss, valid_mask


def _unitvec_to_cylinder(
    uvw: np.ndarray,
    caxis: np.ndarray,
    paxis: np.ndarray,
    radius: float,
    tvec: np.ndarray,
    tVec_s: np.ndarray = constants.zeros_3x1,
    tVec_c: np.ndarray = constants.zeros_3x1,
    rmat_s: np.ndarray = constants.identity_3x3,
) -> np.ndarray:
    """
    get point where unitvector uvw
    intersect the cylindrical detector.
    this will give points which are
    outside the actual panel. the points
    will be clipped to the panel later

    Parameters
    ----------
    uvw : numpy.ndarray
    unit vectors stacked row wise (nx3) shape

    Returns
    -------
    numpy.ndarray
    (x,y,z) vectors point which intersect with
    the cylinder with (nx3) shape
    """
    naxis = np.cross(caxis, paxis)
    naxis = naxis / np.linalg.norm(naxis)

    tvec_c_l = np.dot(rmat_s, tVec_c)

    delta = tvec - (radius * naxis + np.squeeze(tVec_s) + np.squeeze(tvec_c_l))
    num = uvw.shape[0]
    cx = np.atleast_2d(caxis).T

    delta_t = np.tile(delta, [num, 1])

    t1 = np.dot(uvw, delta.T)
    t2 = np.squeeze(np.dot(uvw, cx))
    t3 = np.squeeze(np.dot(delta, cx))
    t4 = np.dot(uvw, cx)

    A = np.squeeze(1 - t4**2)
    B = t1 - t2 * t3
    C = radius**2 - np.linalg.norm(delta) ** 2 + t3**2

    mask = np.abs(A) < 1e-10
    beta = np.zeros(
        [
            num,
        ]
    )

    beta[~mask] = (B[~mask] + np.sqrt(B[~mask] ** 2 + A[~mask] * C)) / A[~mask]

    beta[mask] = np.nan
    return np.tile(beta, [3, 1]).T * uvw


def _clip_to_cylindrical_detector(
    uvw: np.ndarray,
    tVec_d: np.ndarray,
    caxis: np.ndarray,
    paxis: np.ndarray,
    radius: float,
    physical_size: np.ndarray,
    angle_extent: float,
    tVec_s: np.ndarray = constants.zeros_3x1,
    tVec_c: np.ndarray = constants.zeros_3x1,
    rmat_s: np.ndarray = constants.identity_3x3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    takes in the intersection points uvw
    with the cylindrical detector and
    prunes out points which don't actually
    hit the actual panel

    Parameters
    ----------
    uvw : numpy.ndarray
    unit vectors stacked row wise (nx3) shape

    Returns
    -------
    numpy.ndarray
    (x,y,z) vectors point which fall on panel
    with (mx3) shape
    """
    # first get rid of points which are above
    # or below the detector
    naxis = np.cross(caxis, paxis)
    num = uvw.shape[0]

    cx = np.atleast_2d(caxis).T
    nx = np.atleast_2d(naxis).T

    tvec_c_l = np.dot(rmat_s, tVec_c)

    delta = tVec_d - (radius * naxis + np.squeeze(tVec_s) + np.squeeze(tvec_c_l))

    delta_t = np.tile(delta, [num, 1])

    uvwp = uvw - delta_t
    dp = np.dot(uvwp, cx)

    uvwpxy = uvwp - np.tile(dp, [1, 3]) * np.tile(cx, [1, num]).T

    size = physical_size
    tvec = np.atleast_2d(tVec_d).T

    # ycomp = uvwp - np.tile(tVec_d,[num, 1])
    mask1 = np.squeeze(np.abs(dp) > size[0] * 0.5)
    uvwp[mask1, :] = np.nan

    # next get rid of points that fall outside
    # the polar angle range

    ang = np.dot(uvwpxy, nx) / radius
    ang[np.abs(ang) > 1.0] = np.sign(ang[np.abs(ang) > 1.0])

    ang = np.arccos(ang)
    mask2 = np.squeeze(ang >= angle_extent)
    mask = np.logical_or(mask1, mask2)
    res = uvw.copy()
    res[mask, :] = np.nan

    return res, ~mask


def _dewarp_from_cylinder(
    uvw: np.ndarray,
    tVec_d: np.ndarray,
    caxis: np.ndarray,
    paxis: np.ndarray,
    radius: float,
    tVec_s: np.ndarray = constants.zeros_3x1,
    tVec_c: np.ndarray = constants.zeros_3x1,
    rmat_s: np.ndarray = constants.identity_3x3,
):
    """
    routine to convert cylindrical coordinates
    to cartesian coordinates in image frame
    """
    naxis = np.cross(caxis, paxis)
    naxis = naxis / np.linalg.norm(naxis)

    cx = np.atleast_2d(caxis).T
    px = np.atleast_2d(paxis).T
    nx = np.atleast_2d(naxis).T
    num = uvw.shape[0]

    tvec_c_l = np.dot(rmat_s, tVec_c)

    delta = tVec_d - (radius * naxis + np.squeeze(tVec_s) + np.squeeze(tvec_c_l))

    delta_t = np.tile(delta, [num, 1])

    uvwp = uvw - delta_t

    uvwpxy = uvwp - np.tile(np.dot(uvwp, cx), [1, 3]) * np.tile(cx, [1, num]).T

    sgn = np.sign(np.dot(uvwpxy, px))
    sgn[sgn == 0.0] = 1.0
    ang = np.dot(uvwpxy, nx) / radius
    ang[np.abs(ang) > 1.0] = np.sign(ang[np.abs(ang) > 1.0])
    ang = np.arccos(ang)
    xcrd = np.squeeze(radius * ang * sgn)
    ycrd = np.squeeze(np.dot(uvwp, cx))
    return np.vstack((xcrd, ycrd)).T


def _warp_to_cylinder(
    cart: np.ndarray,
    tVec_d: np.ndarray,
    radius: float,
    caxis: np.ndarray,
    paxis: np.ndarray,
    tVec_s: np.ndarray = constants.zeros_3x1,
    rmat_s: np.ndarray = constants.identity_3x3,
    tVec_c: np.ndarray = constants.zeros_3x1,
    normalize: bool = True,
) -> np.ndarray:
    """
    routine to convert cartesian coordinates
    in image frame to cylindrical coordinates
    """
    tvec = np.atleast_2d(tVec_d).T
    if tVec_s.ndim == 1:
        tVec_s = np.atleast_2d(tVec_s).T
    if tVec_c.ndim == 1:
        tVec_c = np.atleast_2d(tVec_c).T
    num = cart.shape[0]
    naxis = np.cross(paxis, caxis)
    x = cart[:, 0]
    y = cart[:, 1]
    th = x / radius
    xp = radius * np.sin(th)
    xn = radius * (1 - np.cos(th))

    ccomp = np.tile(y, [3, 1]).T * np.tile(caxis, [num, 1])
    pcomp = np.tile(xp, [3, 1]).T * np.tile(paxis, [num, 1])
    ncomp = np.tile(xn, [3, 1]).T * np.tile(naxis, [num, 1])
    cart3d = pcomp + ccomp + ncomp

    tVec_c_l = np.dot(rmat_s, tVec_c)

    res = cart3d + np.tile(tvec - tVec_s - tVec_c_l, [1, num]).T

    if normalize:
        return res / np.tile(np.linalg.norm(res, axis=1), [3, 1]).T
    else:
        return res


def _dvecToDetectorXYcylinder(
    dVec_cs: np.ndarray,
    tVec_d: np.ndarray,
    caxis: np.ndarray,
    paxis: np.ndarray,
    radius: float,
    physical_size: np.ndarray,
    angle_extent: float,
    tVec_s: np.ndarray = constants.zeros_3x1,
    tVec_c: np.ndarray = constants.zeros_3x1,
    rmat_s: np.ndarray = constants.identity_3x3,
) -> tuple[np.ndarray, np.ndarray]:

    cvec = _unitvec_to_cylinder(
        dVec_cs,
        caxis,
        paxis,
        radius,
        tVec_d,
        tVec_s=tVec_s,
        tVec_c=tVec_c,
        rmat_s=rmat_s,
    )

    cvec_det, valid_mask = _clip_to_cylindrical_detector(
        cvec,
        tVec_d,
        caxis,
        paxis,
        radius,
        physical_size,
        angle_extent,
        tVec_s=tVec_s,
        tVec_c=tVec_c,
        rmat_s=rmat_s,
    )

    xy_det = _dewarp_from_cylinder(
        cvec_det,
        tVec_d,
        caxis,
        paxis,
        radius,
        tVec_s=tVec_s,
        tVec_c=tVec_c,
        rmat_s=rmat_s,
    )

    return xy_det, valid_mask
