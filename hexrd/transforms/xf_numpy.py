#! /usr/bin/env python
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

# ??? do we want to set np.seterr(invalid='ignore') to avoid nan warnings?
from __future__ import absolute_import, print_function, division

import numpy as np
from numpy import float_ as npfloat
from numpy import int_ as npint

from . import constants as cnst
from .transforms_definitions import xf_api

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# ???: quicker to use conditionals below or just put in the identity matrix
#      and multiply through?
def _beam_to_crystal(vecs, rmat_b=None, rmat_s=None, rmat_c=None):
    """
    Helper function to take vectors defined in the BEAM frame through LAB
    to either SAMPLE or CRYSTAL

    """
    orig_dims = vecs.ndim
    vecs = np.atleast_2d(vecs)
    nvecs = len(vecs)
    if rmat_s is not None:
        rmat_s = np.squeeze(rmat_s)
        if rmat_s.ndim == 3:
            # !!!: this implies individual sample frames for each vector
            assert len(rmat_s) == nvecs, \
                "if specifying an array of rmat_s, dimensions must be " + \
                "(%d, 3, 3), not (%d, %d, %d)" \
                % tuple([nvecs] + list(rmat_s.shape))

    # take to lab frame (row order)
    # !!! rmat_b is defined as taking components from BEAM to LAB
    if rmat_b is not None:
        vecs = np.dot(vecs, rmat_b.T)

    # to go to CRYSTAL in column vec order (hstacked gvec_l):
    #
    # gvec_c = np.dot(rmat_c.T, np.dot(rmat_s.T, np.dot(rmat_b, gvec_b)))
    #
    # rmat_s = np.dot(rchi, rome)
    #
    # --> in row vec order (vstacked gvec_l, C order):
    #
    # gvec_l = np.dot(gvec_b, np.dot(rmat_b.T, np.dot(rmat_s, rmat_c)))
    if rmat_s is not None:
        if rmat_s.ndim > 2:
            for i in range(nvecs):
                vecs[i] = np.dot(vecs[i], rmat_s[i])
        else:
            vecs = np.dot(vecs, rmat_s)
    if rmat_c is not None:
        return np.dot(vecs, rmat_c)

    return vecs[0] if orig_dims == 1 else vecs


def _crystal_to_lab(gvecs,
                    rmat_s, rmat_c,
                    bmat=None, vmat_inv=None):
    """
    gvecs is (n, 3)

    rmat_s are either (3, 3) or (n, 3, 3)

    if bmat is not None, gvecs are assumed to be hkls
    Takes a list of reciprocal lattice vectors components in crystal frame to
    the specified detector-relative frame, subject to the conditions:

    1) the reciprocal lattice vector must be able to satisfy a bragg condition
    2) the associated diffracted beam must intersect the detector plane

    Parameters
    ----------
    gvecs : array_like
        Concatenated triplets of G-vector components in either the
        CRYSTAL FRAME or RECIPROCAL FRAME (see optional kwarg `bmat` below).
        The shape when cast as an ndarray is (n, 3), representing n vectors.
    rmat_s : array_like
        The COB matrix taking components in the SAMPLE FRAME to the LAB FRAME.
        This can be either (3, 3) or (n, 3, 3). In the latter case, each of the
        n input G-vectors is transformed using the associated entry in
        `rmat_s`.
    rmat_c : array_like
        The (3, 3) COB matrix taking components in the
        CRYSTAL FRAME to the SAMPLE FRAME.
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

    Returns
    -------
    array_like
        The (n, 3) array of G-vectors components in the LAB FRAME as specified
        by `rmat_s` and `rmat_c`.  Note that resulting vector components are
        not normalized.

    Raises
    ------
    AssertionError
        If `rmat_s` has dimension 3, but the first is != n.

    Notes
    -----

    To go to the LAB FRAME from the CRYSTAL FRAME in column vec order (hstacked
    gvec_c):

        gvec_l = np.dot(np.dot(rmat_c.T, np.dot(rmat_s.T, rmat_b)), gvec_b)

     rmat_s = np.dot(rchi, rome)

     --> in row vec order (vstacked gvec_l):

     gvec_l = np.dot(gvec_b, np.dot(rmat_b.T, np.dot(rmat_s, rmat_c)))

    """

    # catch 1-d input and grab number of input vectors
    gvec_c = np.atleast_2d(gvecs)
    nvecs = len(gvecs)

    # initialize transformed gvec arrays
    gvec_s = np.empty_like(gvecs)
    gvec_l = np.empty_like(gvecs)

    # squash out the case where rmat_s.shape is (1, 3, 3)
    rmat_s = np.squeeze(rmat_s)

    # if bmat is specified, input are components in reiprocal lattice (h, k, l)
    if bmat is not None:
        gvec_c = np.dot(gvec_c, bmat.T)

    # CRYSTAL FRAME --> SAMPLE FRAME
    gvec_s = np.dot(gvec_c, rmat_c.T)
    if vmat_inv is not None:
        gvec_s = np.dot(gvec_s, vmat_inv.T)

    # SAMPLE FRAME --> LAB FRAME
    if rmat_s.ndim > 2:
        # individual rmat_s for each vector
        assert len(rmat_s) == nvecs, \
            "len(rmat_s) must be %d for 3-d arg; you gave %d" \
            % (nvecs, len(rmat_s))
        for i in range(nvecs):
            gvec_l[i] = np.dot(gvec_s[i], rmat_s[i].T)
    else:
        # single rmat_s
        gvec_l = np.dot(gvec_s, rmat_s.T)
    return gvec_l


def _rmat_s_helper(ome, chi=None):
    """
    simple utility to avoid multiplying by identity for chi=0 when
    calculating sample rotation matrices
    """
    if chi is None:
        return np.array([make_rmat_of_expmap(i*cnst.lab_y) for i in ome])
    else:
        return make_sample_rmat(chi, ome)


def _z_project(x, y):
    return np.cos(x) * np.sin(y) - np.sin(x) * np.cos(y)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================


@xf_api
def angles_to_gvec(
        angs,
        beam_vec=None, eta_vec=None,
        chi=None, rmat_c=None):

    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.eta_vec

    orig_ndim = angs.ndim
    angs = np.atleast_2d(angs)
    nvecs, dim = angs.shape

    # make vectors in BEAM FRAME
    gvec_b = np.vstack([
        [np.cos(0.5*angs[:, 0]) * np.cos(angs[:, 1])],
        [np.cos(0.5*angs[:, 0]) * np.sin(angs[:, 1])],
        [np.sin(0.5*angs[:, 0])]]).T

    # need rmat_b
    rmat_b = make_beam_rmat(beam_vec, eta_vec)

    # handle sample frame(s)
    rmat_s = None
    if dim > 2:
        rmat_s = _rmat_s_helper(angs[:, 2], chi=chi)
    result = _beam_to_crystal(gvec_b, rmat_b=rmat_b,
                              rmat_s=rmat_s, rmat_c=rmat_c)

    return result[0] if orig_ndim == 1 else result


@xf_api
def angles_to_dvec(
        angs,
        beam_vec=None, eta_vec=None,
        chi=None, rmat_c=None):

    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.eta_vec

    angs = np.atleast_2d(angs)
    nvecs, dim = angs.shape

    # make vectors in beam frame
    dvec_b = np.vstack([
        [np.sin(angs[:, 0]) * np.cos(angs[:, 1])],
        [np.sin(angs[:, 0]) * np.sin(angs[:, 1])],
        [-np.cos(angs[:, 0])]]).T

    # need rmat_b
    rmat_b = make_beam_rmat(beam_vec, eta_vec)

    # handle sample frame(s)
    rmat_s = None
    if dim > 2:
        rmat_s = _rmat_s_helper(angs[:, 2], chi=chi)
    return _beam_to_crystal(dvec_b, beam_vec=beam_vec, eta_vec=eta_vec,
                            rmat_s=rmat_s, rmat_c=rmat_c)


@xf_api
def gvec_to_xy(gvec_c,
               rmat_d, rmat_s, rmat_c,
               tvec_d, tvec_s, tvec_c,
               beam_vec=None,
               vmat_inv=None,
               bmat=None):

    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec

    ztol = cnst.epsf

    # catch 1-d input case and initialize return array with NaNs
    gvec_c = np.atleast_2d(gvec_c)
    retval = np.nan * np.ones_like(gvec_c)

    nvec_l = rmat_d[:, 2]  # detector normal (LAB FRAME)
    bhat_l = unit_vector(beam_vec.flatten())  # unit beam vector

    # need CRYSTAL frame origin.  If rmat_s is 3-d, this will be a list
    # !!!: use _crystal_to_lab helper with trivial rmat_c
    P0_l = _crystal_to_lab(tvec_c, rmat_s, np.eye(3))  # CRYSTAL FRAME origin
    P3_l = tvec_d  # DETECTOR FRAME origin

    # form unit reciprocal lattice vectors in lab frame (w/o translation)
    if bmat is None:
        # got hkls as input
        ghat_l = _crystal_to_lab(
            unit_vector(gvec_c), rmat_s, rmat_c,
            bmat=None, vmat_inv=vmat_inv
            )
    else:
        # got G-vectors in CRYSTAL FRAME as input
        ghat_l = unit_vector(
            _crystal_to_lab(
                gvec_c, rmat_s, rmat_c, bmat=bmat, vmat_inv=vmat_inv
                )
            )

    # dot with beam vector (upstream, cone axis)
    bdot = np.dot(ghat_l, -bhat_l)

    # see who can diffract; initialize output array with NaNs
    can_diffract = np.logical_and(bdot >= ztol, bdot <= 1. - ztol)
    if np.any(can_diffract):
        # subset of feasible reciprocal lattice vectors
        adm_ghat_l = np.atleast_2d(ghat_l[can_diffract, :])

        # initialize diffracted beam vector array
        dvec_l = np.empty_like(adm_ghat_l)
        for i, v in enumerate(adm_ghat_l):
            dvec_l[i] = np.dot(make_binary_rmat(v), -bhat_l)
            pass

        '''       displacement vector calculation below
        '''

        # first check for non-instersections and mitigate divide-by-zero
        # ???: better to use np.divide and feed NaNs through?
        denom = np.dot(dvec_l, nvec_l)
        dzero = abs(denom) < ztol
        denom[dzero] = 1.
        cant_intersect = denom > 0.  # index to dvec_l that can't hit det

        # displacement scaling (along dvec_l)
        u = np.dot(P3_l - P0_l, nvec_l) / denom

        # filter out non-intersections, fill with NaNs
        u[np.logical_or(dzero, cant_intersect)] = np.nan

        # diffracted beam points IN DETECTOR FRAME
        P2_l = P0_l + np.tile(u, (3, 1)).T * dvec_l
        P2_d = np.dot(P2_l - tvec_d, rmat_d)

        # put feasible transformed gvec intersections into return array
        retval[can_diffract, :] = P2_d
    return retval[:, :2]


@xf_api
def xy_to_gvec(xy_d,
               rmat_d, rmat_s,
               tvec_d, tvec_s, tvec_c,
               rmat_b=None,
               distortion=None,
               output_ref=False):

    # catch 1-d input and grab number of input vectors
    xy_d = np.atleast_2d(xy_d)
    npts = len(xy_d)

    # need beam vector
    bhat_l = cnst.beam_vec
    if rmat_b is not None:
        bhat_l = -rmat_b[:, 2]
    else:
        rmat_b = cnst.identity_3x3

    # if a distortion function is supplied, apply unwarping
    if distortion is not None:
        xy_d = distortion.unwarp(xy_d)

    # form in-plane vectors for detector points list in DETECTOR FRAME
    P2_d = np.hstack([xy_d, np.zeros((npts, 1))])

    # define points for ray calculations
    P2_l = np.dot(P2_d, rmat_d.T) + tvec_d  # inputs in LAB FRAME
    P0_l = np.dot(tvec_c, rmat_s.T) + tvec_s  # origin of CRYSTAL FRAME

    # diffraction unit vector components in LAB FRAME ans BEAM FRAME
    dhat_l = unit_vector(P2_l - P0_l)
    dhat_b = np.dot(dhat_l, rmat_b)

    # get bragg angle and azimuth of diffracted beam
    tth = np.arccos(np.dot(bhat_l.T, dhat_l)).flatten()
    eta = np.arctan2(dhat_b[1, :], dhat_b[0, :]).flatten()

    # get G-vectors by Laue condition
    ghat_l = unit_vector(dhat_l - bhat_l)

    if output_ref:
        # angles for reference frame
        dhat_ref_l = unit_vector(P2_l)
        dhat_ref_b = np.dot(dhat_ref_l, rmat_b)
        tth_ref = np.arccos(np.dot(bhat_l.T, unit_vector(P2_l))).flatten()
        eta_ref = np.arctan2(dhat_ref_b[1, :], dhat_ref_b[0, :]).flatten()
        return (tth, eta), ghat_l, (tth_ref, eta_ref)
    else:
        return (tth, eta), ghat_l


@xf_api
def solve_omega(gvecs, chi, rmat_c, wavelength,
                bmat=None, vmat_inv=None, rmat_b=None):
    gvecs = np.atleast_2d(gvecs)

    # sin and cos of the oscillation axis tilt
    cchi = np.cos(chi)
    schi = np.sin(chi)

    # transform input to sampe frame and normalize
    gvec_s = _crystal_to_lab(gvecs, cnst.identity_3x3, rmat_c,
                             bmat=bmat, vmat_inv=vmat_inv)
    ghat_s = unit_vector(gvec_s)
    one_by_dsp = row_norm(gvec_s)

    # sin of the Bragg angle using wavelength and d-spacings in Bragg's Law
    sintht = 0.5 * wavelength * one_by_dsp

    # calculate coefficients for harmonic equation
    # !!!: should all be 1-d
    if rmat_b is not None:
        '''       NON-STANDARD FRAME
        '''
        bhat_l = -rmat_b[:, 2]

        # coefficients for harmonic equation
        a = ghat_s[2, :]*bhat_l[0] \
            + schi*ghat_s[0, :]*bhat_l[1] \
            - cchi*ghat_s[0, :]*bhat_l[2]
        b = ghat_s[0, :]*bhat_l[0] \
            - schi*ghat_s[2, :]*bhat_l[1] \
            + cchi*ghat_s[2, :]*bhat_l[2]
        c = -sintht \
            - cchi*ghat_s[1, :]*bhat_l[1] \
            - schi*ghat_s[1, :]*bhat_l[2]
    else:
        '''       STANDARD FRAME; bhat_l = [0, 0, -1]
        '''
        a = cchi*ghat_s[0, :]
        b = -cchi*ghat_s[2, :]
        c = schi*ghat_s[1, :] - sintht

    # form solution
    ab_mag = np.sqrt(a*a + b*b)
    phase_ang = np.arctan2(b, a)
    rhs = c / ab_mag
    rhs[abs(rhs) > 1.] = np.nan
    rhs_ang = np.arcsin(rhs)  # will give NaN for abs(rhs) >  1. + 0.5*epsf

    # write ome angle output arrays (NaNs persist here)
    ome0 = rhs_ang - phase_ang
    ome1 = np.pi - rhs_ang - phase_ang

    # both solutions are invalid for the same inputs, so mark using ome0
    valid_solutions = ~np.isnan(ome0)

    # calculate etas
    if np.any(valid_solutions):
        # initialize eta arrays
        eta0 = np.nan * np.ones_like(ome0)
        eta1 = np.nan * np.ones_like(ome1)

        vs_idx_array = np.tile(valid_solutions, (1, 2)).flatten()

        num_valid = sum(valid_solutions)
        tmp_gvec = np.tile(ghat_s, (1, 2))[:, vs_idx_array]
        all_ome = np.hstack([ome0, ome1])

        # calculate the SAMPLE FRAME COB matrices for each omega
        rmat_s = make_sample_rmat(chi, all_ome[vs_idx_array])

        # return unit G-vectors in LAB FRAME
        ghat_l = _crystal_to_lab(tmp_gvec,
                                 rmat_s, cnst.identity_3x3,
                                 bmat=None, vmat_inv=None)

        # if non-standard beam frame is specified, transform ghat_l to
        # BEAM FRAME in place
        if rmat_b is not None:
            ghat_l = np.dot(ghat_l, rmat_b)

        # get etas in BEAM FRAME using arctan2
        all_eta = np.arctan2(ghat_l[:, 1], ghat_l[:, 0])

        # assign solutions to output array
        eta0[valid_solutions] = all_eta[:num_valid]
        eta1[valid_solutions] = all_eta[num_valid:]

        # make assoc tth array
        tth = 2.*np.arcsin(sintht).flatten()
        tth[~valid_solutions] = np.nan

        sol0 = np.vstack([tth.flatten(), eta0.flatten(), ome0.flatten()]).T
        sol1 = np.vstack([tth.flatten(), eta1.flatten(), ome1.flatten()]).T
        return sol0, sol1

    else:
        # ???: is this what we should do here?
        return ome0.flatten(), ome1.flatten()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@xf_api
def angular_difference(ang_list0, ang_list1, units=None):
    period = cnst.period_for_unit(units)
    # take difference as arrays
    diffAngles = np.atleast_1d(ang_list0) - np.atleast_1d(ang_list1)

    return abs(np.remainder(diffAngles + 0.5*period, period) - 0.5*period)


@xf_api
def map_angle(ang, range=None, units=None):
    period = cnst.period_for_unit(units)

    ang = np.atleast_1d(npfloat(ang))

    # if we have a specified angular range, use it
    if range is not None:
        angRange = np.atleast_1d(npfloat(args[0]))

        # divide of multiples of period
        ang = ang - npint(ang / period) * period

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
        # shouldn't all this be equivalent to:
        #   retval = np.mod(ang - lb, period) + lb ????
        # note the particular case below for range (-0.5*period, +0.5*period)
        # where lb would be -0.5*period.

    else:
        retval = np.mod(ang + 0.5*period, period) - 0.5*period
    return retval

@xf_api
def row_norm(vec_in):
    if len(vec_in.shape) > 2:
        raise ValueError(
                "incorrect shape: arg must be 1-d or 2-d, yours is %d"
                % (len(vec_in.shape))
        )
    return np.sqrt(np.sum(np.asarray(vec_in)**2, axis=-1))


@xf_api
def unit_vector(vec_in):
    orig_dims = vec_in.ndim
    if vec_in.ndim not in [1,2]:
        # Make sure that dimensions are supported
        raise ValueError(
            "incorrect arg shape; must be 1-d or 2-d, yours is %d-d"
            % (vec_in.ndim)
        )

    a = np.atleast_2d(vec_in)
    n = a.shape[1]

    # calculate row norms and prevent divide by zero
    nrm = np.sqrt(np.sum(a*a, axis=1))
    nrm[nrm <= cnst.epsf] = 1.
    normalized = a/nrm[:,np.newaxis]

    return normalized[0] if orig_dims == 1 else normalized


@xf_api
def make_sample_rmat(chi, ome):
    # angle chi about LAB X
    cchi = np.cos(chi)
    schi = np.sin(chi)
    rchi = np.array([[1., 0., 0.],
                     [0., cchi, -schi],
                     [0., schi, cchi]], dtype = np.double)

    # angle ome about SAMPLE Y ([0., cos(chi), sin(chi)] in LAB)
    ome = np.atleast_1d(ome)
    if len(ome) > 1:
        rmat_s = np.array(
            [np.dot(rchi, make_rmat_of_expmap(i*cnst.lab_y)) for i in ome]
        )
    else:
        come = np.cos(ome)
        some = np.sin(ome)
        rome = np.array([[come, 0., some],
                         [0., 1., 0.],
                         [-some, 0., come]], dtype = np.double)
        rmat_s = np.dot(rchi, rome)
    return rmat_s


@xf_api
def make_rmat_of_expmap(exp_map):
    phi = np.sqrt(
        exp_map[0]*exp_map[0]
        + exp_map[1]*exp_map[1]
        + exp_map[2]*exp_map[2]
        )
    if phi > cnst.epsf:
        wmat = np.array([
            [0., -exp_map[2], exp_map[1]],
            [exp_map[2], 0., -exp_map[0]],
            [-exp_map[1], exp_map[0], 0.],
            ])
        rmat = \
            cnst.identity_3x3 \
            + (np.sin(phi)/phi)*wmat \
            + ((1. - np.cos(phi))/(phi*phi))*np.dot(wmat, wmat)
    else:
        rmat = cnst.identity_3x3
    return rmat


@xf_api
def make_binary_rmat(axis):
    if axis.shape != (3,):
        raise ValueError('Axis input does not have 3 components')
    return 2*np.outer(axis, axis) - cnst.identity_3x3


@xf_api
def make_beam_rmat(bvec_l, evec_l):
    # normalize input
    bhat_l = unit_vector(bvec_l)
    ehat_l = unit_vector(evec_l)

    # find Ye as cross(ehat_l, bhat_l), normalize if kosher
    Ye = np.cross(ehat_l, bhat_l)
    if np.sqrt(np.sum(Ye*Ye)) < cnst.sqrt_epsf:
        raise RuntimeError("bvec_l and evec_l must NOT be colinear!")
    Ye = unit_vector(Ye)

    # find Xe as cross(bhat_l, Ye)
    Xe = np.cross(bhat_l, Ye)

    return np.vstack([Xe, Ye, -bhat_l])


@xf_api
def angles_in_range(angles, starts, stops, degrees=True):
    tau = 360.0 if degrees else 2*np.pi
    nw = len(starts)
    na = len(angles)
    in_range = np.zeros((na), dtype=bool)
    for i in range(nw):
        amin = starts[i]
        amax = stops[i]
        for j in range(na):
            a = angles[j]
            acheck = amin + np.mod(a - amin, tau)
            if acheck <= amax:
                in_range[j] = True
    return in_range


@xf_api
def validate_angle_ranges(ang_list, start_angs, stop_angs, ccw=True):
    # Prefer ravel over flatten because flatten never skips the copy
    ang_list = np.asarray(ang_list).ravel()
    startAngs = np.asarray(start_angs).ravel()
    stopAngs = np.asarray(stop_angs).ravel()

    n_ranges = len(start_angs)
    assert len(stop_angs) == n_ranges, \
        "length of min and max angular limits must match!"

    # to avoid warnings in >=, <= later down, mark nans;
    # need these to trick output to False in the case of nan input
    nan_mask = np.isnan(ang_list)

    reflInRange = np.zeros(ang_list.shape, dtype=bool)

    # bin length for chunking
    binLen = np.pi / 2.

    # in plane vectors defining wedges
    x0 = np.vstack([np.cos(startAngs), np.sin(startAngs)])
    x1 = np.vstack([np.cos(stopAngs), np.sin(stopAngs)])

    # dot products
    dp = np.sum(x0 * x1, axis=0)
    if np.any(dp >= 1. - cnst.sqrt_epsf) and n_ranges > 1:
        # ambiguous case
        raise RuntimeError(
            "Improper usage; at least one of your ranges"
            + "is already 360 degrees!"
        )
    elif dp[0] >= 1. - cnst.sqrt_epsf and n_ranges == 1:
        # trivial case!
        reflInRange = np.ones(ang_list.shape, dtype=bool)
        reflInRange[nan_mask] = False
    else:
        # solve for arc lengths
        # ...note: no zeros should have made it here
        a = x0[0, :]*x1[1, :] - x0[1, :]*x1[0, :]
        b = x0[0, :]*x1[0, :] + x0[1, :]*x1[1, :]
        phi = np.arctan2(b, a)

        arclen = 0.5*np.pi - phi          # these are clockwise
        cw_phis = arclen < 0
        arclen[cw_phis] = 2*np.pi + arclen[cw_phis]   # all positive (CW) now
        if not ccw:
            arclen = 2*np.pi - arclen

        if sum(arclen) > 2*np.pi:
            raise RuntimeWarning(
                "Specified angle ranges sum to > 360 degrees"
                + ", which is suspect..."
            )

        # check that there are no more thandp = np.zeros(n_ranges)
        for i in range(n_ranges):
            # number or subranges using 'binLen'
            numSubranges = int(np.ceil(arclen[i]/binLen))

            # check remaider
            binrem = np.remainder(arclen[i], binLen)
            if binrem == 0:
                finalBinLen = binLen
            else:
                finalBinLen = binrem

            # if clockwise, negate bin length
            if not ccw:
                binLen = -binLen
                finalBinLen = -finalBinLen

            # Create sub ranges on the fly to avoid ambiguity in dot product
            # for wedges >= 180 degrees
            subRanges = np.array(
                [startAngs[i] + binLen*j for j in range(numSubranges)]
                + [startAngs[i] + binLen*(numSubranges - 1) + finalBinLen])

            for k in range(numSubranges):
                zStart = _z_project(ang_list, subRanges[k])
                zStop = _z_project(ang_list, subRanges[k + 1])
                if ccw:
                    zStart[nan_mask] = 999.
                    zStop[nan_mask] = -999.
                    reflInRange = \
                        reflInRange | np.logical_and(zStart <= 0, zStop >= 0)
                else:
                    zStart[nan_mask] = -999.
                    zStop[nan_mask] = 999.
                    reflInRange = \
                        reflInRange | np.logical_and(zStart >= 0, zStop <= 0)
    return reflInRange


@xf_api
def rotate_vecs_about_axis(angle, axis, vecs):
    angle = np.atleast_1d(angle)
    # nvecs = vecs.shape[1]  # assume column vecs

    # quaternion components
    q0 = np.cos(0.5*angle)
    q1 = np.sin(0.5*angle)
    qv = np.tile(q1, (3, 1)) * axis

    # component perpendicular to axes (inherits shape of vecs)
    vp0 = vecs[0, :] \
        - axis[0, :]*axis[0, :]*vecs[0, :] \
        - axis[0, :]*axis[1, :]*vecs[1, :] \
        - axis[0, :]*axis[2, :]*vecs[2, :]
    vp1 = vecs[1, :] \
        - axis[1, :]*axis[1, :]*vecs[1, :] \
        - axis[1, :]*axis[0, :]*vecs[0, :] \
        - axis[1, :]*axis[2, :]*vecs[2, :]
    vp2 = vecs[2, :] \
        - axis[2, :]*axis[2, :]*vecs[2, :] \
        - axis[2, :]*axis[0, :]*vecs[0, :] \
        - axis[2, :]*axis[1, :]*vecs[1, :]

    # dot product with components along; cross product with components normal
    qdota = \
        (axis[0, :]*vecs[0, :]
            + axis[1, :]*vecs[1, :]
            + axis[2, :]*vecs[2, :]) \
        * \
        (axis[0, :]*qv[0, :]
            + axis[1, :]*qv[1, :]
            + axis[2, :]*qv[2, :])
    qcrossn = np.vstack([qv[1, :]*vp2 - qv[2, :]*vp1,
                         qv[2, :]*vp0 - qv[0, :]*vp2,
                         qv[0, :]*vp1 - qv[1, :]*vp0])

    # quaternion formula
    v_rot = np.tile(q0*q0 - q1*q1, (3, 1)) * vecs \
        + 2. * np.tile(qdota, (3, 1)) * qv \
        + 2. * np.tile(q0, (3, 1)) * qcrossn
    return v_rot


@xf_api
def quat_product_matrix(q, mult='right'):
    if mult == 'right':
        qmat = np.array([[ q[0], -q[1], -q[2], -q[3]],
                         [ q[1],  q[0],  q[3], -q[2]],
                         [ q[2], -q[3],  q[0],  q[1]],
                         [ q[3],  q[2], -q[1],  q[0]],
                         ])
    elif mult == 'left':
        qmat = np.array([[ q[0], -q[1], -q[2], -q[3]],
                         [ q[1],  q[0], -q[3],  q[2]],
                         [ q[2],  q[3],  q[0], -q[1]],
                         [ q[3], -q[2],  q[1],  q[0]],
                         ])
    return qmat


@xf_api
def quat_distance(q1, q2, qsym):
    # qsym from PlaneData objects are (4, nsym)
    # convert symmetries to (4, 4) qprod matrices
    nsym = qsym.shape[1]
    rsym = np.zeros((nsym, 4, 4))
    for i in range(nsym):
        rsym[i, :, :] = quat_product_matrix(qsym[:, i], mult='right')

    # inverse of q1 in matrix form
    q1i = quat_product_matrix(
        np.r_[1, -1, -1, -1]*np.atleast_1d(q1).flatten(),
        mult='right'
    )

    # Do R * Gc, store as vstacked equivalent quaternions (nsym, 4)
    q2s = np.dot(rsym, q2)

    # Calculate the class of misorientations for full symmetrically equivalent
    # q1 and q2 as:
    #
    #     q2*q1^(-1)
    #
    # using matrix notation (4, 4) * (4, nsym)
    eqv_mis = np.dot(q1i, q2s.T)

    # find the largest scalar component and return arccos
    return 2*np.arccos(eqv_mis[0, np.argmax(abs(eqv_mis[0, :]))])
