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

import numpy as np

from hexrd import constants
from hexrd import matrixutil as mutil
from hexrd import gridutil as gutil

from hexrd.crystallography import processWavelength

from hexrd.transforms import xf
from hexrd.transforms import xfcapi

from hexrd import distortion

from hexrd.constants import USE_NUMBA
if USE_NUMBA:
    import numba


dFunc_ref = distortion.dummy
dParams_ref = []

d2r = piby180 = np.pi/180.
r2d = 1.0/d2r

epsf = constants.epsf            # ~2.2e-16
ten_epsf = 10 * epsf             # ~2.2e-15
sqrt_epsf = constants.sqrt_epsf  # ~1.5e-8

bHat_l_DFLT = constants.beam_vec.flatten()
eHat_l_DFLT = constants.eta_vec.flatten()

_memo_hkls = {}


def _zproject(x, y):
    return np.cos(x) * np.sin(y) - np.sin(x) * np.cos(y)


def validateAngleRanges(angList, startAngs, stopAngs, ccw=True):
    """
    Indetify angles that fall within specified ranges.

    A better way to go.  find out if an angle is in the range
    CCW or CW from start to stop

    There is, of course an ambigutiy if the start and stop angle are
    the same; we treat them as implying 2*pi
    """
    angList = np.atleast_1d(angList).flatten()      # needs to have len
    startAngs = np.atleast_1d(startAngs).flatten()  # needs to have len
    stopAngs = np.atleast_1d(stopAngs).flatten()    # needs to have len

    n_ranges = len(startAngs)
    assert len(stopAngs) == n_ranges, \
        "length of min and max angular limits must match!"

    # to avoid warnings in >=, <= later down, mark nans;
    # need these to trick output to False in the case of nan input
    nan_mask = np.isnan(angList)

    reflInRange = np.zeros(angList.shape, dtype=bool)

    # bin length for chunking
    binLen = np.pi / 2.

    # in plane vectors defining wedges
    x0 = np.vstack([np.cos(startAngs), np.sin(startAngs)])
    x1 = np.vstack([np.cos(stopAngs), np.sin(stopAngs)])

    # dot products
    dp = np.sum(x0 * x1, axis=0)
    if np.any(dp >= 1. - sqrt_epsf) and n_ranges > 1:
        # ambiguous case
        raise RuntimeError(
            "Improper usage; " +
            "at least one of your ranges is alread 360 degrees!")
    elif dp[0] >= 1. - sqrt_epsf and n_ranges == 1:
        # trivial case!
        reflInRange = np.ones(angList.shape, dtype=bool)
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
                "Specified angle ranges sum to > 360 degrees, " +
                "which is suspect...")

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
                + [startAngs[i] + binLen*(numSubranges - 1) + finalBinLen]
                )

            for k in range(numSubranges):
                zStart = _zproject(angList, subRanges[k])
                zStop = _zproject(angList, subRanges[k + 1])
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


def simulateOmeEtaMaps(omeEdges, etaEdges, planeData, expMaps,
                       chi=0.,
                       etaTol=None, omeTol=None,
                       etaRanges=None, omeRanges=None,
                       bVec=xf.bVec_ref, eVec=xf.eta_ref, vInv=xf.vInv_ref):
    """
    Simulate spherical maps.

    Parameters
    ----------
    omeEdges : TYPE
        DESCRIPTION.
    etaEdges : TYPE
        DESCRIPTION.
    planeData : TYPE
        DESCRIPTION.
    expMaps : (3, n) ndarray
        DESCRIPTION.
    chi : TYPE, optional
        DESCRIPTION. The default is 0..
    etaTol : TYPE, optional
        DESCRIPTION. The default is None.
    omeTol : TYPE, optional
        DESCRIPTION. The default is None.
    etaRanges : TYPE, optional
        DESCRIPTION. The default is None.
    omeRanges : TYPE, optional
        DESCRIPTION. The default is None.
    bVec : TYPE, optional
        DESCRIPTION. The default is xf.bVec_ref.
    eVec : TYPE, optional
        DESCRIPTION. The default is xf.eta_ref.
    vInv : TYPE, optional
        DESCRIPTION. The default is xf.vInv_ref.

    Returns
    -------
    eta_ome : TYPE
        DESCRIPTION.

    Notes
    -----
    all angular info is entered in degrees

    ??? might want to creat module-level angluar unit flag
    ??? might want to allow resvers delta omega

    """
    # convert to radians
    etaEdges = np.radians(np.sort(etaEdges))
    omeEdges = np.radians(np.sort(omeEdges))

    omeIndices = list(range(len(omeEdges)))
    etaIndices = list(range(len(etaEdges)))

    i_max = omeIndices[-1]
    j_max = etaIndices[-1]

    etaMin = etaEdges[0]
    etaMax = etaEdges[-1]
    omeMin = omeEdges[0]
    omeMax = omeEdges[-1]
    if omeRanges is None:
        omeRanges = [[omeMin, omeMax], ]

    if etaRanges is None:
        etaRanges = [[etaMin, etaMax], ]

    # signed deltas IN RADIANS
    del_ome = omeEdges[1] - omeEdges[0]
    del_eta = etaEdges[1] - etaEdges[0]

    delOmeSign = np.sign(del_eta)

    # tolerances are in degrees (easier)
    if omeTol is None:
        omeTol = abs(del_ome)
    else:
        omeTol = np.radians(omeTol)
    if etaTol is None:
        etaTol = abs(del_eta)
    else:
        etaTol = np.radians(etaTol)

    # pixel dialtions
    dpix_ome = round(omeTol / abs(del_ome))
    dpix_eta = round(etaTol / abs(del_eta))

    i_dil, j_dil = np.meshgrid(np.arange(-dpix_ome, dpix_ome + 1),
                               np.arange(-dpix_eta, dpix_eta + 1))

    # get symmetrically expanded hkls from planeData
    sym_hkls = planeData.getSymHKLs()
    nhkls = len(sym_hkls)

    # make things C-contiguous for use in xfcapi functions
    expMaps = np.array(expMaps.T, order='C')
    nOrs = len(expMaps)

    bMat = np.array(planeData.latVecOps['B'], order='C')
    wlen = planeData.wavelength

    bVec = np.array(bVec.flatten(), order='C')
    eVec = np.array(eVec.flatten(), order='C')
    vInv = np.array(vInv.flatten(), order='C')

    eta_ome = np.zeros((nhkls, max(omeIndices), max(etaIndices)), order='C')
    for iHKL in range(nhkls):
        these_hkls = np.ascontiguousarray(sym_hkls[iHKL].T, dtype=float)
        for iOr in range(nOrs):
            rMat_c = xfcapi.makeRotMatOfExpMap(expMaps[iOr, :])
            angList = np.vstack(
                xfcapi.oscillAnglesOfHKLs(these_hkls, chi, rMat_c, bMat, wlen,
                                          beamVec=bVec, etaVec=eVec, vInv=vInv)
                )
            if not np.all(np.isnan(angList)):
                #
                angList[:, 1] = xf.mapAngle(
                        angList[:, 1],
                        [etaEdges[0], etaEdges[0]+2*np.pi])
                angList[:, 2] = xf.mapAngle(
                        angList[:, 2],
                        [omeEdges[0], omeEdges[0]+2*np.pi])
                #
                # do eta ranges
                angMask_eta = np.zeros(len(angList), dtype=bool)
                for etas in etaRanges:
                    angMask_eta = np.logical_or(
                        angMask_eta,
                        xf.validateAngleRanges(angList[:, 1], etas[0], etas[1])
                    )

                # do omega ranges
                ccw = True
                angMask_ome = np.zeros(len(angList), dtype=bool)
                for omes in omeRanges:
                    if omes[1] - omes[0] < 0:
                        ccw = False
                    angMask_ome = np.logical_or(
                        angMask_ome,
                        xf.validateAngleRanges(
                                angList[:, 2], omes[0], omes[1], ccw=ccw)
                    )

                # mask angles list, hkls
                angMask = np.logical_and(angMask_eta, angMask_ome)

                culledTTh = angList[angMask, 0]
                culledEta = angList[angMask, 1]
                culledOme = angList[angMask, 2]

                for iTTh in range(len(culledTTh)):
                    culledEtaIdx = np.where(etaEdges - culledEta[iTTh] > 0)[0]
                    if len(culledEtaIdx) > 0:
                        culledEtaIdx = culledEtaIdx[0] - 1
                        if culledEtaIdx < 0:
                            culledEtaIdx = None
                    else:
                        culledEtaIdx = None
                    culledOmeIdx = np.where(omeEdges - culledOme[iTTh] > 0)[0]
                    if len(culledOmeIdx) > 0:
                        if delOmeSign > 0:
                            culledOmeIdx = culledOmeIdx[0] - 1
                        else:
                            culledOmeIdx = culledOmeIdx[-1]
                        if culledOmeIdx < 0:
                            culledOmeIdx = None
                    else:
                        culledOmeIdx = None

                    if culledEtaIdx is not None and culledOmeIdx is not None:
                        if dpix_ome > 0 or dpix_eta > 0:
                            i_sup = omeIndices[culledOmeIdx] + \
                                np.array([i_dil.flatten()], dtype=int)
                            j_sup = etaIndices[culledEtaIdx] + \
                                np.array([j_dil.flatten()], dtype=int)

                            # catch shit that falls off detector...
                            # maybe make this fancy enough to wrap at 2pi?
                            idx_mask = np.logical_and(
                                np.logical_and(i_sup >= 0, i_sup < i_max),
                                np.logical_and(j_sup >= 0, j_sup < j_max))
                            eta_ome[iHKL,
                                    i_sup[idx_mask],
                                    j_sup[idx_mask]] = 1.
                        else:
                            eta_ome[iHKL,
                                    omeIndices[culledOmeIdx],
                                    etaIndices[culledEtaIdx]] = 1.
                            pass  # close conditional on pixel dilation
                        pass  # close conditional on ranges
                    pass  # close for loop on valid reflections
                pass  # close conditional for valid angles
    return eta_ome


def _fetch_hkls_from_planedata(pd):
    if pd not in _memo_hkls:
        _memo_hkls[pd] = np.ascontiguousarray(
            np.hstack(pd.getSymHKLs(withID=True)).T,
            dtype=float)
    return _memo_hkls[pd]


def _filter_hkls_eta_ome(hkls, angles, eta_range, ome_range):
    """
    given a set of hkls and angles, filter them by the
    eta and omega ranges
    """
    # do eta ranges
    angMask_eta = np.zeros(len(angles), dtype=bool)
    for etas in eta_range:
        angMask_eta = np.logical_or(
            angMask_eta,
            xf.validateAngleRanges(angles[:, 1], etas[0], etas[1])
        )

    # do omega ranges
    ccw = True
    angMask_ome = np.zeros(len(angles), dtype=bool)
    for omes in ome_range:
        if omes[1] - omes[0] < 0:
            ccw = False
        angMask_ome = np.logical_or(
            angMask_ome,
            xf.validateAngleRanges(angles[:, 2], omes[0], omes[1], ccw=ccw)
        )

    # mask angles list, hkls
    angMask = np.logical_and(angMask_eta, angMask_ome)

    allAngs = angles[angMask, :]
    allHKLs = np.vstack([hkls, hkls])[angMask, :]

    return allAngs, allHKLs


def _project_on_detector_plane(allAngs,
                               rMat_d, rMat_c, chi,
                               tVec_d, tVec_c, tVec_s,
                               distortion,
                               beamVec=constants.beam_vec):
    """
    utility routine for projecting a list of (tth, eta, ome) onto the
    detector plane parameterized by the args
    """
    gVec_cs = xfcapi.anglesToGVec(allAngs,
                                  chi=chi,
                                  rMat_c=rMat_c,
                                  bHat_l=beamVec)

    rMat_ss = xfcapi.makeOscillRotMatArray(chi, allAngs[:, 2])

    tmp_xys = xfcapi.gvecToDetectorXYArray(
        gVec_cs, rMat_d, rMat_ss, rMat_c,
        tVec_d, tVec_s, tVec_c,
        beamVec=beamVec)

    valid_mask = ~(np.isnan(tmp_xys[:, 0]) | np.isnan(tmp_xys[:, 1]))

    det_xy = np.atleast_2d(tmp_xys[valid_mask, :])

    # FIXME: distortion kludge
    if distortion is not None and len(distortion) == 2:
        det_xy = distortion[0](det_xy,
                               distortion[1],
                               invert=True)
    return det_xy, rMat_ss, valid_mask


def simulateGVecs(pd, detector_params, grain_params,
                  ome_range=[(-np.pi, np.pi), ],
                  ome_period=(-np.pi, np.pi),
                  eta_range=[(-np.pi, np.pi), ],
                  panel_dims=[(-204.8, -204.8), (204.8, 204.8)],
                  pixel_pitch=(0.2, 0.2),
                  distortion=(dFunc_ref, dParams_ref)):
    """
    returns valid_ids, valid_hkl, valid_ang, valid_xy, ang_ps

    panel_dims are [(xmin, ymin), (xmax, ymax)] in mm

    pixel_pitch is [row_size, column_size] in mm

    simulate the monochormatic scattering for a specified

        - space group
        - wavelength
        - orientation
        - strain
        - position
        - detector parameters
        - oscillation axis tilt (chi)

    subject to

        - omega (oscillation) ranges (list of (min, max) tuples)
        - eta (azimuth) ranges

    pd................a hexrd.crystallography.PlaneData instance
    detector_params...a (10,) ndarray containing the tilt angles (3),
                      translation (3), chi (1), and sample frame translation
                      (3) parameters
    grain_params......a (12,) ndarray containing the exponential map (3),
                      translation (3), and inverse stretch tensor compnents
                      in Mandel-Voigt notation (6).

    * currently only one panel is supported, but this will likely change soon
    """
    bMat = pd.latVecOps['B']
    wlen = pd.wavelength
    full_hkls = _fetch_hkls_from_planedata(pd)

    # extract variables for convenience
    rMat_d = xfcapi.makeDetectorRotMat(detector_params[:3])
    tVec_d = np.ascontiguousarray(detector_params[3:6])
    chi = detector_params[6]
    tVec_s = np.ascontiguousarray(detector_params[7:10])
    rMat_c = xfcapi.makeRotMatOfExpMap(grain_params[:3])
    tVec_c = np.ascontiguousarray(grain_params[3:6])
    vInv_s = np.ascontiguousarray(grain_params[6:12])

    # first find valid G-vectors
    angList = np.vstack(
        xfcapi.oscillAnglesOfHKLs(
            full_hkls[:, 1:], chi, rMat_c, bMat, wlen, vInv=vInv_s
            )
        )
    allAngs, allHKLs = _filter_hkls_eta_ome(
        full_hkls, angList, eta_range, ome_range
        )

    if len(allAngs) == 0:
        valid_ids = []
        valid_hkl = []
        valid_ang = []
        valid_xy = []
        ang_ps = []
    else:
        # ??? preallocate for speed?
        det_xy, rMat_s, on_plane = _project_on_detector_plane(
            allAngs,
            rMat_d, rMat_c, chi,
            tVec_d, tVec_c, tVec_s,
            distortion
            )
        #
        on_panel_x = np.logical_and(
            det_xy[:, 0] >= panel_dims[0][0],
            det_xy[:, 0] <= panel_dims[1][0]
            )
        on_panel_y = np.logical_and(
            det_xy[:, 1] >= panel_dims[0][1],
            det_xy[:, 1] <= panel_dims[1][1]
            )
        on_panel = np.logical_and(on_panel_x, on_panel_y)
        #
        op_idx = np.where(on_panel)[0]
        #
        valid_ang = allAngs[op_idx, :]
        valid_ang[:, 2] = xf.mapAngle(valid_ang[:, 2], ome_period)
        valid_ids = allHKLs[op_idx, 0]
        valid_hkl = allHKLs[op_idx, 1:]
        valid_xy = det_xy[op_idx, :]
        ang_ps = angularPixelSize(valid_xy, pixel_pitch,
                                  rMat_d, rMat_s,
                                  tVec_d, tVec_s, tVec_c,
                                  distortion=distortion)

    return valid_ids, valid_hkl, valid_ang, valid_xy, ang_ps


def simulateLauePattern(hkls, bMat,
                        rmat_d, tvec_d,
                        panel_dims, panel_buffer=5,
                        minEnergy=8, maxEnergy=24,
                        rmat_s=np.eye(3),
                        grain_params=None,
                        distortion=None,
                        beamVec=None):

    if beamVec is None:
        beamVec = xfcapi.bVec_ref

    # parse energy ranges
    multipleEnergyRanges = False
    if hasattr(maxEnergy, '__len__'):
        assert len(maxEnergy) == len(minEnergy), \
            'energy cutoff ranges must have the same length'
        multipleEnergyRanges = True
        lmin = []
        lmax = []
        for i in range(len(maxEnergy)):
            lmin.append(processWavelength(maxEnergy[i]))
            lmax.append(processWavelength(minEnergy[i]))
    else:
        lmin = processWavelength(maxEnergy)
        lmax = processWavelength(minEnergy)

    # process crystal rmats and inverse stretches
    if grain_params is None:
        grain_params = np.atleast_2d(
            [0., 0., 0.,
             0., 0., 0.,
             1., 1., 1., 0., 0., 0.
             ]
        )

    n_grains = len(grain_params)

    # dummy translation vector... make input
    tvec_s = np.zeros((3, 1))

    # number of hkls
    nhkls_tot = hkls.shape[1]

    # unit G-vectors in crystal frame
    ghat_c = mutil.unitVector(np.dot(bMat, hkls))

    # pre-allocate output arrays
    xy_det = np.nan*np.ones((n_grains, nhkls_tot, 2))
    hkls_in = np.nan*np.ones((n_grains, 3, nhkls_tot))
    angles = np.nan*np.ones((n_grains, nhkls_tot, 2))
    dspacing = np.nan*np.ones((n_grains, nhkls_tot))
    energy = np.nan*np.ones((n_grains, nhkls_tot))

    """
    LOOP OVER GRAINS
    """

    for iG, gp in enumerate(grain_params):
        rmat_c = xfcapi.makeRotMatOfExpMap(gp[:3])
        tvec_c = gp[3:6].reshape(3, 1)
        vInv_s = mutil.vecMVToSymm(gp[6:].reshape(6, 1))

        # stretch them: V^(-1) * R * Gc
        ghat_s_str = mutil.unitVector(
            np.dot(vInv_s, np.dot(rmat_c, ghat_c))
        )
        ghat_c_str = np.dot(rmat_c.T, ghat_s_str)

        # project
        dpts = xfcapi.gvecToDetectorXY(ghat_c_str.T,
                                       rmat_d, rmat_s, rmat_c,
                                       tvec_d, tvec_s, tvec_c,
                                       beamVec=beamVec).T

        # check intersections with detector plane
        canIntersect = ~np.isnan(dpts[0, :])
        npts_in = sum(canIntersect)

        if np.any(canIntersect):
            dpts = dpts[:, canIntersect].reshape(2, npts_in)
            dhkl = hkls[:, canIntersect].reshape(3, npts_in)

            # back to angles
            tth_eta, gvec_l = xfcapi.detectorXYToGvec(
                dpts.T,
                rmat_d, rmat_s,
                tvec_d, tvec_s, tvec_c,
                beamVec=beamVec)
            tth_eta = np.vstack(tth_eta).T

            # warp measured points
            if distortion is not None:
                if len(distortion) == 2:
                    dpts = distortion[0](dpts, distortion[1], invert=True)

            # plane spacings and energies
            dsp = 1. / mutil.columnNorm(np.dot(bMat, dhkl))
            wlen = 2*dsp*np.sin(0.5*tth_eta[:, 0])

            # find on spatial extent of detector
            xTest = np.logical_and(
                dpts[0, :] >= -0.5*panel_dims[1] + panel_buffer,
                dpts[0, :] <= 0.5*panel_dims[1] - panel_buffer)
            yTest = np.logical_and(
                dpts[1, :] >= -0.5*panel_dims[0] + panel_buffer,
                dpts[1, :] <= 0.5*panel_dims[0] - panel_buffer)

            onDetector = np.logical_and(xTest, yTest)
            if multipleEnergyRanges:
                validEnergy = np.zeros(len(wlen), dtype=bool)
                for i in range(len(lmin)):
                    validEnergy = validEnergy | \
                        np.logical_and(wlen >= lmin[i], wlen <= lmax[i])
                    pass
            else:
                validEnergy = np.logical_and(wlen >= lmin, wlen <= lmax)
                pass

            # index for valid reflections
            keepers = np.where(np.logical_and(onDetector, validEnergy))[0]

            # assign output arrays
            xy_det[iG][keepers, :] = dpts[:, keepers].T
            hkls_in[iG][:, keepers] = dhkl[:, keepers]
            angles[iG][keepers, :] = tth_eta[keepers, :]
            dspacing[iG, keepers] = dsp[keepers]
            energy[iG, keepers] = processWavelength(wlen[keepers])
            pass
        pass
    return xy_det, hkls_in, angles, dspacing, energy


if USE_NUMBA:
    @numba.njit
    def _expand_pixels(original, w, h, result):
        hw = 0.5 * w
        hh = 0.5 * h
        for el in range(len(original)):
            x, y = original[el, 0], original[el, 1]
            result[el*4 + 0, 0] = x - hw
            result[el*4 + 0, 1] = y - hh
            result[el*4 + 1, 0] = x + hw
            result[el*4 + 1, 1] = y - hh
            result[el*4 + 2, 0] = x + hw
            result[el*4 + 2, 1] = y + hh
            result[el*4 + 3, 0] = x - hw
            result[el*4 + 3, 1] = y + hh

        return result

    @numba.jit
    def _compute_max(tth, eta, result):
        period = 2.0 * np.pi
        hperiod = np.pi
        for el in range(0, len(tth), 4):
            max_tth = np.abs(tth[el + 0] - tth[el + 3])
            eta_diff = eta[el + 0] - eta[el + 3]
            max_eta = np.abs(
                np.remainder(eta_diff + hperiod, period) - hperiod
            )
            for i in range(3):
                curr_tth = np.abs(tth[el + i] - tth[el + i + 1])
                eta_diff = eta[el + i] - eta[el + i + 1]
                curr_eta = np.abs(
                    np.remainder(eta_diff + hperiod, period) - hperiod
                )
                max_tth = np.maximum(curr_tth, max_tth)
                max_eta = np.maximum(curr_eta, max_eta)
            result[el//4, 0] = max_tth
            result[el//4, 1] = max_eta

        return result

    def angularPixelSize(
            xy_det, xy_pixelPitch,
            rMat_d, rMat_s,
            tVec_d, tVec_s, tVec_c,
            distortion=None, beamVec=None, etaVec=None):
        """
        Calculate angular pixel sizes on a detector.

        * choices to beam vector and eta vector specs have been supressed
        * assumes xy_det in UNWARPED configuration
        """
        xy_det = np.atleast_2d(xy_det)
        if distortion is not None and len(distortion) == 2:
            xy_det = distortion[0](xy_det, distortion[1])
        if beamVec is None:
            beamVec = xfcapi.bVec_ref
        if etaVec is None:
            etaVec = xfcapi.eta_ref

        xy_expanded = np.empty((len(xy_det) * 4, 2), dtype=xy_det.dtype)
        xy_expanded = _expand_pixels(
            xy_det,
            xy_pixelPitch[0], xy_pixelPitch[1],
            xy_expanded)
        gvec_space, _ = xfcapi.detectorXYToGvec(
            xy_expanded,
            rMat_d, rMat_s,
            tVec_d, tVec_s, tVec_c,
            beamVec=beamVec, etaVec=etaVec)
        result = np.empty_like(xy_det)
        return _compute_max(gvec_space[0], gvec_space[1], result)
else:
    def angularPixelSize(xy_det, xy_pixelPitch,
                         rMat_d, rMat_s,
                         tVec_d, tVec_s, tVec_c,
                         distortion=None, beamVec=None, etaVec=None):
        """
        Calculate angular pixel sizes on a detector.

        * choices to beam vector and eta vector specs have been supressed
        * assumes xy_det in UNWARPED configuration
        """
        xy_det = np.atleast_2d(xy_det)
        if distortion is not None and len(distortion) == 2:
            xy_det = distortion[0](xy_det, distortion[1])
        if beamVec is None:
            beamVec = xfcapi.bVec_ref
        if etaVec is None:
            etaVec = xfcapi.eta_ref

        xp = np.r_[-0.5,  0.5,  0.5, -0.5] * xy_pixelPitch[0]
        yp = np.r_[-0.5, -0.5,  0.5,  0.5] * xy_pixelPitch[1]

        diffs = np.array([[3, 3, 2, 1],
                          [2, 0, 1, 0]])

        ang_pix = np.zeros((len(xy_det), 2))

        for ipt, xy in enumerate(xy_det):
            xc = xp + xy[0]
            yc = yp + xy[1]

            tth_eta, gHat_l = xfcapi.detectorXYToGvec(
                np.vstack([xc, yc]).T,
                rMat_d, rMat_s,
                tVec_d, tVec_s, tVec_c,
                beamVec=beamVec, etaVec=etaVec)
            delta_tth = np.zeros(4)
            delta_eta = np.zeros(4)
            for j in range(4):
                delta_tth[j] = abs(
                    tth_eta[0][diffs[0, j]] - tth_eta[0][diffs[1, j]]
                )
                delta_eta[j] = xf.angularDifference(
                    tth_eta[1][diffs[0, j]], tth_eta[1][diffs[1, j]]
                )

            ang_pix[ipt, 0] = np.amax(delta_tth)
            ang_pix[ipt, 1] = np.amax(delta_eta)
        return ang_pix


if USE_NUMBA:
    @numba.njit
    def _coo_build_window_jit(frame_row, frame_col, frame_data,
                              min_row, max_row, min_col, max_col,
                              result):
        n = len(frame_row)
        for i in range(n):
            if ((min_row <= frame_row[i] <= max_row) and
                    (min_col <= frame_col[i] <= max_col)):
                new_row = frame_row[i] - min_row
                new_col = frame_col[i] - min_col
                result[new_row, new_col] = frame_data[i]

        return result

    def _coo_build_window(frame_i, min_row, max_row, min_col, max_col):
        window = np.zeros(
            ((max_row - min_row + 1), (max_col - min_col + 1)),
            dtype=np.int16
        )

        return _coo_build_window_jit(frame_i.row, frame_i.col, frame_i.data,
                                     min_row, max_row, min_col, max_col,
                                     window)
else:  # not USE_NUMBA
    def _coo_build_window(frame_i, min_row, max_row, min_col, max_col):
        mask = ((min_row <= frame_i.row) & (frame_i.row <= max_row) &
                (min_col <= frame_i.col) & (frame_i.col <= max_col))
        new_row = frame_i.row[mask] - min_row
        new_col = frame_i.col[mask] - min_col
        new_data = frame_i.data[mask]
        window = np.zeros(
            ((max_row - min_row + 1), (max_col - min_col + 1)),
            dtype=np.int16
        )
        window[new_row, new_col] = new_data

        return window


def make_reflection_patches(instr_cfg, tth_eta, ang_pixel_size,
                            omega=None,
                            tth_tol=0.2, eta_tol=1.0,
                            rMat_c=np.eye(3), tVec_c=np.zeros((3, 1)),
                            distortion=None,
                            npdiv=1, quiet=False,
                            compute_areas_func=gutil.compute_areas,
                            beamVec=None):
    """
    Make angular patches on a detector.

    panel_dims are [(xmin, ymin), (xmax, ymax)] in mm

    pixel_pitch is [row_size, column_size] in mm

    FIXME: DISTORTION HANDING IS STILL A KLUDGE!!!

    patches are:

                 delta tth
    d  ------------- ... -------------
    e  | x | x | x | ... | x | x | x |
    l  ------------- ... -------------
    t                 .
    a                 .
                     .
    e  ------------- ... -------------
    t  | x | x | x | ... | x | x | x |
    a  ------------- ... -------------

    outputs are:
        (tth_vtx, eta_vtx),
        (x_vtx, y_vtx),
        connectivity,
        subpixel_areas,
        (x_center, y_center),
        (i_row, j_col)
    """
    npts = len(tth_eta)

    # detector frame
    rMat_d = xfcapi.makeRotMatOfExpMap(
        np.r_[instr_cfg['detector']['transform']['tilt']]
        )
    tVec_d = np.r_[instr_cfg['detector']['transform']['translation']]
    pixel_size = instr_cfg['detector']['pixels']['size']

    frame_nrows = instr_cfg['detector']['pixels']['rows']
    frame_ncols = instr_cfg['detector']['pixels']['columns']

    panel_dims = (
        -0.5*np.r_[frame_ncols*pixel_size[1], frame_nrows*pixel_size[0]],
        0.5*np.r_[frame_ncols*pixel_size[1], frame_nrows*pixel_size[0]]
        )
    row_edges = np.arange(frame_nrows + 1)[::-1]*pixel_size[1] \
        + panel_dims[0][1]
    col_edges = np.arange(frame_ncols + 1)*pixel_size[0] \
        + panel_dims[0][0]

    # sample frame
    chi = instr_cfg['oscillation_stage']['chi']
    tVec_s = np.r_[instr_cfg['oscillation_stage']['translation']]

    # beam vector
    if beamVec is None:
        beamVec = xfcapi.bVec_ref

    # data to loop
    # ...WOULD IT BE CHEAPER TO CARRY ZEROS OR USE CONDITIONAL?
    if omega is None:
        full_angs = np.hstack([tth_eta, np.zeros((npts, 1))])
    else:
        full_angs = np.hstack([tth_eta, omega.reshape(npts, 1)])

    patches = []
    for angs, pix in zip(full_angs, ang_pixel_size):
        ndiv_tth = npdiv*np.ceil(tth_tol/np.degrees(pix[0]))
        ndiv_eta = npdiv*np.ceil(eta_tol/np.degrees(pix[1]))

        tth_del = np.arange(0, ndiv_tth + 1)*tth_tol/float(ndiv_tth) \
            - 0.5*tth_tol
        eta_del = np.arange(0, ndiv_eta + 1)*eta_tol/float(ndiv_eta) \
            - 0.5*eta_tol

        # store dimensions for convenience
        #   * etas and tths are bin vertices, ome is already centers
        sdims = [len(eta_del) - 1, len(tth_del) - 1]

        # FOR ANGULAR MESH
        conn = gutil.cellConnectivity(
            sdims[0],
            sdims[1],
            origin='ll'
        )

        # meshgrid args are (cols, rows), a.k.a (fast, slow)
        m_tth, m_eta = np.meshgrid(tth_del, eta_del)
        npts_patch = m_tth.size

        # calculate the patch XY coords from the (tth, eta) angles
        # * will CHEAT and ignore the small perturbation the different
        #   omega angle values causes and simply use the central value
        gVec_angs_vtx = np.tile(angs, (npts_patch, 1)) \
            + np.radians(
                np.vstack(
                    [m_tth.flatten(),
                     m_eta.flatten(),
                     np.zeros(npts_patch)]
                ).T
            )

        xy_eval_vtx, rmats_s, on_plane = _project_on_detector_plane(
                gVec_angs_vtx,
                rMat_d, rMat_c,
                chi,
                tVec_d, tVec_c, tVec_s,
                distortion,
                beamVec=beamVec)

        areas = compute_areas_func(xy_eval_vtx, conn)

        # EVALUATION POINTS
        #   * for lack of a better option will use centroids
        tth_eta_cen = gutil.cellCentroids(
            np.atleast_2d(gVec_angs_vtx[:, :2]),
            conn
        )

        gVec_angs = np.hstack(
            [tth_eta_cen, np.tile(angs[2], (len(tth_eta_cen), 1))]
        )

        xy_eval, rmats_s, on_plane = _project_on_detector_plane(
                gVec_angs,
                rMat_d, rMat_c,
                chi,
                tVec_d, tVec_c, tVec_s,
                distortion,
                beamVec=beamVec)

        row_indices = gutil.cellIndices(row_edges, xy_eval[:, 1])
        col_indices = gutil.cellIndices(col_edges, xy_eval[:, 0])

        # append patch data to list
        patches.append(
            ((gVec_angs_vtx[:, 0].reshape(m_tth.shape),
              gVec_angs_vtx[:, 1].reshape(m_tth.shape)),
             (xy_eval_vtx[:, 0].reshape(m_tth.shape),
              xy_eval_vtx[:, 1].reshape(m_tth.shape)),
             conn,
             areas.reshape(sdims[0], sdims[1]),
             (xy_eval[:, 0].reshape(sdims[0], sdims[1]),
              xy_eval[:, 1].reshape(sdims[0], sdims[1])),
             (row_indices.reshape(sdims[0], sdims[1]),
              col_indices.reshape(sdims[0], sdims[1])))
        )
        pass    # close loop over angles
    return patches


def extract_detector_transformation(detector_params):
    """
    Construct arrays from detector parameters.

    goes from 10 vector of detector parames OR instrument config dictionary
    (from YAML spec) to affine transformation arrays

    Parameters
    ----------
    detector_params : TYPE
        DESCRIPTION.

    Returns
    -------
    rMat_d : TYPE
        DESCRIPTION.
    tVec_d : TYPE
        DESCRIPTION.
    chi : TYPE
        DESCRIPTION.
    tVec_s : TYPE
        DESCRIPTION.

    """
    # extract variables for convenience
    if isinstance(detector_params, dict):
        rMat_d = xfcapi.makeRotMatOfExpMap(
            np.array(detector_params['detector']['transform']['tilt'])
            )
        tVec_d = np.r_[detector_params['detector']['transform']['translation']]
        chi = detector_params['oscillation_stage']['chi']
        tVec_s = np.r_[detector_params['oscillation_stage']['translation']]
    else:
        assert len(detector_params >= 10), \
            "list of detector parameters must have length >= 10"
        rMat_d = xfcapi.makeRotMatOfExpMap(detector_params[:3])
        tVec_d = np.ascontiguousarray(detector_params[3:6])
        chi = detector_params[6]
        tVec_s = np.ascontiguousarray(detector_params[7:10])
    return rMat_d, tVec_d, chi, tVec_s
