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

from hexrd.core import constants
from hexrd.core import matrixutil as mutil

from hexrd.core.material.crystallography import processWavelength

from hexrd.core.transforms import xfcapi

from hexrd.core.deprecation import deprecated


simlp = 'hexrd.hedm.instrument.hedm_instrument.HEDMInstrument.simulate_laue_pattern'

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

validateAngleRanges = xfcapi.validate_angle_ranges

@deprecated(new_func=simlp, removal_date='2026-01-01')
def simulateLauePattern(
    hkls,
    bMat,
    rmat_d,
    tvec_d,
    panel_dims,
    panel_buffer=5,
    minEnergy=8,
    maxEnergy=24,
    rmat_s=np.eye(3),
    grain_params=None,
    distortion=None,
    beamVec=None,
):

    if beamVec is None:
        beamVec = constants.beam_vec

    # parse energy ranges
    multipleEnergyRanges = False
    if hasattr(maxEnergy, '__len__'):
        assert len(maxEnergy) == len(
            minEnergy
        ), 'energy cutoff ranges must have the same length'
        multipleEnergyRanges = True
        lmin = [processWavelength(e) for e in maxEnergy]
        lmax = [processWavelength(e) for e in minEnergy]
    else:
        lmin = processWavelength(maxEnergy)
        lmax = processWavelength(minEnergy)

    # process crystal rmats and inverse stretches
    if grain_params is None:
        grain_params = np.atleast_2d(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )

    n_grains = len(grain_params)

    # dummy translation vector... make input
    tvec_s = np.zeros((3, 1))

    # number of hkls
    nhkls_tot = hkls.shape[1]

    # unit G-vectors in crystal frame
    ghat_c = mutil.unitVector(np.dot(bMat, hkls))

    # pre-allocate output arrays
    xy_det = np.nan * np.ones((n_grains, nhkls_tot, 2))
    hkls_in = np.nan * np.ones((n_grains, 3, nhkls_tot))
    angles = np.nan * np.ones((n_grains, nhkls_tot, 2))
    dspacing = np.nan * np.ones((n_grains, nhkls_tot))
    energy = np.nan * np.ones((n_grains, nhkls_tot))

    """
    LOOP OVER GRAINS
    """

    for iG, gp in enumerate(grain_params):
        rmat_c = xfcapi.make_rmat_of_expmap(gp[:3])
        tvec_c = gp[3:6].reshape(3, 1)
        vInv_s = mutil.vecMVToSymm(gp[6:].reshape(6, 1))

        # stretch them: V^(-1) * R * Gc
        ghat_s_str = mutil.unitVector(np.dot(vInv_s, np.dot(rmat_c, ghat_c)))
        ghat_c_str = np.dot(rmat_c.T, ghat_s_str)

        # project
        dpts = xfcapi.gvec_to_xy(
            ghat_c_str.T,
            rmat_d,
            rmat_s,
            rmat_c,
            tvec_d,
            tvec_s,
            tvec_c,
            beam_vec=beamVec,
        ).T

        # check intersections with detector plane
        canIntersect = ~np.isnan(dpts[0, :])
        npts_in = sum(canIntersect)

        if np.any(canIntersect):
            dpts = dpts[:, canIntersect].reshape(2, npts_in)
            dhkl = hkls[:, canIntersect].reshape(3, npts_in)

            rmat_b = xfcapi.make_beam_rmat(beamVec, constants.eta_vec)

            # back to angles
            tth_eta, gvec_l = xfcapi.xy_to_gvec(
                dpts.T, rmat_d, rmat_s, tvec_d, tvec_s, tvec_c, rmat_b=rmat_b
            )
            tth_eta = np.vstack(tth_eta).T

            # warp measured points
            if distortion is not None:
                dpts = distortion.apply_inverse(dpts)

            # plane spacings and energies
            dsp = 1.0 / mutil.columnNorm(np.dot(bMat, dhkl))
            wlen = 2 * dsp * np.sin(0.5 * tth_eta[:, 0])

            # find on spatial extent of detector
            xTest = np.logical_and(
                dpts[0, :] >= -0.5 * panel_dims[1] + panel_buffer,
                dpts[0, :] <= 0.5 * panel_dims[1] - panel_buffer,
            )
            yTest = np.logical_and(
                dpts[1, :] >= -0.5 * panel_dims[0] + panel_buffer,
                dpts[1, :] <= 0.5 * panel_dims[0] - panel_buffer,
            )

            onDetector = np.logical_and(xTest, yTest)
            if multipleEnergyRanges:
                validEnergy = np.zeros(len(wlen), dtype=bool)
                for i in range(len(lmin)):
                    validEnergy = validEnergy | np.logical_and(
                        wlen >= lmin[i], wlen <= lmax[i]
                    )
            else:
                validEnergy = np.logical_and(wlen >= lmin, wlen <= lmax)

            # index for valid reflections
            keepers = np.where(np.logical_and(onDetector, validEnergy))[0]

            # assign output arrays
            xy_det[iG][keepers, :] = dpts[:, keepers].T
            hkls_in[iG][:, keepers] = dhkl[:, keepers]
            angles[iG][keepers, :] = tth_eta[keepers, :]
            dspacing[iG, keepers] = dsp[keepers]
            energy[iG, keepers] = processWavelength(wlen[keepers])
    return xy_det, hkls_in, angles, dspacing, energy
