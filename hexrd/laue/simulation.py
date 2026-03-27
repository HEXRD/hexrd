from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Optional, Sequence

from hexrd.core.instrument.detector import Detector
from hexrd.core.material.crystallography import PlaneData

if TYPE_CHECKING:
    from hexrd.core.instrument.hedm_instrument import HEDMInstrument
import hexrd.core.constants as ct
from hexrd.core.transforms.xfcapi import gvec_to_xy, make_rmat_of_expmap, make_beam_rmat, xy_to_gvec
from hexrd.core import matrixutil as mutil

def simulate_laue_pattern_on_panel(
    detector: Detector,
    crystal_data: PlaneData | Sequence[NDArray[np.float64]],
    minEnergy: float | list[float] = 5.0,
    maxEnergy: float | list[float] = 35.0,
    rmat_s=None,
    tvec_s=None,
    grain_params=None,
    beam_vec=None,
):
    """ """
    if isinstance(crystal_data, PlaneData):
        plane_data = crystal_data

        # grab the expanded list of hkls from plane_data
        hkls = np.hstack(plane_data.getSymHKLs())

        # and the unit plane normals (G-vectors) in CRYSTAL FRAME
        gvec_c = np.dot(plane_data.latVecOps['B'], hkls)

        # Filter out g-vectors going in the wrong direction. `gvec_to_xy()` used
        # to do this, but not anymore.
        to_keep = np.dot(gvec_c.T, detector.bvec) <= 0

        hkls = hkls[:, to_keep]
        gvec_c = gvec_c[:, to_keep]
    elif len(crystal_data) == 2:
        # !!! should clean this up
        hkls = np.array(crystal_data[0])
        bmat = crystal_data[1]
        gvec_c = np.dot(bmat, hkls)
    else:
        raise RuntimeError(f'argument list not understood: {crystal_data=}')
    nhkls_tot: int = hkls.shape[1]

    # parse energy ranges
    # TODO: allow for spectrum parsing
    multipleEnergyRanges = False
    lmin: list[float] = []
    lmax: list[float] = []

    if isinstance(maxEnergy, (list, tuple, np.ndarray)):
        if not isinstance(minEnergy, (list, tuple, np.ndarray)):
            raise TypeError('minEnergy must be array-like if maxEnergy is')
        if len(maxEnergy) != len(minEnergy):
            raise ValueError('maxEnergy and minEnergy must be same length')
        multipleEnergyRanges = True

        for max_energy, min_energy in zip(maxEnergy, minEnergy):
            lmin.append(ct.keVToAngstrom(max_energy))
            lmax.append(ct.keVToAngstrom(min_energy))
    else:
        lmin = ct.keVToAngstrom(maxEnergy)
        lmax = ct.keVToAngstrom(minEnergy)

    # parse grain parameters kwarg
    if grain_params is None:
        grain_params = np.atleast_2d(np.hstack([np.zeros(6), ct.identity_6x1]))
    n_grains = len(grain_params)

    # sample rotation
    if rmat_s is None:
        rmat_s = ct.identity_3x3

    # dummy translation vector... make input
    if tvec_s is None:
        tvec_s = ct.zeros_3

    # beam vector
    if beam_vec is None:
        beam_vec = ct.beam_vec

    # =========================================================================
    # LOOP OVER GRAINS
    # =========================================================================

    # pre-allocate output arrays
    xy_det   = np.full((n_grains, nhkls_tot, 2), np.nan)
    hkls_in  = np.full((n_grains, 3, nhkls_tot), np.nan)
    angles   = np.full((n_grains, nhkls_tot, 2), np.nan)
    dspacing = np.full((n_grains, nhkls_tot),    np.nan)
    energy   = np.full((n_grains, nhkls_tot),    np.nan)

    for iG, gp in enumerate(grain_params):
        rmat_c = make_rmat_of_expmap(gp[:3])
        tvec_c = gp[3:6].reshape(3, 1)
        vInv_s = mutil.vecMVToSymm(gp[6:].reshape(6, 1))

        # stretch them: V^(-1) * R * Gc
        gvec_s_str = np.dot(vInv_s, np.dot(rmat_c, gvec_c))
        ghat_c_str = mutil.unitVector(np.dot(rmat_c.T, gvec_s_str))

        # project
        dpts = gvec_to_xy(
            ghat_c_str.T,
            detector.rmat,
            rmat_s,
            rmat_c,
            detector.tvec,
            tvec_s,
            tvec_c,
            beam_vec=beam_vec,
        )

        # check intersections with detector plane
        canIntersect = ~np.isnan(dpts[:, 0])
        npts_in = sum(canIntersect)

        if np.any(canIntersect):
            dpts = dpts[canIntersect, :].reshape(npts_in, 2)
            dhkl = hkls[:, canIntersect].reshape(3, npts_in)

            rmat_b = make_beam_rmat(beam_vec, ct.eta_vec)
            # back to angles
            tth_eta, gvec_l = xy_to_gvec(
                dpts,
                detector.rmat,
                rmat_s,
                detector.tvec,
                tvec_s,
                tvec_c,
                rmat_b=rmat_b,
            )
            tth_eta = np.vstack(tth_eta).T

            # warp measured points
            if detector.distortion is not None:
                dpts = detector.distortion.apply_inverse(dpts)

            # plane spacings and energies
            dsp = 1.0 / mutil.rowNorm(gvec_s_str[:, canIntersect].T)
            wlen = 2 * dsp * np.sin(0.5 * tth_eta[:, 0])

            # clip to detector panel
            _, on_panel = detector.clip_to_panel(dpts, buffer_edges=True)

            if multipleEnergyRanges:
                validEnergy = np.zeros(len(wlen), dtype=bool)
                for l_min, l_max in zip(lmin, lmax):
                    in_energy_range = np.logical_and(
                        wlen >= l_min, wlen <= l_max
                    )
                    validEnergy = validEnergy | in_energy_range
            else:
                validEnergy = np.logical_and(wlen >= lmin, wlen <= lmax)

            # index for valid reflections
            keepers = np.where(np.logical_and(on_panel, validEnergy))[0]

            # assign output arrays
            xy_det[iG][keepers, :] = dpts[keepers, :]
            hkls_in[iG][:, keepers] = dhkl[:, keepers]
            angles[iG][keepers, :] = tth_eta[keepers, :]
            dspacing[iG, keepers] = dsp[keepers]
            energy[iG, keepers] = ct.keVToAngstrom(wlen[keepers])
    return xy_det, hkls_in, angles, dspacing, energy

def simulate_laue_pattern_on_instrument(
    instrument: HEDMInstrument,
    crystal_data: PlaneData | Sequence[NDArray[np.float64]],
    minEnergy: float = 5.0,
    maxEnergy: float = 35.0,
    rmat_s: Optional[NDArray[np.float64]] = None,
    grain_params: Optional[NDArray[np.float64]] = None,
):
    """
    Simulate Laue diffraction over the instrument.

    Parameters
    ----------
    crystal_data : TYPE
        DESCRIPTION.
    minEnergy : TYPE, optional
        DESCRIPTION. The default is 5..
    maxEnergy : TYPE, optional
        DESCRIPTION. The default is 35..
    rmat_s : TYPE, optional
        DESCRIPTION. The default is None.
    grain_params : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    xy_det, hkls_in, angles, dspacing, energy

    TODO: revisit output; dict, or concatenated list?
    """
    results = dict.fromkeys(instrument.detectors)
    for det_key, panel in instrument.detectors.items():
        results[det_key] = simulate_laue_pattern_on_panel(
            panel,
            crystal_data,
            minEnergy=minEnergy,
            maxEnergy=maxEnergy,
            rmat_s=rmat_s,
            tvec_s=instrument.tvec,
            grain_params=grain_params,
            beam_vec=instrument.beam_vector,
        )
    return results
