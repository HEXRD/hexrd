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


from typing import Optional, Union, Any, Generator
from hexrd.core.material.crystallography import PlaneData
from hexrd.core.distortion.distortionabc import DistortionABC

import numba
import numpy as np
import numba

from hexrd.core import constants
from hexrd.core import matrixutil as mutil
from hexrd.core import rotations as rot
from hexrd.core import gridutil as gutil

from hexrd.hed.xrdutil.utils import _project_on_detector_plane
from hexrd.core.material.crystallography import processWavelength, PlaneData

from hexrd.core.transforms import xfcapi
from hexrd.core.valunits import valWUnit

from hexrd.core import distortion as distortion_pkg

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

# =============================================================================
# CLASSES
# =============================================================================


class EtaOmeMaps(object):
    """
    find-orientations loads pickled eta-ome data, but CollapseOmeEta is not
    pickleable, because it holds a list of ReadGE, each of which holds a
    reference to an open file object, which is not pickleable.
    """

    def __init__(self, ome_eta_archive: str):
        ome_eta: np.ndarray = np.load(ome_eta_archive, allow_pickle=True)

        planeData_args = ome_eta['planeData_args']
        planeData_hkls = ome_eta['planeData_hkls']
        self.planeData = PlaneData(planeData_hkls, *planeData_args)
        self.planeData.exclusions = ome_eta['planeData_excl']
        self.dataStore = ome_eta['dataStore']
        self.iHKLList = ome_eta['iHKLList']
        self.etaEdges = ome_eta['etaEdges']
        self.omeEdges = ome_eta['omeEdges']
        self.etas = ome_eta['etas']
        self.omegas = ome_eta['omegas']

    def save_eta_ome_maps(self, filename: str) -> None:
        """
        eta_ome.dataStore
        eta_ome.planeData
        eta_ome.iHKLList
        eta_ome.etaEdges
        eta_ome.omeEdges
        eta_ome.etas
        eta_ome.omegas
        """
        args = np.array(self.planeData.getParams(), dtype=object)[:4]
        args[2] = valWUnit('wavelength', 'length', args[2], 'angstrom')
        hkls = np.vstack([i['hkl'] for i in self.planeData.hklDataList]).T
        save_dict = {
            'dataStore': self.dataStore,
            'etas': self.etas,
            'etaEdges': self.etaEdges,
            'iHKLList': self.iHKLList,
            'omegas': self.omegas,
            'omeEdges': self.omeEdges,
            'planeData_args': args,
            'planeData_hkls': hkls,
            'planeData_excl': self.planeData.exclusions,
        }
        np.savez_compressed(filename, **save_dict)


# =============================================================================
# FUNCTIONS
# =============================================================================


def _zproject(x: np.ndarray, y: np.ndarray):
    return np.cos(x) * np.sin(y) - np.sin(x) * np.cos(y)


def zproject_sph_angles(
    invecs: np.ndarray,
    chi: float = 0.0,
    method: str = 'stereographic',
    source: str = 'd',
    use_mask: bool = False,
    invert_z: bool = False,
    rmat: Optional[np.ndarray] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Projects spherical angles to 2-d mapping.

    Parameters
    ----------
    invec : array_like
        The (n, 3) array of input points, interpreted via the 'source' kwarg.
    chi : scalar, optional
        The inclination angle of the sample frame. The default is 0..
    method : str, optional
        Mapping type spec, either 'stereographic' or 'equal-area'.
        The default is 'stereographic'.
    source : str, optional
        The type specifier of the input vectors, either 'd', 'q', or 'g'.
            'd' signifies unit diffraction vectors as (2theta, eta, omega),
            'q' specifies unit scattering vectors as (2theta, eta, omega),
            'g' specifies unit vectors in the sample frame as (x, y, z).
        The default is 'd'.
    use_mask : bool, optional
        If True, trim points not on the +z hemishpere (polar angles > 90).
        The default is False.
    invert_z : bool, optional
        If True, invert the Z-coordinates of the unit vectors calculated from
        the input angles. The default is False.
    rmat : numpy.ndarry, shape=(3, 3), optional
        Array representing a change of basis (rotation) to appy to the
        calculated unit vectors. The default is None.

    Raises
    ------
    RuntimeError
        If method not in ('stereographic', 'equal-area').

    Returns
    -------
    numpy.ndarray or tuple
        If use_mask = False, then the array of n mapped input points with shape
        (n, 2).  If use_mask = True, then the first element is the ndarray of
        mapped points with shape (<=n, 2), and the second is a bool array with
        shape (n,) marking the point that fell on the upper hemishpere.
        .

    Notes
    -----
    CAVEAT: +Z axis projections only!!!
    TODO: check mask application.
    """
    assert isinstance(source, str), "source kwarg must be a string"

    invecs = np.atleast_2d(invecs)
    if source.lower() == 'd':
        spts_s = xfcapi.angles_to_dvec(invecs, chi=chi)
    elif source.lower() == 'q':
        spts_s = xfcapi.angles_to_gvec(invecs, chi=chi)
    elif source.lower() == 'g':
        spts_s = invecs

    if rmat is not None:
        spts_s = np.dot(spts_s, rmat.T)

    if invert_z:
        spts_s[:, 2] = -spts_s[:, 2]

    # filter based on hemisphere
    if use_mask:
        pzi = spts_s[:, 2] <= 0
        spts_s = spts_s[pzi, :]

    if method.lower() == 'stereographic':
        ppts = np.vstack(
            [
                spts_s[:, 0] / (1.0 - spts_s[:, 2]),
                spts_s[:, 1] / (1.0 - spts_s[:, 2]),
            ]
        ).T
    elif method.lower() == 'equal-area':
        chords = spts_s + np.tile([0, 0, 1], (len(spts_s), 1))
        scl = np.tile(mutil.rowNorm(chords), (2, 1)).T
        ucrd = mutil.unitVector(
            np.hstack([chords[:, :2], np.zeros((len(spts_s), 1))]).T
        )

        ppts = ucrd[:2, :].T * scl
    else:
        raise RuntimeError(f"method '{method}' not recognized")

    if use_mask:
        return ppts, pzi
    else:
        return ppts


def make_polar_net(
    ndiv: int = 24, projection: str = 'stereographic', max_angle: float = 120.0
) -> np.ndarray:
    """
    TODO: options for generating net boundaries; fixed to Z proj.
    """
    ndiv_tth = int(np.floor(0.5 * ndiv)) + 1
    wtths = np.radians(np.linspace(0, 1, num=ndiv_tth, endpoint=True) * max_angle)
    wetas = np.radians(np.linspace(-1, 1, num=ndiv + 1, endpoint=True) * 180.0)
    weta_gen = np.radians(np.linspace(-1, 1, num=181, endpoint=True) * 180.0)
    pts = []
    for eta in wetas:
        net_ang = np.vstack([[wtths[0], wtths[-1]], np.tile(eta, 2), np.zeros(2)]).T
        pts.append(zproject_sph_angles(net_ang, method=projection, source='d'))
        pts.append(np.nan * np.ones((1, 2)))
    for tth in wtths[1:]:
        net_ang = np.vstack(
            [tth * np.ones_like(weta_gen), weta_gen, np.zeros_like(weta_gen)]
        ).T
        pts.append(zproject_sph_angles(net_ang, method=projection, source='d'))
        pts.append(nans_1x2)

    return np.vstack(pts)


validateAngleRanges = xfcapi.validate_angle_ranges


@deprecated(removal_date='2025-01-01')
def simulateOmeEtaMaps(
    omeEdges,
    etaEdges,
    planeData,
    expMaps,
    chi=0.0,
    etaTol=None,
    omeTol=None,
    etaRanges=None,
    omeRanges=None,
    bVec=constants.beam_vec,
    eVec=constants.eta_vec,
    vInv=constants.identity_6x1,
):
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
        DESCRIPTION. The default is [0, 0, -1].
    eVec : TYPE, optional
        DESCRIPTION. The default is [1, 0, 0].
    vInv : TYPE, optional
        DESCRIPTION. The default is [1, 1, 1, 0, 0, 0].

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
        omeRanges = [
            [omeMin, omeMax],
        ]

    if etaRanges is None:
        etaRanges = [
            [etaMin, etaMax],
        ]

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

    i_dil, j_dil = np.meshgrid(
        np.arange(-dpix_ome, dpix_ome + 1), np.arange(-dpix_eta, dpix_eta + 1)
    )

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
            rMat_c = xfcapi.make_rmat_of_expmap(expMaps[iOr, :])
            angList = np.vstack(
                xfcapi.oscill_angles_of_hkls(
                    these_hkls,
                    chi,
                    rMat_c,
                    bMat,
                    wlen,
                    beam_vec=bVec,
                    eta_vec=eVec,
                    v_inv=vInv,
                )
            )
            if not np.all(np.isnan(angList)):
                #
                angList[:, 1] = rot.mapAngle(
                    angList[:, 1], [etaEdges[0], etaEdges[0] + 2 * np.pi]
                )
                angList[:, 2] = rot.mapAngle(
                    angList[:, 2], [omeEdges[0], omeEdges[0] + 2 * np.pi]
                )
                #
                # do eta ranges
                angMask_eta = np.zeros(len(angList), dtype=bool)
                for etas in etaRanges:
                    angMask_eta = np.logical_or(
                        angMask_eta,
                        xfcapi.validate_angle_ranges(angList[:, 1], etas[0], etas[1]),
                    )

                # do omega ranges
                ccw = True
                angMask_ome = np.zeros(len(angList), dtype=bool)
                for omes in omeRanges:
                    if omes[1] - omes[0] < 0:
                        ccw = False
                    angMask_ome = np.logical_or(
                        angMask_ome,
                        xfcapi.validate_angle_ranges(
                            angList[:, 2], omes[0], omes[1], ccw=ccw
                        ),
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
                            i_sup = omeIndices[culledOmeIdx] + np.array(
                                [i_dil.flatten()], dtype=int
                            )
                            j_sup = etaIndices[culledEtaIdx] + np.array(
                                [j_dil.flatten()], dtype=int
                            )

                            # catch shit that falls off detector...
                            # maybe make this fancy enough to wrap at 2pi?
                            idx_mask = np.logical_and(
                                np.logical_and(i_sup >= 0, i_sup < i_max),
                                np.logical_and(j_sup >= 0, j_sup < j_max),
                            )
                            eta_ome[iHKL, i_sup[idx_mask], j_sup[idx_mask]] = 1.0
                        else:
                            eta_ome[
                                iHKL,
                                omeIndices[culledOmeIdx],
                                etaIndices[culledEtaIdx],
                            ] = 1.0
    return eta_ome


def _fetch_hkls_from_planedata(pd: PlaneData):
    return np.hstack(pd.getSymHKLs(withID=True)).T


def _filter_hkls_eta_ome(
    hkls: np.ndarray,
    angles: np.ndarray,
    eta_range: list[tuple[float]],
    ome_range: list[tuple[float]],
    return_mask: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    given a set of hkls and angles, filter them by the
    eta and omega ranges
    """
    angMask_eta = np.zeros(len(angles), dtype=bool)
    for etas in eta_range:
        angMask_eta = np.logical_or(
            angMask_eta,
            xfcapi.validate_angle_ranges(angles[:, 1], etas[0], etas[1]),
        )

    ccw = True
    angMask_ome = np.zeros(len(angles), dtype=bool)
    for omes in ome_range:
        if omes[1] - omes[0] < 0:
            ccw = False
        angMask_ome = np.logical_or(
            angMask_ome,
            xfcapi.validate_angle_ranges(angles[:, 2], omes[0], omes[1], ccw=ccw),
        )

    angMask = np.logical_and(angMask_eta, angMask_ome)

    allAngs = angles[angMask, :]
    allHKLs = np.vstack([hkls, hkls])[angMask, :]

    if return_mask:
        return allAngs, allHKLs, angMask
    else:
        return allAngs, allHKLs


def _dvec_to_angs(
    dvecs: np.ndarray, bvec: np.ndarray, evec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    convert diffraction vectors to (tth, eta)
    angles in the 'eta' frame
    dvecs is assumed to have (nx3) shape
    """
    num = dvecs.shape[0]
    exb = np.cross(evec, bvec)
    exb = exb / np.linalg.norm(exb)
    bxexb = np.cross(bvec, exb)
    bxexb = bxexb / np.linalg.norm(bxexb)

    dp = np.dot(bvec, dvecs.T)
    dp[np.abs(dp) > 1.0] = np.sign(dp[np.abs(dp) > 1.0])
    tth = np.arccos(dp)

    dvecs_p = dvecs - np.tile(dp, [3, 1]).T * np.tile(bvec, [num, 1])

    dpx = np.dot(bxexb, dvecs_p.T)
    dpy = np.dot(exb, dvecs_p.T)
    eta = np.arctan2(dpy, dpx)

    return tth, eta


def simulateGVecs(
    pd: PlaneData,
    detector_params: np.ndarray,
    grain_params: np.ndarray,
    ome_range: list[tuple[float]] = [
        (-np.pi, np.pi),
    ],
    ome_period: tuple[float] = (-np.pi, np.pi),
    eta_range: list[tuple[float]] = [
        (-np.pi, np.pi),
    ],
    panel_dims: list[tuple[float]] = [(-204.8, -204.8), (204.8, 204.8)],
    pixel_pitch: tuple[float] = (0.2, 0.2),
    distortion: DistortionABC = None,
    beam_vector: np.ndarray = constants.beam_vec,
    energy_correction: Union[dict, None] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    rMat_d = xfcapi.make_detector_rmat(detector_params[:3])
    tVec_d = np.ascontiguousarray(detector_params[3:6])
    chi = detector_params[6]
    tVec_s = np.ascontiguousarray(detector_params[7:10])
    rMat_c = xfcapi.make_rmat_of_expmap(grain_params[:3])
    tVec_c = np.ascontiguousarray(grain_params[3:6])
    vInv_s = np.ascontiguousarray(grain_params[6:12])
    beam_vector = np.ascontiguousarray(beam_vector)

    # Apply an energy correction according to grain position
    corrected_wlen = apply_correction_to_wavelength(
        wlen,
        energy_correction,
        tVec_s,
        tVec_c,
    )

    # first find valid G-vectors
    angList = np.vstack(
        xfcapi.oscill_angles_of_hkls(
            full_hkls[:, 1:],
            chi,
            rMat_c,
            bMat,
            corrected_wlen,
            v_inv=vInv_s,
            beam_vec=beam_vector,
        )
    )
    allAngs, allHKLs = _filter_hkls_eta_ome(full_hkls, angList, eta_range, ome_range)

    if len(allAngs) == 0:
        valid_ids = []
        valid_hkl = []
        valid_ang = []
        valid_xy = []
        ang_ps = []
    else:
        # ??? preallocate for speed?
        det_xy, rMat_ss, _ = _project_on_detector_plane(
            allAngs,
            rMat_d,
            rMat_c,
            chi,
            tVec_d,
            tVec_c,
            tVec_s,
            distortion,
            beamVec=beam_vector,
        )

        on_panel = np.logical_and(
            np.logical_and(
                det_xy[:, 0] >= panel_dims[0][0],
                det_xy[:, 0] <= panel_dims[1][0],
            ),
            np.logical_and(
                det_xy[:, 1] >= panel_dims[0][1],
                det_xy[:, 1] <= panel_dims[1][1],
            ),
        )

        op_idx = np.where(on_panel)[0]

        valid_ang = allAngs[op_idx, :]
        valid_ang[:, 2] = xfcapi.mapAngle(valid_ang[:, 2], ome_period)
        valid_ids = allHKLs[op_idx, 0]
        valid_hkl = allHKLs[op_idx, 1:]
        valid_xy = det_xy[op_idx, :]
        ang_ps = angularPixelSize(
            valid_xy,
            pixel_pitch,
            rMat_d,
            # Provide only the first sample rotation matrix to angularPixelSize
            # Perhaps this is something that can be improved in the future?
            rMat_ss[0],
            tVec_d,
            tVec_s,
            tVec_c,
            distortion=distortion,
            beamVec=beam_vector,
        )

    return valid_ids, valid_hkl, valid_ang, valid_xy, ang_ps


@numba.njit(nogil=True, cache=True)
def _expand_pixels(
    original: np.ndarray, w: float, h: float, result: np.ndarray
) -> np.ndarray:
    hw = 0.5 * w
    hh = 0.5 * h
    for el in range(len(original)):
        x, y = original[el, 0], original[el, 1]
        result[el * 4 + 0, 0] = x - hw
        result[el * 4 + 0, 1] = y - hh
        result[el * 4 + 1, 0] = x + hw
        result[el * 4 + 1, 1] = y - hh
        result[el * 4 + 2, 0] = x + hw
        result[el * 4 + 2, 1] = y + hh
        result[el * 4 + 3, 0] = x - hw
        result[el * 4 + 3, 1] = y + hh

    return result


@numba.njit(nogil=True, cache=True)
def _compute_max(tth: np.ndarray, eta: np.ndarray, result: np.ndarray) -> np.ndarray:
    period = 2.0 * np.pi
    hperiod = np.pi
    for el in range(0, len(tth), 4):
        max_tth = np.abs(tth[el + 0] - tth[el + 3])
        eta_diff = eta[el + 0] - eta[el + 3]
        max_eta = np.abs(np.remainder(eta_diff + hperiod, period) - hperiod)
        for i in range(3):
            curr_tth = np.abs(tth[el + i] - tth[el + i + 1])
            eta_diff = eta[el + i] - eta[el + i + 1]
            curr_eta = np.abs(np.remainder(eta_diff + hperiod, period) - hperiod)
            max_tth = np.maximum(curr_tth, max_tth)
            max_eta = np.maximum(curr_eta, max_eta)
        result[el // 4, 0] = max_tth
        result[el // 4, 1] = max_eta

    return result


def angularPixelSize(
    xy_det: np.ndarray,
    xy_pixelPitch: tuple[float],
    rMat_d: np.ndarray,
    rMat_s: np.ndarray,
    tVec_d: np.ndarray,
    tVec_s: np.ndarray,
    tVec_c: np.ndarray,
    distortion: DistortionABC = None,
    beamVec: np.ndarray = None,
    etaVec: np.ndarray = None,
) -> np.ndarray:
    """
    Calculate angular pixel sizes on a detector.

    * choices to beam vector and eta vector specs have been supressed
    * assumes xy_det in UNWARPED configuration
    """
    xy_det = np.atleast_2d(xy_det)
    if distortion is not None:  # !!! check this logic
        xy_det = distortion.apply(xy_det)
    if beamVec is None:
        beamVec = constants.beam_vec
    if etaVec is None:
        etaVec = constants.eta_vec

    # Verify that rMat_s is only 2D (a single matrix).
    # Arrays of matrices were previously provided, which `xy_to_gvec`
    # cannot currently handle.
    if rMat_s.ndim != 2:
        msg = (
            f'rMat_s should have 2 dimensions, but has {rMat_s.ndim} '
            'dimensions instead'
        )
        raise ValueError(msg)

    xy_expanded = np.empty((len(xy_det) * 4, 2), dtype=xy_det.dtype)
    xy_expanded = _expand_pixels(
        xy_det, xy_pixelPitch[0], xy_pixelPitch[1], xy_expanded
    )

    rmat_b = xfcapi.make_beam_rmat(beamVec, etaVec)

    gvec_space, _ = xfcapi.xy_to_gvec(
        xy_expanded,
        rMat_d,
        rMat_s,
        tVec_d,
        tVec_s,
        tVec_c,
        rmat_b=rmat_b,
    )
    result = np.empty_like(xy_det)
    return _compute_max(gvec_space[0], gvec_space[1], result)


def make_reflection_patches(
    instr_cfg: dict[str, Any],
    tth_eta: np.ndarray,
    ang_pixel_size: np.ndarray,
    omega: Optional[np.ndarray] = None,
    tth_tol: float = 0.2,
    eta_tol: float = 1.0,
    rmat_c: np.ndarray = np.eye(3),
    tvec_c: np.ndarray = np.zeros((3, 1)),
    npdiv: int = 1,
    quiet: bool = False,  # TODO: Remove this parameter - it isn't used
    compute_areas_func: np.ndarray = gutil.compute_areas,
) -> Generator[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    None,
    None,
]:
    """Make angular patches on a detector.

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

    # detector quantities
    rmat_d = xfcapi.make_rmat_of_expmap(
        np.r_[instr_cfg['detector']['transform']['tilt']]
    )
    tvec_d = np.r_[instr_cfg['detector']['transform']['translation']]
    pixel_size = instr_cfg['detector']['pixels']['size']

    frame_nrows = instr_cfg['detector']['pixels']['rows']
    frame_ncols = instr_cfg['detector']['pixels']['columns']

    panel_dims = (
        -0.5 * np.r_[frame_ncols * pixel_size[1], frame_nrows * pixel_size[0]],
        0.5 * np.r_[frame_ncols * pixel_size[1], frame_nrows * pixel_size[0]],
    )
    row_edges = np.arange(frame_nrows + 1)[::-1] * pixel_size[1] + panel_dims[0][1]
    col_edges = np.arange(frame_ncols + 1) * pixel_size[0] + panel_dims[0][0]

    # handle distortion
    distortion = None
    if distortion_key in instr_cfg['detector']:
        distortion_cfg = instr_cfg['detector'][distortion_key]
        if distortion_cfg is not None:
            try:
                func_name = distortion_cfg['function_name']
                dparams = distortion_cfg['parameters']
                distortion = distortion_pkg.get_mapping(func_name, dparams)
            except KeyError:
                raise RuntimeError("problem with distortion specification")

    # sample frame
    chi = instr_cfg['oscillation_stage']['chi']
    tvec_s = np.r_[instr_cfg['oscillation_stage']['translation']]
    bvec = np.r_[instr_cfg['beam']['vector']]

    # data to loop
    # ??? WOULD IT BE CHEAPER TO CARRY ZEROS OR USE CONDITIONAL?
    if omega is None:
        full_angs = np.hstack([tth_eta, np.zeros((len(tth_eta), 1))])
    else:
        full_angs = np.hstack([tth_eta, omega.reshape(len(tth_eta), 1)])

    for angs, pix in zip(full_angs, ang_pixel_size):
        # calculate bin edges for patch based on local angular pixel size
        # tth
        ntths, tth_edges = gutil.make_tolerance_grid(
            bin_width=np.degrees(pix[0]),
            window_width=tth_tol,
            num_subdivisions=npdiv,
        )

        # eta
        netas, eta_edges = gutil.make_tolerance_grid(
            bin_width=np.degrees(pix[1]),
            window_width=eta_tol,
            num_subdivisions=npdiv,
        )

        # FOR ANGULAR MESH
        conn = gutil.cellConnectivity(netas, ntths, origin='ll')

        # meshgrid args are (cols, rows), a.k.a (fast, slow)
        m_tth, m_eta = np.meshgrid(tth_edges, eta_edges)
        npts_patch = m_tth.size

        # calculate the patch XY coords from the (tth, eta) angles
        # !!! will CHEAT and ignore the small perturbation the different
        #     omega angle values causes and simply use the central value
        gVec_angs_vtx = np.tile(angs, (npts_patch, 1)) + np.radians(
            np.vstack([m_tth.flatten(), m_eta.flatten(), np.zeros(npts_patch)]).T
        )

        xy_eval_vtx, _, _ = _project_on_detector_plane(
            gVec_angs_vtx,
            rmat_d,
            rmat_c,
            chi,
            tvec_d,
            tvec_c,
            tvec_s,
            distortion,
            beamVec=bvec,
        )

        areas = compute_areas_func(xy_eval_vtx, conn)

        # EVALUATION POINTS
        # !!! for lack of a better option will use centroids
        tth_eta_cen = gutil.cellCentroids(np.atleast_2d(gVec_angs_vtx[:, :2]), conn)

        gVec_angs = np.hstack([tth_eta_cen, np.tile(angs[2], (len(tth_eta_cen), 1))])

        xy_eval, _, _ = _project_on_detector_plane(
            gVec_angs,
            rmat_d,
            rmat_c,
            chi,
            tvec_d,
            tvec_c,
            tvec_s,
            distortion,
            beamVec=bvec,
        )

        row_indices = gutil.cellIndices(row_edges, xy_eval[:, 1])
        col_indices = gutil.cellIndices(col_edges, xy_eval[:, 0])

        yield (
            (
                (
                    gVec_angs_vtx[:, 0].reshape(m_tth.shape),
                    gVec_angs_vtx[:, 1].reshape(m_tth.shape),
                ),
                (
                    xy_eval_vtx[:, 0].reshape(m_tth.shape),
                    xy_eval_vtx[:, 1].reshape(m_tth.shape),
                ),
                conn,
                areas.reshape(netas, ntths),
                (
                    xy_eval[:, 0].reshape(netas, ntths),
                    xy_eval[:, 1].reshape(netas, ntths),
                ),
                (
                    row_indices.reshape(netas, ntths),
                    col_indices.reshape(netas, ntths),
                ),
            )
        )


def extract_detector_transformation(
    detector_params: Union[dict[str, Any], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
        rMat_d = xfcapi.make_rmat_of_expmap(
            np.array(detector_params['detector']['transform']['tilt'])
        )
        tVec_d = np.r_[detector_params['detector']['transform']['translation']]
        chi = detector_params['oscillation_stage']['chi']
        tVec_s = np.r_[detector_params['oscillation_stage']['translation']]
    else:
        assert len(
            detector_params >= 10
        ), "list of detector parameters must have length >= 10"
        rMat_d = xfcapi.make_rmat_of_expmap(detector_params[:3])
        tVec_d = np.ascontiguousarray(detector_params[3:6])
        chi = detector_params[6]
        tVec_s = np.ascontiguousarray(detector_params[7:10])
    return rMat_d, tVec_d, chi, tVec_s


def apply_correction_to_wavelength(
    wavelength: float,
    energy_correction: Union[dict, None],
    tvec_s: np.ndarray,
    tvec_c: np.ndarray,
) -> float:
    """Apply an energy correction to the wavelength according to grain position

    The energy correction dict appears as follows:

        {
            # The beam energy gradient center, in millimeters,
            # along the specified axis
            'intercept': 0.0,

            # The slope of the beam energy gradient along the
            # specified axis, in eV/mm.
            'slope': 0.0,

            # The specified axis for the beam energy gradient,
            # either 'x' or 'y'.
            'axis': 'y',
        }

    If the energy_correction dict is `None`, then no correction
    is performed.
    """
    if not energy_correction:
        # No correction
        return wavelength

    # 'c' here is the conversion factor between keV and angstrom
    c = constants.keVToAngstrom(1)

    ind = 1 if energy_correction['axis'] == 'y' else 0

    # Correct wavelength according to grain position. Position is in mm.
    position = tvec_c[ind] + tvec_s[ind] - energy_correction['intercept']

    # The slope is in eV/mm. Convert to keV.
    adjustment = position * energy_correction['slope'] / 1e3

    # Convert to keV, apply the adjustment, and then convert back to wavelength
    return c / (c / wavelength + adjustment)
