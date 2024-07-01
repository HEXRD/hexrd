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


from typing import Optional, Union, Tuple

import numba
import numpy as np

from hexrd import constants
from hexrd import matrixutil as mutil
from hexrd import gridutil as gutil

from hexrd.material.crystallography import processWavelength, PlaneData

from hexrd.transforms import xf
from hexrd.transforms import xfcapi
from hexrd.valunits import valWUnit

from hexrd import distortion as distortion_pkg


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

    def save(self, filename: str) -> None:
        self.save_eta_ome_maps(self, filename)

    @staticmethod
    def save_eta_ome_maps(eta_ome, filename: str) -> None:
        """
        eta_ome.dataStore
        eta_ome.planeData
        eta_ome.iHKLList
        eta_ome.etaEdges
        eta_ome.omeEdges
        eta_ome.etas
        eta_ome.omegas
        """
        args = np.array(eta_ome.planeData.getParams(), dtype=object)[:4]
        args[2] = valWUnit('wavelength', 'length', args[2], 'angstrom')
        hkls = np.vstack([i['hkl'] for i in eta_ome.planeData.hklDataList]).T
        save_dict = {
            'dataStore': eta_ome.dataStore,
            'etas': eta_ome.etas,
            'etaEdges': eta_ome.etaEdges,
            'iHKLList': eta_ome.iHKLList,
            'omegas': eta_ome.omegas,
            'omeEdges': eta_ome.omeEdges,
            'planeData_args': args,
            'planeData_hkls': hkls,
            'planeData_excl': eta_ome.planeData.exclusions,
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
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        spts_s = xfcapi.anglesToDVec(invecs, chi=chi)
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
    npts_s = len(spts_s)

    if method.lower() == 'stereographic':
        ppts = np.vstack(
            [
                spts_s[:, 0] / (1.0 - spts_s[:, 2]),
                spts_s[:, 1] / (1.0 - spts_s[:, 2]),
            ]
        ).T
    elif method.lower() == 'equal-area':
        chords = spts_s + np.tile([0, 0, 1], (npts_s, 1))
        scl = np.tile(xfcapi.rowNorm(chords), (2, 1)).T
        ucrd = mutil.unitVector(
            np.hstack([chords[:, :2], np.zeros((len(spts_s), 1))]).T
        )

        ppts = ucrd[:2, :].T * scl
    else:
        raise RuntimeError("method '%s' not recognized" % method)

    if use_mask:
        return ppts, pzi
    else:
        return ppts


def make_polar_net(ndiv=24, projection='stereographic', max_angle=120.0):
    """
    TODO: options for generating net boundaries; fixed to Z proj.
    """
    ndiv_tth = int(np.floor(0.5 * ndiv)) + 1
    wtths = np.radians(
        np.linspace(0, 1, num=ndiv_tth, endpoint=True) * max_angle
    )
    wetas = np.radians(np.linspace(-1, 1, num=ndiv + 1, endpoint=True) * 180.0)
    weta_gen = np.radians(np.linspace(-1, 1, num=181, endpoint=True) * 180.0)
    pts = []
    for eta in wetas:
        net_angs = np.vstack(
            [[wtths[0], wtths[-1]], np.tile(eta, 2), np.zeros(2)]
        ).T
        projp = zproject_sph_angles(net_angs, method=projection, source='d')
        pts.append(projp)
        pts.append(np.nan * np.ones((1, 2)))
    for tth in wtths[1:]:
        net_angs = np.vstack(
            [tth * np.ones_like(weta_gen), weta_gen, np.zeros_like(weta_gen)]
        ).T
        projp = zproject_sph_angles(net_angs, method=projection, source='d')
        pts.append(projp)
        pts.append(nans_1x2)
    '''
    # old method
    for tth in wtths:
        net_angs = np.vstack([tth*np.ones_like(wetas),
                              wetas,
                              piby2*np.ones_like(wetas)]).T
        projp = zproject_sph_angles(net_angs, method=projection)
        pts.append(projp)
    '''
    pts = np.vstack(pts)
    return pts


def validateAngleRanges(angList, startAngs, stopAngs, ccw=True):
    """
    Indetify angles that fall within specified ranges.

    A better way to go.  find out if an angle is in the range
    CCW or CW from start to stop

    There is, of course an ambigutiy if the start and stop angle are
    the same; we treat them as implying 2*pi
    """
    angList = np.atleast_1d(angList).flatten()  # needs to have len
    startAngs = np.atleast_1d(startAngs).flatten()  # needs to have len
    stopAngs = np.atleast_1d(stopAngs).flatten()  # needs to have len

    n_ranges = len(startAngs)
    assert (
        len(stopAngs) == n_ranges
    ), "length of min and max angular limits must match!"

    # to avoid warnings in >=, <= later down, mark nans;
    # need these to trick output to False in the case of nan input
    nan_mask = np.isnan(angList)

    reflInRange = np.zeros(angList.shape, dtype=bool)

    # bin length for chunking
    binLen = np.pi / 2.0

    # in plane vectors defining wedges
    x0 = np.vstack([np.cos(startAngs), np.sin(startAngs)])
    x1 = np.vstack([np.cos(stopAngs), np.sin(stopAngs)])

    # dot products
    dp = np.sum(x0 * x1, axis=0)
    if np.any(dp >= 1.0 - sqrt_epsf) and n_ranges > 1:
        # ambiguous case
        raise RuntimeError(
            "Improper usage; "
            + "at least one of your ranges is alread 360 degrees!"
        )
    elif dp[0] >= 1.0 - sqrt_epsf and n_ranges == 1:
        # trivial case!
        reflInRange = np.ones(angList.shape, dtype=bool)
        reflInRange[nan_mask] = False
    else:
        # solve for arc lengths
        # ...note: no zeros should have made it here
        a = x0[0, :] * x1[1, :] - x0[1, :] * x1[0, :]
        b = x0[0, :] * x1[0, :] + x0[1, :] * x1[1, :]
        phi = np.arctan2(b, a)

        arclen = 0.5 * np.pi - phi  # these are clockwise
        cw_phis = arclen < 0
        arclen[cw_phis] = 2 * np.pi + arclen[cw_phis]  # all positive (CW) now
        if not ccw:
            arclen = 2 * np.pi - arclen

        if sum(arclen) > 2 * np.pi:
            raise RuntimeWarning(
                "Specified angle ranges sum to > 360 degrees, "
                + "which is suspect..."
            )

        # check that there are no more thandp = np.zeros(n_ranges)
        for i in range(n_ranges):
            # number or subranges using 'binLen'
            numSubranges = int(np.ceil(arclen[i] / binLen))

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
                [startAngs[i] + binLen * j for j in range(numSubranges)]
                + [startAngs[i] + binLen * (numSubranges - 1) + finalBinLen]
            )

            for k in range(numSubranges):
                zStart = _zproject(angList, subRanges[k])
                zStop = _zproject(angList, subRanges[k + 1])
                if ccw:
                    zStart[nan_mask] = 999.0
                    zStop[nan_mask] = -999.0
                    reflInRange = reflInRange | np.logical_and(
                        zStart <= 0, zStop >= 0
                    )
                else:
                    zStart[nan_mask] = -999.0
                    zStop[nan_mask] = 999.0
                    reflInRange = reflInRange | np.logical_and(
                        zStart >= 0, zStop <= 0
                    )
    return reflInRange


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
            rMat_c = xfcapi.makeRotMatOfExpMap(expMaps[iOr, :])
            angList = np.vstack(
                xfcapi.oscillAnglesOfHKLs(
                    these_hkls,
                    chi,
                    rMat_c,
                    bMat,
                    wlen,
                    beamVec=bVec,
                    etaVec=eVec,
                    vInv=vInv,
                )
            )
            if not np.all(np.isnan(angList)):
                #
                angList[:, 1] = xfcapi.mapAngle(
                    angList[:, 1], [etaEdges[0], etaEdges[0] + 2 * np.pi]
                )
                angList[:, 2] = xfcapi.mapAngle(
                    angList[:, 2], [omeEdges[0], omeEdges[0] + 2 * np.pi]
                )
                #
                # do eta ranges
                angMask_eta = np.zeros(len(angList), dtype=bool)
                for etas in etaRanges:
                    angMask_eta = np.logical_or(
                        angMask_eta,
                        xf.validateAngleRanges(
                            angList[:, 1], etas[0], etas[1]
                        ),
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
                            eta_ome[iHKL, i_sup[idx_mask], j_sup[idx_mask]] = (
                                1.0
                            )
                        else:
                            eta_ome[
                                iHKL,
                                omeIndices[culledOmeIdx],
                                etaIndices[culledEtaIdx],
                            ] = 1.0
                            pass  # close conditional on pixel dilation
                        pass  # close conditional on ranges
                    pass  # close for loop on valid reflections
                pass  # close conditional for valid angles
    return eta_ome


def _fetch_hkls_from_planedata(pd):
    return np.hstack(pd.getSymHKLs(withID=True)).T


def _filter_hkls_eta_ome(
    hkls, angles, eta_range, ome_range, return_mask=False
):
    """
    given a set of hkls and angles, filter them by the
    eta and omega ranges
    """
    # do eta ranges
    angMask_eta = np.zeros(len(angles), dtype=bool)
    for etas in eta_range:
        angMask_eta = np.logical_or(
            angMask_eta, xf.validateAngleRanges(angles[:, 1], etas[0], etas[1])
        )

    # do omega ranges
    ccw = True
    angMask_ome = np.zeros(len(angles), dtype=bool)
    for omes in ome_range:
        if omes[1] - omes[0] < 0:
            ccw = False
        angMask_ome = np.logical_or(
            angMask_ome,
            xf.validateAngleRanges(angles[:, 2], omes[0], omes[1], ccw=ccw),
        )

    # mask angles list, hkls
    angMask = np.logical_and(angMask_eta, angMask_ome)

    allAngs = angles[angMask, :]
    allHKLs = np.vstack([hkls, hkls])[angMask, :]

    if return_mask:
        return allAngs, allHKLs, angMask
    else:
        return allAngs, allHKLs


def _project_on_detector_plane(
    allAngs,
    rMat_d,
    rMat_c,
    chi,
    tVec_d,
    tVec_c,
    tVec_s,
    distortion,
    beamVec=constants.beam_vec,
):
    """
    utility routine for projecting a list of (tth, eta, ome) onto the
    detector plane parameterized by the args
    """
    gVec_cs = xfcapi.angles_to_gvec(
        allAngs, chi=chi, rmat_c=rMat_c, beam_vec=beamVec
    )

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
    allAngs,
    chi,
    tVec_d,
    caxis,
    paxis,
    radius,
    physical_size,
    angle_extent,
    distortion,
    beamVec=constants.beam_vec,
    etaVec=constants.eta_vec,
    tVec_s=constants.zeros_3x1,
    rmat_s=constants.identity_3x3,
    tVec_c=constants.zeros_3x1,
):
    """
    utility routine for projecting a list of (tth, eta, ome) onto the
    detector plane parameterized by the args. this function does the
    computation for a cylindrical detector
    """
    dVec_cs = xfcapi.anglesToDVec(
        allAngs, chi=chi, rMat_c=np.eye(3), bHat_l=beamVec, eHat_l=etaVec
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


def _dvecToDetectorXYcylinder(
    dVec_cs,
    tVec_d,
    caxis,
    paxis,
    radius,
    physical_size,
    angle_extent,
    tVec_s=constants.zeros_3x1,
    tVec_c=constants.zeros_3x1,
    rmat_s=constants.identity_3x3,
):

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


def _unitvec_to_cylinder(
    uvw,
    caxis,
    paxis,
    radius,
    tvec,
    tVec_s=constants.zeros_3x1,
    tVec_c=constants.zeros_3x1,
    rmat_s=constants.identity_3x3,
):
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
    uvw,
    tVec_d,
    caxis,
    paxis,
    radius,
    physical_size,
    angle_extent,
    tVec_s=constants.zeros_3x1,
    tVec_c=constants.zeros_3x1,
    rmat_s=constants.identity_3x3,
):
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

    delta = tVec_d - (
        radius * naxis + np.squeeze(tVec_s) + np.squeeze(tvec_c_l)
    )

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
    uvw,
    tVec_d,
    caxis,
    paxis,
    radius,
    tVec_s=constants.zeros_3x1,
    tVec_c=constants.zeros_3x1,
    rmat_s=constants.identity_3x3,
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

    delta = tVec_d - (
        radius * naxis + np.squeeze(tVec_s) + np.squeeze(tvec_c_l)
    )

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
    cart,
    tVec_d,
    radius,
    caxis,
    paxis,
    tVec_s=constants.zeros_3x1,
    rmat_s=constants.identity_3x3,
    tVec_c=constants.zeros_3x1,
    normalize=True,
):
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


def _dvec_to_angs(dvecs, bvec, evec):
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

    return (tth, eta)


def simulateGVecs(
    pd,
    detector_params,
    grain_params,
    ome_range=[
        (-np.pi, np.pi),
    ],
    ome_period=(-np.pi, np.pi),
    eta_range=[
        (-np.pi, np.pi),
    ],
    panel_dims=[(-204.8, -204.8), (204.8, 204.8)],
    pixel_pitch=(0.2, 0.2),
    distortion=None,
):
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
        det_xy, rMat_s, _ = _project_on_detector_plane(
            allAngs, rMat_d, rMat_c, chi, tVec_d, tVec_c, tVec_s, distortion
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
            rMat_s,
            tVec_d,
            tVec_s,
            tVec_c,
            distortion=distortion,
        )

    return valid_ids, valid_hkl, valid_ang, valid_xy, ang_ps


# TODO: Deprecate this function
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
        beamVec = xfcapi.bVec_ref

    # parse energy ranges
    multipleEnergyRanges = False
    if hasattr(maxEnergy, '__len__'):
        assert len(maxEnergy) == len(
            minEnergy
        ), 'energy cutoff ranges must have the same length'
        multipleEnergyRanges = True
        lmin = [processWavelength(e) for e in minEnergy]
        lmax = [processWavelength(e) for e in maxEnergy]
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
        rmat_c = xfcapi.makeRotMatOfExpMap(gp[:3])
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

            # back to angles
            tth_eta, gvec_l = xfcapi.detectorXYToGvec(
                dpts.T, rmat_d, rmat_s, tvec_d, tvec_s, tvec_c, beamVec=beamVec
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


@numba.njit(nogil=True, cache=True)
def _expand_pixels(original, w, h, result):
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
def _compute_max(tth, eta, result):
    period = 2.0 * np.pi
    hperiod = np.pi
    for el in range(0, len(tth), 4):
        max_tth = np.abs(tth[el + 0] - tth[el + 3])
        eta_diff = eta[el + 0] - eta[el + 3]
        max_eta = np.abs(np.remainder(eta_diff + hperiod, period) - hperiod)
        for i in range(3):
            curr_tth = np.abs(tth[el + i] - tth[el + i + 1])
            eta_diff = eta[el + i] - eta[el + i + 1]
            curr_eta = np.abs(
                np.remainder(eta_diff + hperiod, period) - hperiod
            )
            max_tth = np.maximum(curr_tth, max_tth)
            max_eta = np.maximum(curr_eta, max_eta)
        result[el // 4, 0] = max_tth
        result[el // 4, 1] = max_eta

    return result


def angularPixelSize(
    xy_det,
    xy_pixelPitch,
    rMat_d,
    rMat_s,
    tVec_d,
    tVec_s,
    tVec_c,
    distortion=None,
    beamVec=None,
    etaVec=None,
):
    """
    Calculate angular pixel sizes on a detector.

    * choices to beam vector and eta vector specs have been supressed
    * assumes xy_det in UNWARPED configuration
    """
    xy_det = np.atleast_2d(xy_det)
    if distortion is not None:  # !!! check this logic
        xy_det = distortion.apply(xy_det)
    if beamVec is None:
        beamVec = xfcapi.bVec_ref
    if etaVec is None:
        etaVec = xfcapi.eta_ref

    xy_expanded = np.empty((len(xy_det) * 4, 2), dtype=xy_det.dtype)
    xy_expanded = _expand_pixels(
        xy_det, xy_pixelPitch[0], xy_pixelPitch[1], xy_expanded
    )
    gvec_space, _ = xfcapi.detectorXYToGvec(
        xy_expanded,
        rMat_d,
        rMat_s,
        tVec_d,
        tVec_s,
        tVec_c,
        beamVec=beamVec,
        etaVec=etaVec,
    )
    result = np.empty_like(xy_det)
    return _compute_max(gvec_space[0], gvec_space[1], result)


def make_reflection_patches(
    instr_cfg,
    tth_eta,
    ang_pixel_size,
    omega=None,
    tth_tol=0.2,
    eta_tol=1.0,
    rmat_c=np.eye(3),
    tvec_c=np.zeros((3, 1)),
    npdiv=1,
    quiet=False,
    compute_areas_func=gutil.compute_areas,
):
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
    npts = len(tth_eta)

    # detector quantities
    rmat_d = xfcapi.makeRotMatOfExpMap(
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
    row_edges = (
        np.arange(frame_nrows + 1)[::-1] * pixel_size[1] + panel_dims[0][1]
    )
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
        full_angs = np.hstack([tth_eta, np.zeros((npts, 1))])
    else:
        full_angs = np.hstack([tth_eta, omega.reshape(npts, 1)])

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
            np.vstack(
                [m_tth.flatten(), m_eta.flatten(), np.zeros(npts_patch)]
            ).T
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
        tth_eta_cen = gutil.cellCentroids(
            np.atleast_2d(gVec_angs_vtx[:, :2]), conn
        )

        gVec_angs = np.hstack(
            [tth_eta_cen, np.tile(angs[2], (len(tth_eta_cen), 1))]
        )

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
        assert len(
            detector_params >= 10
        ), "list of detector parameters must have length >= 10"
        rMat_d = xfcapi.makeRotMatOfExpMap(detector_params[:3])
        tVec_d = np.ascontiguousarray(detector_params[3:6])
        chi = detector_params[6]
        tVec_s = np.ascontiguousarray(detector_params[7:10])
    return rMat_d, tVec_d, chi, tVec_s
