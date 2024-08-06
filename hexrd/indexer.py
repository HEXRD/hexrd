#!/usr/bin/env python3
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
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
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

import logging

import multiprocessing

import numpy as np
import numba

import timeit

from hexrd import constants
from hexrd import rotations
from hexrd.transforms import xfcapi


# =============================================================================
# Parameters
# =============================================================================
omega_period_DFLT = np.radians(np.r_[-180.0, 180.0])

paramMP = None
nCPUs_DFLT = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)


# =============================================================================
# Methods
# =============================================================================
def paintGrid(
    quats,
    etaOmeMaps,
    threshold=None,
    bMat=None,
    omegaRange=None,
    etaRange=None,
    omeTol=constants.d2r,
    etaTol=constants.d2r,
    omePeriod=omega_period_DFLT,
    doMultiProc=False,
    nCPUs=None,
    debug=False,
):
    r"""
    Spherical map-based indexing algorithm, i.e. paintGrid.

    Given a list of trial orientations `quats` and an eta-omega intensity map
    object `etaOmeMaps`, this method executes a test to produce a completeness
    ratio for each orientation across the spherical inensity maps.

    Parameters
    ----------
    quats : (4, N) ndarray
        hstacked array of trial orientations in the form of unit quaternions.
    etaOmeMaps : object
        an spherical map object of type `hexrd.instrument.GenerateEtaOmeMaps`.
    threshold : float, optional
        threshold value on the etaOmeMaps.
    bMat : (3, 3) ndarray, optional
        the COB matrix from the reciprocal lattice to the reference crystal
        frame.  In not provided, the B in the planeData class in the etaOmeMaps
        is used.
    omegaRange : array_like, optional
        list of valid omega ranges in radians,
        e.g. np.radians([(-60, 60), (120, 240)])
    etaRange : array_like, optional
        list of valid eta ranges in radians,
        e.g. np.radians([(-85, 85), (95, 265)])
    omeTol : float, optional
        the tolerance to use in the omega dimension in radians.  Default is
        1 degree (0.017453292519943295)
    etaTol : float, optional
        the tolerance to use in the eta dimension in radians.  Default is
        1 degree (0.017453292519943295)
    omePeriod : (2, ) array_like, optional
        the period to use for omega angles in radians,
        e.g. np.radians([-180, 180])
    doMultiProc : bool, optional
        flag for enabling multiprocessing
    nCPUs : int, optional
        number of processes to use in case doMultiProc = True
    debug : bool, optional
        debugging mode flag

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    retval : (N, ) list
        completeness score list for `quats`.


    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a Greek symbol like :math:`\omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    >>> print("a\nb")
    a
    b
    """
    quats = np.atleast_2d(quats)
    if quats.size == 4:
        quats = quats.reshape(4, 1)

    planeData = etaOmeMaps.planeData

    # !!! these are master hklIDs
    hklIDs = np.asarray(etaOmeMaps.iHKLList)
    hklList = planeData.getHKLs(*hklIDs).tolist()
    hkl_idx = planeData.getHKLID(planeData.getHKLs(*hklIDs).T, master=False)
    nHKLS = len(hklIDs)

    numEtas = len(etaOmeMaps.etaEdges) - 1
    numOmes = len(etaOmeMaps.omeEdges) - 1

    if threshold is None:
        threshold = np.zeros(nHKLS)
        for i in range(nHKLS):
            threshold[i] = np.mean(
                np.r_[
                    np.mean(etaOmeMaps.dataStore[i]),
                    np.median(etaOmeMaps.dataStore[i]),
                ]
            )
    elif threshold is not None and not hasattr(threshold, '__len__'):
        threshold = threshold * np.ones(nHKLS)
    elif hasattr(threshold, '__len__'):
        if len(threshold) != nHKLS:
            raise RuntimeError("threshold list is wrong length!")
        else:
            print("INFO: using list of threshold values")
    else:
        raise RuntimeError(
            "unknown threshold option. should be a list of numbers or None"
        )
    if bMat is None:
        bMat = planeData.latVecOps['B']

    # ???
    # not positive why these are needed
    etaIndices = np.arange(numEtas)
    omeIndices = np.arange(numOmes)

    omeMin = None
    omeMax = None
    if omegaRange is None:  # FIXME
        omeMin = [
            np.min(etaOmeMaps.omeEdges),
        ]
        omeMax = [
            np.max(etaOmeMaps.omeEdges),
        ]
    else:
        omeMin = [omegaRange[i][0] for i in range(len(omegaRange))]
        omeMax = [omegaRange[i][1] for i in range(len(omegaRange))]
    if omeMin is None:
        omeMin = [
            -np.pi,
        ]
        omeMax = [
            np.pi,
        ]
    omeMin = np.asarray(omeMin)
    omeMax = np.asarray(omeMax)

    etaMin = None
    etaMax = None
    if etaRange is not None:
        etaMin = [etaRange[i][0] for i in range(len(etaRange))]
        etaMax = [etaRange[i][1] for i in range(len(etaRange))]
    if etaMin is None:
        etaMin = [
            -np.pi,
        ]
        etaMax = [
            np.pi,
        ]
    etaMin = np.asarray(etaMin)
    etaMax = np.asarray(etaMax)

    multiProcMode = nCPUs_DFLT > 1 and doMultiProc

    if multiProcMode:
        nCPUs = nCPUs or nCPUs_DFLT
        chunksize = min(quats.shape[1] // nCPUs, 10)
        logger.info(
            "using multiprocessing with %d processes and a chunk size of %d",
            nCPUs,
            chunksize,
        )
    else:
        logger.info("running in serial mode")
        nCPUs = 1

    # Get the symHKLs for the selected hklIDs
    symHKLs = planeData.getSymHKLs()
    symHKLs = [symHKLs[id] for id in hkl_idx]
    # Restructure symHKLs into a flat NumPy HKL array with
    # each HKL stored contiguously (C-order instead of F-order)
    # symHKLs_ix provides the start/end index for each subarray
    # of symHKLs.
    symHKLs_ix = np.add.accumulate([0] + [s.shape[1] for s in symHKLs])
    symHKLs = np.vstack([s.T for s in symHKLs])

    # Pack together the common parameters for processing
    params = {
        'symHKLs': symHKLs,
        'symHKLs_ix': symHKLs_ix,
        'wavelength': planeData.wavelength,
        'hklList': hklList,
        'omeMin': omeMin,
        'omeMax': omeMax,
        'omeTol': omeTol,
        'omeIndices': omeIndices,
        'omePeriod': omePeriod,
        'omeEdges': etaOmeMaps.omeEdges,
        'etaMin': etaMin,
        'etaMax': etaMax,
        'etaTol': etaTol,
        'etaIndices': etaIndices,
        'etaEdges': etaOmeMaps.etaEdges,
        'etaOmeMaps': np.stack(etaOmeMaps.dataStore),
        'bMat': bMat,
        'threshold': np.asarray(threshold),
    }

    # do the mapping
    start = timeit.default_timer()
    retval = None
    if multiProcMode:
        # multiple process version
        pool = multiprocessing.Pool(nCPUs, paintgrid_init, (params,))
        retval = pool.map(paintGridThis, quats.T, chunksize=chunksize)
        pool.close()
    else:
        # single process version.
        global paramMP
        paintgrid_init(params)  # sets paramMP
        retval = list(map(paintGridThis, quats.T))
        paramMP = None  # clear paramMP
    elapsed = timeit.default_timer() - start
    logger.info("paintGrid took %.3f seconds", elapsed)

    return retval


def _meshgrid2d(x, y):
    """
    Special-cased implementation of np.meshgrid.

    For just two arguments, (x, y). Found to be about 3x faster on some simple
    test arguments.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    r1 : TYPE
        DESCRIPTION.
    r2 : TYPE
        DESCRIPTION.

    """
    x, y = (np.asarray(x), np.asarray(y))
    shape = (len(y), len(x))
    dt = np.result_type(x, y)
    r1, r2 = (np.empty(shape, dt), np.empty(shape, dt))
    r1[...] = x[np.newaxis, :]
    r2[...] = y[:, np.newaxis]
    return (r1, r2)


def _normalize_ranges(starts, stops, offset, ccw=False):
    """
    Range normalization.

    Normalize in the range [offset, 2*pi+offset[ the ranges defined
    by starts and stops.

    Checking if an angle lies inside a range can be done in a way that
    is more efficient than using validateAngleRanges.

    Note this function assumes that ranges don't overlap.

    Parameters
    ----------
    starts : TYPE
        DESCRIPTION.
    stops : TYPE
        DESCRIPTION.
    offset : TYPE
        DESCRIPTION.
    ccw : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    if ccw:
        starts, stops = stops, starts

    # results are in the range of [0, 2*np.pi]
    if not np.all(starts < stops):
        raise ValueError('Invalid angle ranges')

    # If there is a range that spans more than 2*pi,
    # return the full range
    two_pi = 2 * np.pi
    if np.any((starts + two_pi) < stops + 1e-8):
        return np.array([offset, two_pi + offset])

    starts = np.mod(starts - offset, two_pi) + offset
    stops = np.mod(stops - offset, two_pi) + offset

    order = np.argsort(starts)
    result = np.hstack(
        (starts[order, np.newaxis], stops[order, np.newaxis])
    ).ravel()
    # at this point, result is in its final form unless there
    # is wrap-around in the last segment. Handle this case:
    if result[-1] < result[-2]:
        new_result = np.empty((len(result) + 2,), dtype=result.dtype)
        new_result[0] = offset
        new_result[1] = result[-1]
        new_result[2:-1] = result[0:-1]
        new_result[-1] = offset + two_pi
        result = new_result

    if not np.all(starts[1:] > stops[0:-2]):
        raise ValueError('Angle ranges overlap')

    return result


def paintgrid_init(params):
    """
    Initialize global variables for paintGrid.

    Parameters
    ----------
    params : dict
        multiprocessing parameter dictionary.

    Returns
    -------
    None.
    """
    global paramMP
    paramMP = params

    # create valid_eta_spans, valid_ome_spans from etaMin/Max and omeMin/Max
    # this allows using faster checks in the code.
    # TODO: build valid_eta_spans and valid_ome_spans directly in paintGrid
    #       instead of building etaMin/etaMax and omeMin/omeMax. It may also
    #       be worth handling range overlap and maybe "optimize" ranges if
    #       there happens to be contiguous spans.
    paramMP['valid_eta_spans'] = _normalize_ranges(
        paramMP['etaMin'], paramMP['etaMax'], -np.pi
    )

    paramMP['valid_ome_spans'] = _normalize_ranges(
        paramMP['omeMin'], paramMP['omeMax'], min(paramMP['omePeriod'])
    )
    return


###############################################################################
#
# paintGridThis contains the bulk of the process to perform for paintGrid for a
# given quaternion. This is also used as the basis for multiprocessing, as the
# work is split in a per-quaternion basis among different processes.
# The remainding arguments are marshalled into the module variable "paramMP".
#
# There is a version of PaintGridThis using numba, and another version used
# when numba is not available. The numba version should be noticeably faster.
###############################################################################


@numba.njit(nogil=True, cache=True)
def _check_dilated(eta, ome, dpix_eta, dpix_ome, etaOmeMap, threshold):
    """Part of paintGridThis.

    check if there exists a sample over the given threshold in the etaOmeMap
    at (eta, ome), with a tolerance of (dpix_eta, dpix_ome) samples.

    Note this function is "numba friendly" and will be jitted when using numba.

    TODO: currently behaves like "np.any" call for values above threshold.
    There is some ambigutiy if there are NaNs in the dilation range, but it
    hits a value above threshold first.  Is that ok???

    FIXME: works in non-numba implementation of paintGridThis only
    <JVB 2017-04-27>
    """
    i_max, j_max = etaOmeMap.shape
    ome_start, ome_stop = (
        max(ome - dpix_ome, 0),
        min(ome + dpix_ome + 1, i_max),
    )
    eta_start, eta_stop = (
        max(eta - dpix_eta, 0),
        min(eta + dpix_eta + 1, j_max),
    )

    for i in range(ome_start, ome_stop):
        for j in range(eta_start, eta_stop):
            if etaOmeMap[i, j] > threshold:
                return 1
            if np.isnan(etaOmeMap[i, j]):
                return -1
    return 0


def paintGridThis(quat):
    """Single instance paintGrid call.

    Note that this version does not use omeMin/omeMax to specify the valid
    angles. It uses "valid_eta_spans" and "valid_ome_spans". These are
    precomputed and make for a faster check of ranges than
    "validateAngleRanges"
    """
    symHKLs = paramMP['symHKLs']  # the HKLs
    symHKLs_ix = paramMP['symHKLs_ix']  # index partitioning of symHKLs
    bMat = paramMP['bMat']
    wavelength = paramMP['wavelength']
    omeEdges = paramMP['omeEdges']
    omeTol = paramMP['omeTol']
    omePeriod = paramMP['omePeriod']
    valid_eta_spans = paramMP['valid_eta_spans']
    valid_ome_spans = paramMP['valid_ome_spans']
    omeIndices = paramMP['omeIndices']
    etaEdges = paramMP['etaEdges']
    etaTol = paramMP['etaTol']
    etaIndices = paramMP['etaIndices']
    etaOmeMaps = paramMP['etaOmeMaps']
    threshold = paramMP['threshold']

    # dpix_ome and dpix_eta are the number of pixels for the tolerance in
    # ome/eta. Maybe we should compute this per run instead of per
    # quaternion
    del_ome = abs(omeEdges[1] - omeEdges[0])
    del_eta = abs(etaEdges[1] - etaEdges[0])
    dpix_ome = int(round(omeTol / del_ome))
    dpix_eta = int(round(etaTol / del_eta))

    # FIXME
    debug = False
    if debug:
        print(
            "using ome, eta dilitations of (%d, %d) pixels"
            % (dpix_ome, dpix_eta)
        )

    # get the equivalent rotation of the quaternion in matrix form (as
    # expected by oscillAnglesOfHKLs

    rMat = rotations.rotMatOfQuat(quat.T if quat.ndim == 2 else quat)

    # Compute the oscillation angles of all the symHKLs at once
    oangs_pair = xfcapi.oscill_angles_of_hkls(
        symHKLs, 0.0, rMat, bMat, wavelength
    )
    # pdb.set_trace()
    return _filter_and_count_hits(
        oangs_pair[0],
        oangs_pair[1],
        symHKLs_ix,
        etaEdges,
        valid_eta_spans,
        valid_ome_spans,
        omeEdges,
        omePeriod,
        etaOmeMaps,
        etaIndices,
        omeIndices,
        dpix_eta,
        dpix_ome,
        threshold,
    )


@numba.njit(nogil=True, cache=True)
def _find_in_range(value, spans):
    """
    Find the index in spans where value >= spans[i] and value < spans[i].

    spans is an ordered array where spans[i] <= spans[i+1]
    (most often < will hold).

    If value is not in the range [spans[0], spans[-1]], then
    -2 is returned.

    This is equivalent to "bisect_right" in the bisect package, in which
    code it is based, and it is somewhat similar to NumPy's searchsorted,
    but non-vectorized
    """
    if value < spans[0] or value >= spans[-1]:
        return -2

    # from the previous check, we know 0 is not a possible result
    li = 0
    ri = len(spans)

    while li < ri:
        mi = (li + ri) // 2
        if value < spans[mi]:
            ri = mi
        else:
            li = mi + 1

    return li


@numba.njit(nogil=True, cache=True)
def _angle_is_hit(
    ang,
    eta_offset,
    ome_offset,
    hkl,
    valid_eta_spans,
    valid_ome_spans,
    etaEdges,
    omeEdges,
    etaOmeMaps,
    etaIndices,
    omeIndices,
    dpix_eta,
    dpix_ome,
    threshold,
):
    """Perform work on one of the angles.

    This includes:

    - filtering nan values

    - filtering out angles not in the specified spans

    - checking that the discretized angle fits into the sensor range (maybe
        this could be merged with the previous test somehow, for extra speed)

    - actual check for a hit, using dilation for the tolerance.

    Note the function returns both, if it was a hit and if it passed the
    filtering, as we'll want to discard the filtered values when computing
    the hit percentage.

    CAVEAT: added map-based nan filtering to _check_dilated; this may not
    be the best option.  Perhaps filter here? <JVB 2017-04-27>

    """
    tth, eta, ome = ang

    if np.isnan(tth):
        return 0, 0

    eta = _map_angle(eta, eta_offset)
    if _find_in_range(eta, valid_eta_spans) & 1 == 0:
        # index is even: out of valid eta spans
        return 0, 0

    ome = _map_angle(ome, ome_offset)
    if _find_in_range(ome, valid_ome_spans) & 1 == 0:
        # index is even: out of valid ome spans
        return 0, 0

    # discretize the angles
    eta_idx = _find_in_range(eta, etaEdges) - 1
    if eta_idx < 0:
        # out of range
        return 0, 0

    ome_idx = _find_in_range(ome, omeEdges) - 1
    if ome_idx < 0:
        # out of range
        return 0, 0

    eta = etaIndices[eta_idx]
    ome = omeIndices[ome_idx]
    isHit = _check_dilated(
        eta, ome, dpix_eta, dpix_ome, etaOmeMaps[hkl], threshold[hkl]
    )
    if isHit == -1:
        return 0, 0
    else:
        return isHit, 1


@numba.njit(nogil=True, cache=True)
def _filter_and_count_hits(
    angs_0,
    angs_1,
    symHKLs_ix,
    etaEdges,
    valid_eta_spans,
    valid_ome_spans,
    omeEdges,
    omePeriod,
    etaOmeMaps,
    etaIndices,
    omeIndices,
    dpix_eta,
    dpix_ome,
    threshold,
):
    """Accumulate completeness scores.

    assumes:
    we want etas in -pi -> pi range
    we want omes in ome_offset -> ome_offset + 2*pi range

    Instead of creating an array with the angles of angs_0 and angs_1
    interleaved, in this numba version calls for both arrays are performed
    getting the angles from angs_0 and angs_1. this is done in this way to
    reuse hkl computation. This may not be that important, though.

    """
    eta_offset = -np.pi
    ome_offset = np.min(omePeriod)
    hits = 0
    total = 0
    curr_hkl_idx = 0
    end_curr = symHKLs_ix[1]
    count = len(angs_0)

    for i in range(count):
        if i >= end_curr:
            curr_hkl_idx += 1
            end_curr = symHKLs_ix[curr_hkl_idx + 1]

        # first solution
        hit, not_filtered = _angle_is_hit(
            angs_0[i],
            eta_offset,
            ome_offset,
            curr_hkl_idx,
            valid_eta_spans,
            valid_ome_spans,
            etaEdges,
            omeEdges,
            etaOmeMaps,
            etaIndices,
            omeIndices,
            dpix_eta,
            dpix_ome,
            threshold,
        )
        hits += hit
        total += not_filtered

        # second solution
        hit, not_filtered = _angle_is_hit(
            angs_1[i],
            eta_offset,
            ome_offset,
            curr_hkl_idx,
            valid_eta_spans,
            valid_ome_spans,
            etaEdges,
            omeEdges,
            etaOmeMaps,
            etaIndices,
            omeIndices,
            dpix_eta,
            dpix_ome,
            threshold,
        )
        hits += hit
        total += not_filtered

    return float(hits) / float(total) if total != 0 else 0.0


@numba.njit(nogil=True, cache=True)
def _map_angle(angle, offset):
    """Numba-firendly equivalent to xf.mapAngle."""
    return np.mod(angle - offset, 2 * np.pi) + offset
