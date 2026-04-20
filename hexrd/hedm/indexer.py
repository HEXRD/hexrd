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
"""Spherical map-based orientation indexing for HEDM (paintGrid algorithm).

This module implements the paintGrid algorithm, which scores a set of trial
crystal orientations against measured eta-omega intensity maps produced by
High Energy X-Ray Diffraction (HEXRD) experiments.

For each trial orientation, the algorithm:

1. Computes the predicted diffraction angles (tth, eta, omega) for every
   symmetry-equivalent HKL reflection using the orientation matrix and the
   reciprocal-lattice B-matrix.
2. Filters predictions that fall outside the specified valid eta/omega ranges.
3. Checks, with a pixel-dilation tolerance, whether each surviving prediction
   coincides with measured intensity above a per-HKL threshold in the
   corresponding eta-omega map.
4. Returns a completeness score (hits / valid predictions) for each orientation.

The main public entry point is :func:`paintGrid`.  Internal helpers
:func:`paintgrid_init` and :func:`paintGridThis` support both serial and
multiprocessing execution modes.  The innermost pixel-checking and angle-
mapping routines are JIT-compiled with Numba for performance.
"""

from __future__ import annotations

import logging
import multiprocessing
import timeit
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
import numba

from hexrd.core import constants
from hexrd.core import rotations
from hexrd.core.transforms import xfcapi

# =============================================================================
# Parameters
# =============================================================================
omega_period_DFLT: NDArray[np.float64] = np.radians(np.r_[-180.0, 180.0])


class PaintGridParams(TypedDict):
    """Typed parameters dictionary shared across paintGrid workers.

    The first 18 fields are populated by :func:`paintGrid` and passed to
    :func:`paintgrid_init`.  The final two fields (``valid_eta_spans`` and
    ``valid_ome_spans``) are computed and added by :func:`paintgrid_init`
    before any worker calls :func:`paintGridThis`.
    """

    symHKLs: NDArray[np.int_]
    symHKLs_ix: NDArray[np.intp]
    wavelength: float
    omeMin: NDArray[np.float64]
    omeMax: NDArray[np.float64]
    omeTol: float
    omeIndices: NDArray[np.intp]
    omePeriod: NDArray[np.float64]
    omeEdges: NDArray[np.float64]
    etaMin: NDArray[np.float64]
    etaMax: NDArray[np.float64]
    etaTol: float
    etaIndices: NDArray[np.intp]
    etaEdges: NDArray[np.float64]
    etaOmeMaps: NDArray[np.float64]
    bMat: NDArray[np.float64]
    threshold: NDArray[np.float64]
    valid_eta_spans: NDArray[np.float64]
    valid_ome_spans: NDArray[np.float64]


paramMP: PaintGridParams | None = None
nCPUs_DFLT: int = multiprocessing.cpu_count()
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Methods
# =============================================================================
def paintGrid(
    quats: NDArray[np.float64],
    etaOmeMaps: Any,
    threshold: float | NDArray[np.float64] | None = None,
    bMat: NDArray[np.float64] | None = None,
    omegaRange: NDArray[np.float64] | None = None,
    etaRange: NDArray[np.float64] | None = None,
    omeTol: float = constants.d2r,
    etaTol: float = constants.d2r,
    omePeriod: NDArray[np.float64] = omega_period_DFLT,
    doMultiProc: bool = False,
    nCPUs: int | None = None,
    debug: bool = False,
) -> list[float]:
    r"""
    Spherical map-based indexing algorithm, i.e. paintGrid.

    Given a list of trial orientations `quats` and an eta-omega intensity map
    object `etaOmeMaps`, this method executes a test to produce a completeness
    ratio for each orientation across the spherical intensity maps.

    Parameters
    ----------
    quats : (4, N) ndarray
        hstacked array of trial orientations in the form of unit quaternions.
    etaOmeMaps : object
        a spherical map object of type
        ``hexrd.hedm.instrument.GenerateEtaOmeMaps``.
    threshold : float or array_like or None, optional
        Intensity threshold(s) applied to each HKL's eta-omega map when
        deciding whether a predicted reflection constitutes a hit.

        - ``None`` (default): a per-HKL threshold is computed automatically
          as the mean of the per-map mean and median.
        - scalar ``float``: the same threshold is used for every HKL.
        - sequence of length ``nHKLS``: one threshold value per HKL.
    bMat : (3, 3) ndarray, optional
        the COB matrix from the reciprocal lattice to the reference crystal
        frame.  If not provided, the B in the planeData class in the
        etaOmeMaps is used.
    omegaRange : array_like, optional
        list of valid omega ranges in radians,
        e.g. ``np.radians([(-60, 60), (120, 240)])``.
        Defaults to the full range spanned by the omega edges in
        ``etaOmeMaps``.
    etaRange : array_like, optional
        list of valid eta ranges in radians,
        e.g. ``np.radians([(-85, 85), (95, 265)])``.
        Defaults to ``[-pi, pi]``.
    omeTol : float, optional
        the tolerance to use in the omega dimension in radians.  Default is
        1 degree (0.017453292519943295).
    etaTol : float, optional
        the tolerance to use in the eta dimension in radians.  Default is
        1 degree (0.017453292519943295).
    omePeriod : (2, ) array_like, optional
        the period to use for omega angles in radians,
        e.g. ``np.radians([-180, 180])``.
    doMultiProc : bool, optional
        flag for enabling multiprocessing.  Requires more than one CPU to
        have any effect.  Default is ``False``.
    nCPUs : int or None, optional
        number of processes to use when ``doMultiProc=True``.  Defaults to
        ``multiprocessing.cpu_count()`` when ``None``.
    debug : bool, optional
        debugging mode flag.  Currently unused; reserved for future use.

    Raises
    ------
    RuntimeError
        If ``threshold`` is a sequence whose length does not match the number
        of HKLs in ``etaOmeMaps``, or if ``threshold`` is an unrecognised
        type.

    Returns
    -------
    retval : (N, ) list of float
        Completeness score list for `quats`.  Each value is in ``[0, 1]``
        and represents the fraction of symmetry-equivalent reflections that
        were predicted within the valid angle ranges *and* found to have
        intensity above threshold in the corresponding eta-omega map bin
        (after applying the dilation tolerance).

    Notes
    -----
    The completeness score for a given orientation is computed as::

        score = hits / total_valid

    where ``total_valid`` counts predicted reflections that pass the eta and
    omega range filters, and ``hits`` counts the subset of those for which
    the dilated pixel neighbourhood in the corresponding eta-omega map
    contains at least one value above ``threshold``.

    Both the first and second oscillation-angle solutions returned by
    ``xfcapi.oscill_angles_of_hkls`` are evaluated independently.  Bins
    containing ``NaN`` values are treated as misses and excluded from
    ``total_valid``.

    When ``doMultiProc=True``, the quaternion array is partitioned into
    chunks and distributed across ``nCPUs`` worker processes.  Each worker
    shares a read-only parameter dictionary initialised by
    :func:`paintgrid_init`.
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
    elif threshold is not None and not hasattr(threshold, "__len__"):
        threshold = threshold * np.ones(nHKLS)
    elif hasattr(threshold, "__len__"):
        if len(threshold) != nHKLS:
            raise RuntimeError("threshold list is wrong length!")
        else:
            logging.debug("Using list of threshold values")
    else:
        raise RuntimeError(
            "unknown threshold option. should be a list of numbers or None"
        )
    if bMat is None:
        bMat = planeData.latVecOps["B"]

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
        logger.debug(
            "using multiprocessing with %d processes and a chunk size of %d",
            nCPUs,
            chunksize,
        )
    else:
        logger.debug("running in serial mode")
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
        "symHKLs": symHKLs,
        "symHKLs_ix": symHKLs_ix,
        "wavelength": planeData.wavelength,
        "omeMin": omeMin,
        "omeMax": omeMax,
        "omeTol": omeTol,
        "omeIndices": omeIndices,
        "omePeriod": omePeriod,
        "omeEdges": etaOmeMaps.omeEdges,
        "etaMin": etaMin,
        "etaMax": etaMax,
        "etaTol": etaTol,
        "etaIndices": etaIndices,
        "etaEdges": etaOmeMaps.etaEdges,
        "etaOmeMaps": np.stack(etaOmeMaps.dataStore),
        "bMat": bMat,
        "threshold": np.asarray(threshold),
    }

    # do the mapping
    start = timeit.default_timer()
    retval = None
    if multiProcMode:
        # multiple process version
        pool = constants.mp_context.Pool(nCPUs, paintgrid_init, (params,))
        retval = pool.map(paintGridThis, quats.T, chunksize=chunksize)
        pool.close()
        pool.join()
    else:
        # single process version.
        global paramMP
        paintgrid_init(params)  # sets paramMP
        retval = list(map(paintGridThis, quats.T))
        paramMP = None  # clear paramMP
    elapsed = timeit.default_timer() - start
    logger.info("paintGrid took %.3f seconds", elapsed)

    return retval


def _meshgrid2d(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Special-cased implementation of np.meshgrid for exactly two arguments.

    Found to be about 3x faster than ``np.meshgrid`` on typical inputs
    because it avoids the general-case overhead of the NumPy implementation.

    Parameters
    ----------
    x : (N,) array_like
        1-D array of x-coordinates (broadcast along columns).
    y : (M,) array_like
        1-D array of y-coordinates (broadcast along rows).

    Returns
    -------
    r1 : (M, N) ndarray
        2-D array in which each row is a copy of ``x``.
    r2 : (M, N) ndarray
        2-D array in which each column is a copy of ``y``.
    """
    x, y = (np.asarray(x), np.asarray(y))
    shape = (len(y), len(x))
    dt = np.result_type(x, y)
    r1, r2 = (np.empty(shape, dt), np.empty(shape, dt))
    r1[...] = x[np.newaxis, :]
    r2[...] = y[:, np.newaxis]
    return (r1, r2)


def _normalize_ranges(
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    offset: float,
    ccw: bool = False,
) -> NDArray[np.float64]:
    """
    Normalize angle ranges into the interval ``[offset, offset + 2*pi)``.

    Converts a set of ``(start, stop)`` angle ranges into a flat, sorted
    array of ``[start, stop, start, stop, ...]`` pairs mapped into the
    canonical window ``[offset, offset + 2*pi)``.  The resulting array can
    be used with :func:`_find_in_range` for efficient membership tests:
    an angle lies inside a valid span when ``_find_in_range`` returns an
    odd index.

    Parameters
    ----------
    starts : (K,) array_like
        Start angles for each range, in radians.  Must satisfy
        ``starts[i] < stops[i]`` for all ``i``.
    stops : (K,) array_like
        Stop angles for each range, in radians.
    offset : float
        The lower bound of the target interval, in radians.
        The output is normalised to ``[offset, offset + 2*pi)``.
    ccw : bool, optional
        If ``True``, treat ranges as counter-clockwise, which effectively
        swaps ``starts`` and ``stops`` before processing.  Default is
        ``False``.

    Raises
    ------
    ValueError
        If any ``starts[i] >= stops[i]`` (invalid range), or if the
        normalised ranges overlap.

    Returns
    -------
    result : (2*K,) or (2*K+2,) ndarray
        Flat array of normalised ``[start, stop]`` pairs, sorted by start
        angle.  An extra leading segment ``[offset, wrap_stop]`` is
        prepended when the last range wraps around past ``offset + 2*pi``.
        If any single range spans more than ``2*pi``, the full interval
        ``[offset, offset + 2*pi]`` is returned instead.
    """
    if ccw:
        starts, stops = stops, starts

    # results are in the range of [0, 2*np.pi]
    if not np.all(starts < stops):
        raise ValueError("Invalid angle ranges")

    # If there is a range that spans more than 2*pi,
    # return the full range
    two_pi = 2 * np.pi
    if np.any((starts + two_pi) < stops + 1e-8):
        return np.array([offset, two_pi + offset])

    starts = np.mod(starts - offset, two_pi) + offset
    stops = np.mod(stops - offset, two_pi) + offset

    order = np.argsort(starts)
    result = np.hstack((starts[order, np.newaxis], stops[order, np.newaxis])).ravel()
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
        raise ValueError("Angle ranges overlap")

    return result


def paintgrid_init(params: PaintGridParams) -> None:
    """
    Initialize global variables for paintGrid.

    Sets the module-level ``paramMP`` dictionary used by
    :func:`paintGridThis` in both serial and multiprocessing execution
    modes.  Also pre-computes ``valid_eta_spans`` and ``valid_ome_spans``
    from the raw ``etaMin``/``etaMax`` and ``omeMin``/``omeMax`` entries in
    ``params`` so that angle validity checks inside the hot loop are faster.

    Parameters
    ----------
    params : dict
        Parameter dictionary assembled by :func:`paintGrid`.  Must contain
        at least the keys ``"etaMin"``, ``"etaMax"``, ``"omeMin"``,
        ``"omeMax"``, and ``"omePeriod"``.

    Returns
    -------
    None
    """
    global paramMP
    paramMP = params

    # create valid_eta_spans, valid_ome_spans from etaMin/Max and omeMin/Max
    # this allows using faster checks in the code.
    # TODO: build valid_eta_spans and valid_ome_spans directly in paintGrid
    #       instead of building etaMin/etaMax and omeMin/omeMax. It may also
    #       be worth handling range overlap and maybe "optimize" ranges if
    #       there happens to be contiguous spans.
    paramMP["valid_eta_spans"] = _normalize_ranges(
        paramMP["etaMin"], paramMP["etaMax"], -np.pi
    )

    paramMP["valid_ome_spans"] = _normalize_ranges(
        paramMP["omeMin"], paramMP["omeMax"], min(paramMP["omePeriod"])
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
def _check_dilated(
    eta: int,
    ome: int,
    dpix_eta: int,
    dpix_ome: int,
    etaOmeMap: NDArray[np.float64],
    threshold: float,
) -> int:
    """Check for intensity above threshold in a dilated pixel neighbourhood.

    Scans the rectangular region ``[ome-dpix_ome, ome+dpix_ome] x
    [eta-dpix_eta, eta+dpix_eta]`` in ``etaOmeMap`` and returns 1 if any
    pixel exceeds ``threshold``, -1 if a ``NaN`` is encountered before a
    hit, and 0 if no value above threshold is found.

    This function is Numba-JIT-compiled (``@numba.njit``) for performance
    and is called from :func:`_angle_is_hit`.

    Parameters
    ----------
    eta : int
        Column index (eta pixel) of the centre of the dilation window.
    ome : int
        Row index (omega pixel) of the centre of the dilation window.
    dpix_eta : int
        Half-width of the dilation window in the eta dimension (pixels).
    dpix_ome : int
        Half-width of the dilation window in the omega dimension (pixels).
    etaOmeMap : (M, N) ndarray
        2-D intensity map for a single HKL family, indexed as
        ``[ome_pixel, eta_pixel]``.
    threshold : float
        Intensity threshold; a pixel must strictly exceed this value to
        count as a hit.

    Returns
    -------
    int
        ``1`` if at least one pixel in the window exceeds ``threshold``,
        ``-1`` if a ``NaN`` is encountered before any hit is found,
        ``0`` if no pixel in the window exceeds ``threshold``.

    Notes
    -----
    TODO: currently behaves like ``np.any`` call for values above threshold.
    There is some ambiguity if there are NaNs in the dilation range, but it
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


def paintGridThis(quat: NDArray[np.float64]) -> float:
    """Score a single trial orientation against the eta-omega maps.

    Computes the completeness of the orientation represented by ``quat``
    by predicting the oscillation angles for all symmetry-equivalent HKL
    reflections and checking each prediction against the eta-omega intensity
    maps stored in the shared ``paramMP`` dictionary.

    This function is designed to be called via ``map``/``Pool.map`` over
    each column of the ``quats`` array in :func:`paintGrid`.  All parameters
    other than the quaternion are read from the module-level ``paramMP``
    dictionary, which must be initialised by :func:`paintgrid_init` before
    calling this function.

    Parameters
    ----------
    quat : (4,) ndarray
        Unit quaternion representing a single trial orientation.

    Returns
    -------
    float
        Completeness score in ``[0, 1]``: the fraction of symmetry-
        equivalent reflections that (a) fall within the valid eta and omega
        ranges and (b) have intensity above threshold in the corresponding
        eta-omega map bin (with pixel-dilation tolerance applied).
        Returns ``0.0`` if no reflections pass the validity filters.

    Notes
    -----
    Uses ``valid_eta_spans`` and ``valid_ome_spans`` from ``paramMP`` rather
    than the raw ``omeMin``/``omeMax`` arrays.  These pre-normalised spans
    allow faster range membership tests via :func:`_find_in_range`.
    """
    symHKLs = paramMP["symHKLs"]  # the HKLs
    symHKLs_ix = paramMP["symHKLs_ix"]  # index partitioning of symHKLs
    bMat = paramMP["bMat"]
    wavelength = paramMP["wavelength"]
    omeEdges = paramMP["omeEdges"]
    omeTol = paramMP["omeTol"]
    omePeriod = paramMP["omePeriod"]
    valid_eta_spans = paramMP["valid_eta_spans"]
    valid_ome_spans = paramMP["valid_ome_spans"]
    omeIndices = paramMP["omeIndices"]
    etaEdges = paramMP["etaEdges"]
    etaTol = paramMP["etaTol"]
    etaIndices = paramMP["etaIndices"]
    etaOmeMaps = paramMP["etaOmeMaps"]
    threshold = paramMP["threshold"]

    # dpix_ome and dpix_eta are the number of pixels for the tolerance in
    # ome/eta. Maybe we should compute this per run instead of per
    # quaternion
    del_ome = abs(omeEdges[1] - omeEdges[0])
    del_eta = abs(etaEdges[1] - etaEdges[0])
    dpix_ome = int(round(omeTol / del_ome))
    dpix_eta = int(round(etaTol / del_eta))

    # get the equivalent rotation of the quaternion in matrix form (as
    # expected by oscillAnglesOfHKLs

    rMat = rotations.rotMatOfQuat(quat.T if quat.ndim == 2 else quat)

    # Compute the oscillation angles of all the symHKLs at once
    oangs_pair = xfcapi.oscill_angles_of_hkls(
        symHKLs, 0.0, rMat, bMat, wavelength
    )
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
def _find_in_range(value: float, spans: NDArray[np.float64]) -> int:
    """
    Binary search for the interval in ``spans`` that contains ``value``.

    Returns the index ``i`` such that ``spans[i-1] <= value < spans[i]``,
    which corresponds to ``bisect_right`` from the standard library.
    This is the non-vectorised, Numba-friendly equivalent of
    ``np.searchsorted(spans, value, side='right')``.

    An odd return value means ``value`` falls *inside* a valid span (i.e.
    between a start and a stop), while an even return value means it falls
    *outside* — assuming ``spans`` was built by :func:`_normalize_ranges`
    as an interleaved ``[start, stop, start, stop, ...]`` array.

    Parameters
    ----------
    value : float
        The angle (or other scalar) to locate, in radians.
    spans : (2*K,) ndarray
        Sorted, non-overlapping array of ``[start, stop]`` pairs as
        produced by :func:`_normalize_ranges`.

    Returns
    -------
    int
        Index in ``spans`` such that ``spans[index-1] <= value < spans[index]``,
        or ``-2`` if ``value`` is outside the range
        ``[spans[0], spans[-1])``.
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
    ang: NDArray[np.float64],
    eta_offset: float,
    ome_offset: float,
    hkl: int,
    valid_eta_spans: NDArray[np.float64],
    valid_ome_spans: NDArray[np.float64],
    etaEdges: NDArray[np.float64],
    omeEdges: NDArray[np.float64],
    etaOmeMaps: NDArray[np.float64],
    etaIndices: NDArray[np.intp],
    omeIndices: NDArray[np.intp],
    dpix_eta: int,
    dpix_ome: int,
    threshold: NDArray[np.float64],
) -> tuple[int, int]:
    """Determine whether a single predicted reflection angle is a hit.

    Applies a chain of filters to the predicted ``(tth, eta, ome)`` triple
    and, if the angle passes all filters, calls :func:`_check_dilated` to
    test for intensity above threshold in the eta-omega map.

    Filtering steps (in order):

    1. Reject ``NaN`` values in ``tth``.
    2. Map ``eta`` into ``[eta_offset, eta_offset + 2*pi)`` and reject if
       outside the valid eta spans.
    3. Map ``ome`` into ``[ome_offset, ome_offset + 2*pi)`` and reject if
       outside the valid omega spans.
    4. Discretise the mapped angles to pixel indices; reject if out of the
       map bounds.
    5. Check the dilated pixel neighbourhood for a hit.

    Parameters
    ----------
    ang : (3,) ndarray
        Predicted ``(tth, eta, ome)`` angles in radians for one reflection.
    eta_offset : float
        Lower bound of the canonical eta interval in radians (typically
        ``-pi``).
    ome_offset : float
        Lower bound of the canonical omega interval in radians (typically
        ``min(omePeriod)``).
    hkl : int
        Index into ``etaOmeMaps`` (and ``threshold``) for the current HKL
        family.
    valid_eta_spans : (2*K,) ndarray
        Sorted interleaved array of valid eta ``[start, stop]`` pairs as
        produced by :func:`_normalize_ranges`.
    valid_ome_spans : (2*K,) ndarray
        Sorted interleaved array of valid omega ``[start, stop]`` pairs.
    etaEdges : (numEtas+1,) ndarray
        Eta bin-edge array in radians.
    omeEdges : (numOmes+1,) ndarray
        Omega bin-edge array in radians.
    etaOmeMaps : (nHKLS, numOmes, numEtas) ndarray
        Stacked eta-omega intensity maps, one slice per HKL family.
    etaIndices : (numEtas,) ndarray
        Array of valid eta pixel indices (``np.arange(numEtas)``).
    omeIndices : (numOmes,) ndarray
        Array of valid omega pixel indices (``np.arange(numOmes)``).
    dpix_eta : int
        Dilation half-width in the eta direction (pixels).
    dpix_ome : int
        Dilation half-width in the omega direction (pixels).
    threshold : (nHKLS,) ndarray
        Per-HKL intensity threshold values.

    Returns
    -------
    is_hit : int
        ``1`` if the angle is a hit (intensity above threshold within the
        dilation window), ``0`` otherwise.
    not_filtered : int
        ``1`` if the angle passed all validity filters and was actually
        tested, ``0`` if it was discarded by a filter.

    Notes
    -----
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
    angs_0: NDArray[np.float64],
    angs_1: NDArray[np.float64],
    symHKLs_ix: NDArray[np.intp],
    etaEdges: NDArray[np.float64],
    valid_eta_spans: NDArray[np.float64],
    valid_ome_spans: NDArray[np.float64],
    omeEdges: NDArray[np.float64],
    omePeriod: NDArray[np.float64],
    etaOmeMaps: NDArray[np.float64],
    etaIndices: NDArray[np.intp],
    omeIndices: NDArray[np.intp],
    dpix_eta: int,
    dpix_ome: int,
    threshold: NDArray[np.float64],
) -> float:
    """Accumulate completeness hits across all symmetry-equivalent reflections.

    Iterates over both oscillation-angle solutions (``angs_0`` and
    ``angs_1``) for every symmetry-equivalent HKL reflection and delegates
    to :func:`_angle_is_hit` for the per-angle validity check and hit
    detection.  Tracks which HKL family each reflection belongs to using
    the partition index array ``symHKLs_ix``.

    Parameters
    ----------
    angs_0 : (N, 3) ndarray
        First oscillation-angle solution array.  Each row is a
        ``(tth, eta, ome)`` triple in radians for one symmetry-equivalent
        HKL reflection.
    angs_1 : (N, 3) ndarray
        Second oscillation-angle solution array, same layout as ``angs_0``.
    symHKLs_ix : (nHKLS+1,) ndarray of int
        Cumulative index array partitioning rows of ``angs_0``/``angs_1``
        by HKL family.  Reflections ``symHKLs_ix[k] : symHKLs_ix[k+1]``
        belong to HKL family ``k``.
    etaEdges : (numEtas+1,) ndarray
        Eta bin-edge array in radians.
    valid_eta_spans : (2*K,) ndarray
        Normalised valid eta spans from :func:`_normalize_ranges`.
    valid_ome_spans : (2*K,) ndarray
        Normalised valid omega spans from :func:`_normalize_ranges`.
    omeEdges : (numOmes+1,) ndarray
        Omega bin-edge array in radians.
    omePeriod : (2,) array_like
        ``[ome_min, ome_max]`` defining the omega period in radians.
    etaOmeMaps : (nHKLS, numOmes, numEtas) ndarray
        Stacked eta-omega intensity maps, one slice per HKL family.
    etaIndices : (numEtas,) ndarray
        Valid eta pixel indices (``np.arange(numEtas)``).
    omeIndices : (numOmes,) ndarray
        Valid omega pixel indices (``np.arange(numOmes)``).
    dpix_eta : int
        Dilation half-width in the eta direction (pixels).
    dpix_ome : int
        Dilation half-width in the omega direction (pixels).
    threshold : (nHKLS,) ndarray
        Per-HKL intensity threshold values.

    Returns
    -------
    float
        Completeness score: ``hits / total_valid``, where ``total_valid``
        is the number of reflections that passed all validity filters.
        Returns ``0.0`` if no reflections are valid (avoids division by
        zero).
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
def _map_angle(angle: float, offset: float) -> float:
    """Numba-friendly equivalent to ``xf.mapAngle``.

    Maps ``angle`` into the half-open interval ``[offset, offset + 2*pi)``
    using a modulo operation.

    Parameters
    ----------
    angle : float
        Input angle in radians.
    offset : float
        Lower bound of the target interval in radians.

    Returns
    -------
    float
        ``angle`` mapped into ``[offset, offset + 2*pi)``.
    """
    return np.mod(angle - offset, 2 * np.pi) + offset
