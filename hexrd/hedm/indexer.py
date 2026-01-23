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
import timeit

import numpy as np
import numba

from scipy.spatial.transform import Rotation as R

from hexrd.core import constants
from hexrd.core.transforms import xfcapi


omega_period_DFLT = np.radians(np.r_[-180.0, 180.0])

paramMP = None
nCPUs_DFLT = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)


# @numba.njit(cache=True, nogil=True)
def _evaluate_windows_numba(
    eta_idx,
    ome_idx,
    hkl_ids,
    valid,
    nan_masks,
    gt_masks,
    ds_shapes,
    dpix_eta,
    dpix_ome,
):
    """
    Numba-accelerated evaluation of window hits.

    Parameters
    ----------
    eta_idx, ome_idx : int64[:]
        Bin indices for each candidate (same length).
    hkl_ids : int64[:]
        Parent HKL index for each candidate.
    valid : bool[:]
        Validity mask for candidates.
    nan_masks, gt_masks : list of 2D boolean arrays
        Precomputed per-HKL masks.
    ds_shapes : int64[:, :]
        Shape (ome, eta) per HKL.
    dpix_eta, dpix_ome : int
        Pixel tolerances.

    Returns
    -------
    hits : int
    total : int
    """
    hits = 0
    total = 0

    for i in range(valid.shape[0]):
        if not valid[i]:
            continue

        o0 = max(0, ome_idx[i] - dpix_ome)
        o1 = min(ome_idx[i] + dpix_ome + 1, ds_shapes[hkl_ids[i], 0])

        e0 = max(0, eta_idx[i] - dpix_eta)
        e1 = min(eta_idx[i] + dpix_eta + 1, ds_shapes[hkl_ids[i], 1])

        if np.any(nan_masks[hkl_ids[i]][o0:o1, e0:e1]):
            hits -= 1
        elif np.any(gt_masks[hkl_ids[i]][o0:o1, e0:e1]):
            hits += 1
            total += 1
        else:
            total += 1

    return hits, total

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
):
    quats = np.atleast_2d(quats)
    if quats.size == 4:
        quats = quats.reshape(4, 1)

    planeData = etaOmeMaps.planeData

    hklIDs = np.asarray(etaOmeMaps.iHKLList)
    hkl_idx = planeData.getHKLID(planeData.getHKLs(*hklIDs).T, master=False)
    nHKLS = len(hklIDs)

    numEtas = len(etaOmeMaps.etaEdges) - 1
    numOmes = len(etaOmeMaps.omeEdges) - 1

    if threshold is None:
        threshold = np.zeros(nHKLS)
        for i in range(nHKLS):
            ds = etaOmeMaps.dataStore[i]
            threshold[i] = np.mean(np.r_[np.mean(ds), np.median(ds)])
    elif not hasattr(threshold, "__len__"):
        threshold = threshold * np.ones(nHKLS)
    elif len(threshold) != nHKLS:
        raise RuntimeError("threshold list is wrong length!")

    if bMat is None:
        bMat = planeData.latVecOps["B"]

    symHKLs = planeData.getSymHKLs()
    symHKLs = [symHKLs[i] for i in hkl_idx]
    symHKLs_ix = np.add.accumulate([0] + [s.shape[1] for s in symHKLs])
    symHKLs = np.vstack([s.T for s in symHKLs])

    hkl_for_sym = np.repeat(np.arange(nHKLS), np.diff(symHKLs_ix))

    data_store = etaOmeMaps.dataStore
    nan_masks = [np.isnan(ds) for ds in data_store]
    gt_masks = [ds > threshold[i] for i, ds in enumerate(data_store)]

    ds_shapes = np.array(
        [ds.shape for ds in etaOmeMaps.dataStore],
        dtype=np.int64
    )

    # --- ranges and tolerances (same semantics) ---
    if omegaRange is None:
        omeMin = np.array([np.min(etaOmeMaps.omeEdges)])
        omeMax = np.array([np.max(etaOmeMaps.omeEdges)])
    else:
        omeMin = omegaRange[:, 0]
        omeMax = omegaRange[:, 1]

    if etaRange is None:
        etaMin = np.array([-np.pi])
        etaMax = np.array([np.pi])
    else:
        etaMin = etaRange[:, 0]
        etaMax = etaRange[:, 1]

    valid_eta_spans = _normalize_ranges(etaMin, etaMax, -np.pi)
    valid_ome_spans = _normalize_ranges(omeMin, omeMax, min(omePeriod))

    dpix_ome = int(round(omeTol / abs(etaOmeMaps.omeEdges[1] - etaOmeMaps.omeEdges[0])))
    dpix_eta = int(round(etaTol / abs(etaOmeMaps.etaEdges[1] - etaOmeMaps.etaEdges[0])))
    ome_offset = np.min(omePeriod)

    def interleave_two_solutions(a0, a1):
        out = np.empty((a0.shape[0] * 2, 3), dtype=a0.dtype)
        out[0::2, :] = a0
        out[1::2, :] = a1
        return out

    retval = []
    start = timeit.default_timer()

    for quat in quats.T:
        rmat = R.from_quat(np.roll(quat, -1)).as_matrix()

        oangs0, oangs1 = xfcapi.oscill_angles_of_hkls(symHKLs, 0.0, rmat, bMat, planeData.wavelength)

        oangs = interleave_two_solutions(oangs0, oangs1)
        hkl_ids = np.repeat(hkl_for_sym, 2)

        tth = oangs[:, 0]
        eta = oangs[:, 1]
        ome = oangs[:, 2]

        valid = ~np.isnan(tth)

        eta = np.mod(eta + np.pi, 2 * np.pi) - np.pi
        ome = np.mod(ome - ome_offset, 2 * np.pi) + ome_offset

        eta_span_idx = np.searchsorted(valid_eta_spans, eta, side='right')
        ome_span_idx = np.searchsorted(valid_ome_spans, ome, side='right')

        valid &= (eta_span_idx & 1) == 1
        valid &= (ome_span_idx & 1) == 1

        eta_idx = np.searchsorted(etaOmeMaps.etaEdges, eta, side='right') - 1
        ome_idx = np.searchsorted(etaOmeMaps.omeEdges, ome, side='right') - 1

        valid &= eta_idx >= 0
        valid &= ome_idx >= 0
        valid &= eta_idx < numEtas
        valid &= ome_idx < numOmes

        eta_idx = eta_idx.astype(np.int64, copy=False)
        ome_idx = ome_idx.astype(np.int64, copy=False)

        hits, total = _evaluate_windows_numba(
            eta_idx,
            ome_idx,
            hkl_ids,
            valid,
            nan_masks,
            gt_masks,
            ds_shapes,
            dpix_eta,
            dpix_ome,
        )
        retval.append(0.0 if total == 0 else hits / total)

    elapsed = timeit.default_timer() - start
    logger.info("paintGrid took %.3f seconds", elapsed)

    return retval


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

    if not np.all(starts < stops):
        raise ValueError("Invalid angle ranges")

    if np.any(stops - starts >= 2 * np.pi - 1e-8):
        return np.array([offset, offset + 2 * np.pi])

    starts = (starts - offset) % (2 * np.pi) + offset
    stops  = (stops  - offset) % (2 * np.pi) + offset

    if not np.all(starts[1:] > stops[:-2]):
        raise ValueError("Angle ranges overlap")

    order = np.argsort(starts)
    result = np.concatenate((starts[order], stops[order]))

    if result[-1] < result[-2]:
        result = np.concatenate((
            [offset, result[-1]],
            result[:-1],
            [offset + 2 * np.pi],
        ))

    return result

