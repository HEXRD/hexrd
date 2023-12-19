#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:06:02 2020

@author: bernier2
"""

from collections import namedtuple
import logging

import numpy as np

from hexrd.material.crystallography import PlaneData
from hexrd.rotations import misorientation


# =============================================================================
# ORIENTATION LISTS
# =============================================================================


def compare_quaternion_lists(new_quats, ref_quats, tol=0.05):
    """
    Test an array of quaternions against reference values.

    Parameters
    ----------
    new_quats : array_like, (n, 4)
        The (n, 4) array of unit quaternions to test against reference.
    ref_quats : array_like, Nn, 4)
        The (n, 4) array of reference unit quaternions.
    tol : scalar, optional
        The angular tolerance in degrees for the comparison.
        The default is 0.05.

    Raises
    ------
    RuntimeError
        Will raise if either the length of `new_quats` does not match the
        reference, or the minimum misorientation of a member of `new_quats`
        w.r.t. the reference is greater than `tol`.

    Returns
    -------
    None.

    """
    nquats = len(ref_quats)  # 3 for multiruby case

    # FIRST CHECK THAT NUMBER OF ORIENTATIONS MATCHES
    if len(new_quats) != nquats:
        raise RuntimeError(
            "Incorrect number of orientations found; should be %d" % nquats
            + ", currently found %d" % len(new_quats)
        )

    # NEXT CHECK THE ACTUAL MISORIENTATIONS
    # !!! order may be different
    for i, nq in enumerate(new_quats):
        ang, mis = misorientation(nq.reshape(4, 1), ref_quats.T)
        if np.min(ang) > np.radians(tol):
            raise RuntimeError(
                "Misorientation for test orientation %d " % i
                + "is greater than threshold"
            )

# =============================================================================
# ETA-OMEGA MAPS
# =============================================================================
EOMap = namedtuple('EOMap',
                   ['data', 'eta', 'eta_edges', 'omega', 'omega_edges',
                    'hkl_indices', 'plane_data']
)

_keys = [
    'dataStore',
    'etas',
    'etaEdges',
    'iHKLList',
    'omegas',
    'omeEdges',
    'planeData_args',
    'planeData_hkls'
]


def load(npz):
    """load eta-omega map from file"""
    e = np.load(npz, allow_pickle=True)
    return EOMap(
        e['dataStore'],
        e['etas'],
        e['etaEdges'],
        e['omegas'],
        e['omeEdges'],
        e['iHKLList'],
        plane_data(e)
    )


def compare(e1, e2):
    """compare eta-omega maps"""
    return Comparison(e1, e2).compare()


def compare_plane_data(pd1, pd2):
    """Compare two plane data instances"""
    raise NotImplementedError


def plane_data(e):
    hkls = e['planeData_hkls']
    args = e['planeData_args']
    if len(args) > 4:
        args = args[:4]
    return PlaneData(hkls, *args)


class Comparison:

        def __init__(self, e1, e2):
            self.e1 = e1
            self.e2 = e2
            self.tol = 1.0e-6

        def compare(self):
            """Compare whether maps are same or not"""
            same = self.eta()[0] and self.omega()[0] and self.data()[0]
            return same

        def eta(self):
            """compare etas"""
            eta1 = self.e1.eta
            eta2 = self.e2.eta
            l1, l2 = len(eta1), len(eta2)
            if l1 != l2:
                msg = "eta: lengths differ: %d and %d" % (l1, l2)
                logging.info(msg)
                return False, msg

            nrmdiff = np.linalg.norm(eta1 - eta2)
            if nrmdiff < self.tol:
                return True, "eta: same"
            else:
                msg = "eta: norm of difference: %s" % nrmdiff
                logging.info(msg)
                return False, msg

        def omega(self):
            """compare omegas"""
            omega1 = self.e1.omega
            omega2 = self.e2.omega
            l1, l2 = len(omega1), len(omega2)
            if l1 != l2:
                msg = "omega: lengths differ: %d and %d" % (l1, l2)
                logging.info(msg)
                return False, msg

            nrmdiff = np.linalg.norm(omega1 - omega2)
            if nrmdiff < self.tol:
                return True, "omega: same"
            else:
                msg = "omega: norm of difference: %s" % nrmdiff
                logging.info(msg)
                return False, msg

        def hkl_indices(self):
            hkl1, hkl2  = self.e1.hkl_indices, self.e2.hkl_indices
            n1, n2 = len(hkl1), len(hkl2)
            if n1 != n2:
                return False, "hkl: lengths differ: %d and %d" % (n1, n2)
            for i in range(n1):
                if hkl1[i] != hkl2[i]:
                    return False, "hkl: indices not the same"

            return True, "hkl: same"

        def data(self):
            d1, d2 = self.e1.data, self.e2.data
            if d1.shape != d2.shape:
                msg = "data shapes do not match: " % (d1.shape, d2.shape)
                logging.info(msg)
                return False, msg

            for ind in range(d1.shape[0]):
                d1i, d2i  = d1[ind], d2[ind]
                nnan1 = np.count_nonzero(np.isnan(d1i))
                nnan2 = np.count_nonzero(np.isnan(d2i))
                # print("number nans: ", nnan1, nnan2)
                if nnan1 > 0:
                    d1i = np.nan_to_num(d1i)
                if nnan2 > 0:
                    d2i = np.nan_to_num(d1i)

                nnz1 = np.count_nonzero(d1i)
                nnz2 = np.count_nonzero(d2i)
                if nnz1 != nnz2:
                    msg = "data: map %d: number nonzero differ: %d, %d" % (ind, nnz1, nnz2)
                    logging.info(msg)
                    return False, msg

                overlapping = d1i.astype(bool) | d2i.astype(bool)
                nnz = np.count_nonzero(overlapping)
                if nnz != nnz1:
                    msg = "data: map %d: overlaps differ: %d, %d" % (ind, nnz1, nnz)
                    logging.info(msg)
                    return False, msg

                d1over = d1i[overlapping]
                d2over = d2i[overlapping]
                diff = np.linalg.norm(d1over - d2over)
                if diff < self.tol:
                    return True, "data: same"
                else:
                    msg = "data: map %s: map values differ" % (ind)
                    logging.info(msg)
                    return False, msg


            return True, "data: same"
