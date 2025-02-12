#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:53:26 2020

@author: joel
"""
import argparse

import numpy as np

from hexrd.hedm import config
from hexrd.hedm.fitgrains import fit_grains
from hexrd.core import matrixutil as mutil
from hexrd.core import rotations as rot


def compare_grain_fits(
    fit_grain_params, ref_grain_params, mtol=1.0e-4, ctol=1.0e-3, vtol=1.0e-4
):
    """
    Executes comparison between reference and fit grain parameters for ff-HEDM
    for the same initial parameters.

    Parameters
    ----------
    fit_grain_params : array_like, (n, 12)
        The fit grain parameters to be tested.
    ref_grain_params : array_like, (n, 12)
        The reference grain parameters (see Notes below).

    Returns
    -------
    bool
        True is successful comparison

    Notes
    -----
    The fitgrains action currently returns
        grain_id, completeness, chisq, grain_params.
    We will have to assume that the grain_ids are in the *same order* as the
    reference, which can be enforces by running the comparison using the
    reference orientation list.
    """
    fit_grain_params = np.atleast_2d(fit_grain_params)
    ref_grain_params = np.atleast_2d(ref_grain_params)
    cresult = False
    ii = 0
    for fg, rg in zip(fit_grain_params, ref_grain_params):
        # test_orientation
        quats = rot.quatOfExpMap(np.vstack([fg[:3], rg[:3]]).T)
        ang, mis = rot.misorientation(
            quats[:, 0].reshape(4, 1), quats[:, 1].reshape(4, 1)
        )
        if ang <= mtol:
            cresult = True
        else:
            print("orientations for grain %d do not agree." % ii)
            return cresult

        # test position
        if np.linalg.norm(fg[3:6] - rg[3:6]) > ctol:
            print("centroidal coordinates for grain %d do not agree." % ii)
            return False

        # test strain
        vmat_fit = mutil.symmToVecMV(
            np.linalg.inv(mutil.vecMVToSymm(fg[6:])), scale=False
        )
        vmat_ref = mutil.symmToVecMV(
            np.linalg.inv(mutil.vecMVToSymm(rg[6:])), scale=False
        )
        if np.linalg.norm(vmat_fit - vmat_ref, ord=1) > vtol:
            print("stretch components for grain %d do not agree." % ii)
            return False

        # index grain id
        ii += 1
    return cresult


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Montage of spot data for a specifed G-vector family"
    )

    parser.add_argument('cfg_file', help="yaml HEDM config filename", type=str)
    parser.add_argument(
        'gt_ref', help="reference grain table filename", type=str
    )

    parser.add_argument(
        '-m',
        '--misorientation',
        help="misorientation threshold",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        '-c',
        '--centroid',
        help="centroid threshold",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        '-v', '--stretch', help="stretch threshold", type=float, default=1.0e-4
    )

    args = parser.parse_args()

    cfg_file = args.cfg_file
    gt_ref = args.gt_ref
    mtol = args.misorientation
    ctol = args.centroid
    vtol = args.stretch

    # load the config object
    cfg = config.open(cfg_file)[0]
    grains_table = np.loadtxt(gt_ref, ndmin=2)
    ref_grain_params = grains_table[:, 3:15]
    gresults = fit_grains(
        cfg,
        grains_table,
        show_progress=False,
        ids_to_refine=None,
        write_spots_files=False,
    )
    cresult = compare_grain_fits(
        np.vstack([i[-1] for i in gresults]),
        ref_grain_params,
        mtol=mtol,
        ctol=ctol,
        vtol=vtol,
    )
    if cresult:
        print("test passed")
    else:
        print("test failed")
