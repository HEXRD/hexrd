#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""
import os
import logging
import multiprocessing
import numpy as np
import timeit

from hexrd import constants as cnst
from hexrd import instrument
from hexrd.transforms import xfcapi
from hexrd.fitting import fitGrain, objFuncFitGrain, gFlag_ref

logger = logging.getLogger(__name__)


# multiprocessing fit funcs
def fit_grain_FF_init(params):
    """
    Broadcast the fitting parameters as globals for multiprocessing

    Parameters
    ----------
    params : dict
        The dictionary of fitting parameters.

    Returns
    -------
    None.

    Notes
    -----
    See fit_grain_FF_reduced for specification.
    """
    global paramMP
    paramMP = params


def fit_grain_FF_cleanup():
    """
    Tears down the global fitting parameters.
    """
    global paramMP
    del paramMP


def fit_grain_FF_reduced(grain_id):
    """
    Perform non-linear least-square fit for the specified grain.

    Parameters
    ----------
    grain_id : int
        The grain id.

    Returns
    -------
    grain_id : int
        The grain id.
    completeness : float
        The ratio of predicted to measured (observed) Bragg reflections.
    chisq: float
        Figure of merit describing the sum of squared residuals for each Bragg
        reflection in the form (x, y, omega) normalized by the total number of
        degrees of freedom.
    grain_params : array_like
        The optimized grain parameters
        [<orientation [3]>, <centroid [3]> <inverse stretch [6]>].

    Notes
    -----
    input parameters are
    [plane_data, instrument, imgser_dict,
    tth_tol, eta_tol, ome_tol, npdiv, threshold]
    """
    grains_table = paramMP['grains_table']
    plane_data = paramMP['plane_data']
    instrument = paramMP['instrument']
    imgser_dict = paramMP['imgser_dict']
    tth_tol = paramMP['tth_tol']
    eta_tol = paramMP['eta_tol']
    ome_tol = paramMP['ome_tol']
    npdiv = paramMP['npdiv']
    refit = paramMP['refit']
    threshold = paramMP['threshold']
    eta_ranges = paramMP['eta_ranges']
    ome_period = paramMP['ome_period']
    analysis_dirname = paramMP['analysis_dirname']
    spots_filename = paramMP['spots_filename']

    grain = grains_table[grain_id]
    grain_params = grain[3:15]

    for tols in zip(tth_tol, eta_tol, ome_tol):
        complvec, results = instrument.pull_spots(
            plane_data, grain_params,
            imgser_dict,
            tth_tol=tols[0],
            eta_tol=tols[1],
            ome_tol=tols[2],
            npdiv=npdiv, threshold=threshold,
            eta_ranges=eta_ranges,
            ome_period=ome_period,
            dirname=analysis_dirname, filename=spots_filename % grain_id,
            save_spot_list=False,
            quiet=True, check_only=False, interp='nearest')

        # ======= DETERMINE VALID REFLECTIONS =======

        culled_results = dict.fromkeys(results)
        num_refl_tot = 0
        num_refl_valid = 0
        for det_key in culled_results:
            panel = instrument.detectors[det_key]

            '''
            grab panel results:
                peak_id
                hkl_id
                hkl
                sum_int
                max_int,
                pred_angs,
                meas_angs,
                meas_xy
            '''
            presults = results[det_key]
            nrefl = len(presults)

            # make data arrays
            refl_ids = np.empty(nrefl)
            max_int = np.empty(nrefl)
            for i, spot_data in enumerate(presults):
                refl_ids[i] = spot_data[0]
                max_int[i] = spot_data[4]

            valid_refl_ids = refl_ids >= 0

            # find unsaturated spots on this panel
            unsat_spots = np.ones(len(valid_refl_ids), dtype=bool)
            if panel.saturation_level is not None:
                unsat_spots[valid_refl_ids] = \
                    max_int[valid_refl_ids] < panel.saturation_level

            idx = np.logical_and(valid_refl_ids, unsat_spots)

            # if an overlap table has been written, load it and use it
            overlaps = np.zeros_like(idx, dtype=bool)
            try:
                ot = np.load(
                    os.path.join(
                        analysis_dirname, os.path.join(
                            det_key, 'overlap_table.npz'
                        )
                    )
                )
                for key in ot.keys():
                    for this_table in ot[key]:
                        these_overlaps = np.where(
                            this_table[:, 0] == grain_id)[0]
                        if len(these_overlaps) > 0:
                            mark_these = np.array(
                                this_table[these_overlaps, 1], dtype=int
                            )
                            otidx = [
                                np.where(refl_ids == mt)[0]
                                for mt in mark_these
                            ]
                            overlaps[otidx] = True
                idx = np.logical_and(idx, ~overlaps)
                # logger.info("found overlap table for '%s'", det_key)
            except(IOError, IndexError):
                # logger.info("no overlap table found for '%s'", det_key)
                pass

            # attach to proper dict entry
            # FIXME: want to avoid looping again here
            culled_results[det_key] = [presults[i] for i in np.where(idx)[0]]
            num_refl_tot += len(valid_refl_ids)
            num_refl_valid += sum(valid_refl_ids)

            pass  # now we have culled data

        # CAVEAT: completeness from pullspots only; incl saturated and overlaps
        # <JVB 2015-12-15>
        completeness = num_refl_valid / float(num_refl_tot)

        # ======= DO LEASTSQ FIT =======

        if num_refl_valid <= 12:    # not enough reflections to fit... exit
            return grain_id, completeness, np.inf, grain_params
        else:
            grain_params = fitGrain(
                    grain_params, instrument, culled_results,
                    plane_data.latVecOps['B'], plane_data.wavelength
                )
            # get chisq
            # TODO: do this while evaluating fit???
            chisq = objFuncFitGrain(
                    grain_params[gFlag_ref], grain_params, gFlag_ref,
                    instrument,
                    culled_results,
                    plane_data.latVecOps['B'], plane_data.wavelength,
                    ome_period,
                    simOnly=False, return_value_flag=2)
            pass  # end conditional on fit
        pass  # end tolerance looping

    if refit is not None:
        # first get calculated x, y, ome from previous solution
        # NOTE: this result is a dict
        xyo_det_fit_dict = objFuncFitGrain(
            grain_params[gFlag_ref], grain_params, gFlag_ref,
            instrument,
            culled_results,
            plane_data.latVecOps['B'], plane_data.wavelength,
            ome_period,
            simOnly=True, return_value_flag=2)

        # make dict to contain new culled results
        culled_results_r = dict.fromkeys(culled_results)
        num_refl_valid = 0
        for det_key in culled_results_r:
            presults = culled_results[det_key]

            ims = imgser_dict[det_key]
            ome_step = sum(np.r_[-1, 1]*ims.metadata['omega'][0, :])

            xyo_det = np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in presults])
            )

            xyo_det_fit = xyo_det_fit_dict[det_key]

            xpix_tol = refit[0]*panel.pixel_size_col
            ypix_tol = refit[0]*panel.pixel_size_row
            fome_tol = refit[1]*ome_step

            # define difference vectors for spot fits
            x_diff = abs(xyo_det[:, 0] - xyo_det_fit['calc_xy'][:, 0])
            y_diff = abs(xyo_det[:, 1] - xyo_det_fit['calc_xy'][:, 1])
            ome_diff = np.degrees(
                xfcapi.angularDifference(xyo_det[:, 2],
                                         xyo_det_fit['calc_omes'])
                )

            # filter out reflections with centroids more than
            # a pixel and delta omega away from predicted value
            idx_new = np.logical_and(
                x_diff <= xpix_tol,
                np.logical_and(y_diff <= ypix_tol,
                               ome_diff <= fome_tol)
                               )

            # attach to proper dict entry
            culled_results_r[det_key] = [
                presults[i] for i in np.where(idx_new)[0]
            ]

            num_refl_valid += sum(idx_new)
            pass

        # only execute fit if left with enough reflections
        if num_refl_valid > 12:
            grain_params = fitGrain(
                grain_params, instrument, culled_results_r,
                plane_data.latVecOps['B'], plane_data.wavelength
            )
            # get chisq
            # TODO: do this while evaluating fit???
            chisq = objFuncFitGrain(
                    grain_params[gFlag_ref],
                    grain_params, gFlag_ref,
                    instrument,
                    culled_results_r,
                    plane_data.latVecOps['B'], plane_data.wavelength,
                    ome_period,
                    simOnly=False, return_value_flag=2)
            pass
        pass  # close refit conditional
    return grain_id, completeness, chisq, grain_params


def fit_grains(cfg,
               force=False, clean=False,
               show_progress=False, ids_to_refine=None):
    """
    Performs optimization of grain parameters.

    operates on a single HEDM config block
    """
    grains_filename = os.path.join(
        cfg.analysis_dir, 'grains.out'
    )

    # grab imageseries dict
    imsd = cfg.image_series

    # grab instrument
    instr = cfg.instrument.hedm

    # process plane data
    plane_data = cfg.material.plane_data
    tth_max = cfg.fit_grains.tth_max
    if isinstance(tth_max, bool):
        if tth_max:
            max_tth = instrument.max_tth(instr)
            plane_data.tThMax = max_tth
            logger.info("\tsetting the maximum 2theta to instrument"
                        + " maximum: %.2f degrees",
                        np.degrees(max_tth))
        else:
            logger.info("\tnot adjusting exclusions in planeData")
    else:
        # a value for tth max has been specified
        plane_data.exclusions = None
        plane_data.tThMax = np.radians(tth_max)
        logger.info("\tsetting the maximum 2theta to %.2f degrees",
                    tth_max)

    # make output directories
    if not os.path.exists(cfg.analysis_dir):
        os.mkdir(cfg.analysis_dir)
        for det_key in instr.detectors:
            os.mkdir(os.path.join(cfg.analysis_dir, det_key))
    else:
        # make sure panel dirs exist under analysis dir
        for det_key in instr.detectors:
            if not os.path.exists(os.path.join(cfg.analysis_dir, det_key)):
                os.mkdir(os.path.join(cfg.analysis_dir, det_key))

    # grab eta ranges and ome_period
    eta_ranges = np.radians(cfg.find_orientations.eta.range)

    # handle omega period
    # !!! we assume all detector ims have the same ome ranges, so any will do!
    oims = next(iter(imsd.values()))
    ome_period = np.radians(oims.omega[0, 0] + np.r_[0., 360.])

    # number of processes
    ncpus = cfg.multiprocessing

    # threshold for fitting
    threshold = cfg.fit_grains.threshold

    # some conditions for arg handling
    existing_analysis = os.path.exists(grains_filename)
    new_with_estimate = not existing_analysis \
        and cfg.fit_grains.estimate is not None
    new_without_estimate = not existing_analysis \
        and cfg.fit_grains.estimate is None
    force_with_estimate = force and cfg.fit_grains.estimate is not None
    force_without_estimate = force and cfg.fit_grains.estimate is None

    # handle args
    if clean or force_without_estimate or new_without_estimate:
        # need accepted orientations from indexing in this case
        if clean:
            logger.info(
                "'clean' specified; ignoring estimate and using default"
            )
        elif force_without_estimate:
            logger.info(
                "'force' option specified, but no initial estimate; "
                + "using default"
            )
        try:
            qbar = np.loadtxt(
                'accepted_orientations_' + cfg.analysis_id + '.dat',
                ndmin=2).T

            gw = instrument.GrainDataWriter(grains_filename)
            for i_g, q in enumerate(qbar.T):
                phi = 2*np.arccos(q[0])
                n = xfcapi.unitRowVector(q[1:])
                grain_params = np.hstack(
                    [phi*n, cnst.zeros_3, cnst.identity_6x1]
                )
                gw.dump_grain(int(i_g), 1., 0., grain_params)
            gw.close()
        except(IOError):
            raise(RuntimeError,
                  "indexing results '%s' not found!"
                  % 'accepted_orientations_' + cfg.analysis_id + '.dat')
    elif force_with_estimate or new_with_estimate:
        grains_filename = cfg.fit_grains.estimate
    elif existing_analysis and not (clean or force):
        raise(RuntimeError,
              "fit results '%s' exist, " % grains_filename
              + "but --clean or --force options not specified")

    # load grains table
    grains_table = np.loadtxt(grains_filename, ndmin=2)
    if ids_to_refine is not None:
        grains_table = np.atleast_2d(grains_table[ids_to_refine, :])
    spots_filename = "spots_%05d.out"
    params = dict(
            grains_table=grains_table,
            plane_data=plane_data,
            instrument=instr,
            imgser_dict=imsd,
            tth_tol=cfg.fit_grains.tolerance.tth,
            eta_tol=cfg.fit_grains.tolerance.eta,
            ome_tol=cfg.fit_grains.tolerance.omega,
            npdiv=cfg.fit_grains.npdiv,
            refit=cfg.fit_grains.refit,
            threshold=threshold,
            eta_ranges=eta_ranges,
            ome_period=ome_period,
            analysis_dirname=cfg.analysis_dir,
            spots_filename=spots_filename)

    # =====================================================================
    # EXECUTE MP FIT
    # =====================================================================

    # DO FIT!
    if len(grains_table) == 1 or ncpus == 1:
        logger.info("\tstarting serial fit")
        start = timeit.default_timer()
        fit_grain_FF_init(params)
        fit_results = list(
            map(fit_grain_FF_reduced,
                np.array(grains_table[:, 0], dtype=int))
        )
        fit_grain_FF_cleanup()
        elapsed = timeit.default_timer() - start
    else:
        nproc = min(ncpus, len(grains_table))
        chunksize = max(1, len(grains_table)//ncpus)
        logger.info("\tstarting fit on %d processes with chunksize %d",
                    nproc, chunksize)
        start = timeit.default_timer()
        pool = multiprocessing.Pool(
            nproc,
            fit_grain_FF_init,
            (params, )
        )
        fit_results = pool.map(
            fit_grain_FF_reduced,
            np.array(grains_table[:, 0], dtype=int),
            chunksize=chunksize
        )
        pool.close()
        pool.join()
        elapsed = timeit.default_timer() - start
    logger.info("fitting took %f seconds", elapsed)

    # =====================================================================
    # WRITE OUTPUT
    # =====================================================================

    gw = instrument.GrainDataWriter(
        os.path.join(cfg.analysis_dir, 'grains.out')
    )
    for fit_result in fit_results:
        gw.dump_grain(*fit_result)
        pass
    gw.close()
