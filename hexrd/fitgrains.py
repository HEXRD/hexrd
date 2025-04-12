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
import warnings

from hexrd import rotations
from hexrd.fitting import fitGrain, objFuncFitGrain, gFlag_ref
from hexrd.spot_tracking.assign_spots import assign_spots_to_hkls
from hexrd.spot_tracking.track_spots import track_combine_and_chunk_spots

logger = logging.getLogger(__name__)

# These are file names that were hardwired in the code. I am putting them
# here so they will be easier to find if we want to make them user inputs
# at some point later.

OVERLAP_TABLE_FILE = 'overlap_table.npz'
SPOTS_OUT_FILE = "spots_%05d.out"


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
    print('Running fit_grains for grain ID:', grain_id)
    grains_table = paramMP['grains_table']
    plane_data = paramMP['plane_data']
    instrument = paramMP['instrument']
    imgser_dict = paramMP['imgser_dict']
    tth_tol = paramMP['tth_tol']
    eta_tol = paramMP['eta_tol']
    ome_tol = paramMP['ome_tol']
    refit = paramMP['refit']
    ome_period = paramMP['ome_period']
    analysis_dirname = paramMP['analysis_dirname']

    grain = grains_table[grain_id]
    grain_params = grain[3:15]

    for tols in zip(tth_tol, eta_tol, ome_tol):
        results = extract_spots(grain_id, grain_params, tols, paramMP)

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
                            det_key, OVERLAP_TABLE_FILE
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
            except (IOError, IndexError):
                # logger.info("no overlap table found for '%s'", det_key)
                pass

            # attach to proper dict entry
            # FIXME: want to avoid looping again here
            culled_results[det_key] = [presults[i] for i in np.where(idx)[0]]
            num_refl_tot += len(valid_refl_ids)
            num_refl_valid += sum(valid_refl_ids)
            # now we have culled data

        # CAVEAT: completeness from pullspots only; incl saturated and overlaps
        # <JVB 2015-12-15>
        try:
            completeness = num_refl_valid / float(num_refl_tot)
        except ZeroDivisionError:
            raise RuntimeError(
                "simulated number of relfections is 0; "
                + "check instrument config or grain parameters"
            )

        # ======= DO LEASTSQ FIT =======

        if num_refl_valid <= 12:    # not enough reflections to fit... exit
            warnings.warn(
                f'Not enough valid reflections ({num_refl_valid}) to fit, '
                f'exiting',
                RuntimeWarning
            )
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

            if not presults:
                culled_results_r[det_key] = []
                continue

            ims = next(iter(imgser_dict.values()))  # grab first for the omes
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
                rotations.angularDifference(xyo_det[:, 2],
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

    return grain_id, completeness, chisq, grain_params


def extract_spots(
    grain_id: int,
    grain_params: np.ndarray,
    tols: list[float],
    params: dict,
) -> dict[str, list]:
    # Extract the spots either by running pull_spots(), or by obtaining the
    # spots from a data file (if one was specified)
    if params.get('spots_data_file'):
        # There's a spots data file with spots already identified. Use that.
        return _assign_spots_data(grain_params, tols, params)
    else:
        # Run pull_spots() and find the spots.
        return _run_pull_spots(grain_id, grain_params, tols, params)


def _assign_spots_data(
    grain_params: np.ndarray,
    tols: list[float],
    params: dict,
) -> dict[str, list]:
    spots_filename = params['spots_data_file']
    plane_data = params['plane_data']
    instr = params['instrument']
    ims_dict = params['imgser_dict']
    eta_ranges = params['eta_ranges']
    eta_period = (-np.pi, np.pi)
    omega_period = params['ome_period']

    first_ims = next(iter(ims_dict.values()))
    omegas = first_ims.metadata['omega']

    # Just assume the whole omega range is used
    omega_ranges = [omega_period]

    num_images = len(first_ims)

    # Track, combine, and chunk spots into subpanels
    spot_arrays = track_combine_and_chunk_spots(
        spots_filename,
        instr,
        num_images,
        omegas,
    )

    tolerances = np.asarray(tols)

    # Now assign spots to HKLs
    assigned_spots = assign_spots_to_hkls(
        spot_arrays,
        instr,
        tolerances,
        eta_period,
        plane_data,
        np.atleast_2d(grain_params),
        eta_ranges,
        omega_ranges,
        omega_period,
        num_images,
    )

    """
    To match the current `pull_spots()` output, we reorder
    the output to be more like the following:

    peak_id
    hkl_id
    hkl
    sum_int
    max_int
    pred_angs
    meas_angs
    meas_xy

    FIXME: we can modernize/simplify this sometime? But it might
    take some effort.
    """
    results = {}
    for det_key, spots in assigned_spots.items():
        # We only ran for one grain
        spots = spots[0]

        data_list = []
        for i, hkl in enumerate(spots['hkls']):
            peak_id = i
            hkl_id = i  # FIXME: hkl_id has no meaning in this setup
            sum_int = spots['sum_int'][i]
            max_int = spots['max_int'][i]
            pred_angs = spots['sim_angs'][i]
            meas_angs = spots['meas_angs'][i]
            meas_xy = spots['meas_xys'][i][:2]
            data_list.append([
                peak_id,
                hkl_id,
                hkl,
                sum_int,
                max_int,
                pred_angs,
                meas_angs,
                meas_xy,
            ])

        # Put in some fillers for invalid peaks
        for i in range(spots['num_hkls_skipped']):
            data_list.append([
                -1,
                -1,
                [0, 0, 0],
                0,
                0,
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan],
            ])

        results[det_key] = data_list

    return results


def _run_pull_spots(
    grain_id: int,
    grain_params: np.ndarray,
    tols: list[float],
    params: dict,
) -> dict[str, list]:
    plane_data = params['plane_data']
    instrument = params['instrument']
    imgser_dict = params['imgser_dict']
    npdiv = params['npdiv']
    threshold = params['threshold']
    eta_ranges = params['eta_ranges']
    ome_period = params['ome_period']
    analysis_dirname = params['analysis_dirname']
    prefix = params['spots_filename']
    spots_filename = None if prefix is None else prefix % grain_id

    complvec, results = instrument.pull_spots(
        plane_data, grain_params,
        imgser_dict,
        tth_tol=tols[0],
        eta_tol=tols[1],
        ome_tol=tols[2],
        npdiv=npdiv, threshold=threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        dirname=analysis_dirname, filename=spots_filename,
        return_spot_list=False,
        quiet=True, check_only=False, interp='nearest')

    return results


def fit_grains(cfg,
               grains_table,
               show_progress=False,
               ids_to_refine=None,
               write_spots_files=True,
               check_if_canceled_func=None):
    """
    Performs optimization of grain parameters.

    operates on a single HEDM config block

    The `check_if_canceled_func` has the following signature:

        check_if_canceled_func() -> bool

    If it returns `True`, it indicates that fit_grains should be canceled.
    This is done by terminating the multiprocessing processes.

    If `check_if_canceled_func` is set, multiprocessing will be performed,
    even if there is only one grain or one process, so that it will be
    cancelable.
    """
    grains_table = np.atleast_2d(grains_table)

    # grab imageseries dict
    imsd = cfg.image_series

    # grab instrument
    instr = cfg.instrument.hedm

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

    if ids_to_refine is not None:
        grains_table = np.atleast_2d(grains_table[ids_to_refine, :])

    spots_filename = SPOTS_OUT_FILE if write_spots_files else None
    params = dict(
            grains_table=grains_table,
            plane_data=cfg.material.plane_data,
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

    run_serial = (
        (len(grains_table) == 1 or ncpus == 1)
        and not check_if_canceled_func
    )
    if cfg.fit_grains.spots_data_file is not None:
        # Use the spots data file instead of `pull_spots()`
        params['spots_data_file'] = cfg.fit_grains.spots_data_file

    # =====================================================================
    # EXECUTE MP FIT
    # =====================================================================

    # DO FIT!
    if run_serial:
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

        async_result = pool.map_async(
            fit_grain_FF_reduced,
            np.array(grains_table[:, 0], dtype=int),
            chunksize=chunksize
        )
        while not async_result.ready():
            if check_if_canceled_func and check_if_canceled_func():
                pool.terminate()
                logger.info('Fit grains canceled.')
                # Perform an early return if we need to cancel.
                return None

            async_result.wait(0.25)

        fit_results = async_result.get()
        pool.close()
        pool.join()
        elapsed = timeit.default_timer() - start
    logger.info("fitting took %f seconds", elapsed)
    return fit_results
