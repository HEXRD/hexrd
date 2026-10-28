#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:04:10 2017

@author: bernier2
"""

import os
import gc
import logging
import multiprocessing
import numpy as np
import timeit
import warnings
from queue import Empty, Full

from collections import defaultdict

from hexrd.core import constants
from hexrd.core import instrument
from hexrd.core import rotations
from .fitting import fitGrain, objFuncFitGrain, gFlag_ref
from hexrd.core.instrument._spotutils import (
    _SpotData,
    sparse_frame_view,
    clear_sparse_frame_views,
)
from hexrd.hedm._progress import (
    activity,
    check_cancelled,
    current_dashboard,
    detach_dashboard,
    diagnostic,
    progress_bar,
    report_grain,
    report_fit_metrics,
    stage,
)

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
    if multiprocessing.current_process().name != 'MainProcess':
        detach_dashboard()
    from hexrd.hedm._fit_progress import FitReporter, set_reporter

    queue = params.get('progress_queue')
    set_reporter(
        FitReporter(
            queue,
            lazy=params.get('lazy_images', False),
            cache_limit=params.get('image_cache_limit', 0),
        )
        if params.get('collect_metrics', current_dashboard() is not None)
        else None
    )


def fit_grain_FF_cleanup():
    """
    Tears down the global fitting parameters.
    """
    global paramMP
    del paramMP
    from hexrd.hedm._fit_progress import set_reporter

    set_reporter(None)


def _grain_event(grain_id, state, index, passes, completeness=None, chisq=None):
    if state == 'Extracting':
        paramMP['progress_pass'] = index
    event = (int(grain_id), state, index, passes, completeness, chisq)
    from hexrd.hedm._fit_progress import current_reporter

    reporter = current_reporter()
    if reporter is not None:
        reporter.grain(grain_id, state, index, passes)
    queue = paramMP.get('progress_queue')
    if queue is None:
        report_grain(event)
        check_cancelled()
    else:
        try:
            queue.put_nowait(event)
        except Full:
            # Progress must never stall a worker. Final counts come from results.
            pass


def _consume_fit_progress(queue, bar):
    completed = 0
    while queue is not None:
        try:
            event = queue.get_nowait()
        except Empty:
            break
        if isinstance(event, dict) and event.get('kind') == 'fit_metrics':
            report_fit_metrics(event)
            continue
        report_grain(event)
        grain, state, index, passes, *_ = event
        activity(f'Grain {grain}: {state.lower()} · pass {index}/{passes}')
        completed += state in ('Done', 'Insufficient reflections')
    if completed:
        bar.update(completed)
    return completed


def fit_grain_FF_reduced(grain_id):
    try:
        try:
            result = _fit_grain_FF_reduced(grain_id)
        finally:
            _write_pending_spots(grain_id)
    except BaseException:
        _grain_event(grain_id, 'Failed', 0, len(paramMP['tth_tol']))
        raise
    passes = len(paramMP['tth_tol'])
    state = 'Insufficient reflections' if np.isinf(result[2]) else 'Done'
    _grain_event(
        grain_id,
        state,
        paramMP.get('progress_pass', passes),
        passes,
        result[1],
        result[2],
    )
    return result


def _fit_grain_with_metrics(grain_id):
    """Return a reliable final snapshot on the required result channel.

    Routine telemetry is nonblocking and may be dropped when its queue fills.
    The final cumulative counters travel with the fit result, so the parent
    can reconcile them even if a worker's last telemetry update was dropped.
    """
    from hexrd.hedm._fit_progress import current_reporter

    result = fit_grain_FF_reduced(grain_id)
    reporter = current_reporter()
    return result, dict(reporter.data) if reporter is not None else None


def _write_pending_spots(grain_id):
    """Persist the last extraction once, rather than overwriting each pass.

    Early exits and fitting errors still publish the last complete extraction.
    The records precede the last fit/refit, just as in the reference workflow.
    """
    results = paramMP.pop('pending_spot_data', None)
    if results is None:
        return
    from hexrd.core.instrument.hedm_instrument import PatchDataWriter

    filename = paramMP['spots_filename'] % grain_id
    for key in paramMP['instrument'].detectors:
        directory = os.path.join(paramMP['analysis_dirname'], key)
        os.makedirs(directory, exist_ok=True)
        writer = PatchDataWriter(os.path.join(directory, filename))
        try:
            records = results.get(key, [])
            if isinstance(records, _SpotData):
                writer.dump_packed(records)
            else:
                for record in records:
                    writer.dump_patch(*record[:7], record[8], record[7])
        finally:
            writer.close()


def _fit_grain_FF_reduced(grain_id):
    steps = _fit_grain_steps(grain_id)
    response = None
    try:
        while True:
            try:
                request = steps.send(response)
            except StopIteration as finished:
                return finished.value
            response = paramMP['instrument'].pull_spots(*request[0], **request[1])
    finally:
        steps.close()


def _fit_grain_steps(grain_id):
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
    prefix = paramMP['spots_filename']
    spots_filename = None if prefix is None else prefix % grain_id

    return_pull_spots_data: bool = paramMP.get('return_pull_spots_data', False)

    grain = grains_table[grain_id]
    grain_params = grain[3:15]

    from hexrd.hedm._fit_progress import current_reporter

    reporter = current_reporter()
    passes = min(len(tth_tol), len(eta_tol), len(ome_tol))
    for pass_index, tols in enumerate(zip(tth_tol, eta_tol, ome_tol), 1):
        _grain_event(grain_id, 'Extracting', pass_index, passes)
        complvec, results = yield (
            (plane_data, grain_params, imgser_dict),
            dict(
                tth_tol=tols[0],
                eta_tol=tols[1],
                ome_tol=tols[2],
                npdiv=npdiv,
                threshold=threshold,
                eta_ranges=eta_ranges,
                ome_period=ome_period,
                dirname=analysis_dirname,
                filename=None,
                return_spot_list=False,
                check_only=False,
                interp='nearest',
                _packed=True,
            ),
        )
        if spots_filename is not None:
            paramMP['pending_spot_data'] = results

        if reporter is not None:
            reporter.phase('Culling reflections', force=True)

        culled_results, num_refl_valid, completeness = determine_valid_reflections(
            results, instrument, analysis_dirname
        )

        # ======= DO LEASTSQ FIT =======

        if num_refl_valid <= 12:  # not enough reflections to fit... exit
            if reporter is not None:
                reporter.pass_complete(skipped=passes - pass_index)
            warnings.warn(
                f'Not enough valid reflections ({num_refl_valid}) to fit, ' f'exiting',
                RuntimeWarning,
            )
            result = (grain_id, completeness, np.inf, grain_params)
            if return_pull_spots_data:
                result += ((complvec, results),)
            return result
        else:
            _grain_event(grain_id, 'Fitting', pass_index, passes, completeness)
            grain_params = fitGrain(
                grain_params,
                instrument,
                culled_results,
                plane_data.latVecOps['B'],
                plane_data.wavelength,
            )
            # get chisq
            # TODO: do this while evaluating fit???
            chisq = objFuncFitGrain(
                grain_params[gFlag_ref],
                grain_params,
                gFlag_ref,
                instrument,
                culled_results,
                plane_data.latVecOps['B'],
                plane_data.wavelength,
                ome_period,
                simOnly=False,
                return_value_flag=2,
            )
            if reporter is not None:
                reporter.pass_complete()

    if refit is not None:
        _grain_event(grain_id, 'Refitting', passes, passes, completeness, chisq)
        # first get calculated x, y, ome from previous solution
        # NOTE: this result is a dict
        xyo_det_fit_dict = objFuncFitGrain(
            grain_params[gFlag_ref],
            grain_params,
            gFlag_ref,
            instrument,
            culled_results,
            plane_data.latVecOps['B'],
            plane_data.wavelength,
            ome_period,
            simOnly=True,
            return_value_flag=2,
        )

        culled_results_r, num_refl_valid = refine_valid_reflections(
            culled_results, xyo_det_fit_dict, instrument, imgser_dict, refit
        )

        # only execute fit if left with enough reflections
        if num_refl_valid > 12:
            grain_params = fitGrain(
                grain_params,
                instrument,
                culled_results_r,
                plane_data.latVecOps['B'],
                plane_data.wavelength,
            )
            # get chisq
            # TODO: do this while evaluating fit???
            chisq = objFuncFitGrain(
                grain_params[gFlag_ref],
                grain_params,
                gFlag_ref,
                instrument,
                culled_results_r,
                plane_data.latVecOps['B'],
                plane_data.wavelength,
                ome_period,
                simOnly=False,
                return_value_flag=2,
            )
        if reporter is not None:
            reporter.refit_complete()
    result = (grain_id, completeness, chisq, grain_params)
    if return_pull_spots_data:
        result += ((complvec, results),)
    return result


def determine_valid_reflections(results, instrument, analysis_dirname):
    """Process results from `pull_spots()`

    PARAMETERS
    ----------
    results: dict
       output from `pull_spots()`
    instrument:
       the HEDM instrument
    analysis_dirname: str or Path
       the analysis output directory

    RETURNS
    -------
    culled_results: dict
       spot information for valid reflections
    """

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
        if isinstance(presults, _SpotData):
            refl_ids, max_int = presults.peak_id, presults.max_int.astype(
                np.float64, copy=False
            )
        else:
            refl_ids = np.empty(nrefl)
            max_int = np.empty(nrefl)
            for i, spot_data in enumerate(presults):
                refl_ids[i] = spot_data[0]
                max_int[i] = spot_data[4]

        valid_refl_ids = refl_ids >= 0

        # find unsaturated spots on this panel
        unsat_spots = np.ones(len(valid_refl_ids), dtype=bool)
        if panel.saturation_level is not None:
            unsat_spots[valid_refl_ids] = (
                max_int[valid_refl_ids] < panel.saturation_level
            )

        idx = np.logical_and(valid_refl_ids, unsat_spots)

        # if an overlap table has been written, load it and use it
        overlaps = np.zeros_like(idx, dtype=bool)
        try:
            ot = np.load(
                os.path.join(
                    analysis_dirname,
                    os.path.join(det_key, OVERLAP_TABLE_FILE),
                )
            )
            for key in ot.keys():
                for this_table in ot[key]:
                    these_overlaps = np.where(this_table[:, 0] == grain_id)[0]
                    if len(these_overlaps) > 0:
                        mark_these = np.array(this_table[these_overlaps, 1], dtype=int)
                        otidx = [np.where(refl_ids == mt)[0] for mt in mark_these]
                        overlaps[otidx] = True
            idx = np.logical_and(idx, ~overlaps)
            # logger.info("found overlap table for '%s'", det_key)
        except (IOError, IndexError):
            # logger.info("no overlap table found for '%s'", det_key)
            pass

        # attach to proper dict entry
        # FIXME: want to avoid looping again here
        culled_results[det_key] = (
            presults.take(idx)
            if isinstance(presults, _SpotData)
            else [presults[i] for i in np.where(idx)[0]]
        )
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

    return culled_results, num_refl_valid, completeness


def refine_valid_reflections(
    culled_results: dict,
    xyo_det_fit_dict: dict,
    instrument: instrument.HEDMInstrument,
    imgser_dict: dict,
    refit: tuple[float, float],
) -> tuple[dict, int]:
    """Refine valid reflections list using pixel distances

    Parameters
    ----------
    culled_results, xyo_det_fit_dict, instrument, imgser_dict, refit

    Returns
    -------
    culled_results_r, num_refl_valid
    """
    tol_xy, tol_ome = refit

    # make dict to contain new culled results
    culled_results_r = defaultdict(list)
    num_refl_valid = 0
    for det_key, presults in culled_results.items():
        if not presults:
            continue

        panel = instrument.detectors[det_key]
        xyo_det_fit = xyo_det_fit_dict[det_key]

        ims = next(iter(imgser_dict.values()))  # grab first for the omes
        ome_step = sum(np.array([-1, 1]) * ims.metadata['omega'][0, :])

        if isinstance(presults, _SpotData):
            xyo_det = presults.as_fit_data()['meas_xyo']
        else:
            xyo_det = np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in presults])
            )

        xpix_tol = tol_xy * panel.pixel_size_col
        ypix_tol = tol_xy * panel.pixel_size_row
        fome_tol = tol_ome * ome_step

        # define difference vectors for spot fits
        x_diff = abs(xyo_det[:, 0] - xyo_det_fit['calc_xy'][:, 0])
        y_diff = abs(xyo_det[:, 1] - xyo_det_fit['calc_xy'][:, 1])
        ome_diff = np.degrees(
            rotations.angularDifference(xyo_det[:, 2], xyo_det_fit['calc_omes'])
        )

        # filter out reflections with centroids more than
        # a pixel and delta omega away from predicted value
        idx_new = (x_diff <= xpix_tol) & (y_diff <= ypix_tol) & (ome_diff <= fome_tol)

        # attach to proper dict entry
        culled_results_r[det_key] = (
            presults.take(idx_new)
            if isinstance(presults, _SpotData)
            else [presults[i] for i in np.where(idx_new)[0]]
        )

        num_refl_valid += sum(idx_new)

    return culled_results_r, num_refl_valid


def fit_grains(
    experiment,
    grains_table,
    material=None,
    show_progress=False,
    ids_to_refine=None,
    write_spots_files=True,
    check_if_canceled_func=None,
    return_pull_spots_data: bool = False,
):
    """
    Performs optimization of grain parameters.

    operates on a single HEDM config block

    The `check_if_canceled_func` has the following signature:

        check_if_canceled_func() -> bool

    If it returns `True`, it indicates that fit_grains should be canceled.
    This is done by terminating the multiprocessing processes.

    In the default measurement mode, if `check_if_canceled_func` is set,
    multiprocessing will be performed,
    even if there is only one grain or one process, so that it will be
    cancelable.

    ``fit_grains.measurement_mode: staged`` prepares exact peak measurements
    in shared frame order before each fit pass. This opt-in mode currently uses
    one process and checks cancellation during planning, I/O and optimization.
    """
    from hexrd.hedm._image_cache import ImagePlan

    plan = ImagePlan()
    try:
        return _fit_grains(
            experiment,
            grains_table,
            material,
            show_progress,
            ids_to_refine,
            write_spots_files,
            check_if_canceled_func,
            return_pull_spots_data,
            plan,
        )
    finally:
        clear_sparse_frame_views()
        plan.close()


def _fit_grains(
    experiment,
    grains_table,
    material,
    show_progress,
    ids_to_refine,
    write_spots_files,
    check_if_canceled_func,
    return_pull_spots_data,
    image_plan,
):
    grains_table = np.atleast_2d(grains_table)
    clear_sparse_frame_views()

    with stage('Load images'):
        # grab imageseries dict
        imsd = experiment.image_series

        # grab instrument
        instr = experiment.instrument

        # grab eta ranges and ome_period
        eta_ranges = experiment.find_orientations.eta.range

        # handle omega period
        # !!! we assume all detector ims have the same ome ranges, so any will do!
        oims = next(iter(imsd.values()))
        ome_period = np.radians(oims.omega[0, 0] + np.r_[0.0, 360.0])

        # number of processes
        ncpus = experiment.max_workers
        staged = getattr(experiment.fit_grains, 'measurement_mode', 'grain') == 'staged'
        if staged:
            ncpus = 1

        # threshold for fitting
        threshold = experiment.fit_grains.threshold

        if material is None:
            material = experiment.get_active_material()

        if ids_to_refine is not None:
            grains_table = np.atleast_2d(grains_table[ids_to_refine, :])

        from hexrd.hedm.experiment import ImageCacheSettings

        settings = getattr(experiment.fit_grains, 'image_cache', ImageCacheSettings())
        if staged:
            from dataclasses import replace

            settings = replace(settings, mode='lazy')
        imsd, ncpus = image_plan.prepare(
            imsd,
            settings,
            min(ncpus, max(1, len(grains_table))),
            force_process=check_if_canceled_func is not None and not staged,
        )
        context = image_plan.context
        spots_filename = SPOTS_OUT_FILE if write_spots_files else None
        params = dict(
            grains_table=grains_table,
            plane_data=material._material.planeData,
            instrument=instr,
            imgser_dict=imsd,
            tth_tol=experiment.fit_grains.tolerance.tth,
            eta_tol=experiment.fit_grains.tolerance.eta,
            ome_tol=experiment.fit_grains.tolerance.omega,
            npdiv=experiment.fit_grains.npdiv,
            refit=experiment.fit_grains.refit,
            threshold=threshold,
            eta_ranges=eta_ranges,
            ome_period=ome_period,
            analysis_dirname=experiment.analysis_dir,
            spots_filename=spots_filename,
            return_pull_spots_data=return_pull_spots_data,
            collect_metrics=current_dashboard() is not None,
            lazy_images=image_plan.lazy,
            image_cache_limit=image_plan.cache.max_bytes if image_plan.lazy else 0,
        )
        if staged:
            params['staged_workspace'] = min(
                512 * 1024**2, max(16 * 1024**2, image_plan.cache.max_bytes)
            )
            image_plan.cache.close()
            temporary_cache = min(128 * 1024**2, params['staged_workspace'] // 4)
            image_plan.cache.max_bytes = temporary_cache
            image_plan.cache.metadata_limit = temporary_cache // 2
            params['image_cache_limit'] = temporary_cache
            diagnostic('Measurement mode', 'Segment and measure before each fit pass')
            diagnostic(
                'Staged workspace', f"{params['staged_workspace'] / 1024**2:.0f} MiB"
            )
            diagnostic(
                'Image cache total',
                f'{temporary_cache / 1024**2:.0f} MiB (indexes and current-frame chunks)',
            )
            diagnostic('Image cache / worker', f'{temporary_cache / 1024**2:.0f} MiB')

        if (len(grains_table) > 1 and ncpus != 1) or check_if_canceled_func:
            if not image_plan.lazy and context.get_start_method() == 'fork':
                # For frame cache, we need to load in all of the data up-front
                # so it can use fork multiprocessing to share with the other
                # processes. Otherwise, every process will load in the data on
                # its own. Accessing one frame in the imageseries is currently
                # all we need to do to trigger frame caches to load in all the data.
                with progress_bar(
                    total=len(imsd),
                    unit='panels',
                    desc='Loading frames',
                    disable=not show_progress,
                ) as load_bar:
                    for name, ims in imsd.items():
                        activity(f'Loading detector {name}')
                        if sparse_frame_view(ims) is None:
                            ims[0]
                        load_bar.update(1)
                # Share compiled pixel-size kernels and LLVM's initialized pages
                # rather than loading/compiling them independently in each worker.
                from hexrd.hedm.xrdutil.utils import _expand_pixels, _compute_max

                _expand_pixels(np.zeros((1, 2)), 1.0, 1.0, np.empty((4, 2)))
                _compute_max(np.zeros(4), np.zeros(4), np.empty((1, 2)))

    # =====================================================================
    # EXECUTE MP FIT
    # =====================================================================

    diagnostic('Grains', len(grains_table))
    diagnostic('Workers', min(ncpus, len(grains_table)))
    diagnostic('Tolerance passes', len(params['tth_tol']))
    dashboard = current_dashboard()
    if dashboard is not None:
        with dashboard.lock:
            dashboard.grains.clear()
            dashboard.fit_workers.clear()
            dashboard.fit_grain_metrics.clear()
            dashboard.fit_started = timeit.default_timer()
            dashboard.fit_passes_planned = len(grains_table) * min(
                len(params[key]) for key in ('tth_tol', 'eta_tol', 'ome_tol')
            )
            dashboard.finished_grains.clear()
            dashboard.diagnostics['Grains finished'] = '0'
            dashboard.diagnostics['Insufficient reflections'] = '0'
    with stage('Fit grains'), progress_bar(
        total=len(grains_table),
        unit='grains',
        desc='Fitting grains',
        disable=not show_progress,
    ) as bar:
        # DO FIT!
        if staged:
            from hexrd.hedm._staged_spots import fit_staged, StagedCancelled

            start = timeit.default_timer()
            try:
                fit_results = fit_staged(params, bar, check_if_canceled_func)
            except StagedCancelled:
                return None
            elapsed = timeit.default_timer() - start
        elif (len(grains_table) == 1 or ncpus == 1) and not check_if_canceled_func:
            logger.info("\tstarting serial fit")
            start = timeit.default_timer()
            fit_grain_FF_init(params)
            try:
                fit_results = []
                for grain_id in np.array(grains_table[:, 0], dtype=int):
                    check_cancelled()
                    fit_results.append(fit_grain_FF_reduced(grain_id))
                    bar.update(1)
            finally:
                fit_grain_FF_cleanup()
            elapsed = timeit.default_timer() - start
        else:
            nproc = min(ncpus, len(grains_table))
            chunksize = max(1, len(grains_table) // (64 * nproc))

            logger.info(
                "\tstarting fit on %d processes with chunksize %d", nproc, chunksize
            )
            start = timeit.default_timer()
            freeze = context.get_start_method() == 'fork' and gc.get_freeze_count() == 0
            progress_queue = (
                context.Queue(maxsize=256)
                if current_dashboard() is not None or show_progress
                else None
            )
            if progress_queue is not None:
                params['progress_queue'] = progress_queue
            if freeze:
                gc.freeze()
            try:
                pool = context.Pool(nproc, fit_grain_FF_init, (params,))
            except BaseException:
                if progress_queue is not None:
                    progress_queue.close()
                    progress_queue.join_thread()
                raise
            finally:
                # Workers keep their frozen inherited heap. Restore the caller's
                # collector immediately, including when pool construction fails.
                if freeze:
                    gc.unfreeze()
            try:
                async_result = pool.map_async(
                    (
                        _fit_grain_with_metrics
                        if dashboard is not None
                        else fit_grain_FF_reduced
                    ),
                    np.array(grains_table[:, 0], dtype=int),
                    chunksize=chunksize,
                )
                completed = 0
                while not async_result.ready():
                    check_cancelled()
                    completed += _consume_fit_progress(progress_queue, bar)
                    if check_if_canceled_func and check_if_canceled_func():
                        logger.info('Fit grains canceled.')
                        # Perform an early return if we need to cancel.
                        return None

                    async_result.wait(0.25)

                check_cancelled()
                fit_results = async_result.get()
                if dashboard is not None:
                    records = []
                    for result, snapshot in fit_results:
                        records.append(result)
                        if snapshot is not None:
                            report_fit_metrics(snapshot)
                    fit_results = records
                completed += _consume_fit_progress(progress_queue, bar)
                bar.update(max(0, len(fit_results) - completed))
                for result in fit_results:
                    passes = len(params['tth_tol'])
                    state = (
                        'Insufficient reflections' if np.isinf(result[2]) else 'Done'
                    )
                    previous = (
                        dashboard.grains.get(int(result[0])) if dashboard else None
                    )
                    index = (
                        previous[2]
                        if previous is not None
                        else (0 if state == 'Insufficient reflections' else passes)
                    )
                    report_grain(
                        (int(result[0]), state, index, passes, result[1], result[2])
                    )
                pool.close()
            finally:
                pool.terminate()
                pool.join()
                if progress_queue is not None:
                    progress_queue.close()
                    progress_queue.join_thread()
            elapsed = timeit.default_timer() - start
    logger.info("fitting took %f seconds", elapsed)
    diagnostic('Grains finished', len(fit_results))
    diagnostic(
        'Insufficient reflections', sum(np.isinf(result[2]) for result in fit_results)
    )
    if return_pull_spots_data:
        spots_data = {}
        for result in fit_results:
            signals, panels = result[4]
            panels = {
                key: list(value) if isinstance(value, _SpotData) else value
                for key, value in panels.items()
            }
            spots_data[result[0]] = (signals, panels)
        fit_results = [result[:4] for result in fit_results]
        return fit_results, spots_data
    return fit_results
