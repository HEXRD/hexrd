"""Bounded grain fitting with shared extraction, optimization and reporting."""

import os
import gc
import logging
import multiprocessing
import numpy as np
import timeit
import warnings
from queue import Empty, Full

from collections import defaultdict
from contextlib import ExitStack

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
    """Initialize fit-local parameters and process-local progress reporting."""
    global paramMP
    paramMP = params
    if multiprocessing.current_process().name != 'MainProcess':
        detach_dashboard()
    from hexrd.hedm._fit_progress import FitReporter, set_reporter

    queue = params.get('progress_queue')
    set_reporter(
        FitReporter(queue)
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
    _grain_event(*_result_event(result, passes, paramMP.get('progress_pass', passes)))
    return result


def _result_event(result, passes, index=None):
    """Format completion consistently for serial, staged and IPC results."""
    grain, completeness, chisq = result[:3]
    state = 'Insufficient reflections' if np.isinf(chisq) else 'Done'
    if index is None:
        index = 0 if state == 'Insufficient reflections' else passes
    return int(grain), state, index, passes, completeness, chisq


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
    """Yield extraction requests and fit the returned grain-dependent spots."""
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

    def optimize(reflections):
        fitted = fitGrain(
            grain_params,
            instrument,
            reflections,
            plane_data.latVecOps['B'],
            plane_data.wavelength,
        )
        chisq = objFuncFitGrain(
            fitted[gFlag_ref],
            fitted,
            gFlag_ref,
            instrument,
            reflections,
            plane_data.latVecOps['B'],
            plane_data.wavelength,
            ome_period,
            simOnly=False,
            return_value_flag=2,
        )
        return fitted, chisq

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
            grain_params, chisq = optimize(culled_results)

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
            grain_params, chisq = optimize(culled_results_r)
    result = (grain_id, completeness, chisq, grain_params)
    if return_pull_spots_data:
        result += ((complvec, results),)
    return result


def determine_valid_reflections(results, instrument, analysis_dirname):
    """Cull invalid, saturated and overlapping spots; report pre-cull completeness."""
    culled_results = dict.fromkeys(results)
    num_refl_tot = 0
    num_refl_valid = 0
    for det_key in culled_results:
        panel = instrument.detectors[det_key]

        presults = results[det_key]

        # make data arrays
        if isinstance(presults, _SpotData):
            refl_ids, max_int = presults.peak_id, presults.max_int.astype(
                np.float64, copy=False
            )
        else:
            refl_ids, max_int = (
                np.array([(spot[0], spot[4]) for spot in presults], dtype=float)
                .reshape(-1, 2)
                .T
            )
        valid_refl_ids = refl_ids >= 0
        idx = valid_refl_ids.copy()
        if panel.saturation_level is not None:
            idx &= max_int < panel.saturation_level

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


def _fit_parallel(params, context, ncpus, bar, show_progress, check_if_canceled_func):
    """Own worker resources and reconcile results with sampled progress."""
    dashboard = current_dashboard()
    nproc = min(ncpus, len(params['grains_table']))
    chunksize = max(1, len(params['grains_table']) // (64 * nproc))

    logger.info("\tstarting fit on %d processes with chunksize %d", nproc, chunksize)
    freeze = context.get_start_method() == 'fork' and gc.get_freeze_count() == 0
    progress_queue = (
        context.Queue(maxsize=256)
        if dashboard is not None or show_progress
        else None
    )
    if progress_queue is not None:
        params['progress_queue'] = progress_queue
    with ExitStack() as cleanup:
        if progress_queue is not None:
            cleanup.callback(progress_queue.join_thread)
            cleanup.callback(progress_queue.close)
        if freeze:
            gc.freeze()
        try:
            pool = context.Pool(nproc, fit_grain_FF_init, (params,))
        finally:
            # Workers keep their frozen heap; always restore the caller's GC.
            if freeze:
                gc.unfreeze()
        cleanup.callback(pool.join)
        cleanup.callback(pool.terminate)
        async_result = pool.map_async(
            (
                _fit_grain_with_metrics
                if dashboard is not None
                else fit_grain_FF_reduced
            ),
            np.array(params['grains_table'][:, 0], dtype=int),
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
            previous = dashboard.grains.get(int(result[0])) if dashboard else None
            report_grain(
                _result_event(
                    result,
                    len(params['tth_tol']),
                    previous[2] if previous is not None else None,
                )
            )
        pool.close()
    return fit_results


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

    image_plan = ImagePlan()
    try:
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
            staged = (
                getattr(experiment.fit_grains, 'measurement_mode', 'grain') == 'staged'
            )
            if staged:
                ncpus = 1

            # threshold for fitting
            threshold = experiment.fit_grains.threshold

            if material is None:
                material = experiment.get_active_material()

            if ids_to_refine is not None:
                grains_table = np.atleast_2d(grains_table[ids_to_refine, :])

            from hexrd.hedm.experiment import ImageCacheSettings

            settings = getattr(
                experiment.fit_grains, 'image_cache', ImageCacheSettings()
            )
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
            )
            if staged:
                params['staged_workspace'] = min(
                    512 * 1024**2, max(16 * 1024**2, image_plan.cache.max_bytes)
                )
                image_plan.cache.close()
                temporary_cache = min(128 * 1024**2, params['staged_workspace'] // 4)
                image_plan.cache.max_bytes = temporary_cache
                image_plan.cache.metadata_limit = temporary_cache // 2
                image_plan.cache.chunk_workers = settings.read_workers
                params['reuse_pixels'] = settings.reuse_pixels
                diagnostic(
                    'Measurement mode', 'Segment and measure before each fit pass'
                )
                diagnostic('Maximum chunk read workers', settings.read_workers)
                diagnostic(
                    'Staged workspace',
                    f"{params['staged_workspace'] / 1024**2:.0f} MiB",
                )
                diagnostic(
                    'Image cache total',
                    f'{temporary_cache / 1024**2:.0f} MiB (indexes and current-frame chunks)',
                )
                diagnostic(
                    'Image cache / worker', f'{temporary_cache / 1024**2:.0f} MiB'
                )

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
                dashboard.finished_grains.clear()
                dashboard.diagnostics['Grains finished'] = '0'
                dashboard.diagnostics['Insufficient reflections'] = '0'
        with stage('Fit grains'), progress_bar(
            total=len(grains_table),
            unit='grains',
            desc='Fitting grains',
            disable=not show_progress,
        ) as bar:
            start = timeit.default_timer()
            if staged:
                from hexrd.hedm._staged_spots import fit_staged, StagedCancelled

                try:
                    fit_results = fit_staged(params, bar, check_if_canceled_func)
                except StagedCancelled:
                    return None
            elif (len(grains_table) == 1 or ncpus == 1) and not check_if_canceled_func:
                logger.info("\tstarting serial fit")
                fit_grain_FF_init(params)
                try:
                    fit_results = []
                    for grain_id in np.array(grains_table[:, 0], dtype=int):
                        check_cancelled()
                        fit_results.append(fit_grain_FF_reduced(grain_id))
                        bar.update(1)
                finally:
                    fit_grain_FF_cleanup()
            else:
                fit_results = _fit_parallel(
                    params, context, ncpus, bar, show_progress, check_if_canceled_func
                )
                if fit_results is None:
                    return None
        logger.info("fitting took %f seconds", timeit.default_timer() - start)
        diagnostic('Grains finished', len(fit_results))
        diagnostic(
            'Insufficient reflections',
            sum(np.isinf(result[2]) for result in fit_results),
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
    finally:
        clear_sparse_frame_views()
        image_plan.close()
