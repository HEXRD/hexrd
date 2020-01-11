from __future__ import absolute_import

import copy
import logging
import multiprocessing as mp
from multiprocessing.queues import Empty
import os
import time

import numpy as np
from scipy.linalg.matfuncs import logm

from hexrd.instrument import io
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd.matrixutil import vecMVToSymm
from hexrd.utils.progressbar import Bar, ETA, ProgressBar, ReverseBar
from hexrd.transforms import xfcapi
from hexrd.fitting.grains import fitGrain, objFuncFitGrain
from hexrd.rotations import angleAxisOfRotMat, rotMatOfQuat

logger = logging.getLogger(__name__)

# grain parameter refinement flags
gFlag_ref = np.array([1, 1, 1,
                      1, 1, 1,
                      1, 1, 1, 1, 1, 1], dtype=bool)
# grain parameter scalings
gScl_ref = np.array([1., 1., 1.,
                     1., 1., 1.,
                     1., 1., 1., 0.01, 0.01, 0.01])

# minimum number of reflections to allow fitting
min_nrefl = 24  # 2X number of parameters


descr = 'Extracts G vectors, grain position and strain'
example = """
examples:
    hexrd fit-grains configuration.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('fit-grains', description = descr, help = descr)
    p.add_argument(
        'yml', type=str,
        help='YAML configuration file'
        )
    p.add_argument(
        '-g', '--grains', type=str, default=None,
        help="comma-separated list of IDs to refine, defaults to all"
        )
    p.add_argument(
        '-q', '--quiet', action='store_true',
        help="don't report progress in terminal"
        )
    p.add_argument(
        '-c', '--clean', action='store_true',
        help='overwrites existing analysis, including frame cache'
        )
    p.add_argument(
        '-f', '--force', action='store_true',
        help='overwrites existing analysis, exlcuding frame cache'
        )
    p.add_argument(
        '-p', '--profile', action='store_true',
        help='runs the analysis with cProfile enabled',
        )
    p.set_defaults(func=execute)


def execute(args, parser):
    import logging
    import os
    import sys

    import yaml

    from hexrd import config


    # load the configuration settings
    cfgs = config.open(args.yml)

    # configure logging to the console:
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.quiet:
        log_level = logging.ERROR
    logger = logging.getLogger('hexrd')
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL if args.quiet else log_level)
    cf = logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S')
    ch.setFormatter(cf)
    logger.addHandler(ch)

    # ...make this an attribute in cfg?
    analysis_id = '%s_%s' %(
        cfgs[0].analysis_name.strip().replace(' ', '-'),
        cfgs[0].material.active.strip().replace(' ', '-'),
        )

    # if find-orientations has not already been run, do so:
    quats_f = os.path.join(
        cfgs[0].working_dir,
        'accepted_orientations_%s.dat' %analysis_id
        )
    if not os.path.exists(quats_f):
        logger.info("Missing %s, running find-orientations", quats_f)
        logger.removeHandler(ch)
        from . import findorientations
        findorientations.execute(args, parser)
        logger.addHandler(ch)

    logger.info('=== begin fit-grains ===')

    clobber = args.force or args.clean
    for cfg in cfgs:
        # prepare the analysis directory
        if os.path.exists(cfg.analysis_dir) and not clobber:
            logger.error(
                'Analysis "%s" at %s already exists.'
                ' Change yml file or specify "force"',
                cfg.analysis_name, cfg.analysis_dir
                )
            sys.exit()
        if not os.path.exists(cfg.analysis_dir):
            os.makedirs(cfg.analysis_dir)

        logger.info('*** begin analysis "%s" ***', cfg.analysis_name)

        # configure logging to file for this particular analysis
        logfile = os.path.join(
            cfg.working_dir,
            cfg.analysis_name,
            'fit-grains.log'
            )
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(log_level)
        ff = logging.Formatter(
                '%(asctime)s - %(name)s - %(message)s',
                '%m-%d %H:%M:%S'
                )
        fh.setFormatter(ff)
        logger.info("logging to %s", logfile)
        logger.addHandler(fh)

        if args.profile:
            import cProfile as profile, pstats, StringIO
            pr = profile.Profile()
            pr.enable()

        # process the data
        gid_list = None
        if args.grains is not None:
            gid_list = [int(i) for i in args.grains.split(',')]

        fit_grains(
            cfg,
            force=args.force,
            clean=args.clean,
            show_progress=not args.quiet,
            ids_to_refine=gid_list,
            )

        if args.profile:
            pr.disable()
            s = StringIO.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
            logger.info('%s', s.getvalue())

        # stop logging for this particular analysis
        fh.flush()
        fh.close()
        logger.removeHandler(fh)

        logger.info('*** end analysis "%s" ***', cfg.analysis_name)

    logger.info('=== end fit-grains ===')
    # stop logging to the console
    ch.flush()
    ch.close()
    logger.removeHandler(ch)

# =============================================================================
# FUNCTIONS
# =============================================================================


def get_job_queue(cfg, ids_to_refine=None):
    job_queue = mp.JoinableQueue()
    # load the queue
    try:
        # use an estimate of the grain parameters, if available
        estimate_f = cfg.fit_grains.estimate
        grain_params_list = np.atleast_2d(np.loadtxt(estimate_f))
        n_quats = len(grain_params_list)
        n_jobs = 0
        for grain_params in grain_params_list:
            grain_id = grain_params[0]
            if ids_to_refine is None or grain_id in ids_to_refine:
                job_queue.put((grain_id, grain_params[3:15]))
                n_jobs += 1
        logger.info(
            'fitting grains using "%s" for the initial estimate',
            estimate_f
            )
    except (ValueError, IOError):
        # no estimate available, use orientations and defaults
        logger.info('fitting grains using default initial estimate')

        # ...make this an attribute in cfg?
        analysis_id = '%s_%s' % (
            cfg.analysis_name.strip().replace(' ', '-'),
            cfg.material.active.strip().replace(' ', '-'),
            )

        # load quaternion file
        quats = np.atleast_2d(
            np.loadtxt(
                os.path.join(
                    cfg.working_dir,
                    'accepted_orientations_%s.dat' % analysis_id
                    )
                )
            )
        n_quats = len(quats)
        n_jobs = 0
        phi, n = angleAxisOfRotMat(rotMatOfQuat(quats.T))
        for i, (phi, n) in enumerate(zip(phi, n.T)):
            if ids_to_refine is None or i in ids_to_refine:
                exp_map = phi*n
                grain_params = np.hstack(
                    [exp_map, 0., 0., 0., 1., 1., 1., 0., 0., 0.]
                    )
                job_queue.put((i, grain_params))
                n_jobs += 1
    logger.info("fitting grains for %d of %d orientations", n_jobs, n_quats)
    return job_queue, n_jobs


def get_data(cfg, show_progress=False, force=False, clean=False):

    # instrument and plane data from config
    # set planedata exclusions
    plane_data = cfg.material.plane_data
    tth_max = cfg.fit_grains.tth_max
    if tth_max is True:
        pd.exclusions = np.zeros_like(pd.exclusions, dtype=bool)
        pd.exclusions = plane_data.getTTh() > cfg.instrument.hedm.max_tth()
    elif tth_max > 0:
        pd.exclusions = np.zeros_like(pd.exclusions, dtype=bool)
        pd.exclusions = plane_data.getTTh() >= np.radians(tth_max)

    pkwargs = {
        'analysis_directory': cfg.analysis_dir,
        'eta_range': np.radians(cfg.find_orientations.eta.range),
        'eta_tol': cfg.fit_grains.tolerance.eta,
        'fit_only': cfg.fit_grains.fit_only,
        'npdiv': cfg.fit_grains.npdiv,
        'omega_period': np.radians(cfg.find_orientations.omega.period),
        'omega_tol': cfg.fit_grains.tolerance.omega,
        'overlap_table': os.path.join(cfg.analysis_dir, 'overlap_table.npz'),
        'panel_buffer': cfg.fit_grains.panel_buffer,
        'plane_data': plane_data,
        'refit_tol': cfg.fit_grains.refit,
        'spots_stem': 'spots_%05d.out',
        'threshold': cfg.fit_grains.threshold,
        'tth_tol': cfg.fit_grains.tolerance.tth,
        }
    return cfg.image_series, cfg.instrument.hedm, pkwargs


def fit_grains(cfg,
               force=False,
               clean=False,
               show_progress=False,
               ids_to_refine=None
):
    # load the data
    imgser_dict, instr, pkwargs = get_data(cfg, show_progress, force, clean)
    job_queue, njobs = get_job_queue(cfg, ids_to_refine)

    # log this before starting progress bar
    ncpus = cfg.multiprocessing
    ncpus = ncpus if ncpus < njobs else njobs
    logger.info(
        'will use %d of %d processors', ncpus, mp.cpu_count()
        )
    if ncpus == 1:
        logger.info('multiprocessing disabled')

    # echo some of the fitting options
    if cfg.fit_grains.fit_only:
        logger.info('\t**fitting only; will not pull spots')
    if cfg.fit_grains.refit is not None:
        msg = 'will perform refit excluding spots > ' + \
              '%.2f pixels and ' % cfg.fit_grains.refit[0] + \
              '%.2f frames from expected values' % cfg.fit_grains.refit[1]
        logger.info(msg)

    start = time.time()
    pbar = None
    if show_progress:
        pbar = ProgressBar(
            widgets=[Bar('>'), ' ', ETA(), ' ', ReverseBar('<')],
            maxval=njobs
            ).start()

    # finally start processing data
    if ncpus == 1:
        # no multiprocessing
        results = []
        w = FitGrainsWorker(
            job_queue, results,
            imgser_dict, instr,
            copy.deepcopy(pkwargs),
            progressbar=pbar
            )
        w.run()
    else:
        # multiprocessing
        manager = mp.Manager()
        results = manager.list()
        for i in range(ncpus):
            # lets make a deep copy of the pkwargs, just in case:
            w = FitGrainsWorkerMP(job_queue, results,
                                  imgser_dict, instr,
                                  copy.deepcopy(pkwargs))
            w.daemon = True
            w.start()
    while True:
        n_res = len(results)
        if show_progress:
            pbar.update(n_res)
        if n_res == njobs:
            break
        time.sleep(0.1)  # ??? check this?
    job_queue.join()

    # call to grain tabel output
    write_grains_file(cfg, results)

    if show_progress:
        pbar.finish()
    elapsed = time.time() - start
    logger.info('processed %d grains in %g minutes', n_res, elapsed/60)


def write_grains_file(cfg, results, output_name=None):
    # record the results to file
    # results: (id, grain_params, compl, emat, resd)
    if output_name is None:
        f = open(os.path.join(cfg.analysis_dir, 'grains.out'), 'w')
    else:
        f = open(os.path.join(cfg.analysis_dir, output_name), 'w')
    gw = io.GrainDataWriter(f)
    for result in sorted(results):
        gw.dump_grain(result[0], result[2], result[4], result[1])
        pass
    gw.close()


# =============================================================================
# CLASSES
# =============================================================================

class FitGrainsWorker(object):
    """
    Wrapper class for looped grains fitting
    """
    def __init__(self, jobs, results, imgser_dict, instr, pkwargs, **kwargs):
        # (id, g_refined, compl, eMat, resd) in sorted(results)
        self._jobs = jobs
        self._results = results

        # a dict containing the rest of the parameters
        self._p = pkwargs

        # set dictionary from input, have to make it OmegaImageSeries
        # FIXME this should probably be done in config
        self._imgsd = dict.fromkeys(imgser_dict)

        # handle panel buffer input fron cfg
        # !!! panel buffer setting is global and assumes same type of panel!
        pbuff_arr = np.array(self._p['panel_buffer'])
        for det_key, panel in instr.detectors.iteritems():
            panel.panel_buffer = pbuff_arr
            self._imgsd[det_key] = OmegaImageSeries(imgser_dict[det_key])
        buff_str = str(pbuff_arr)
        # logger.info("\tset panel buffer for %s to: %s", det_key, buff_str)
        self._instr = instr

        # lets make a couple shortcuts:
        # !!! is it still necessary to re-cast bmat?
        self._p['bmat'] = np.ascontiguousarray(
            self._p['plane_data'].latVecOps['B']
        )
        self._p['wlen'] = self._p['plane_data'].wavelength
        self._pbar = kwargs.get('progressbar', None)

    def pull_spots(self, grain_id, grain_params, iteration):
        """
        ??? maybe pass interpolation option
        """
        complvec, results = self._instr.pull_spots(
            self._p['plane_data'], grain_params,
            self._imgsd,
            tth_tol=self._p['tth_tol'][iteration],
            eta_tol=self._p['eta_tol'][iteration],
            ome_tol=self._p['omega_tol'][iteration],
            npdiv=self._p['npdiv'],
            threshold=self._p['threshold'],
            eta_ranges=self._p['eta_range'],
            ome_period=self._p['omega_period'],
            dirname=self._p['analysis_directory'],
            filename=self._p['spots_stem'] % grain_id,
            save_spot_list=False, quiet=True,
            check_only=False, interp='nearest')

    def fit_grains(self, grain_id, grain_params, refit_tol=None):
        """
        Executes lsq fits of grains based on spot files

        REFLECTION TABLE

        Cols as follows:
            0:7    ID    PID    H    K    L    sum(int)    max(int)
            7:10   pred tth    pred eta    pred ome
            10:13  meas tth    meas eta    meas ome
            13:17  pred X    pred Y    meas X    meas Y
        """
        # !!! load resuts form spots
        spots_fname_dict = {}
        for det_key in self._instr.detectors.iterkeys():
            spots_fname_dict[det_key] = os.path.join(
                self._p['analysis_directory'],
                os.path.join(
                    det_key,
                    self._p['spots_stem'] % grain_id
                )
            )
        self._culled_results = dict.fromkeys(spots_fname_dict)
        num_refl_tot = 0
        num_refl_valid = 0
        for det_key in self._culled_results:
            panel = self._instr.detectors[det_key]

            presults = np.loadtxt(spots_fname_dict[det_key])

            valid_refl_ids = presults[:, 0] >= 0
            spot_ids = presults[:, 1]

            # find unsaturated spots on this panel
            if panel.saturation_level is None:
                unsat_spots = np.ones(len(valid_refl_ids))
            else:
                unsat_spots = presults[:, 6] < panel.saturation_level

            idx = np.logical_and(valid_refl_ids, unsat_spots)

            # if an overlap table has been written, load it and use it
            overlaps = np.zeros_like(idx, dtype=bool)
            try:
                ot = np.load(
                    os.path.join(
                        self._p['analysis_directory'],
                        'overlap_table.npz'
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
                                np.where(spot_ids == mt)[0]
                                for mt in mark_these
                            ]
                            overlaps[otidx] = True
                idx = np.logical_and(idx, ~overlaps)
                # logger.info("found overlap table for '%s'", det_key)
            except(IOError, IndexError):
                # logger.info("no overlap table found for '%s'", det_key)
                pass

            # attach to proper dict entry
            self._culled_results[det_key] = presults[idx, :]
            num_refl_tot += len(valid_refl_ids)
            num_refl_valid += sum(valid_refl_ids)
            pass  # now we have culled data

        # CAVEAT: completeness from pullspots only; incl saturated and overlaps
        # <JVB 2015-12-15>
        completeness = num_refl_valid / float(num_refl_tot)

        # ======= DO LEASTSQ FIT =======

        if num_refl_valid <= min_nrefl:  # exit if not enough refls to fit
            grain_params_fit = grain_params
            return grain_id, completeness, np.inf, grain_params_fit
        else:
            grain_params_fit = fitGrain(
                    grain_params, self._instr, self._culled_results,
                    self._p['bmat'], self._p['wlen']
                )
        # get chisq
        # TODO: do this while evaluating fit???
        chisq = objFuncFitGrain(
                grain_params_fit[gFlag_ref], grain_params_fit, gFlag_ref,
                self._instr,
                self._culled_results,
                self._p['bmat'], self._p['wlen'],
                self._p['omega_period'],
                simOnly=False, return_value_flag=2)

        if self._p['refit_tol'] is not None:
            # first get calculated x, y, ome from previous solution
            # NOTE: this result is a dict
            xyo_det_fit_dict = objFuncFitGrain(
                grain_params_fit[gFlag_ref], grain_params_fit, gFlag_ref,
                self._instr,
                self._culled_results,
                self._p['bmat'], self._p['wlen'],
                self._p['omega_period'],
                simOnly=True, return_value_flag=2)

            # make dict to contain new culled results
            culled_results_r = dict.fromkeys(self._culled_results)
            num_refl_valid = 0
            for det_key in culled_results_r:
                presults = self._culled_results[det_key]

                ims = self._imgsd[det_key]  # !!! must be OmegaImageSeries
                ome_step = sum(np.r_[-1, 1]*ims.metadata['omega'][0, :])

                # measured vals for pull spots
                xyo_det = presults[:, [15, 16, 12]]

                # previous solutions calc vals
                xyo_det_fit = xyo_det_fit_dict[det_key]

                xpix_tol = self._p['refit_tol'][0]*panel.pixel_size_col
                ypix_tol = self._p['refit_tol'][0]*panel.pixel_size_row
                fome_tol = self._p['refit_tol'][1]*ome_step

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
                culled_results_r[det_key] = presults[idx_new, :]
                num_refl_valid += sum(idx_new)
                pass

            # only execute fit if left with enough reflections
            if num_refl_valid > min_nrefl:
                grain_params_fit = fitGrain(
                    grain_params_fit, self._instr, culled_results_r,
                    self._p['bmat'], self._p['wlen'],
                )
                # get chisq
                # TODO: do this while evaluating fit???
                chisq = objFuncFitGrain(
                        grain_params_fit[gFlag_ref],
                        grain_params_fit, gFlag_ref,
                        self._instr,
                        culled_results_r,
                        self._p['bmat'], self._p['wlen'],
                        self._p['omega_period'],
                        simOnly=False, return_value_flag=2)
                pass  # close check on number of valid refls
            pass  # close refit conditional
        return grain_id, completeness, chisq, grain_params_fit

    def get_e_mat(self, grain_params):
        """
        strain tensor calculation
        """
        return logm(np.linalg.inv(vecMVToSymm(grain_params[6:])))

    def get_residuals(self, grain_params):
        return objFuncFitGrain(grain_params[gFlag_ref],
                               grain_params, gFlag_ref,
                               self._instr,
                               self._culled_results,
                               self._p['bmat'], self._p['wlen'],
                               self._p['omega_period'],
                               simOnly=False, return_value_flag=2)

    def loop(self):
        id, grain_params = self._jobs.get(False)
        iterations = (0, len(self._p['eta_tol']))
        for iteration in range(*iterations):
            # pull spots if asked to, otherwise just fit
            if not self._p['fit_only']:
                self.pull_spots(id, grain_params, iteration)
            # FITTING HERE
            _, compl, chisq, grain_params = self.fit_grains(
                id, grain_params, refit_tol=self._p['refit_tol']
            )
            if compl == 0:
                break
            pass

        # final pull spots if enabled
        if not self._p['fit_only']:
            self.pull_spots(id, grain_params, -1)

        emat = self.get_e_mat(grain_params)
        resd = self.get_residuals(grain_params)

        self._results.append((id, grain_params, compl, emat, resd))
        self._jobs.task_done()

    def run(self):
        n_res = 0
        while True:
            try:
                self.loop()
                n_res += 1
                if self._pbar is not None:
                    self._pbar.update(n_res)
            except Empty:
                break


class FitGrainsWorkerMP(FitGrainsWorker, mp.Process):

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self)
        FitGrainsWorker.__init__(self, *args, **kwargs)
