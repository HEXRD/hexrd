import logging
import os
import sys
from collections import namedtuple
from pathlib import Path

from hexrd.hedm._progress import (
    run_cli,
    FIT_STAGES,
    check_cancelled,
    diagnostic,
    quiet_progress,
    stage,
)

import numpy as np

from hexrd.core import constants as cnst
from hexrd.core import rotations
from hexrd.hedm.experiment import HedmExperiment
from hexrd.core.transforms import xfcapi

descr = 'Extracts G vectors, grain position and strain'
example = """
examples:
    hexrd fit-grains configuration.yml
"""

_flds = [
    "id",
    "completeness",
    "chisq",
    "expmap",
    "centroid",
    "inv_Vs",
    "ln_Vs",
]
_BaseGrainData = namedtuple("_BaseGrainData", _flds)
del _flds


class GrainData(_BaseGrainData):
    """Simple class for storing grain output data

    To read the grains file, use the `load` method, like this:
    > from hexrd.hedm.fitgrains import GrainData
    > gd = GrainData.load("grains.npz")
    """

    def save(self, fname):
        """Save grain data to an np file

        Parameters
        ----------
        fname: path | string
            name of the file to save to
        """
        np.savez(fname, **self._asdict())

    @classmethod
    def load(cls, fname):
        """Return GrainData instance from npz file
        Parameters
        ----------
        fname: path | string
            name of the file to load
        """
        return cls(**np.load(fname))

    @classmethod
    def from_grains_out(cls, fname):
        """Read hexrd grains output file"""
        return cls.from_array(np.loadtxt(fname))

    @classmethod
    def from_array(cls, a):
        """Return GrainData instance from numpy datatype array"""
        return cls(
            id=a[:, 0].astype(int),
            completeness=a[:, 1],
            chisq=a[:, 2],
            expmap=a[:, 3:6],
            centroid=a[:, 6:9],
            inv_Vs=a[:, 9:15],
            ln_Vs=a[:, 15:21],
        )

    def write_grains_out(self, fname):
        """Write a file in grains.out format"""
        from hexrd.core import instrument

        gw = instrument.GrainDataWriter(filename=fname)
        n = len(self.id)
        for i in range(n):
            gparams = np.hstack((self.expmap[i], self.centroid[i], self.inv_Vs[i]))
            gw.dump_grain(self.id[i], self.completeness[i], self.chisq[i], gparams)
        gw.close()

    @property
    def num_grains(self):
        return len(self.id)

    @property
    def quaternions(self):
        """Return quaternions as array(num_grains, 4)."""
        return rotations.quatOfExpMap(self.expmap.T).T

    @property
    def rotation_matrices(self):
        """ "Return rotation matrices from exponential map parameters"""
        #
        # Compute the rotation matrices only once, the first time this is
        # called, and save the results.
        #
        if not hasattr(self, "_rotation_matrices"):
            n = len(self.expmap)
            rmats = np.zeros((n, 3, 3))
            for i in range(n):
                rmats[i] = xfcapi.make_rmat_of_expmap(self.expmap[i])
            self._rotation_matrices = rmats
        return self._rotation_matrices

    @property
    def strain(self):
        """Return symmetric strain tensor as array(num_grains, 6).

        The order of components is `11`, `22`, `33`, `23`, `13`, `23`.
        """
        return self.ln_Vs

    def select(self, min_completeness=0.0, max_chisq=None):
        """Return a new GrainData instance with only selected grains

        PARAMETERS
        ----------
        min_completeness: float, default=0
           minimum value of completeness
        max_chisq: float | None, default=None
           if not None, maximum value for chi-squared

        RETURNS
        -------
        GrainData instance
           new instance for subset of grains meeting selection criteria
        """
        has_chisq = max_chisq is not None
        sel_comp = self.completeness >= min_completeness
        sel = sel_comp & (self.chisq <= max_chisq) if has_chisq else sel_comp

        return __class__(
            self.id[sel],
            self.completeness[sel],
            self.chisq[sel],
            self.expmap[sel],
            self.centroid[sel],
            self.inv_Vs[sel],
            self.ln_Vs[sel],
        )


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser(
        'fit-grains',
        description=descr,
        help=descr,
        epilog='Interactive terminals show the Textual dashboard; redirected output uses plain logs.',
    )
    p.add_argument(
        '--no-tui',
        action='store_true',
        help='use plain logs and exit on completion (for batch runs)',
    )
    p.add_argument('yml', type=str, help='YAML configuration file')
    p.add_argument(
        '-g',
        '--grains',
        type=str,
        default=None,
        help="comma-separated list of IDs to refine, defaults to all",
    )
    p.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help="don't report progress in terminal",
    )
    p.add_argument(
        '-c',
        '--clean',
        action='store_true',
        help='overwrites existing analysis, uses initial orientations',
    )
    p.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='overwrites existing analysis',
    )
    p.add_argument(
        '-p',
        '--profile',
        action='store_true',
        help='runs the analysis with cProfile enabled',
    )
    p.set_defaults(func=execute)


def write_results(
    fit_results, experiment, grains_filename='grains.out', grains_npz='grains.npz'
):
    from hexrd.core import instrument

    instr = experiment.instrument
    analysis_dir = Path(experiment.analysis_dir)
    nfit = len(fit_results)

    # Make output directories: analysis directory and a subdirectory for
    # each panel.
    for det_key in instr.detectors:
        (analysis_dir / det_key).mkdir(parents=True, exist_ok=True)

    gw = instrument.GrainDataWriter(str(analysis_dir / grains_filename))
    gd_array = np.zeros((nfit, 21))
    gwa = instrument.GrainDataWriter(array=gd_array)
    for fit_result in fit_results:
        gw.dump_grain(*fit_result)
        gwa.dump_grain(*fit_result)
    gw.close()
    gwa.close()

    gdata = GrainData.from_array(gd_array)
    gdata.save(str(analysis_dir / grains_npz))


def execute(args, parser):
    return run_cli(
        args,
        lambda dashboard: _execute(args, parser, dashboard),
        stages=FIT_STAGES,
        view='FitGrainsDashboard',
    )


def _execute(args, parser, dashboard=None):
    log_level = (
        logging.ERROR
        if args.quiet
        else (logging.DEBUG if getattr(args, 'debug', False) else logging.INFO)
    )
    logger = logging.getLogger('hexrd')
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(log_level)
    logger.propagate = False
    ch = dashboard or logging.StreamHandler()
    ch.setLevel(logging.CRITICAL if args.quiet else log_level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)
    file_handlers = []
    try:
        if dashboard is not None:
            dashboard.start()
        with quiet_progress(args.quiet or getattr(args, 'no_tui', False)):
            return _run(args, parser, dashboard, logger, file_handlers)
    finally:
        for handler in file_handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.removeHandler(ch)
        ch.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _run(args, parser, dashboard, logger, file_handlers):
    from hexrd.core import instrument
    from hexrd.hedm.fitgrains import fit_grains

    log_level = logger.level
    clobber = args.force or args.clean
    with stage('Configuration'):
        base = HedmExperiment(args.yml)
        experiments = [base] + [
            HedmExperiment(args.yml, study=i) for i in range(len(base.studies))
        ]

    # handle initial state
    experiment = experiments[0]
    analysis_dir = Path(experiment.analysis_dir)

    # use path to grains.out to determine if analysis exists
    grains_filename = analysis_dir / 'grains.out'

    quats_f = Path(experiment.accepted_orientations_file)

    # some conditionals for arg handling
    have_orientations = quats_f.exists()
    existing_analysis = grains_filename.exists()

    def needs_orientations(experiment, existing):
        return args.clean or (
            experiment.fit_grains.estimate is None and (args.force or not existing)
        )

    # if no estimate is supplied, or the clean option is selected, will need
    # the indexing results from find_orientations:
    #   'accepted_orientations_*.dat'
    # result stored in the variable qbar
    if needs_orientations(experiment, existing_analysis):
        if have_orientations:
            try:
                qbar = np.loadtxt(quats_f, ndmin=2).T
            except IOError:
                raise (
                    RuntimeError,
                    "error loading indexing results '%s'" % quats_f,
                )
        else:
            logger.info("Missing %s, running find-orientations", quats_f)
            from hexrd.hedm.find_orientations import (
                find_orientations,
                write_results as write_find_orientations_results,
            )

            results = find_orientations(experiment)
            write_find_orientations_results(results, experiment)
            qbar = results.grain_orientations

    logger.info('=== begin fit-grains ===')

    for experiment in experiments:
        check_cancelled()
        diagnostic('Analysis', experiment.analysis_name)
        analysis_dir = Path(experiment.analysis_dir)

        # Check whether an existing analysis exists.
        grains_filename = analysis_dir / 'grains.out'

        if grains_filename.exists() and not clobber:
            logger.error(
                'Analysis "%s" already exists. ' 'Change yml file or specify "force"',
                experiment.analysis_name,
            )
            sys.exit()

        # Set up analysis directory and output directories.
        analysis_dir.mkdir(parents=True, exist_ok=True)

        instr = experiment.instrument
        for det_key in instr.detectors:
            det_dir = analysis_dir / det_key
            det_dir.mkdir(exist_ok=True)

        with stage('Material'):
            # Set HKLs to use.
            material = experiment.get_active_material()
            plane_data = material._material.planeData
            exclusions = experiment.fit_grains.exclusion_parameters.copy()
            tth_max = experiment.fit_grains.tth_max
            if tth_max is not False:
                exclusions['tthmax'] = (
                    instrument.max_tth(instr)
                    if tth_max is True
                    else np.radians(tth_max)
                )
            if experiment.fit_grains.reset_exclusions:
                plane_data.exclude(**exclusions)
            using_nhkls = np.count_nonzero(~plane_data.exclusions)
            logger.info(f'using {using_nhkls} HKLs')

        logger.info('*** begin analysis "%s" ***', experiment.analysis_name)

        # configure logging to file for this particular analysis
        logfile = analysis_dir / 'fit-grains.log'
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(log_level)
        ff = logging.Formatter('%(asctime)s - %(name)s - %(message)s', '%m-%d %H:%M:%S')
        fh.setFormatter(ff)
        logger.info("logging to %s", logfile)
        logger.addHandler(fh)
        file_handlers.append(fh)

        if args.profile:
            import cProfile as profile
            import pstats
            from io import StringIO

            pr = profile.Profile()
            pr.enable()

        # some conditionals for arg handling
        existing_analysis = grains_filename.exists()
        fit_estimate = experiment.fit_grains.estimate
        if needs_orientations(experiment, existing_analysis):
            # need accepted orientations from indexing in this case
            if args.clean:
                logger.info("'clean' specified; ignoring estimate and using default")
            elif args.force:
                logger.info(
                    "'force' option specified, but no initial estimate; "
                    + "using default"
                )
            try:
                # Write the accepted orientations (in `qbar`) to the
                # grains.out file
                gw = instrument.GrainDataWriter(grains_filename)
                for i_g, q in enumerate(qbar.T):
                    phi = 2 * np.arccos(q[0])
                    n = xfcapi.unit_vector(q[1:])
                    grain_params = np.hstack([phi * n, cnst.zeros_3, cnst.identity_6x1])
                    gw.dump_grain(int(i_g), 1.0, 0.0, grain_params)
                gw.close()
            except IOError:
                raise (
                    RuntimeError,
                    "indexing results '%s' not found!" % str(grains_filename),
                )
        elif fit_estimate is not None:
            grains_filename = fit_estimate
            logger.info("using initial estimate '%s'", fit_estimate)
        # get grain parameters by loading grains table
        try:
            grains_table = np.loadtxt(grains_filename, ndmin=2)
        except IOError:
            raise RuntimeError("problem loading '%s'" % grains_filename)

        # process the data
        gid_list = None
        if args.grains is not None:
            gid_list = [int(i) for i in args.grains.split(',')]

        fit_results = fit_grains(
            experiment,
            grains_table,
            material,
            show_progress=not args.quiet,
            check_if_canceled_func=(lambda: False) if dashboard is not None else None,
            ids_to_refine=gid_list,
        )

        if args.profile:
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
            logger.info('%s', s.getvalue())

        # stop logging for this particular analysis
        fh.flush()
        fh.close()
        logger.removeHandler(fh)

        logger.info('*** end analysis "%s" ***', experiment.analysis_name)

        with stage('Write results'):
            check_cancelled()
            write_results(fit_results, experiment)
        diagnostic('Results', analysis_dir)
        logger.info(
            'Fit %d grain(s); results written to %s', len(fit_results), analysis_dir
        )

    logger.info('=== end fit-grains ===')
