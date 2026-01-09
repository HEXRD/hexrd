import logging
import os
import sys
from collections import namedtuple

import numpy as np

from hexrd.hedm import config
from hexrd.core import config
from hexrd.core import constants as cnst
from hexrd.core import rotations
from hexrd.core import instrument
from hexrd.hedm.findorientations import find_orientations
from hexrd.hedm.fitgrains import fit_grains
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
    p = sub_parsers.add_parser('fit-grains', description=descr, help=descr)
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
    fit_results, cfg, grains_filename='grains.out', grains_npz='grains.npz'
):
    instr = cfg.instrument.hedm
    nfit = len(fit_results)

    # Make output directories: analysis directory and a subdirectory for
    # each panel.
    for det_key in instr.detectors:
        (cfg.analysis_dir / det_key).mkdir(parents=True, exist_ok=True)

    gw = instrument.GrainDataWriter(str(cfg.analysis_dir / grains_filename))
    gd_array = np.zeros((nfit, 21))
    gwa = instrument.GrainDataWriter(array=gd_array)
    for fit_result in fit_results:
        gw.dump_grain(*fit_result)
        gwa.dump_grain(*fit_result)
    gw.close()
    gwa.close()

    gdata = GrainData.from_array(gd_array)
    gdata.save(str(cfg.analysis_dir / grains_npz))


def execute(args, parser):
    clobber = args.force or args.clean

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

    # handle initial state
    cfg = cfgs[0]

    # use path to grains.out to determine if analysis exists
    grains_filename = cfg.find_orientations.grains_file

    # path to accepted_orientations
    quats_f = cfg.find_orientations.accepted_orientations_file

    # some conditionals for arg handling
    have_orientations = quats_f.exists()
    existing_analysis = grains_filename.exists()
    fit_estimate = cfg.fit_grains.estimate
    force_without_estimate = args.force and fit_estimate is None
    new_without_estimate = not existing_analysis and fit_estimate is None

    # if no estimate is supplied, or the clean option is selected, will need
    # the indexing results from find_orientations:
    #   'accepted_orientations_*.dat'
    # result stored in the variable qbar
    if args.clean or force_without_estimate or new_without_estimate:
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
            logger.removeHandler(ch)
            results = find_orientations(cfg)
            qbar = results['qbar']
            logger.addHandler(ch)

    logger.info('=== begin fit-grains ===')

    for cfg in cfgs:

        # Check whether an existing analysis exists.
        grains_filename = cfg.fit_grains.grains_file

        if grains_filename.exists() and not clobber:
            logger.error(
                'Analysis "%s" already exists. ' 'Change yml file or specify "force"',
                cfg.analysis_name,
            )
            sys.exit()

        # Set up analysis directory and output directories.
        cfg.analysis_dir.mkdir(parents=True, exist_ok=True)

        instr = cfg.instrument.hedm
        for det_key in instr.detectors:
            det_dir = cfg.analysis_dir / det_key
            det_dir.mkdir(exist_ok=True)

        # Set HKLs to use.
        if cfg.fit_grains.reset_exclusions:
            excl_p = cfg.fit_grains.exclusion_parameters
            #
            # tth_max can be True, False or a value
            #
            if cfg.fit_grains.tth_max is not False:
                if cfg.fit_grains.tth_max is True:
                    maxtth = instrument.max_tth(cfg.instrument.hedm)
                else:
                    maxtth = np.radians(cfg.fit_grains.tth_max)
                excl_p = excl_p._replace(tthmax=maxtth)

            cfg.material.plane_data.exclude(**excl_p._asdict())
        using_nhkls = np.count_nonzero(
            np.logical_not(cfg.material.plane_data.exclusions)
        )
        logger.info(f'using {using_nhkls} HKLs')

        logger.info('*** begin analysis "%s" ***', cfg.analysis_name)

        # configure logging to file for this particular analysis
        logfile = cfg.fit_grains.logfile
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(log_level)
        ff = logging.Formatter('%(asctime)s - %(name)s - %(message)s', '%m-%d %H:%M:%S')
        fh.setFormatter(ff)
        logger.info("logging to %s", logfile)
        logger.addHandler(fh)

        if args.profile:
            import cProfile as profile
            import pstats
            from io import StringIO

            pr = profile.Profile()
            pr.enable()

        # some conditionals for arg handling
        existing_analysis = grains_filename.exists()
        fit_estimate = cfg.fit_grains.estimate
        new_with_estimate = not existing_analysis and fit_estimate is not None
        new_without_estimate = not existing_analysis and fit_estimate is None
        force_with_estimate = args.force and fit_estimate is not None
        force_without_estimate = args.force and fit_estimate is None
        #
        # ------- handle args
        # - 'clean' indicates ignoring any estimate specified and starting with
        #   the 'accepted_orientations' file.  Will run find-orientations if
        #   it doesn't exist
        # - 'force' means ignore existing analysis directory.  If the config
        #   option "fit_grains:estimate" is None, will use results from
        #   find-orientations.  If 'accepted_orientations' does not exists,
        #   then it runs find-orientations.
        #
        if args.clean or force_without_estimate or new_without_estimate:
            # need accepted orientations from indexing in this case
            if args.clean:
                logger.info("'clean' specified; ignoring estimate and using default")
            elif force_without_estimate:
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
        elif force_with_estimate or new_with_estimate:
            grains_filename = fit_estimate
            logger.info("using initial estimate '%s'", fit_estimate)
        elif existing_analysis and not clobber:
            raise (
                RuntimeError,
                "fit results '%s' exist, " % grains_filename
                + "but --clean or --force options not specified",
            )

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
            cfg,
            grains_table,
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

        write_results(fit_results, cfg)

    logger.info('=== end fit-grains ===')
    # stop logging to the console
    ch.flush()
    ch.close()
    logger.removeHandler(ch)
