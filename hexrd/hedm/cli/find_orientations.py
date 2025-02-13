from __future__ import print_function, division, absolute_import

import os
import logging
import sys

import numpy as np

from hexrd import constants as const
from hexrd import config
from hexrd import instrument
from hexrd.transforms import xfcapi
from hexrd.findorientations import find_orientations, write_scored_orientations


descr = 'Process rotation image series to find grain orientations'
example = """
examples:
    hexrd find-orientations configuration.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser(
        'find-orientations',
        description=descr,
        help=descr
        )
    p.add_argument(
        'yml', type=str,
        help='YAML configuration file'
        )
    p.add_argument(
        '-q', '--quiet', action='store_true',
        help="don't report progress in terminal"
        )
    p.add_argument(
        '-f', '--force', action='store_true',
        help='overwrites existing analysis'
        )
    p.add_argument(
        '-c', '--clean', action='store_true',
        help='overwrites existing analysis, including maps'
        )
    p.add_argument(
        '--hkls', metavar='HKLs', type=str, default=None,
        help="""\
          list hkl entries in the materials file to use for fitting;
          if None, defaults to list specified in the yml file"""
        )
    p.add_argument(
        '-p', '--profile', action='store_true',
        help='runs the analysis with cProfile enabled',
        )
    p.set_defaults(func=execute)


def write_results(results, cfg):
    # Write scored orientations.
    write_scored_orientations(results, cfg)

    # Write accepted orientations.
    qbar_filename = str(cfg.find_orientations.accepted_orientations_file)
    np.savetxt(qbar_filename, results['qbar'].T,
               fmt='%.18e', delimiter='\t')

    # Write grains.out.
    gw = instrument.GrainDataWriter(cfg.find_orientations.grains_file)
    for gid, q in enumerate(results['qbar'].T):
        phi = 2*np.arccos(q[0])
        n = xfcapi.unit_vector(q[1:])
        grain_params = np.hstack([phi*n, const.zeros_3, const.identity_6x1])
        gw.dump_grain(gid, 1., 0., grain_params)
    gw.close()


def execute(args, parser):
    # make sure hkls are passed in as a list of ints
    try:
        if args.hkls is not None:
            args.hkls = [int(i) for i in args.hkls.split(',') if i]
    except AttributeError:
        # called from fit-grains, hkls not passed
        args.hkls = None

    # configure logging to the console
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.quiet:
        log_level = logging.ERROR
    logger = logging.getLogger('hexrd')
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL if args.quiet else log_level)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S')
        )
    logger.addHandler(ch)
    logger.info('=== begin find-orientations ===')

    # load the configuration settings
    cfg = config.open(args.yml)[0]

    # prepare the analysis directory
    quats_f = cfg.find_orientations.accepted_orientations_file

    if (quats_f.exists()) and not (args.force or args.clean):
        logger.error(
            '%s already exists. Change yml file or specify "force" or "clean"',
            quats_f
        )
        sys.exit()

    # Create analysis directory and any intermediates.
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)

    # configure logging to file
    logfile = cfg.find_orientations.logfile
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(message)s',
            '%m-%d %H:%M:%S'
            )
        )
    logger.info("logging to %s", logfile)
    logger.addHandler(fh)

    if args.profile:
        import cProfile as profile
        import pstats
        from io import StringIO
        pr = profile.Profile()
        pr.enable()

    # process the data
    results = find_orientations(
        cfg,
        hkls=args.hkls,
        clean=args.clean,
        profile=args.profile
    )

    # Write out the results
    write_results(results, cfg)

    if args.profile:
        pr.disable()
        s = StringIO.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(50)
        logger.info('%s', s.getvalue())

    # clean up the logging
    fh.flush()
    fh.close()
    logger.removeHandler(fh)
    logger.info('=== end find-orientations ===')
    ch.flush()
    ch.close()
    logger.removeHandler(ch)
