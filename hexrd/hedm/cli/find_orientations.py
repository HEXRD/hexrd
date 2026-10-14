"""The ``hexrd find-orientations`` command."""

import logging
import os
import sys
import cProfile
import io
import pstats
from contextlib import nullcontext
from dataclasses import replace

import numpy as np

from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.find_orientations import find_orientations, write_results

descr = 'Process rotation image series to find grain orientations'
example = """
examples:
    hexrd find-orientations configuration.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('find-orientations', description=descr, help=descr)
    p.add_argument('yml', type=str, help='YAML configuration file')
    p.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="don't report progress in terminal",
    )
    p.add_argument(
        '-f', '--force',
        action='store_true',
        help='overwrites existing analysis',
    )
    p.add_argument(
        '-c', '--clean',
        action='store_true',
        help='overwrites existing analysis, including cached eta-omega maps',
    )
    p.add_argument(
        '--hkls',
        metavar='HKLs',
        type=str,
        default=None,
        help="""\
          list hkl entries in the materials file to use for fitting;
          if None, defaults to list specified in the yml file""",
    )
    p.add_argument(
        '-p', '--profile',
        action='store_true',
        help='runs the analysis with cProfile enabled',
    )
    p.add_argument(
        '--study',
        type=int,
        default=None,
        help='apply the Nth study overlay of a multi-document config',
    )
    p.set_defaults(func=execute)


def execute(args, parser):
    log_level = logging.ERROR if args.quiet else (
        logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
    )
    logger = logging.getLogger('hexrd')
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(log_level)
    logger.propagate = False  # avoid double-printing via the root logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL if args.quiet else log_level)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S')
    )
    logger.addHandler(ch)
    fh = None
    try:
        logger.info('=== begin find-orientations ===')

        experiment = HedmExperiment(
            args.yml,
            study=getattr(args, 'study', None),
            working_dir=os.getcwd(),
            new_file_placement=False,
        )
        if experiment.active_material.dmin is None:
            experiment.active_material.dmin = 0.5
        hkls = getattr(args, 'hkls', None)
        if isinstance(hkls, str):
            hkls = [int(i) for i in hkls.split(',') if i]
        if hkls:
            fo = experiment.find_orientations
            experiment.find_orientations = replace(
                fo,
                orientation_maps=replace(
                    fo.orientation_maps, active_hkls=np.asarray(hkls, dtype=int)
                ),
            )

        accepted = experiment.accepted_orientations_file
        if os.path.exists(accepted) and not (args.force or args.clean):
            logger.error(
                '%s already exists. Change yml file or specify "force" or "clean"',
                accepted,
            )
            sys.exit()

        os.makedirs(experiment.analysis_dir, exist_ok=True)
        logfile = experiment.find_orientations_logfile
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(log_level)
        fh.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(message)s', '%m-%d %H:%M:%S')
        )
        logger.info('logging to %s', logfile)
        logger.addHandler(fh)

        with (
            cProfile.Profile() if getattr(args, 'profile', False) else nullcontext()
        ) as profile:
            results = find_orientations(experiment, clean=args.clean)
            output_dir = write_results(results, experiment)
        if profile is not None:
            summary = io.StringIO()
            pstats.Stats(profile, stream=summary).sort_stats('cumulative').print_stats(50)
            logger.info('%s', summary.getvalue())
        logger.info(
            'found %d grain(s); results written to %s',
            results.num_grains, output_dir,
        )

        logger.info('=== end find-orientations ===')
    finally:
        if fh is not None:
            logger.removeHandler(fh)
            fh.close()
        logger.removeHandler(ch)
        ch.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
