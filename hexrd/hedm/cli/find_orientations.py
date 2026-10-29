"""The ``hexrd find-orientations`` command."""

import os
import sys
from contextlib import ExitStack
from dataclasses import replace

import numpy as np

from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.find_orientations import find_orientations, write_results
from hexrd.hedm._progress import (
    cli_logging, cli_profile, log_file, run_cli, quiet_progress, stage,
)

descr = 'Process rotation image series to find grain orientations'
example = """
examples:
    hexrd find-orientations configuration.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser(
        'find-orientations', description=descr, help=descr,
        epilog='Interactive terminals show a live stage dashboard; redirected '
               'output uses plain logs.',
    )
    p.add_argument('yml', type=str, help='YAML configuration file')
    p.add_argument('--no-tui', action='store_true',
                   help='use plain logs and exit on completion (for batch runs)')
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
    return run_cli(args, lambda dashboard: _execute(args, parser, dashboard))


def _execute(args, parser, dashboard=None):
    with cli_logging(args, dashboard) as logger, ExitStack() as stack:
        logger.info('=== begin find-orientations ===')

        with stage('Configuration'):
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
            stack.enter_context(log_file(logger, experiment.find_orientations_logfile))

        with (
            quiet_progress(args.quiet or getattr(args, 'no_tui', False)),
            cli_profile(getattr(args, 'profile', False), logger),
        ):
            results = find_orientations(experiment, clean=args.clean)
            with stage('Write results'):
                output_dir = write_results(results, experiment)
        logger.info(
            'found %d grain(s); results written to %s',
            results.num_grains, output_dir,
        )

        logger.info('=== end find-orientations ===')
