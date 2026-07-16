"""The `hexrd find-orientations` command: rotation series -> grain orientations.

A thin driver over :mod:`hexrd.hedm.find_orientations`: parse the experiment
config, run the five-stage pipeline, write the results.
"""
import logging
import os
import sys

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
    logger.setLevel(log_level)
    logger.propagate = False  # avoid double-printing via the root logger
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S')
    )
    logger.addHandler(ch)
    logger.info('=== begin find-orientations ===')

    experiment = HedmExperiment(args.yml, study=args.study)
    material = experiment.get_active_material()

    actmat = experiment.active_material.active
    accepted = os.path.join(
        experiment.analysis_dir, f'accepted-orientations-{actmat}.dat'
    )
    if os.path.exists(accepted) and not (args.force or args.clean):
        logger.error(
            '%s already exists. Change yml file or specify "force" or "clean"',
            accepted,
        )
        sys.exit()

    # log to a file in the analysis directory as well
    os.makedirs(experiment.analysis_dir, exist_ok=True)
    logfile = os.path.join(
        experiment.analysis_dir, f'find-orientations-{actmat}.log'
    )
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(message)s', '%m-%d %H:%M:%S')
    )
    logger.info('logging to %s', logfile)
    logger.addHandler(fh)

    results = find_orientations(experiment, material, clean=args.clean)
    output_dir = write_results(results, experiment)
    logger.info(
        'found %d grain(s); results written to %s',
        results.num_grains, output_dir,
    )

    fh.flush()
    fh.close()
    logger.removeHandler(fh)
    logger.info('=== end find-orientations ===')
    ch.flush()
    ch.close()
    logger.removeHandler(ch)
