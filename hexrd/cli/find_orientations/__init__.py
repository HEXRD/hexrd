"""Command to find orientations"""


descr = 'Process diffraction data to find grain orientations'
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
    p.set_defaults(func=execute)

    p.add_argument(
        'yml', type=str,
        help="YAML configuration file; add :<n> to filename to select n'th doc"
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
list hkl entries in the materials file to use for fitting
if None, defaults to list specified in the yml file"""
        )
    p.add_argument(
        '-p', '--profile', action='store_true',
        help='runs the analysis with cProfile enabled',
        )


def execute(args, parser):
    import logging
    import os
    import sys

    import yaml

    from hexrd import config
    from .find_orientations import find_orientations

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
    yinfo = args.yml.split(':', 1)
    yname = yinfo[0]
    ydoc = 0 if len(yinfo) == 1 else int(yinfo[1])
    cfg = config.open(yname)[ydoc]

    # ...make this an attribute in cfg?
    analysis_id = '%s_%s' % (
        cfg.analysis_name.strip().replace(' ', '-'),
        cfg.material.active.strip().replace(' ', '-'),
    )

    # prepare the analysis directory
    quats_f = os.path.join(
        cfg.working_dir,
        'accepted_orientations_%s.dat' % analysis_id
        )
    if os.path.exists(quats_f) and not (args.force or args.clean):
        msg = '%s already exists. '\
              'Change yml file or specify "force" or "clean"' % quats_f
        logger.error(msg)
        sys.exit()
    if not os.path.exists(cfg.working_dir):
        os.makedirs(cfg.working_dir)

    # configure logging to file
    logfile = os.path.join(
        cfg.working_dir,
        'find-orientations_%s.log' % analysis_id
        )
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
        import io
        pr = profile.Profile()
        pr.enable()

    # process the data
    find_orientations(
        cfg, hkls=args.hkls, clean=args.clean, profile=args.profile
    )

    if args.profile:
        pr.disable()
        s = io.StringIO()
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
