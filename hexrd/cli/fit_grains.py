descr = 'Extracts G vectors, grain position and strain'
example = """
examples:
    hexrd fit-grains configuration.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('fit-grains', description=descr, help=descr)
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
        help='overwrites existing analysis, uses initial orientations'
        )
    p.add_argument(
        '-f', '--force', action='store_true',
        help='overwrites existing analysis'
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

    from hexrd import config
    from hexrd.fitgrains import fit_grains

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

    # if find-orientations has not already been run, do so:
    quats_f = os.path.join(
        cfgs[0].working_dir,
        'accepted_orientations_%s.dat' % cfgs[0].analysis_id
        )
    if not os.path.exists(quats_f):
        logger.info("Missing %s, running find-orientations", quats_f)
        logger.removeHandler(ch)
        from . import find_orientations
        find_orientations.execute(args, parser)
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
            import cProfile as profile
            import pstats
            from io import StringIO

            pr = profile.Profile()
            pr.enable()

        # process the data
        gid_list = None
        if args.grains is not None:
            gid_list = [int(i) for i in args.grains.split(',')]
            pass

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
