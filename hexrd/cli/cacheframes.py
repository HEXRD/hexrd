

from hexrd.coreutil import initialize_experiment

descr = 'Caches detector frames in npz format'
example = """
examples:
    hexrd cache-frames configuration.yml
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('cache-frames', description = descr, help = descr)
    p.add_argument(
        'yml', type=str,
        help='YAML configuration file'
        )
    p.add_argument(
        '-q', '--quiet', action='store_true',
        help="don't report progress in terminal"
        )
    p.set_defaults(func=execute)


def execute(args, parser):
    import logging
    import os
    import sys

    import yaml

    from hexrd import config
    from hexrd.cacheframes import cache_frames


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

    logger.info('=== begin cache-frames ===')

    for cfg in cfgs:
        logger.info('*** begin caching for analysis "%s" ***', cfg.analysis_name)

        # configure logging to file for this particular analysis
        logfile = os.path.join(
            cfg.working_dir,
            cfg.analysis_name,
            'cache-frames.log'
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

        # process the data
        pd, reader, detector = initialize_experiment(cfg)
        cache_frames(reader, cfg, show_progress=not args.quiet)

        # stop logging for this particular analysis
        fh.flush()
        fh.close()
        logger.removeHandler(fh)

        logger.info('*** end caching for analysis "%s" ***', cfg.analysis_name)

    logger.info('=== end cache-frames ===')
    # stop logging to the console
    ch.flush()
    ch.close()
    logger.removeHandler(ch)
