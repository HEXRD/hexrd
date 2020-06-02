import sys


help = "Launches the hexrd graphical user interface"

# FIXME: must be updated to work with the new GUI

def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('gui', description=help, help=help)
    p.add_argument(
        '-q', '--quiet', action='store_true',
        help="don't report progress in terminal"
        )
    p.add_argument(
        '--qt', action='store_true',
        help='use the Qt user interface'
        )
    p.add_argument(
        'config', nargs='?', default=None,
        help='yaml configuration file'
        )
    p.set_defaults(func=execute)


def execute(args, parser):
    import logging

    logger = logging.getLogger('hexrd')
    logger.setLevel(logging.DEBUG)

    if args.qt:
        from hexrd.qt import execute

        execute(args)
    else:
        from hexrd.wx import mainapp

        mainapp.execute(*sys.argv[2:])
