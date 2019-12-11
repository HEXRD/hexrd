"""Entry point for hexrd command line interface"""

import argparse
import logging
import sys
import multiprocessing
import warnings

# These can't be relative imports on Windows because of the hack
# in main() for multiprocessing.freeze_support()
from hexrd.cli import help
from hexrd.cli import test
from hexrd.cli import documentation
from hexrd.utils import profiler

from hexrd.cli import find_orientations
from hexrd.cli import fitgrains
from hexrd.cli import gui
from hexrd.cli import pickle23


def main():
    if sys.platform.startswith('win'):
        # Hack for multiprocessing.freeze_support() to work from a
        # setuptools-generated entry point.
        if __name__ != "__main__":
            sys.modules["__main__"] = sys.modules[__name__]
        multiprocessing.freeze_support()

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    p = argparse.ArgumentParser(
        description='High energy diffraction data analysis'
    )
    p.add_argument(
        "--debug",
        action = "store_true",
        help = 'verbose reporting',
    )
    p.add_argument(
        "--inst-profile",
        action="append",
        help='use the following files as source for functions to instrument',
    )
    sub_parsers = p.add_subparsers(
        metavar = 'command',
        dest = 'cmd',
    )

    help.configure_parser(sub_parsers)
    test.configure_parser(sub_parsers)
    documentation.configure_parser(sub_parsers)

    find_orientations.configure_parser(sub_parsers)
    fitgrains.configure_parser(sub_parsers)
    gui.configure_parser(sub_parsers)
    pickle23.configure_parser(sub_parsers)

    try:
        import argcomplete
        argcomplete.autocomplete(p)
    except ImportError:
        pass

    args = p.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = logging.getLogger('hexrd')
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    if args.inst_profile:
        profiler.instrument_all(args.inst_profile)

    args.func(args, p)

    if args.inst_profile:
        profiler.dump_results(args.inst_profile)
