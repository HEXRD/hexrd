"""Entry point for hexrd command line interface"""

import argparse
import logging
import sys
import multiprocessing
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

from hexrd.core.utils import profiler


# Register workflow commands here, outside core.
COMMAND_MODULES = (
    'hexrd.cli.help',
    'hexrd.cli.documentation',
    'hexrd.hedm.cli.find_orientations',
    'hexrd.hedm.cli.fit_grains',
    'hexrd.hedm.cli.preprocess',
    'hexrd.powder.cli.cake_data',
)

try:
    _version = version("hexrd")
except PackageNotFoundError:
    _version = None


def main():
    if sys.platform.startswith('win'):
        # Hack for multiprocessing.freeze_support() to work from a
        # setuptools-generated entry point.
        if __name__ != "__main__":
            sys.modules["__main__"] = sys.modules[__name__]
        multiprocessing.freeze_support()

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    p = argparse.ArgumentParser(description='High energy diffraction data analysis')
    p.add_argument(
        "--debug",
        action="store_true",
        help='verbose reporting',
    )
    p.add_argument(
        "--inst-profile",
        action="append",
        help='use the following files as source for functions to instrument',
    )
    p.add_argument(
        "--version",
        action="version",
        version=f'%(prog)s {_version}',
    )

    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    for module_name in COMMAND_MODULES:
        import_module(module_name).configure_parser(sub_parsers)

    try:
        import argcomplete

        argcomplete.autocomplete(p)
    except ImportError:
        pass

    args = p.parse_args()
    logging.info(f'HEXRD version: {_version}')

    log_level = logging.DEBUG if args.debug else logging.INFO
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    if args.inst_profile:
        profiler.instrument_all(args.inst_profile)

    args.func(args, p)

    if args.inst_profile:
        profiler.dump_results(args.inst_profile)
