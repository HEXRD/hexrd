import argparse

descr = 'Display available HEXRD commands and their help'


def configure_parser(sub_parsers: argparse._SubParsersAction) -> None:
    p = sub_parsers.add_parser('help', description=descr, help=descr)
    p.add_argument(
        'command',
        metavar='COMMAND',
        action="store",
        nargs='?',
        help='print help for COMMAND (same as: hexrd COMMAND -h)',
    )
    p.set_defaults(func=execute)


def execute(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.command:
        parser.print_help()
        return

    import sys
    import subprocess

    result = subprocess.run(
        [sys.executable, '-m', 'hexrd.cli.main', args.command, '-h'], check=False
    )
    if result.returncode:
        parser.exit(result.returncode)
