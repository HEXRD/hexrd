"""Browse documentation"""

import argparse

help = "Launch the HEXRD documentation in a web browser"


def configure_parser(sub_parsers: argparse._SubParsersAction) -> None:
    p = sub_parsers.add_parser('documentation', description=help, help=help)
    p.set_defaults(func=execute)


def execute(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import webbrowser

    webbrowser.open_new_tab('https://hexrd.readthedocs.io/en/latest/')
