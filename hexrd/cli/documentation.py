"""Browse documentation"""

import argparse

help = "Launches the hexrd documentation (work in progress) in a web browser"


def configure_parser(sub_parsers: argparse._SubParsersAction) -> None:
    p = sub_parsers.add_parser('documentation', description=help, help=help)
    p.set_defaults(func=execute)


def execute(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import webbrowser

    import hexrd

    webbrowser.open_new_tab(hexrd.doc_url)
