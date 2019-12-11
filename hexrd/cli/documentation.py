import sys


help = "Launches the hexrd documentation in a web browser"


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('documentation', description=help, help=help)
    p.set_defaults(func=execute)


def execute(args, parser):
    import webbrowser

    import hexrd
    webbrowser.open_new_tab(hexrd.doc_url)
