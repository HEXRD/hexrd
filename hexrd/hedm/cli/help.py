

descr = "Displays a list of available conda commands and their help strings."

def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('help',
                               description = descr,
                               help = descr)
    p.add_argument(
        'command',
        metavar = 'COMMAND',
        action = "store",
        nargs = '?',
        help = "print help information for COMMAND "
               "(same as: conda COMMAND -h)",
    )
    p.set_defaults(func=execute)


def execute(args, parser):
    if not args.command:
        parser.print_help()
        return

    import sys
    import subprocess

    subprocess.call([sys.executable, sys.argv[0], args.command, '-h'])
