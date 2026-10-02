descr = "Displays a list of available hexrd commands and their help strings."


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('help', description=descr, help=descr)
    p.add_argument(
        'command',
        metavar='COMMAND',
        action="store",
        nargs='?',
        help="print help information for COMMAND (same as: hexrd COMMAND -h)",
    )
    p.set_defaults(func=execute)


def execute(args, parser):
    if not args.command:
        parser.print_help()
        return

    parser.parse_args([args.command, '-h'])
