"""Entry point for hexrd command line interface"""
_description = 'Preprocess detector images'

def configure_parser(sub_parsers):
    parser = sub_parsers.add_parser('preprocess', description=_description, help=_description)

    subparsers = parser.add_subparsers(
        dest="profile", required=True, help="Select detector profile"
    )

    parser.set_defaults(func=execute)


def execute(args, parser):
    pass
