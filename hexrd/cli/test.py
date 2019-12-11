


descr = 'runs the hexrd test suite'
example = """
examples:
    hexrd test --verbose
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('test', description = descr, help = descr)
    p.add_argument(
        '-v', '--verbose', action='store_true',
        help="report detailed results in terminal"
        )
    p.set_defaults(func=execute)


def execute(args, parser):
    import unittest

    suite = unittest.TestLoader().discover('hexrd')
    unittest.TextTestRunner(verbosity = args.verbose + 1).run(suite)
