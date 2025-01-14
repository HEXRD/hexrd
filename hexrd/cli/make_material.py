import sys
from hexrd.cli import help

descr = (f'make a new material file in the '
         f'hdf5 format using the command line. '
         f'user specifies the file and crystal '
         f'names')
example = """
examples:
    hexrd make_material --file material.h5 --xtal diamond
"""


def configure_parser(sub_parsers):

    p = sub_parsers.add_parser('make-material',
                               description=descr, help=descr)

    p.add_argument(
        '-f', '--file', type=str,
        default='test.h5',
        help='name of h5 file, default = test.h5'
    )

    p.add_argument(
        '-x', '--xtal', type=str,
        default='xtal',
        help='name of crystal, default = xtal'
    )

    p.set_defaults(func=execute)


def execute(args, parser):
    from hexrd.material.mksupport import mk

    file = args.file
    xtal = args.xtal
    mk(file, xtal)
