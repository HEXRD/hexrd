from hexrd.material.mksupport import mk
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

def make_material():

    p = sub_parsers.add_parser('make_material',
        description = descr, help = descr)

    p.set_defaults(func=execute)

    p.add_argument(
        '-f', '--file', action='store',
        help='name of h5 file'
        )

    p.add_argument(
        '-x', '--xtal', action='store',
        help='name of crystal'
        )

def execute(parser):
    file = parser.parse_args['--file']
    xtal = parser.parse_args['--xtal']
    mk(file, xtal)
