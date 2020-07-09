"""Convert python 2 hexrd pickles to hexrd3"""
import sys
import shutil

descr = r"""modify old material files (pickles) to be compatible with hexrd3;
      it makes a backup and overwrites the original file
"""


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('pickle23',
                               description = descr,
                               help = descr)
    p.set_defaults(func=execute)

    p.add_argument(
        'file', type=str,
        help='name of file to convert'
        )


def execute(args, p):
    """convert module paths to hexrd3"""
    fname = args.file
    fback = fname + ".bak"
    shutil.copy(fname, fback)
    with  open(fname, "w") as fnew:
        with open(fback, "r") as f:
            for  l in f:
                l = l.replace('hexrd.xrd.', 'hexrd.')
                fnew.write(l)
    return
