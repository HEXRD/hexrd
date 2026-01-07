"""Functions helping instrumented profiling of scripts

This functions provide functionality to easily add instrumented
profiling of certain functions. The functions to instrument will be
provided by a yaml file containing the name of the function with its
full dot path.
"""

from __future__ import print_function, absolute_import

import logging
import warnings

try:
    import importlib
except ImportError:
    pass

try:
    import nvtxpy as nvtx
except ImportError:
    pass


def instrument_function(fn_desc):
    """Interpret a record for an instrumented function, and instrument
    accordingly. The record, fn_desc, contains:

    'fn' is the full path to the function
    'color' is the color to use when instrumenting (nvtx feature)

    """

    # we must split 'fn' into the module and the function itself. Note
    # this is not trivial as we may find several levels of objects
    # inside the containing module. Try sequentially...
    full_name = fn_desc['fn']
    color = fn_desc.get('color', 'black')
    color = getattr(nvtx.colors, color, nvtx.colors.black)
    parts = full_name.split('.')

    # last item will always be the function name
    fn_name = parts[-1]

    # number of parts in the path
    path_parts = len(parts) - 1

    # consume as many as possible with import (ignore last part that is
    # the function name)
    pos = 0
    for i in range(1, path_parts + 1):
        try:
            m = importlib.import_module('.'.join(parts[0:i]))
            pos = i
        except ImportError as e:
            break

    # at this point, i points at the starting of the dotted path to the
    # function to instrument... follow the parts till we get to the
    # actual function
    try:
        o = m
        for i in range(pos, path_parts):
            o = getattr(o, parts[i])

        # instrument...
        original = getattr(o, fn_name)
        override = nvtx.profiled(full_name, color=color)(original)
        setattr(o, fn_name, override)
    except AttributeError:
        warnings.warn('Could not instrument "{0}"'.format(full_name))


def parse_file(filename):
    """Parse a file and instrument the associated functions"""
    try:
        import yaml

        with open(filename, 'r') as f:
            cfg = yaml.load(f)

        if 'profile' not in cfg:
            msg = 'profile file "{0}" missing a profile section'
            warnings.warn(msg.format(filename))
            return

        profile_cfg = cfg['profile']
        if 'instrument' in profile_cfg:
            # instrument all
            [
                instrument_function(fn_desc)
                for fn_desc in profile_cfg['instrument']
            ]
    except Exception as e:
        msg = 'Failed to include profile file: {0}'
        warnings.warn(msg.format(filename))
        warnings.warn(str(e))


def instrument_all(filenames):
    """Instrument functions based on a list of profiler configuration files."""
    [parse_file(filename) for filename in filenames]


def dump_results(args):
    logging.info(" STATS ".center(72, '='))
    fmt = "{2:>14}, {1:>8}, {0:<40}"
    logging.info(fmt.format("FUNCTION", "CALLS", "TIME"))
    fmt = "{2:>14F}, {1:>8}, {0:<40}"
    sorted_by_time = sorted(
        nvtx.getstats().iteritems(), key=lambda tup: tup[1][1]
    )
    for key, val in sorted_by_time:
        logging.debug(fmt.format(key, *val))
