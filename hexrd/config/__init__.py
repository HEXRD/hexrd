import yaml

from . import root
from . import utils

"""
Note that we need to use the open() builtin in what was formerly the "open()"
function. So we define the _open(), and then redefine open() to the new
function.
"""
open_file = open


def open(file_name=None):
    """
    Reads configuration settings from a yaml file.

    Returns a list of configuration objects, one for each document section in
    the file.
    """
    if file_name is None:
        return [root.RootConfig({})]

    with open_file(file_name) as f:
        res = []
        for cfg in yaml.load_all(f, Loader=yaml.SafeLoader):
            try:
                # take the previous config section and update with values
                # from the current one
                res.append(utils.merge_dicts(res[-1], cfg))
            except IndexError:
                # this is the first config section
                res.append(cfg)

        return [root.RootConfig(i) for i in res]


def save(config_list, file_name):
    res = [cfg._cfg for cfg in config_list]

    with open_file(file_name, 'w') as f:
        if len(res) > 1:
            yaml.dump_all(res, f)
        else:
            yaml.dump(res, f)
