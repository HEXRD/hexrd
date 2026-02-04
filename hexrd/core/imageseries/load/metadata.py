"""metadata tools for imageseries"""

import os

import yaml
import numpy as np
from numpy.typing import NDArray


def yamlmeta(meta, path=None) -> dict[str, NDArray[np.float64]]:
    """Image sequence metadata

        *path* is a full path or directory used to find the relative location
               of files loaded via the trigger mechanism

    The usual yaml dictionary is returned with the exception that
    if the first word of a multiword string is an exclamation mark ("!"),
    it will trigger further processing determined by the rest of the string.
    Currently only one trigger is used:

    ! load-numpy-object <filename>
      the returned value will the numpy object read from the file
    """
    if path is not None:
        path = os.path.dirname(path)
    else:
        path = '.'

    metad = {}
    for k, v in list(meta.items()):
        # check for triggers
        istrigger = False
        if isinstance(v, str):
            words = v.split()
            istrigger = (words[0] == "!") and (len(words) > 1)

        if v == '++np.array':  # old way used in frame-cache (obsolescent)
            newk = k + '-array'
            metad[k] = np.array(meta.pop(newk))
            metad.pop(newk, None)
        elif istrigger:
            if words[1] == "load-numpy-array":
                fname = os.path.join(path, words[2])
                metad[k] = np.load(fname)
        else:
            metad[k] = v

    return metad
