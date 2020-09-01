"""Base Config class"""

import logging
import os

from .utils import null

logger = logging.getLogger('hexrd.config')

class Config(object):

    _dirty = False

    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def dirty(self):
        return self._dirty

    def get(self, key, default=null):
        args = key.split(':')
        args, item = args[:-1], args[-1]
        temp = self._cfg
        for arg in args:
            temp = temp.get(arg, {})
            # intermediate block may be None:
            temp = {} if temp is None else temp
        try:
            res = temp[item]
        except KeyError:
            if default is not null:
                logger.info(
                    '%s not specified, defaulting to %s', key, default
                    )
                res = temp.get(item, default)
            else:
                raise RuntimeError(
                    '%s must be specified in configuration file' % key
                    )
        return res

    def set(self, key, val):
        args = key.split(':')
        args, item = args[:-1], args[-1]
        temp = self._cfg
        for arg in args:
            temp = temp.get(arg, {})
            # intermediate block may be None:
            temp = {} if temp is None else temp
        if temp.get(item, null) != val:
            temp[item] = val
            self._dirty = True

    def dump(self, filename):
        import yaml

        with open(filename, 'w') as f:
            yaml.dump(self._cfg, f)
        self._dirty = False

    @staticmethod
    def check_filename(fname, wdir):
        """Check whether filename is valid relative to working directory

        fname - the name of the file

        Returns the absolute path of the filename if the file exists

        If fname is an absolute path, use that; otherwise take it as a path relative
        to the working directory.
"""
        temp = fname
        if not os.path.isabs(fname):
            temp = os.path.join(wdir, temp)
        if os.path.exists(temp):
            return temp
        raise IOError(
            'file: "%s" not found' % temp
            )
