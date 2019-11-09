"""Base Config class"""

import logging

from .utils import null

logger = logging.getLogger('hexrd.config')

class Config(object):

    _dirty = False

    def __init__(self, cfg):
        self._cfg = cfg

    def __getitem__(self, key):
        return self.get(key)

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
