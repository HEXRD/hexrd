import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')


class InputConfig(Config):

    @property
    def det_file(self):
        temp = self._cfg.get('input_files:det_file')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp

    @property
    def mat_file(self):
        temp = self._cfg.get('input_files:mat_file')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp

    @property
    def grain_out_file(self):
        temp = self._cfg.get('input_files:grain_out_file')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp
