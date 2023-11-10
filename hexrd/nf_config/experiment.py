import logging
import os
import numpy as np

from .config import Config


logger = logging.getLogger('hexrd.config')


class ExperimentConfig(Config):
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def beam_energy(self):
        return self._cfg.get('experiment:beam_energy')

    @property
    def mat_name(self):
        return self._cfg.get('experiment:mat_name')

    @property
    def ome_range(self):
        ome_start = self._cfg.get('experiment:ome_start', 0)
        ome_end = self._cfg.get('experiment:ome_end', 359.75)
        ome_range = [(ome_start, ome_end)]
        return ome_range

    @property
    def max_tth(self):
        return self._cfg.get('experiment:max_tth', None)

    @property
    def comp_thresh(self):
        temp = self._cfg.get('experiment:comp_thresh', None)
        if temp is None:
            return temp
        elif np.logical_and(temp <= 1.0, temp > 0.0):
            return temp
        else:
            raise RuntimeError(
                'comp_thresh must be None or a number between 0 and 1')

    @property
    def chi2_thresh(self):
        temp = self._cfg.get('experiment:chi2_thresh', None)
        if temp is None:
            return temp
        elif np.logical_and(temp <= 1.0, temp > 0.0):
            return temp
        else:
            raise RuntimeError(
                'chi2_thresh must be None or a number between 0 and 1')

    @property
    def misorientation(self):
        key = self._cfg.get(
            'experiment:misorientation:use_misorientation', False)
        if key is True:
            parms = dict(misorientation_bnd=self.get('experiment:misorientation:bnd', 0.0),
                         misorientation_spacing=self.get('experiment:misorientation:spacing', 0.25))
            return parms
        else:
            return
