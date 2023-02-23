import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')


class ExperimentConfig(Config):

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
    def misorientation(self):
        key = self._cfg.get('experiment:misorientation:use_misorientation')
        if key is True:
            parms = dict(misorientation_bnd='experiment:bnd',
            misorientation_spacing='experiment:spacing')
            return parms
        else:
            parms = dict(misorientation_bnd=0,
            misorientation_spacing=0.1) #won't look at spacing if bnd is 0
            return parms
