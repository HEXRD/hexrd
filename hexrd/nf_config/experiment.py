import logging
import os
import numpy as np

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
    def comp_thresh(self):
        key = 'experiment:comp_thresh'
        temp = self._cfg.get(key, None)
        if temp is not None and np.any(
            np.logical_or(temp > 1.0, temp <= 0.0)
        ):
            raise RuntimeError(
                'comp_thresh must be None or a number between 0 and 1'
            )

        return temp

    @property
    def chi2_thresh(self):
        key = 'experiment:chi2_thresh'
        temp = self._cfg.get(key, None)
        if temp is not None and np.any(
            np.logical_or(temp > 1.0, temp <= 0.0)
        ):
            raise RuntimeError(
                'chi2_thresh must be None or a number between 0 and 1'
            )

        return temp


    @property
    def misorientation(self):
        key = self._cfg.get('experiment:misorientation:use_misorientation')
        if key is True:
            return {
                'misorientation_bnd': 'experiment:bnd',
                'misorientation_spacing': 'experiment:spacing',
            }
        return {
            'misorientation_bnd': None,
            'misorientation_spacing': None,
        }
