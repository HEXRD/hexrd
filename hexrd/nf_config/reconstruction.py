import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')


class ReconstructionConfig(Config):

    @property
    def tomography(self):
        key = self._cfg.get('reconstruction:tomography:use_mask')
        if key is True:
            parms = dict(use_mask = True,
            mask_data_file='reconstruction:tomography:mask_data_file',
            mask_vert_offset='reconstruction:tomography:mask_vert_offset')
            return parms
        else:
            parms = dict(use_mask = False,
            mask_data_file=None,
            mask_vert_offset=None)
            return

    @property
    def cross_sectional_dim(self):
        return self._cfg.get('reconstruction:cross_sectional_dim')

    @property
    def voxel_spacing(self):
        return self._cfg.get('reconstruction:voxel_spacing')

    @property
    def v_bnds(self):
        return self._cfg.get('reconstruction:v_bnds',[0.0,0.0])

    @property
    def beam_stop_y_cen(self):
        return self._cfg.get('reconstruction:beam_stop_y_cen')

    @property
    def beam_stop_width(self):
        return self._cfg.get('reconstruction:beam_stop_parms')
