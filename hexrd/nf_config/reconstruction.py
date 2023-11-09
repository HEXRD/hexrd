import logging
import os
from hexrd import config as Config


logger = logging.getLogger('hexrd.config')


class ReconstructionConfig(Config):

    @property
    def tomography(self):
        key = self._cfg.get('NF_reconstruction:tomography:use_mask')
        if key is True:
            parms = dict(use_mask = True,
            mask_data_file='NF_reconstruction:tomography:mask_data_file',
            mask_vert_offset='NF_reconstruction:tomography:mask_vert_offset',
            project_single_layer = 'NF_reconstruction:tomography:project_single_layer')
            return parms
        else:
            return

    @property
    def cross_sectional_dim(self):
        return self._cfg.get('NF_reconstruction:cross_sectional_dim')

    @property
    def voxel_spacing(self):
        return self._cfg.get('NF_reconstruction:voxel_spacing')

    @property
    def v_bnds(self):
        return self._cfg.get('NF_reconstruction:v_bnds',[0.0,0.0])

    @property
    def beam_stop_y_cen(self):
        return self._cfg.get('NF_reconstruction:beam_stop_y_cen')

    @property
    def beam_stop_width(self):
        return self._cfg.get('NF_reconstruction:beam_stop_width')
