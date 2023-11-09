import logging
import os

from hexrd import config as Config


logger = logging.getLogger('hexrd.config')


class TomographyConfig(Config):

    @property
    def data_folder(self):
        return self._cfg.get('tomography:data_folder')

    @property
    def img_stem(self):
        return self._cfg.get('tomography:img_stem')

    @property
    def bright(self):
        return BrightConfig(self._cfg)

    @property
    def dark(self):
        return DarkConfig(self._cfg)

    @property
    def tomo_images(self):
        return TomoImagesConfig(self._cfg)

    @property
    def num_digits(self):
        return self._cfg.get('tomography:num_digits', 6)

    @property
    def ome_range(self):
        ome_start = self._cfg.get('tomography:ome_start', 0)
        ome_end = self._cfg.get('tomography:ome_end', 359.75)
        ome_range = [(ome_start, ome_end)]
        return ome_range

    @property
    def processing(self):
        return TomoProcessingConfig(self._cfg)

    @property
    def reconstruction(self):
        return TomoReconstructionConfig(self._cfg)


class BrightConfig(Config):

    @property
    def folder(self):
        return self._cfg.get('tomography:bright:folder')

    @property
    def img_start(self):
        return self._cfg.get('tomography:bright:img_start')

    @property
    def num_imgs(self):
        return self._cfg.get('tomography:bright:num_imgs')


class DarkConfig(Config):

    @property
    def folder(self):
        return self._cfg.get('tomography:dark:folder')

    @property
    def img_start(self):
        return self._cfg.get('tomography:dark:img_start')

    @property
    def num_imgs(self):
        return self._cfg.get('tomography:dark:num_imgs')


class TomoImagesConfig(Config):

    @property
    def img_start(self):
        return self._cfg.get('tomography:tomo_images:img_start')

    @property
    def num_imgs(self):
        return self._cfg.get('tomography:tomo_images:num_imgs')

    @property
    def folder(self):
        return self._cfg.get('tomography:tomo_images:folder')

class TomoProcessingConfig(Config):

    @property
    def recon_thresh(self):
        return self._cfg.get('tomography:processing:recon_thresh')

    @property
    def noise_obj_size(self):
        return self._cfg.get('tomography:processing:noise_obj_size', 500)

    @property
    def min_hole_size(self):
        return self._cfg.get('tomography:processing:min_hole_size', 500)

    @property
    def erosion_iter(self):
        return self._cfg.get('tomography:processings:erosion_iter', 1)

    @property
    def dilation_iter(self):
        return self._cfg.get('tomography:processing:dilation_iter', 1)


class TomoReconstructionConfig(Config):

    @property
    def project_single_layer(self):
        return self._cfg.get('tomography:reconstruction:project_single_layer', True)

    @property
    def cross_sectional_dim(self):
        return self._cfg.get('tomography:reconstruction:cross_sectional_dim')

    @property
    def voxel_spacing(self):
        return self._cfg.get('tomography:reconstruction:voxel_spacing')

    @property
    def v_bnds(self):
        return self._cfg.get('tomography:reconstruction:v_bnds', [0.0, 0.0])
