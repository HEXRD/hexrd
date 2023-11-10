import os
import logging
import multiprocessing as mp

from hexrd.constants import shared_ims_key
from hexrd import imageseries

from .config import Config
from .multiprocessing import MultiprocessingConfig
from .reconstruction import ReconstructionConfig
from .images import ImagesConfig
from .experiment import ExperimentConfig
from .input_files import InputConfig
from .tomography import TomographyConfig

logger = logging.getLogger('hexrd.config')


class NFRootConfig(Config):
    def __init__(self, cfg):
        self._cfg = cfg
        
    @property
    def main_dir(self):
        return self._cfg.get('main_dir')

    @property
    def output_dir(self):
        return self._cfg.get('output_dir', self.main_dir)

    @property
    def analysis_name(self):
        return self._cfg.get('analysis_name', 'NF_')

    @property
    def output_plot_check(self):
        return self._cfg.get('output_plot_check')

    @property
    def multiprocessing(self):
        return MultiprocessingConfig(self)

    @property
    def tomography(self):
        return TomographyConfig(self)

    @property
    def reconstruction(self):
        return ReconstructionConfig(self)

    @property
    def images(self):
        return ImagesConfig(self)

    @property
    def experiment(self):
        return ExperimentConfig(self)

    @property
    def input_files(self):
        return InputConfig(self)
