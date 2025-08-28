import os

import numpy as np

from .config import Config
from hexrd.core import imageseries


class Beam(Config):

    BASEKEY = 'beam'

    def __init__(self, cfg):
        super(ImageSeries, self).__init__(cfg)
        self._image_dict = None
