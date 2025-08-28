from .config import Config
from hexrd.core import imageseries

from hexrd.core.constants import shared_ims_key


class ImageSeries(Config):
    BASEKEY = 'image_series'

    def __init__(self, cfg):
        super(ImageSeries, self).__init__(cfg)
        self._image_dict = None

    def get(self, key):
        """Get item with given key."""
        return self._cfg.get(':'.join([self.BASEKEY, key]))

    @property
    def imageseries(self):
        """Return the imageseries dictionary"""
        if self._image_dict is None:
            self._image_dict = dict()
            fmt = self.format
            for ispec in self.data:
                fname = ispec['file']
                args = ispec['args']
                ims = imageseries.open(fname, fmt, **args)
                oms = imageseries.omega.OmegaImageSeries(ims)
                # handle special case for single IMS
                # for use with ROI
                try:
                    panel = oms.metadata['panel']
                    if isinstance(panel, (tuple, list)):
                        panel = '_'.join(panel)
                    elif panel is None:
                        panel = shared_ims_key
                except KeyError:
                    panel = shared_ims_key
                self._image_dict[panel] = oms

        return self._image_dict

    # ========== yaml inputs

    @property
    def data(self):
        return self.get('data')

    @property
    def format(self):
        return self.get('format')
