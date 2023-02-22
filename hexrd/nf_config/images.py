import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')

processing_methods = {
    'gaussian': dict(sigma=2.0, size=3.0),
    'dilations_only': dict(num_erosions=2, num_dilations=3)}

class ImagesConfig(Config):

    @property
    def data_folder(self):
        return self._cfg.get('images:data_folder')

    @property
    def stem(self):
        return self._cfg.get('images:stem')

    @property
    def img_start(self):
        return self._cfg.get('images:img_start')

    @property
    def nframes(self):
        return self._cfg.get('images:nframes', 1440)

    @property
    def processing_parameters(self):
        return ProcessingConfig(self._cfg)


class ProcessingConfig(Config):

    @property
    def num_for_dark(self):
        return self._cfg.get('images:processing_parameters:num_for_dark', 200)

    @property
    def img_threshold(self):
        return self._cfg.get('images:processing_parameters:img_threshold', 0)

    @property
    def ome_dilation_iter(self):
        return self._cfg.get('images:processing_parameters:ome_dilation_iter', 1)

    @property
    def method(self):
        key = 'images:processing_parameters:method'
        try:
            temp = self._cfg.get(key)
            assert len(temp) == 1., \
                "method must have exactly one key"
            if isinstance(temp, dict):
                method_spec = next(iter(list(temp.keys())))
                if method_spec.lower() not in processing_methods:
                    raise RuntimeError(
                        'invalid image processing method "%s"'
                        % method_spec
                    )
                else:
                    return temp
        except:
            raise RuntimeError(
                "Undefined image processing method"
            )
