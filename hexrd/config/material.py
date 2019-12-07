import os

import pickle as cpl

from .config import Config


class MaterialConfig(Config):

    BASEKEY = 'material'

    def get(self, key, **kwargs):
        return self._cfg.get(':'.join([self.BASEKEY, key]), **kwargs)

    @property
    def definitions(self):
        temp = self.get('definitions')
        if not os.path.isabs(temp):
            temp = os.path.join(self._cfg.working_dir, temp)
        if os.path.exists(temp):
            return temp
        raise IOError(
            '"material:definitions": "%s" does not exist'
            )

    @property
    def active(self):
        return self.get('active')

    @property
    def material_dict(self):
        with open(self.definitions, "rb") as matf:
            mat_list = cpl.load(matf, encoding='latin1')
        return dict(list(zip([i.name for i in mat_list], mat_list)))

    @property
    def plane_data(self):
        return self.material_dict[self.active].planeData
