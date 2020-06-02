import os

try:
    import dill as cpl
except(ImportError):
    import pickle as cpl

from .config import Config


class MaterialConfig(Config):
    """Handle material configuration."""

    @property
    def definitions(self):
        """Return the materials database filename."""
        temp = self._cfg.get('material:definitions')
        if not os.path.isabs(temp):
            temp = os.path.join(self._cfg.working_dir, temp)
        if os.path.exists(temp):
            return temp
        raise IOError(
            '"material:definitions": "%s" does not exist'
            )

    @property
    def active(self):
        """Return the active material key."""
        return self._cfg.get('material:active')

    @property
    def plane_data(self):
        """Return the active material PlaneData class."""
        with open(self.definitions, "rb") as matf:
            mat_list = cpl.load(matf)
        return dict(
            zip([i.name for i in mat_list], mat_list)
        )[self.active].planeData
