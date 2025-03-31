import os

import numpy as np

from hexrd.core import material
from hexrd.core.constants import keVToAngstrom
from hexrd.core.valunits import valWUnit

from .config import Config
from .utils import get_exclusion_parameters


DMIN_DFLT = 0.5    # angstrom
TTHW_DFLT = 0.25   # degrees


class MaterialConfig(Config):
    """Handle material configuration."""

    def __init__(self, cfg):
        super().__init__(cfg)
        re, ep = get_exclusion_parameters(self._cfg, 'material')
        self._reset_exclusions, self._exclusion_parameters = re, ep

    @property
    def definitions(self):
        """Return the materials database filename."""
        temp = self._cfg.get('material:definitions')
        if not os.path.isabs(temp):
            temp = os.path.join(self._cfg.working_dir, temp)
        if os.path.exists(temp):
            return temp
        raise IOError(
            f'"material:definitions": "{temp}" does not exist'
        )

    @property
    def active(self):
        """Return the active material key."""
        return self._cfg.get('material:active')

    @property
    def materials(self):
        """Return a dict of materials."""
        #
        # If reset_exclusions is False, we use the material as read from
        # the file. This includes not using the dmin value, which may alter
        # the HKLs available.
        #
        if not hasattr(self, '_materials'):
            kwa = {"f": self.definitions}
            if self.reset_exclusions:
                kwa["dmin"] = valWUnit("dmin", "length", self.dmin, "angstrom")
            self._materials = material.load_materials_hdf5(**kwa)
        return self._materials

    @materials.setter
    def materials(self, mats):
        assert isinstance(mats, dict), "input must be a dict"
        self._materials = mats

    @property
    def dmin(self):
        """Return the specified minimum d-spacing for hkl generation."""
        return self._cfg.get('material:dmin', DMIN_DFLT)

    @property
    def tthw(self):
        """Return the specified tth tolerance."""
        return self._cfg.get('material:tth_width', TTHW_DFLT)

    @property
    def fminr(self):
        """Return the specified tth tolerance."""
        return self._cfg.get('material:min_sfac_ratio', None)

    @property
    def reset_exclusions(self):
        """Flag to use hkls saved in the material"""
        return self._reset_exclusions

    @property
    def exclusion_parameters(self):
        return self._exclusion_parameters

    @property
    def plane_data(self):
        """crystallographic information"""
        #
        # Only generate this once, not on each call.
        #
        if not hasattr(self, "_plane_data"):
            self._plane_data = self._make_plane_data()
        return self._plane_data

    def _make_plane_data(self):
        """Return the active material PlaneData class."""
        pd = self.materials[self.active].planeData
        pd.tThWidth = np.radians(self.tthw)
        if self.reset_exclusions:
            pd.exclude(**self.exclusion_parameters._asdict())
        return pd

    @property
    def beam_energy(self):
        return keVToAngstrom(self.plane_data.wavelength)

    @beam_energy.setter
    def beam_energy(self, x):
        if not isinstance(x, valWUnit):
            x = valWUnit("beam energy", "energy", x, "keV")
        for matl in self.materials.values():
            matl.beamEnergy = x
