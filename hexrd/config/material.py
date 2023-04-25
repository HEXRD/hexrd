import os

import numpy as np

from hexrd import material
from hexrd.constants import keVToAngstrom
from hexrd.valunits import valWUnit

from .config import Config

DMIN_DFLT = 0.5    # angstrom
TTHW_DFLT = 0.25   # degrees


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
    def materials(self):
        """Return a dict of materials."""
        if not hasattr(self, '_materials'):
            self._materials = material.load_materials_hdf5(
                f=self.definitions,
                dmin=valWUnit("dmin", "length", self.dmin, "angstrom")
            )
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
        if self.fminr is not None:
            # !!! exclusions are all False upon generation; setting
            #     the tThMax attribute later won't affect these.
            excl_full = np.array(pd.exclusions, dtype=bool)
            pd.exclusions = None
            mod_f_sq = pd.structFact
            excl_these = mod_f_sq/np.max(mod_f_sq) < self.fminr
            pd.exclusions = np.logical_or(excl_full, excl_these)
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
