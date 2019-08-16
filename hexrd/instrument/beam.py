"""Beam parameters"""

import numpy as np

from hexrd import constants
from hexrd.extensions._transforms_CAPI import unitRowVector

beam_energy_DFLT = 65.351
beam_vec_DFLT = constants.beam_vec


class Beam(object):

    def __init__(self,
                 energy=beam_energy_DFLT,
                 vector=beam_vec_DFLT):
        self._energy = energy
        self._vector = vector

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, x):
        """
        assumes input float in keV
        """
        self._energy = float(x)

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, x):
        x = np.array(x).flatten()
        if len(x) == 3:
            if np.abs(sum(x*x) - 1.) > constants.sqrt_epsf:
                raise RuntimeError("beam vector not a unit vector")
            self._vector = x
        elif len(x) == 2:
            self._vector = self._calc_beam_vec(*x)
        else:
            raise RuntimeError("input must be a unit vector or angle pair")
        assert len(x) == 3
        self._vector = unitRowVector(np.atleast_1d(x).flatten())

    @property
    def wavelength(self):
        return constants.keVToAngstrom(self.energy)

    @wavelength.setter
    def wavelength(self, x):
        """
        in angstrom
        """
        self._energy = constants.keVToAngstrom(x)

    @property
    def angles(self):
        """Azimuth and polar angle of beam vector"""
        nvec = unitRowVector(-self.vector)
        azim = float(
            np.degrees(np.arctan2(nvec[2], nvec[0]))
        )
        pola = float(np.degrees(np.arccos(nvec[1])))
        return azim, pola

    @staticmethod
    def _calc_beam_vec(azim, pola):
        """
        Calculate unit beam propagation vector from
        spherical coordinate spec in DEGREES
        """
        tht = np.radians(azim)
        phi = np.radians(pola)
        bv = np.r_[
            np.sin(phi)*np.cos(tht),
            np.cos(phi),
            np.sin(phi)*np.sin(tht)]
        return -bv
