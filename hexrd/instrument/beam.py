"""Beam parameters"""

import numpy as np

from hexrd import constants
from hexrd.extensions._transforms_CAPI import unitRowVector

class Beam(object):

    def __init__(self, energy, vector):
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


def calc_beam_vec(azim, pola):
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


def calc_angles_from_beam_vec(bvec):
    """
    Return the azimuth and polar angle from a beam
    vector
    """
    bvec = np.atleast_1d(bvec).flatten()
    nvec = unitRowVector(-bvec)
    azim = float(
        np.degrees(np.arctan2(nvec[2], nvec[0]))
    )
    pola = float(np.degrees(np.arccos(nvec[1])))
    return azim, pola
