import numpy as np
from hexrd.material.utils import (calculate_linear_absorption_length,
    calculate_energy_absorption_length)
from hexrd.constants import density, density_compounds

class abstractlayer:
    """abstract class for encode information
    for an arbitrary planar layer of given
    thickness, density, and material

    Parameters
    ----------
    material : str or hexrd.material.Material
        either the formula or a hexrd material instance
    diameter : float
        pinhole diameter in microns
    thickness : float
        pinhole thickness in microns

    """

    def __init__(self,
                 material=None,
                 density=None,
                 thickness=None,
                 readout_length=None,
                 pre_U0=None):
        self._material = material
        self._density = density
        self._thickness = thickness

    @property
    def material(self):
        return self._material

    @property
    def density(self):
        if self._density is None:
            return 0.0
        return self._density

    @property
    def thickness(self):
        if self._thickness is None:
            return 0.0
        return self._thickness

class Pinhole(object):
    """simple class to encode all pinhole
    related parameters

    Parameters
    ----------
    material : str or hexrd.material.Material
        either the formula or a hexrd material instance
    diameter : float
        pinhole diameter in microns
    thickness : float
        pinhole thickness in microns

    Notes
    -----
    [1] Rygg et al., X-ray diffraction at the National 
        Ignition Facility, Rev. Sci. Instrum. 91, 043902 (2020)
    [2] M. Stoeckl, A. A. Solodov
        Readout models for BaFBr0.85I0.15:Eu image plates 
        Rev. Sci. Instrum. 89, 063101 (2018
    """
    def __init__(self,
                 material='Ta',
                 diameter=400,
                 thickness=100):
        self._material = material
        self._diameter = diameter
        self._thickness = thickness

    @property
    def radius(self):
        if self.diameter is None:
            return 0.0
        return 0.5*self.diameter

    @property
    def material(self):
        return self._material

    @property
    def diameter(self):
        if self._diameter is None:
            return 0.0
        return self._diameter

    @property
    def thickness(self):
        if self._thickness is None:
            return 0.0
        return self._thickness

class Filter(abstractlayer):

    def __init__(self, **abstractlayer_kwargs):
        super().__init__(**abstractlayer_kwargs)

class Coating(abstractlayer):

    def __init__(self, **abstractlayer_kwargs):
        super().__init__(**abstractlayer_kwargs)

class Phosphor(abstractlayer):

    def __init__(self, **abstractlayer_kwargs):
        super().__init__(**abstractlayer_kwargs)
        self._readout_length = abstractlayer_kwargs['readout_length']
        self._pre_U0 = abstractlayer_kwargs['pre_U0']

    @property
    def readout_length(self):
        if self._readout_length is None:
            return 0.0
        return self._readout_length

    @property
    def pre_U0(self):
        if self._pre_U0 is None:
            return 0.0
        return self._pre_U0

# class Scintillator(object):
#     """simple class to encode all information related
#     to the detector. we need maximum flexibility here
#     since the detector could be an image plate (omega/NIF),
#     a scintillator/CCD based (DCS), hybrid photon counting
#     detectors (SLAC) etc.

#      Parameters
#     ----------
#     material : str or hexrd.material.Material
#         either the formula or a hexrd material instance
#     diameter : float
#         pinhole diameter in microns
#     thickness : float
#         pinhole thickness in microns

#     """


# default filter and coating materials
FILTER_DEFAULT = {
    'material': 'Ge',
    'density' : density['Ge'],
    'thickness' : 10, # microns
}

COATING_DEFAULT = {
    'material': 'C10H8O4',
    'density' : density_compounds['C10H8O4'],
    'thickness' : 9, # microns
}

PHOSPHOR_DEFAULT = {
    'material' : 'Ba2263F2263Br1923I339C741H1730N247O494',
    'density' : density_compounds['Ba2263F2263Br1923I339C741H1730N247O494'], # g/cc
    'thickness' : 115, # microns
    'readout_length' : 222, #microns,
    'pre_U0' : 0.695
}

"""default physics package for dynamic compression
experiments the template of the other type is commented"""
PHYSICS_PACKAGE_DEFAULT = {
    'type' : 'HED',
    'sample_material' : 'Fe',
    'sample_density' : density['Fe'],
    'sample_thickness' : 15,# in microns
    'window_material' : 'LiF',
    'window_density' : density_compounds['LiF'],
    'window_thickness' : 150, # in microns
}

"""template for HEDM type physics package
"""
# PHYSICS_PACKAGE_DEFAULT = {
#     'type' : 'HEDM',
#     'shape' : 'cylinder', # cuboid
#     'dimension' : 1.0, # radius (mm) for cylinder, width x thickness for cuboid
#     'material' : 'Ti',
#     'density' : density['Ti'],
# }