import numpy as np
from hexrd.material.utils import (
    calculate_energy_absorption_length,
    calculate_linear_absorption_length,
)


class AbstractLayer:
    """abstract class for encode information
    for an arbitrary planar layer of given
    thickness, density, and material

    Parameters
    ----------
    material : str
        either the formula or a material name
    density : float
        density of element in g/cc
    thickness : float
        thickness in microns
    readout_length : float
        the distance of phosphor screen that encodes
        the information from x-rays
    pre_U0 : float
        scale factor for phosphor screen to convert
        intensity to PSL
    formula: str or None
        The chemical formula to use for absorption length calculations.
        If one is not provided, the chemical formula is assumed to be the
        same as the material name.
    """

    def __init__(self,
                 material: str | None = None,
                 density: float | None = None,
                 thickness: float | None = None,
                 readout_length: float | None = None,
                 pre_U0: float | None = None,
                 formula: str | None = None,
    ):
        self._material = material
        self._density = density
        self._thickness = thickness
        self._formula = formula

    @property
    def attributes_to_serialize(self):
        return [
            'material',
            'density',
            'thickness',
            'formula',
        ]

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self._material = material

    @property
    def density(self):
        if self._density is None:
            return 0.0
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    @property
    def thickness(self):
        if self._thickness is None:
            return 0.0
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value

    @property
    def formula(self) -> str:
        return self._formula

    @formula.setter
    def formula(self, value: str):
        self._formula = value

    def absorption_length(self, energy):
        if isinstance(energy, float):
            energy_inp = np.array([energy])
        elif isinstance(energy, list):
            energy_inp = np.array(energy)
        elif isinstance(energy, np.ndarray):
            energy_inp = energy

        # Use the chemical formula if provided. Otherwise, assume the material
        # name is the chemical formula.
        formula = self.formula if self.formula else self.material
        args = (
            self.density,
            formula,
            energy_inp,
        )
        abs_length = calculate_linear_absorption_length(*args)
        if abs_length.shape[0] == 1:
            return abs_length[0]
        else:
            return abs_length

    def energy_absorption_length(self, energy):
        if isinstance(energy, float):
            energy_inp = np.array([energy])
        elif isinstance(energy, list):
            energy_inp = np.array(energy)
        elif isinstance(energy, np.ndarray):
            energy_inp = energy

        # Use the chemical formula if provided. Otherwise, assume the material
        # name is the chemical formula.
        formula = self.formula if self.formula else self.material
        args = (
            self.density,
            formula,
            energy_inp,
        )
        abs_length = calculate_energy_absorption_length(*args)
        if abs_length.shape[0] == 1:
            return abs_length[0]
        else:
            return abs_length

    def serialize(self):
        return {a: getattr(self, a) for a in self.attributes_to_serialize}

    def deserialize(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Filter(AbstractLayer):

    def __init__(self, **abstractlayer_kwargs):
        super().__init__(**abstractlayer_kwargs)


class Coating(AbstractLayer):

    def __init__(self, **abstractlayer_kwargs):
        super().__init__(**abstractlayer_kwargs)


class Phosphor(AbstractLayer):

    def __init__(self, **abstractlayer_kwargs):
        super().__init__(**abstractlayer_kwargs)
        self._readout_length = abstractlayer_kwargs['readout_length']
        self._pre_U0 = abstractlayer_kwargs['pre_U0']

    @property
    def attributes_to_serialize(self):
        return super().attributes_to_serialize + [
            'readout_length',
            'pre_U0',
        ]

    @property
    def readout_length(self):
        if self._readout_length is None:
            return 0.0
        return self._readout_length

    @readout_length.setter
    def readout_length(self, value):
        self._readout_length = value

    @property
    def pre_U0(self):
        if self._pre_U0 is None:
            return 0.0
        return self._pre_U0

    @pre_U0.setter
    def pre_U0(self, value):
        self._pre_U0 = value
