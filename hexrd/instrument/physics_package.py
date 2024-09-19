from abc import abstractmethod
import numpy as np
from hexrd.material.utils import calculate_linear_absorption_length


class AbstractPP:
    """abstract class for the physics package.
    there will be two separate physics package class
    types -- one for HED samples and the other for
    HEDM samples. 

    Parameters
    ----------
    sample_material : str or hexrd.material.Material
        either the formula or a hexrd material instance
    sample_density : float
        density of sample material in g/cc
    sample_thickness : float
        sample thickness in microns
    sample_geometry : FIXME
        FIXME
    pinhole_material : str or hexrd.material.Material, optional
        either the formula or a hexrd material instance
    pinhole_density : float
        density of pinhole material in g/cc
    pinhole_thickness : float
        pinhole thickness in microns
    pinhole_diameter : float
        pinhole diameter in microns
    window_material : str or hexrd.material.Material
        either the formula or a hexrd material instance
    window_density : float
        density of window material in g/cc
    window_thickness : float
        window thickness in microns


    Notes
    -----
    [1] Rygg et al., X-ray diffraction at the National
        Ignition Facility, Rev. Sci. Instrum. 91, 043902 (2020)
    [2] M. Stoeckl, A. A. Solodov
        Readout models for BaFBr0.85I0.15:Eu image plates
        Rev. Sci. Instrum. 89, 063101 (2018
    """
    # Abstract methods that must be redefined in derived classes
    @property
    @abstractmethod
    def type(self):
        pass

    def __init__(self,
                 sample_material=None,
                 sample_density=None,
                 sample_thickness=None,
                 sample_geometry=None,
                 pinhole_material=None,
                 pinhole_density=None,
                 pinhole_thickness=None,
                 pinhole_diameter=None,
                 window_material=None,
                 window_density=None,
                 window_thickness=None,
                 ):
        self._sample_material = sample_material
        self._sample_density = sample_density
        self._sample_thickness = sample_thickness
        self._pinhole_material = pinhole_material
        self._pinhole_density = pinhole_density
        self._pinhole_thickness = pinhole_thickness
        self._pinhole_diameter = pinhole_diameter

    @property
    def attributes_to_serialize(self):
        return [
            'sample_material',
            'sample_density',
            'sample_thickness',
            'pinhole_material',
            'pinhole_density',
            'pinhole_thickness',
            'pinhole_diameter',
        ]

    @property
    def sample_material(self):
        return self._sample_material

    @sample_material.setter
    def sample_material(self, material):
        self._sample_material = material

    @property
    def sample_density(self):
        if self._sample_density is None:
            return 0.0
        return self._sample_density

    @sample_density.setter
    def sample_density(self, density):
        self._sample_density = density

    @property
    def sample_thickness(self):
        if self._sample_thickness is None:
            return 0.0
        return self._sample_thickness

    @sample_thickness.setter
    def sample_thickness(self, value):
        self._sample_thickness = value

    @property
    def pinhole_material(self):
        return self._pinhole_material

    @pinhole_material.setter
    def pinhole_material(self, material):
        self._pinhole_material = material

    @property
    def pinhole_density(self):
        if self._pinhole_density is None:
            return 0.0
        return self._pinhole_density

    @pinhole_density.setter
    def pinhole_density(self, density):
        self._pinhole_density = density

    @property
    def pinhole_thickness(self):
        if self._pinhole_thickness is None:
            return 0.0
        return self._pinhole_thickness

    @pinhole_thickness.setter
    def pinhole_thickness(self, value):
        self._pinhole_thickness = value

    @property
    def pinhole_radius(self):
        if self.pinhole_diameter is None:
            return 0.0
        return 0.5 * self.pinhole_diameter

    @pinhole_radius.setter
    def pinhole_radius(self, value):
        self._pinhole_diameter = 2.0 * value

    @property
    def pinhole_diameter(self):
        if self._pinhole_diameter is None:
            return 0.0
        return self._pinhole_diameter

    @pinhole_diameter.setter
    def pinhole_diameter(self, value):
        self._pinhole_diameter = value

    def absorption_length(self, energy, flag):
        if isinstance(energy, float):
            energy_inp = np.array([energy])
        elif isinstance(energy, list):
            energy_inp = np.array(energy)
        elif isinstance(energy, np.ndarray):
            energy_inp = energy

        if flag.lower() == 'sample':
            args = (self.sample_density,
                    self.sample_material,
                    energy_inp,
                    )
        elif flag.lower() == 'window':
            args = (self.window_density,
                    self.window_material,
                    energy_inp,
                    )
        elif flag.lower() == 'pinhole':
            args = (self.pinhole_density,
                    self.pinhole_material,
                    energy_inp,
                    )
        abs_length = calculate_linear_absorption_length(*args)
        if abs_length.shape[0] == 1:
            return abs_length[0]
        else:
            return abs_length

    def sample_absorption_length(self, energy):
        return self.absorption_length(energy, 'sample')

    def pinhole_absorption_length(self, energy):
        return self.absorption_length(energy, 'pinhole')

    def serialize(self):
        return {a: getattr(self, a) for a in self.attributes_to_serialize}

    def deserialize(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class HEDPhysicsPackage(AbstractPP):

    def __init__(self, **pp_kwargs):
        super().__init__(**pp_kwargs)
        self._window_material = pp_kwargs['window_material']
        self._window_density = pp_kwargs['window_density']
        self._window_thickness = pp_kwargs['window_thickness']

    @property
    def attributes_to_serialize(self):
        return [
            'sample_material',
            'sample_density',
            'sample_thickness',
            'pinhole_material',
            'pinhole_density',
            'pinhole_thickness',
            'pinhole_diameter',
            'window_material',
            'window_density',
            'window_thickness',
        ]

    @property
    def type(self):
        return 'HED'

    @property
    def window_material(self):
        return self._window_material

    @window_material.setter
    def window_material(self, material):
        self._window_material = material

    @property
    def window_density(self):
        if self._window_density is None:
            return 0.0
        return self._window_density

    @window_density.setter
    def window_density(self, density):
        self._window_density = density

    @property
    def window_thickness(self):
        if self._window_thickness is None:
            return 0.0
        return self._window_thickness

    @window_thickness.setter
    def window_thickness(self, thickness):
        self._window_thickness = thickness

    def window_absorption_length(self, energy):
        return self.absorption_length(energy, 'window')


class HEDMPhysicsPackage(AbstractPP):

    def __init__(self, **pp_kwargs):
        super().__init__(**pp_kwargs)
        self._sample_geometry = pp_kwargs['sample_geometry']

    @property
    def attributes_to_serialize(self):
        return [
            'sample_material',
            'sample_density',
            'sample_thickness',
            'sample_geometry',
            'pinhole_material',
            'pinhole_density',
            'pinhole_thickness',
            'pinhole_diameter',
        ]

    @property
    def sample_geometry(self):
        return self._sample_geometry

    @property
    def sample_diameter(self):
        if self.sample_geometry == 'cylinder':
            return self._sample_thickness
        else:
            msg = (f'sample geometry does not have diameter '
                   f'associated with it.')
            print(msg)
            return 

    @property
    def type(self):
        return 'HEDM'
