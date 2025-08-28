from abc import abstractmethod
from dataclasses import dataclass, fields
from functools import partial

import numpy as np
from hexrd.core.material.utils import calculate_linear_absorption_length


# Below are the possible layers
@dataclass
class PhysicsPackageLayer:
    name: str = ''
    material: str = ''
    density: float = 0
    thickness: float = 0
    formula: str | None = None  # chemical formula


@dataclass
class PinholeLayer(PhysicsPackageLayer):
    diameter: float = 0


@dataclass
class HEDMSampleLayer(PhysicsPackageLayer):
    geometry: str = ''


class AbstractPhysicsPackage:
    """abstract class for the physics package.
    there will be two separate physics package class
    types -- one for HED samples and the other for
    HEDM samples.

    Parameters
    ----------
    The parameters are set up so that layer attributes can be accessed
    (via both setters and getters) with `<layer_name>_<attribute>`.

    For example, for sample thickness, you may specify `sample_thickness`.

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

    # If you want to add more layers, you just need to:
    #   1. Add the layer name to this list
    #   2. If the layer requires a special class (like `PinholeLayer`),
    #      specify that in `SPECIAL_LAYERS`
    LAYER_TYPES = []
    SPECIAL_LAYERS = {}

    def __init__(self, **kwargs):
        # The physics packages are set up so that you can access layer
        # attributes via `<layer>_<attr>`. For example, for the sample
        # thickness, you can do `self.sample_thickness`.
        self._setup_layers()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _setup_layers(self):
        super().__setattr__('_layers', {})
        for name in self.LAYER_TYPES:
            self._layers[name] = self.make_layer(name)

    def __getattr__(self, key):
        if not key.startswith('_') and key.count('_') == 1:
            name, attr = key.split('_')
            if name in self._layers:
                return getattr(self._layers[name], attr)

        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if key.count('_') == 1:
            name, attr = key.split('_')
            if name in self._layers:
                setattr(self._layers[name], attr, value)

        # Default behavior is standard
        super().__setattr__(key, value)

    @property
    def attributes_to_serialize(self):
        result = []
        for name, layer in self._layers.items():
            result += [f'{name}_{x.name}' for x in fields(layer)]

        return result

    def make_layer(self, name: str, **kwargs) -> PhysicsPackageLayer:
        cls = self.SPECIAL_LAYERS.get(name, PhysicsPackageLayer)
        return cls(name, **kwargs)

    def serialize(self):
        return {a: getattr(self, a) for a in self.attributes_to_serialize}

    def deserialize(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class HEDPhysicsPackage(AbstractPhysicsPackage):

    # If you want to add more layers, you just need to:
    #   1. Add the layer name to this list
    #   2. If the layer requires a special class (like `PinholeLayer`),
    #      specify that in `SPECIAL_LAYERS`
    # These layer types should be in order, for proper computation of
    # layer standoff.
    LAYER_TYPES = [
        'ablator',
        'heatshield',
        'pusher',
        'sample',
        'reflective',
        'window',
        'pinhole',
    ]
    SPECIAL_LAYERS = {
        'pinhole': PinholeLayer,
    }

    @property
    def type(self):
        return 'HED'

    @property
    def pinhole_radius(self):
        return 0.5 * self.pinhole_diameter

    @pinhole_radius.setter
    def pinhole_radius(self, value):
        self.pinhole_diameter = 2.0 * value

    def absorption_length(self, energy, layer):
        if isinstance(energy, float):
            energy_inp = np.array([energy])
        elif isinstance(energy, list):
            energy_inp = np.array(energy)
        elif isinstance(energy, np.ndarray):
            energy_inp = energy

        density = getattr(self, f'{layer}_density')
        formula = getattr(self, f'{layer}_formula')
        if not formula:
            # Assume the material name is the formula
            formula = getattr(self, f'{layer}_material')

        layer = layer.lower()
        args = (
            density,
            formula,
            energy_inp
        )
        abs_length = calculate_linear_absorption_length(*args)
        if abs_length.shape[0] == 1:
            return abs_length[0]
        else:
            return abs_length

    def layer_standoff(self, layer: str) -> float:
        # Compute layer standoff from the pinhole
        idx = self.LAYER_TYPES.index(layer)
        result = 0.
        for i in range(idx + 1, len(self.LAYER_TYPES) - 1):
            name = self.LAYER_TYPES[i]
            result += self._layers[name].thickness
        return result

    def layer_thickness(self, layer: str) -> float:
        return self._layers[layer].thickness

    def __getattr__(self, key: str):
        if key.endswith('_absorption_length'):
            # Make a function to get the absorption length of the layer
            # For example, you can get the sample absorption length like
            # this: `package.sample_absorption_length(energy)`
            name = key.split('_', 1)[0]
            f = partial(self.absorption_length, layer=name)
            return f

        return super().__getattr__(key)


class HEDMPhysicsPackage(AbstractPhysicsPackage):

    # If you want to add more layers, you just need to:
    #   1. Add the layer name to this list
    #   2. If the layer requires a special class (like `PinholeLayer`),
    #      specify that in `SPECIAL_LAYERS`
    LAYER_TYPES = [
        'sample',
    ]
    SPECIAL_LAYERS = {
        'sample': HEDMSampleLayer,
    }

    @property
    def type(self):
        return 'HEDM'

    @property
    def sample_diameter(self):
        if self.sample_geometry == 'cylinder':
            return self.sample_thickness

        raise Exception(
            'sample geometry does not have diameter associated with it.'
        )
