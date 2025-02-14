from hexrd.core.constants import DENSITY, DENSITY_COMPOUNDS


# default filter and coating materials
class FILTER_DEFAULTS:
    TARDIS = {
        'material': 'Ge',
        'density': DENSITY['Ge'],
        'thickness': 10,  # microns
    }
    PXRDIP = {
        'material': 'Cu',
        'density': DENSITY['Cu'],
        'thickness': 10,  # microns
    }


COATING_DEFAULT = {
    'material': 'C10H8O4',
    'density': DENSITY_COMPOUNDS['C10H8O4'],
    'thickness': 9,  # microns
}

PHOSPHOR_DEFAULT = {
    'material': 'Ba2263F2263Br1923I339C741H1730N247O494',
    'density': DENSITY_COMPOUNDS[
        'Ba2263F2263Br1923I339C741H1730N247O494'
    ],  # g/cc
    'thickness': 115,  # microns
    'readout_length': 222,  # microns
    'pre_U0': 0.695,
}


class PHYSICS_PACKAGE_DEFAULTS:
    # Default physics package for dynamic compression
    HED = {
        'sample_material': 'Fe',
        'sample_density': DENSITY['Fe'],
        'sample_thickness': 15,  # in microns
        'window_material': 'LiF',
        'window_density': DENSITY_COMPOUNDS['LiF'],
        'window_thickness': 150,  # in microns
    }
    # # Template for HEDM type physics package
    # HEDM = {
    #     'sample_material': 'Fe',
    #     'sample_density': DENSITY['Fe'],
    #     'sample_thickness': 1000, # in microns
    #     'sample_geometry': 'cylinder'
    # }


# Default pinhole area correction parameters
class PINHOLE_DEFAULTS:
    TARDIS = {
        'pinhole_material': 'Ta',
        'pinhole_diameter': 400,  # in microns
        'pinhole_thickness': 100,  # in microns
        'pinhole_density': 16.65,  # g/cc
    }
    PXRDIP = {
        'pinhole_material': 'Ta',
        'pinhole_diameter': 130,  # in microns
        'pinhole_thickness': 70,  # in microns
        'pinhole_density': 16.65,  # g/cc
    }
