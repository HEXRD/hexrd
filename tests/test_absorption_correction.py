import numpy as np
import pytest
import yaml

from hexrd.hedm.instrument.hedm_instrument import HEDMInstrument
from hexrd.hedm.instrument.physics_package import HEDPhysicsPackage


@pytest.fixture
def simulated_tardis_dir(example_repo_path):
    return example_repo_path / 'tardis' / 'simulated'


# These are just all current default settings for TARDIS.
PHYSICS_PACKAGE_SETTINGS = {
    'sample_material': 'Fe',
    'sample_density': 7.874,
    'sample_thickness': 15,
    'pinhole_material': 'Ta',
    'pinhole_density': 16.65,
    'pinhole_thickness': 100,
    'pinhole_diameter': 400,
    'window_material': 'LiF',
    'window_density': 2.64,
    'window_thickness': 150,
}

COATING_SETTINGS = {
    'material': 'C10H8O4',
    'density': 1.4,
    'thickness': 9.0,
}

FILTER_SETTINGS = {
    'material': 'Ge',
    'density': 5.323,
    'thickness': 10.0,
}

PHOSPHOR_SETTINGS = {
    'material': 'Ba2263F2263Br1923I339C741H1730N247O494',
    'density': 3.3,
    'thickness': 115.0,
    'readout_length': 222,
    'pre_U0': 0.695,
}


def test_absorption_correction(simulated_tardis_dir, test_data_dir):
    # Load the picks
    with open(simulated_tardis_dir / 'uncalibrated_tardis.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    instr = HEDMInstrument(conf)

    physics_package = HEDPhysicsPackage()
    physics_package.deserialize(**PHYSICS_PACKAGE_SETTINGS)
    instr.physics_package = physics_package

    for panel in instr.detectors.values():
        panel.coating.deserialize(**COATING_SETTINGS)
        panel.filter.deserialize(**FILTER_SETTINGS)
        panel.phosphor.deserialize(**PHOSPHOR_SETTINGS)

    # Now compute the transmissions
    transmissions = instr.calc_transmission()

    # Normalize so that the max transmission across all detectors is 1
    max_transmission = max(
        [np.nanmax(v) for v in transmissions.values()])
    transmissions = {k: v / max_transmission for k, v in transmissions.items()}

    # Now compare to our reference
    expected_transmissions = np.load(
        test_data_dir / 'ideal_tardis_transmissions.npy', allow_pickle=True
    ).item()
    for det_key, transmission_array in transmissions.items():
        expected = expected_transmissions[det_key]
        assert np.allclose(transmission_array, expected, equal_nan=True)
