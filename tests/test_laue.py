from pathlib import Path

import numpy as np
import yaml

import pytest

from hexrd.material.material import load_materials_hdf5, Material
from hexrd.instrument.hedm_instrument import HEDMInstrument
from common import compare_vector_set


@pytest.fixture
def simulated_tardis_path(example_repo_path: Path) -> Path:
    return example_repo_path / 'tardis' / 'simulated'


@pytest.fixture
def lif_grain_params(simulated_tardis_path: Path) -> np.ndarray:
    path = simulated_tardis_path / 'lif_grains_ideal.out'
    grain = np.loadtxt(path, ndmin=2)[0]
    return grain[3:15]


@pytest.fixture
def lif_material(simulated_tardis_path: Path) -> Material:
    path = simulated_tardis_path / 'materials.h5'
    return load_materials_hdf5(path)['LiF']


@pytest.fixture
def tardis_instrument(simulated_tardis_path: Path) -> HEDMInstrument:
    path = simulated_tardis_path / 'ideal_tardis.yml'
    with open(path, 'r') as rf:
        conf = yaml.safe_load(rf)

    return HEDMInstrument(conf)


@pytest.fixture
def expected_simulated_laue_results(test_data_dir: Path) -> dict[str, list]:
    path = test_data_dir / 'expected_simulated_laue_results.npy'
    return np.load(path, allow_pickle=True).item()


def test_simulate_laue_spots(
    tardis_instrument: HEDMInstrument,
    lif_material: Material,
    lif_grain_params: np.ndarray,
    expected_simulated_laue_results: dict[str, np.ndarray],
):
    instr = tardis_instrument
    plane_data = lif_material.planeData

    # Disable all exclusions on the plane data
    plane_data.exclusions = None

    sim_data = instr.simulate_laue_pattern(
        plane_data, minEnergy=5, maxEnergy=35, grain_params=[lif_grain_params]
    )

    # A few manual expected results
    expected = {
        'IMAGE-PLATE-2': [
            {
                'hkl': [-2, -2, 0],
                'energy': 25.89604972,
                'xy_det': [14.22357913, 8.41677193],
            },
            {
                'hkl': [-4, -2, 2],
                'energy': 11.764688099021837,
                'xy_det': [-3.07140086, -20.34804044],
            },
        ],
        'IMAGE-PLATE-4': [
            {
                'hkl': [2, 2, 2],
                'energy': 14.229829125689493,
                'xy_det': [-14.5468251, -15.65452491],
            },
            {
                'hkl': [1, 5, 1],
                'energy': 28.08132246902937,
                'xy_det': [-7.89464842, -8.0019517],
            },
        ],
    }

    for det_key, psim in sim_data.items():
        xy_det, hkls, angles, dspacing, energy = psim

        # Verify a few manually
        for entry in expected[det_key]:
            # Find the hkl that matches
            hkl_idx = None
            for i, row in enumerate(hkls[0].T.tolist()):
                if row == entry['hkl']:
                    hkl_idx = i
                    break

            assert hkl_idx is not None
            assert np.allclose(entry['energy'], energy[0][hkl_idx])
            assert np.allclose(entry['xy_det'], xy_det[0][hkl_idx])

        # Now verify that nothing changed
        ref = expected_simulated_laue_results[det_key]
        (
            result_xy_det,
            result_hkls,
            result_angles,
            result_dspacing,
            result_energy,
        ) = ref

        # Stack all of the coupled data into the same vector before comparing.

        stacked = np.concatenate(
            [
                xy_det,
                hkls.transpose(0, 2, 1),
                angles,
                dspacing[..., None],
                energy[..., None],
            ],
            axis=2,
        )

        results_stacked = np.concatenate(
            [
                result_xy_det,
                result_hkls.transpose(0, 2, 1),
                result_angles,
                result_dspacing[..., None],
                result_energy[..., None],
            ],
            axis=2,
        )

        assert compare_vector_set(
            stacked.transpose(1, 0, 2), results_stacked.transpose(1, 0, 2)
        )
