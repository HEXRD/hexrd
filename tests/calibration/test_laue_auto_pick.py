from pathlib import Path

import numpy as np
import yaml

import pytest

from hexrd.fitting.calibration import LaueCalibrator
from hexrd.material.material import load_materials_hdf5, Material
from hexrd.instrument.hedm_instrument import HEDMInstrument


@pytest.fixture
def simulated_tardis_path(example_repo_path: Path) -> Path:
    return example_repo_path / 'tardis' / 'simulated'


@pytest.fixture
def simulated_tardis_images(
    simulated_tardis_path: Path,
) -> dict[str, np.array]:
    path = simulated_tardis_path / 'tardis_images.npz'
    npz = np.load(path)
    return {k: v for k, v in npz.items()}


@pytest.fixture
def lif_grain_params(simulated_tardis_path: Path) -> np.array:
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
def expected_laue_auto_pick_results(test_data_dir: Path) -> dict[str, list]:
    path = test_data_dir / 'expected_laue_auto_pick_results.npy'
    return np.load(path, allow_pickle=True).item()


def test_autopick_laue_spots(
    tardis_instrument: HEDMInstrument,
    lif_material: Material,
    lif_grain_params: np.array,
    simulated_tardis_images: dict[str, np.array],
    expected_laue_auto_pick_results: dict[str, list],
):
    instr = tardis_instrument
    plane_data = lif_material.planeData
    img_dict = simulated_tardis_images

    # Disable all exclusions on the plane data
    plane_data.exclusions = None

    calibrator = LaueCalibrator(
        instr,
        lif_material,
        lif_grain_params,
        min_energy=5,
        max_energy=35,
    )

    picks = calibrator.autopick_points(
        img_dict,
        tth_tol=5,
        eta_tol=5,
        npdiv=2,
        do_smoothing=True,
        smoothing_sigma=2,
        use_blob_detection=True,
        blob_threshold=0.25,
        fit_peaks=True,
        min_peak_int=0.1,
        fit_tth_tol=0.1,
    )

    # Verify that we have the number of expected auto picks
    assert len(picks['hkls']['IMAGE-PLATE-2']) == 10
    assert len(picks['hkls']['IMAGE-PLATE-4']) == 14

    # Check two hkls from each detector
    comparisons = [
        {
            'det_key': 'IMAGE-PLATE-2',
            'hkl': [-2, 0, 0],
            'point': [52.1089, -152.008],
        },
        {
            'det_key': 'IMAGE-PLATE-2',
            'hkl': [1, -3, 3],
            'point': [42.1898, -34.326],
        },
        {
            'det_key': 'IMAGE-PLATE-4',
            'hkl': [0, 2, 0],
            'point': [23.5841, 112.204],
        },
        {
            'det_key': 'IMAGE-PLATE-4',
            'hkl': [-3, 3, 1],
            'point': [80.188, 154.282],
        },
    ]
    for entry in comparisons:
        det_key = entry['det_key']
        panel = instr.detectors[det_key]

        hkl_idx = None
        for i, row in enumerate(picks['hkls'][det_key]):
            if row == entry['hkl']:
                hkl_idx = i
                break

        assert hkl_idx is not None
        point_deg = np.degrees(
            panel.cart_to_angles(picks['pick_xys'][det_key][hkl_idx])[0][0]
        )
        assert np.allclose(entry['point'], point_deg)

    # Now just verify that everything matches the previous results
    for pick_key in expected_laue_auto_pick_results:
        for det_key in expected_laue_auto_pick_results[pick_key]:
            assert np.allclose(
                picks[pick_key][det_key],
                expected_laue_auto_pick_results[pick_key][det_key],
                equal_nan=True
            )
