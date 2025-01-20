import copy

import h5py
import numpy as np
import yaml

import pytest

from hexrd.core.material.material import load_materials_hdf5
from hexrd.hedm.instrument.hedm_instrument import HEDMInstrument

from hexrd.core.fitting.calibration import InstrumentCalibrator, LaueCalibrator, PowderCalibrator


@pytest.fixture
def calibration_dir(example_repo_path):
    return example_repo_path / 'tardis' / 'simulated'


def test_calibration(calibration_dir, test_data_dir):
    # Load the material
    with h5py.File(calibration_dir / 'materials.h5', 'r') as rf:
        materials = load_materials_hdf5(rf)

    # Load the picks
    with open(
        calibration_dir / 'uncalibrated_tardis.yml', 'r', encoding='utf-8'
    ) as rf:
        conf = yaml.safe_load(rf)

    instrument = HEDMInstrument(conf)

    # Load the images
    img_npz = np.load(calibration_dir / 'tardis_images.npz')
    img_dict = {k: img_npz[k] for k in img_npz}

    # Load picks
    picks = np.load(calibration_dir / 'picks.npy', allow_pickle=True)

    euler_convention = {'axes_order': 'zxz', 'extrinsic': False}

    # Create the calibrators
    calibrators = []
    for pick_data in picks:
        if pick_data['type'] == 'powder':
            kwargs = {
                'instr': instrument,
                'material': materials[pick_data['material']],
                'img_dict': img_dict,
                'default_refinements': pick_data['default_refinements'],
                'tth_distortion': pick_data['tth_distortion'],
                'calibration_picks': pick_data['picks'],
            }
            calibrators.append(PowderCalibrator(**kwargs))

        elif pick_data['type'] == 'laue':
            # gpflags = [i[1] for i in pick_data['refinements']]
            kwargs = {
                'instr': instrument,
                'material': materials[pick_data['material']],
                'grain_params': pick_data['options']['crystal_params'],
                'default_refinements': pick_data['default_refinements'],
                'min_energy': pick_data['options']['min_energy'],
                'max_energy': pick_data['options']['max_energy'],
                'calibration_picks': pick_data['picks'],
                'euler_convention': euler_convention,
            }
            calibrators.append(LaueCalibrator(**kwargs))

    calibrator = InstrumentCalibrator(
        *calibrators,
        # Engineering constraints were actually not being utilized before,
        # due to a bug. Disable them for now.
        # engineering_constraints='TARDIS',
        euler_convention=euler_convention,
    )

    tilt_angle_names = [
        [
            f'IMAGE_PLATE_{n}_euler_z',
            f'IMAGE_PLATE_{n}_euler_xp',
            f'IMAGE_PLATE_{n}_euler_zpp',
        ]
        for n in [2, 4]
    ]
    all_tilt_angle_names = tilt_angle_names[0] + tilt_angle_names[1]

    # The tilts are already ideal. Do not refine.
    orig_params = copy.deepcopy(calibrator.params)
    for name in all_tilt_angle_names:
        calibrator.params[name].vary = False

    # The lattice parameter is actually perfect. Adjust it a little
    # So that it can be corrected.
    calibrator.params['diamond_a'].value = 3.58

    x0 = calibrator.params.valuesdict()
    result = calibrator.run_calibration({})
    x1 = result.params.valuesdict()

    # Parse the data
    tvec_names = [
        [
            f'IMAGE_PLATE_{n}_tvec_x',
            f'IMAGE_PLATE_{n}_tvec_y',
            f'IMAGE_PLATE_{n}_tvec_z',
        ]
        for n in [2, 4]
    ]

    tvecs = {
        'old': [np.array([x0[k] for k in vec_names]) for vec_names in tvec_names],
        'new': [np.array([x1[k] for k in vec_names]) for vec_names in tvec_names],
    }

    grain_param_names = [f'LiF_grain_param_{n}' for n in range(12)]
    grain_params = {
        'old': np.array([x0[name] for name in grain_param_names]),
        'new': np.array([x1[name] for name in grain_param_names]),
    }

    diamond_a_vals = {
        'old': x0['diamond_a'],
        'new': x1['diamond_a'],
    }

    expected = np.load(
        test_data_dir / 'calibration_expected.npy', allow_pickle=True
    )

    assert_errors_are_better(
        tvecs, grain_params, diamond_a_vals, expected.item()
    )


def assert_errors_are_better(
    tvecs, grain_params, diamond_a_vals, expected
):
    """
    Make sure error has decreased during fitting
    """
    # What fraction of the old error we need to have (at worst) for the
    # test to pass. For now, just make sure the error decreased.
    ERROR_TOLERANCE = 1

    tvec_error_2_old = np.linalg.norm(tvecs['old'][0] - expected['tvec_2'])
    tvec_error_2_new = np.linalg.norm(tvecs['new'][0] - expected['tvec_2'])
    tvec_error_4_old = np.linalg.norm(tvecs['old'][1] - expected['tvec_4'])
    tvec_error_4_new = np.linalg.norm(tvecs['new'][1] - expected['tvec_4'])

    assert tvec_error_2_new < tvec_error_2_old * ERROR_TOLERANCE
    assert tvec_error_4_new < tvec_error_4_old * ERROR_TOLERANCE

    grain_param_error_old = np.linalg.norm(
        grain_params['old'] - expected['grain_params']
    )
    grain_param_error_new = np.linalg.norm(
        grain_params['new'] - expected['grain_params']
    )
    assert grain_param_error_new < grain_param_error_old * ERROR_TOLERANCE

    diamond_a_error_old = np.abs(diamond_a_vals['old'] - expected['diamond_a'])
    diamond_a_error_new = np.abs(diamond_a_vals['new'] - expected['diamond_a'])

    # The old diamond setting was actually perfect, but we let it refine
    # The new diamond error should be less than 2%
    assert diamond_a_error_new < diamond_a_error_old * ERROR_TOLERANCE
