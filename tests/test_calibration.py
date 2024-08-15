import h5py
import numpy as np
import yaml

import pytest

from hexrd.material.material import load_materials_hdf5
from hexrd.instrument.hedm_instrument import HEDMInstrument
from hexrd import rotations as rot

from hexrd.fitting.calibration import (
    InstrumentCalibrator,
    LaueCalibrator,
    PowderCalibrator,
)


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
        engineering_constraints='TARDIS',
        euler_convention=euler_convention,
    )

    x0 = calibrator.params.valuesdict()
    result = calibrator.run_calibration({'max_nfev': 1300})
    x1 = result.params.valuesdict()

    # Parse the data
    tilt_angle_names = [
        [
            f'IMAGE_PLATE_{n}_euler_z',
            f'IMAGE_PLATE_{n}_euler_xp',
            f'IMAGE_PLATE_{n}_euler_zpp',
        ]
        for n in [2, 4]
    ]

    rmats = {
        'old': [
            euler_to_rot([x0[k] for k in names]) for names in tilt_angle_names
        ],
        'new': [
            euler_to_rot([x1[k] for k in names]) for names in tilt_angle_names
        ],
    }

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

    # expected_obj = {
    #     'rmat_2': rmats['new'][0],
    #     'rmat_4': rmats['new'][1],
    #     'tvec_2': tvecs['new'][0],
    #     'tvec_4': tvecs['new'][1],
    #     'grain_params': grain_params['new'],
    #     'diamond_a': diamond_a_vals['new'],
    # }

    # np.save(test_data_dir / 'calibration_expected.npy', expected_obj)

    expected = np.load(
        test_data_dir / 'calibration_expected.npy', allow_pickle=True
    )

    assert_errors_are_better(
        rmats, tvecs, grain_params, diamond_a_vals, expected.item()
    )


def euler_to_rot(euler):
    return rot.RotMatEuler(np.array(euler), 'zxz', False, 'degrees').rmat


def assert_errors_are_better(
    rmats, tvecs, grain_params, diamond_a_vals, expected
):
    """
    Make sure error has decreased during fitting
    """
    # What fraction of the old error we need to have (at worst) for the
    # test to pass
    ERROR_TOLERANCE = 0.8

    rmat_error_2_old = np.linalg.norm(
        rmats['old'][0] @ expected['rmat_2'].T - np.eye(3)
    )
    rmat_error_2_new = np.linalg.norm(
        rmats['new'][0] @ expected['rmat_2'].T - np.eye(3)
    )
    rmat_error_4_old = np.linalg.norm(
        rmats['old'][1] @ expected['rmat_4'].T - np.eye(3)
    )
    rmat_error_4_new = np.linalg.norm(
        rmats['new'][1] @ expected['rmat_4'].T - np.eye(3)
    )
    assert rmat_error_2_new < rmat_error_2_old * ERROR_TOLERANCE
    assert rmat_error_4_new < rmat_error_4_old * ERROR_TOLERANCE

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
    assert diamond_a_error_new < diamond_a_error_old * ERROR_TOLERANCE
