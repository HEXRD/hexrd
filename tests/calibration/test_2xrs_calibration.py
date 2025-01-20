import h5py
import numpy as np
import yaml

import pytest

from hexrd.core.material.material import load_materials_hdf5
from hexrd.hedm.instrument.hedm_instrument import HEDMInstrument

from hexrd.core.fitting.calibration import InstrumentCalibrator, PowderCalibrator


@pytest.fixture
def tardis_2xrs_examples_dir(example_repo_path):
    return example_repo_path / 'tardis' / '2xrs'


def test_2xrs_calibration(tardis_2xrs_examples_dir, test_data_dir):
    # This test is currently only verifying that 2XRS calibration
    # runs without errors

    # Load the materials
    with h5py.File(tardis_2xrs_examples_dir / 'platinum.h5', 'r') as rf:
        materials = load_materials_hdf5(rf)

    # Load the picks
    with open(
        tardis_2xrs_examples_dir / 'tardis_2xrs_example.yml', 'r',
        encoding='utf-8',
    ) as rf:
        conf = yaml.safe_load(rf)

    instrument = HEDMInstrument(conf)

    # Use dummy images
    img_dict = {k: {} for k in instrument.detectors}

    # Load picks
    picks = np.load(tardis_2xrs_examples_dir / 'picks.npy', allow_pickle=True)

    euler_convention = {'axes_order': 'zxz', 'extrinsic': False}

    # Create the calibrators
    calibrators = []
    for pick_data in picks:
        kwargs = {
            'instr': instrument,
            'material': materials[pick_data['material']],
            'img_dict': img_dict,
            'default_refinements': pick_data['default_refinements'],
            'tth_distortion': pick_data['tth_distortion'],
            'calibration_picks': pick_data['picks'],
            'xray_source': pick_data['xray_source'],
        }
        calibrators.append(PowderCalibrator(**kwargs))

    calibrator = InstrumentCalibrator(
        *calibrators,
        engineering_constraints='TARDIS',
        euler_convention=euler_convention,
    )

    calibrator.run_calibration({})

    # At some point, we might want to do some test to verify that this
    # stepped in the correct direction. But for now, just running it
    # without any errors is a good step.
