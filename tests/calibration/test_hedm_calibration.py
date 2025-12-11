import h5py
import numpy as np
import yaml

import pytest

import hexrd.core.constants as cnst
from hexrd.material.material import load_materials_hdf5
from hexrd.instrument.hedm_instrument import HEDMInstrument

from hexrd.fitting.calibration import (
    fix_detector_y,
    GrainCalibrator,
    InstrumentCalibrator,
)


@pytest.fixture
def calibration_dir(example_repo_path):
    return example_repo_path / 'dexelas' / 'ruby'


@pytest.fixture
def dexelas_instrument(calibration_dir):
    # Load the instrument
    with open(calibration_dir / 'dual_dexelas_composite.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    return HEDMInstrument(conf)


@pytest.fixture
def ruby_material(calibration_dir):
    # Load the material
    with h5py.File(calibration_dir / 'ruby.h5', 'r') as rf:
        return load_materials_hdf5(rf)['ruby']


@pytest.fixture
def pull_spots_picks(calibration_dir):
    path = calibration_dir
    picks = []
    for i in range(3):
        picks.append(
            {
                'pick_xys': np.load(path / f'grain{i + 1}_picks.npz'),
                'hkls': np.load(path / f'grain{i + 1}_pick_hkls.npz'),
            }
        )
    return picks


@pytest.fixture
def grain_params(calibration_dir):
    return np.load(calibration_dir / 'grain_params.npy')


def test_calibration(
    dexelas_instrument, ruby_material, pull_spots_picks, grain_params
):

    instr = dexelas_instrument

    ome_period = np.array([0, 2 * np.pi])
    euler_convention = {'axes_order': 'xyz', 'extrinsic': True}

    # Create the calibrators
    calibrators = []
    for i, picks in enumerate(pull_spots_picks):
        kwargs = {
            'instr': instr,
            'material': ruby_material,
            'grain_params': grain_params[i],
            'ome_period': ome_period,
            'index': i,
            'euler_convention': euler_convention,
        }
        calibrator = GrainCalibrator(**kwargs)
        calibrator.data_dict = picks
        calibrators.append(calibrator)

    ic = InstrumentCalibrator(
        *calibrators,
        euler_convention=euler_convention,
    )
    params_dict = ic.params

    # Try out a few of the convenience functions on the first calibrator
    calibrator = calibrators[0]
    calibrator.fix_strain_to_identity(params_dict)

    identity = cnst.identity_6x1
    for i, name in enumerate(calibrator.strain_param_names):
        param = params_dict[name]
        assert np.isclose(param.value, identity[i])
        assert param.vary is False

    calibrator.fix_y_to_zero(params_dict)
    for i, name in enumerate(calibrator.translation_param_names):
        param = params_dict[name]
        if i == 1:
            assert np.isclose(param.value, 0)
            assert param.vary is False
        else:
            assert param.vary is True

        param.vary = False

    origin = cnst.zeros_3
    calibrator.fix_translation_to_origin(params_dict)
    for i, name in enumerate(calibrator.translation_param_names):
        param = params_dict[name]
        assert np.isclose(param.value, origin[0])
        assert param.vary is False

    # Now verify that fixing detector y works
    det_y_names = []
    for det_key in instr.detectors:
        name = det_key.replace('-', '_')
        det_y_names.append(f'{name}_tvec_y')

    for name in det_y_names:
        param = params_dict[name]
        assert param.vary is True

    fix_detector_y(instr, params_dict)
    for name in det_y_names:
        param = params_dict[name]
        assert param.vary is False

        # Set it back for the actual calibration
        param.vary = True

    # Get the initial values
    x0 = ic.params.valuesdict()

    # Now run the calibration
    odict = {}
    results = ic.run_calibration(odict=odict)

    # Verify that at least one value changed
    x1 = results.params.valuesdict()
    check_key = 'ruby_0_grain_param_0'
    assert not np.isclose(x0[check_key], x1[check_key])

    # Just verify it was successful, and that the cost and optimality are low
    assert results.success
    assert results.cost < 10
    assert results.optimality < 1e-3
