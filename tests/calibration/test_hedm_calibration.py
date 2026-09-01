import copy

import h5py
import numpy as np
import yaml

import pytest

import hexrd.core.constants as cnst
from hexrd.core.material.material import load_materials_hdf5
from hexrd.core.instrument.hedm_instrument import HEDMInstrument

from hexrd.core.fitting.calibration import (
    fix_detector_y,
    InstrumentCalibrator,
    MultiInstrumentCalibrator,
)
from hexrd.core.fitting.calibration.relative_constraints import (
    RelativeConstraintsType,
)
from hexrd.hedm.fitting.calibration import GrainCalibrator

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


def test_multi_instrument_calibration(
    dexelas_instrument, ruby_material, pull_spots_picks, grain_params
):
    # Build two "scans" out of the ruby dataset, each with its own instrument
    # copy and its own grains, and calibrate them together with a shared
    # detector geometry but independent oscillation stages.
    ome_period = np.array([0, 2 * np.pi])
    euler_convention = {'axes_order': 'xyz', 'extrinsic': True}

    instr_a = copy.deepcopy(dexelas_instrument)
    instr_b = copy.deepcopy(dexelas_instrument)

    # Give each scan a distinct rotation axis to exercise the per-scan chi.
    instr_a.chi = np.radians(0.1)
    instr_b.chi = np.radians(-0.2)

    # Perturb scan B's detector geometry. Because geometry is shared (created
    # only from the first scan), this should be synced back to scan A's
    # geometry when the multi-calibrator is constructed.
    for det in instr_b.detectors.values():
        det.tvec = det.tvec + np.array([0.5, -0.3, 1.0])
        det.tilt = det.tilt + np.array([0.01, 0.0, 0.0])

    def make_ic(instr, grain_indices):
        calibs = []
        for i in grain_indices:
            gc = GrainCalibrator(
                instr=instr,
                material=ruby_material,
                grain_params=grain_params[i],
                ome_period=ome_period,
                index=i,
                euler_convention=euler_convention,
            )
            gc.data_dict = pull_spots_picks[i]
            calibs.append(gc)
        return InstrumentCalibrator(*calibs, euler_convention=euler_convention)

    ic_a = make_ic(instr_a, [0])
    ic_b = make_ic(instr_b, [1, 2])

    multi = MultiInstrumentCalibrator([ic_a, ic_b], labels=['scanA', 'scanB'])
    params = multi.params

    # Stage params are per-scan prefixed; there is no unprefixed stage param.
    assert 'scanA_instr_chi' in params
    assert 'scanB_instr_chi' in params
    assert 'scanA_instr_tvec_z' in params
    assert 'scanB_instr_tvec_z' in params
    assert 'instr_chi' not in params

    # Detector geometry params exist once (unprefixed, shared).
    for det_key in instr_a.detectors:
        name = det_key.replace('-', '_')
        assert f'{name}_tvec_x' in params
        assert f'scanA_{name}_tvec_x' not in params
        assert f'scanB_{name}_tvec_x' not in params

    # Grain params are independent per scan (one grain in A, two in B).
    assert 'ruby_0_grain_param_0' in params
    assert 'ruby_1_grain_param_0' in params
    assert 'ruby_2_grain_param_0' in params

    # On construction, scan B's perturbed geometry is synced to scan A's.
    for da, db in zip(instr_a.detectors.values(), instr_b.detectors.values()):
        np.testing.assert_allclose(da.tvec, db.tvec)
        np.testing.assert_allclose(da.rmat, db.rmat)

    # ... but the per-scan rotation axes stay independent.
    assert not np.isclose(instr_a.chi, instr_b.chi)

    # Refine each scan's rotation axis independently (the per-scan stage is the
    # headline feature). These are fixed by default, like the single-instrument
    # calibrator.
    multi.params['scanA_instr_chi'].vary = True
    multi.params['scanB_instr_chi'].vary = True

    x0 = multi.params.valuesdict()
    resid0 = np.linalg.norm(multi.residual())
    results = multi.run_calibration()
    resid1 = np.linalg.norm(multi.residual())
    x1 = results.params.valuesdict()

    # The combined residual decreased.
    assert resid1 < resid0

    # The geometry stayed shared through the whole fit: both instruments are
    # still identical afterward.
    for da, db in zip(instr_a.detectors.values(), instr_b.detectors.values()):
        np.testing.assert_allclose(da.tvec, db.tvec)
        np.testing.assert_allclose(da.rmat, db.rmat)

    # The per-scan rotation axes were refined independently.
    assert not np.isclose(x0['scanA_instr_chi'], x1['scanA_instr_chi'])
    assert not np.isclose(x0['scanB_instr_chi'], x1['scanB_instr_chi'])

    # At least one grain parameter moved during the calibration.
    assert not np.isclose(x0['ruby_0_grain_param_0'], x1['ruby_0_grain_param_0'])


def _make_ruby_ic(
    instr, grain_indices, ruby_material, grain_params, pull_spots_picks,
    relative_constraints_type=RelativeConstraintsType.none,
):
    ome_period = np.array([0, 2 * np.pi])
    euler_convention = {'axes_order': 'xyz', 'extrinsic': True}
    calibs = []
    for i in grain_indices:
        gc = GrainCalibrator(
            instr=instr,
            material=ruby_material,
            grain_params=grain_params[i],
            ome_period=ome_period,
            index=i,
            euler_convention=euler_convention,
        )
        gc.data_dict = pull_spots_picks[i]
        calibs.append(gc)
    return InstrumentCalibrator(
        *calibs,
        euler_convention=euler_convention,
        relative_constraints_type=relative_constraints_type,
    )


def test_multi_instrument_mismatched_detectors(
    dexelas_instrument, ruby_material, pull_spots_picks, grain_params
):
    # Each calibrator's instrument must share the same detector layout, since
    # geometry params are only emitted from calibrators[0] and read back by key.
    instr_a = copy.deepcopy(dexelas_instrument)
    instr_b = copy.deepcopy(dexelas_instrument)

    # Give scan B an extra detector key that scan A does not have.
    existing_key = next(iter(instr_b.detectors))
    extra_panel = copy.deepcopy(instr_b.detectors[existing_key])
    instr_b._detectors['extra_panel'] = extra_panel

    ic_a = _make_ruby_ic(
        instr_a, [0], ruby_material, grain_params, pull_spots_picks
    )
    ic_b = _make_ruby_ic(
        instr_b, [1], ruby_material, grain_params, pull_spots_picks
    )

    with pytest.raises(ValueError, match='detector layout'):
        MultiInstrumentCalibrator([ic_a, ic_b])


def test_multi_instrument_mismatched_relative_constraints(
    dexelas_instrument, ruby_material, pull_spots_picks, grain_params
):
    # All calibrators must agree on relative_constraints, since only
    # calibrators[0]'s configuration actually takes effect.
    instr_a = copy.deepcopy(dexelas_instrument)
    instr_b = copy.deepcopy(dexelas_instrument)

    ic_a = _make_ruby_ic(
        instr_a, [0], ruby_material, grain_params, pull_spots_picks,
        relative_constraints_type=RelativeConstraintsType.none,
    )
    ic_b = _make_ruby_ic(
        instr_b, [1], ruby_material, grain_params, pull_spots_picks,
        relative_constraints_type=RelativeConstraintsType.system,
    )

    with pytest.raises(ValueError, match='relative_constraints'):
        MultiInstrumentCalibrator([ic_a, ic_b])
