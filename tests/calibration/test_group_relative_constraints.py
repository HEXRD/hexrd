import copy
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from hexrd import imageseries
from hexrd.fitting.calibration import (
    InstrumentCalibrator,
    PowderCalibrator,
)
from hexrd.fitting.calibration.relative_constraints import (
    RelativeConstraintsType,
)
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument import HEDMInstrument
from hexrd.material import load_materials_hdf5, Material
from hexrd.rotations import rotMatOfExpMap
from hexrd.utils.hdf5 import unwrap_h5_to_dict


@pytest.fixture
def dexelas_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'dexelas'


@pytest.fixture
def ceria_examples_path(dexelas_examples_path: Path) -> Path:
    return dexelas_examples_path / 'ceria'


@pytest.fixture
def ceria_example_data(ceria_examples_path: Path) -> dict[str, np.ndarray]:
    data_filenames = {
        'ff1': 'ceria_ff1.h5',
        'ff2': 'ceria_ff2.h5',
    }

    datasets = {}
    for key, filename in data_filenames.items():
        path = ceria_examples_path / filename
        with h5py.File(path, 'r') as rf:
            datasets[key] = rf['data'][()]

    return datasets


@pytest.fixture
def dexelas_composite_instrument(ceria_examples_path: Path) -> HEDMInstrument:
    instr_path = (
        ceria_examples_path / 'dexelas.yml'
    )
    with open(instr_path, 'r') as rf:
        config = yaml.safe_load(rf)

    return HEDMInstrument(config)


@pytest.fixture
def ceria_material(ceria_examples_path: Path) -> Material:
    path = ceria_examples_path / 'ceria.h5'
    materials = load_materials_hdf5(path)
    return materials['CeO2']


@pytest.fixture
def ceria_picks(ceria_examples_path: Path) -> dict:
    path = ceria_examples_path / 'picks.h5'
    with h5py.File(path) as rf:
        output = {}
        # We need the picks in cartesian format
        unwrap_h5_to_dict(rf['/cartesian/CeO2 powder'], output)

    return output


def test_group_relative_constraints(
    dexelas_composite_instrument: HEDMInstrument,
    ceria_example_data: dict[str, np.ndarray],
    ceria_material: Material,
    ceria_picks: dict,
):
    # For this test, we will first perform a calibration without relative
    # constraints between the subpanels, to show that they do all move without
    # any relative constraints, and then we will add relative constraints, and
    # show that they only move relative to each other.
    instr = dexelas_composite_instrument
    monolithic_img_dict = ceria_example_data
    material = ceria_material

    # Local testing shows that the rmat_diff applied to each of the
    # original panel rmats does indeed result in the new relative
    # rmats almost exactly. However, the rmat diff between one
    # panel and another panel seems to have a slightly bigger
    # difference after the transformation. Thus we need to use a
    # little higher tolerance.
    relative_rmats_atol = 1e-3

    # Keep a deep copy of the original instrument around
    orig_instr = copy.deepcopy(instr)

    # Break up the image data into separate images for each detector
    # It's easiest to do this using hexrd's imageseries and
    # ProcessedImageSeries
    ims_dict = {}
    for monolithic_det_key, img in monolithic_img_dict.items():
        ims = imageseries.open(None, format='array', data=img)
        for det_key, panel in instr.detectors.items():
            if det_key.startswith(monolithic_det_key):
                ims_dict[det_key] = ProcessedImageSeries(
                    ims, oplist=[('rectangle', panel.roi)]
                )

    # Create the img_dict
    img_dict = {k: v[0] for k, v in ims_dict.items()}

    def make_calibrator(instr: HEDMInstrument) -> InstrumentCalibrator:
        # Calibrator for calibrating the Ceria powder lines
        powder_calibrator = PowderCalibrator(
            instr=instr,
            material=material,
            # The img_dict is actually only used for auto-picking points, which
            # we are not doing in this test, but we will provide it anyways...
            img_dict=img_dict,
            calibration_picks=ceria_picks,
        )

        # Use the extrinsic XYZ Euler angle convention
        extrinsic = True
        euler_convention = ('xyz', extrinsic)

        # Overall calibrator for whole instrument and experimental setup
        calibrator = InstrumentCalibrator(
            powder_calibrator,
            engineering_constraints=None,
            euler_convention=euler_convention,
        )

        # Let's not vary the lattice parameter of ceria
        params = calibrator.params
        for name in powder_calibrator.param_names:
            params[name].vary = False

        return calibrator

    calibrator = make_calibrator(instr)

    # Find all translations and rotation matrices relative to the
    # first detector.
    translation_suffixes = [
        '_tvec_x',
        '_tvec_y',
        '_tvec_z',
    ]
    tilt_suffixes = [
        '_euler_x',
        '_euler_y',
        '_euler_z',
    ]

    def detector_translation_keys(det_key: str) -> list[str]:
        return [f'{det_key}{x}' for x in translation_suffixes]

    def detector_tilt_keys(det_key: str) -> list[str]:
        return [f'{det_key}{x}' for x in tilt_suffixes]

    # Make all translations and tilts refinable.
    for det_key in instr.detectors:
        all_keys = detector_translation_keys(det_key) + detector_tilt_keys(
            det_key
        )
        for key in all_keys:
            calibrator.params[key].vary = True

    # Keep track of the original relative translations/tilts with respect
    # to the first detector. They should all vary after calibration.
    comparee_keys = {
        'ff1': 'ff1_0_0',
        'ff2': 'ff2_1_1',
    }

    def compute_relative_translations(instr: HEDMInstrument) -> dict:
        ret = {}
        for group in instr.detector_groups:
            comparee_key = comparee_keys[group]
            comparee_tvec = instr.detectors[comparee_key].tvec

            relative_translations = {}
            for det_key, panel in instr.detectors.items():
                relative_translations[det_key] = panel.tvec - comparee_tvec

            ret[group] = relative_translations

        return ret

    def compute_relative_rmats(instr: HEDMInstrument) -> dict:
        ret = {}
        for group in instr.detector_groups:
            comparee_key = comparee_keys[group]
            comparee_rmat = instr.detectors[comparee_key].rmat

            relative_rmats = {}
            for det_key, panel in instr.detectors.items():
                # Compute transformation required to get this rmat to the
                # comparee
                relative_rmats[det_key] = panel.rmat @ comparee_rmat.T

            ret[group] = relative_rmats

        return ret

    orig_relative_translations = compute_relative_translations(instr)
    orig_relative_rmats = compute_relative_rmats(instr)

    # FIXME: can we reduce quality to make this test faster?
    calibration_options = {}
    calibrator.run_calibration(calibration_options)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    # Relative tilts and rmats should all be different. Verify this.
    for group in instr.detector_groups:
        for key in instr.detectors_in_group(group):
            if key == comparee_keys[group]:
                continue

            assert not np.allclose(
                orig_relative_translations[group][key],
                new_relative_translations[group][key],
            )
            assert not np.allclose(
                orig_relative_rmats[group][key],
                new_relative_rmats[group][key],
                atol=relative_rmats_atol,
            )

    # Verify that two subpanels from different groups do not match
    assert not np.allclose(
        orig_relative_translations['ff1']['ff1_0_0'],
        new_relative_translations['ff1']['ff2_0_0'],
    )
    assert not np.allclose(
        orig_relative_rmats['ff1']['ff1_0_0'],
        new_relative_rmats['ff1']['ff2_0_0'],
        atol=relative_rmats_atol,
    )

    #### VARY GROUP TRANSLATION ####
    # Now, constrain relative tilts but allow relative translations
    instr = copy.deepcopy(orig_instr)
    calibrator = make_calibrator(instr)

    calibrator.relative_constraints_type = RelativeConstraintsType.group

    group_names = instr.detector_groups
    orig_centers = {k: instr.mean_group_center(k) for k in group_names}
    orig_tvecs = {k: v.tvec for k, v in instr.detectors.items()}
    orig_rmats = {k: v.rmat for k, v in instr.detectors.items()}

    tvec_names = (
        'ff1_tvec_x',
        'ff1_tvec_y',
        'ff1_tvec_z',
        'ff2_tvec_x',
        'ff2_tvec_y',
        'ff2_tvec_z',
    )

    tilt_names = (
        'ff1_euler_x',
        'ff1_euler_y',
        'ff1_euler_z',
        'ff2_euler_x',
        'ff2_euler_y',
        'ff2_euler_z',
    )
    for tvec_name in tvec_names:
        calibrator.params[tvec_name].vary = True

    for tilt_name in tilt_names:
        calibrator.params[tilt_name].vary = False

    # Run the calibration
    calibrator.run_calibration(calibration_options)

    # The new centers should not match
    for group in group_names:
        new_center = instr.mean_group_center(group)
        assert not np.allclose(orig_centers[group], new_center)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    # absolute tvecs should not match, but relative tvecs should
    # absolute and relative rmats should match
    for group in instr.detector_groups:
        for key, panel in instr.detectors_in_group(group).items():
            if key == comparee_keys[group]:
                continue

            assert not np.allclose(panel.tvec, orig_tvecs[key])
            assert np.allclose(
                orig_relative_translations[group][key],
                new_relative_translations[group][key],
            )
            assert np.allclose(orig_rmats[key], panel.rmat)
            assert np.allclose(
                orig_relative_rmats[group][key],
                new_relative_rmats[group][key],
                atol=relative_rmats_atol,
            )

    # Verify that two subpanels from different groups do not match
    assert not np.allclose(
        orig_relative_translations['ff1']['ff1_0_0'],
        new_relative_translations['ff1']['ff2_0_0'],
    )
    assert not np.allclose(
        orig_relative_rmats['ff1']['ff1_0_0'],
        new_relative_rmats['ff1']['ff2_0_0'],
        atol=relative_rmats_atol,
    )

    #### VARY GROUP TILTS ####
    # Now, constrain center translations but allow relative tilt
    instr = copy.deepcopy(orig_instr)
    calibrator = make_calibrator(instr)

    calibrator.relative_constraints_type = RelativeConstraintsType.group

    orig_centers = {k: instr.mean_group_center(k) for k in group_names}
    orig_tvecs = {k: v.tvec for k, v in instr.detectors.items()}
    orig_rmats = {k: v.rmat for k, v in instr.detectors.items()}

    # Now allow tilts to vary, but not the center. The center should
    # remain the same.
    for tvec_name in tvec_names:
        calibrator.params[tvec_name].vary = False

    for tilt_name in tilt_names:
        calibrator.params[tilt_name].vary = True

    relative_params = calibrator.relative_constraints.params
    orig_group_tilts = {
        'ff1': relative_params['ff1']['tilt'].copy(),
        'ff2': relative_params['ff2']['tilt'].copy(),
    }

    # Run the calibration
    calibrator.run_calibration(calibration_options)

    # The new centers should match
    for group in group_names:
        new_center = instr.mean_group_center(group)
        assert np.allclose(orig_centers[group], new_center)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    new_group_tilts = {
        'ff1': relative_params['ff1']['tilt'].copy(),
        'ff2': relative_params['ff2']['tilt'].copy(),
    }

    tilt_rmat_diffs = {}
    for group, tilt in new_group_tilts.items():
        orig_tilt = orig_group_tilts[group]
        tilt_rmat_diffs[group] = (
            rotMatOfExpMap(tilt) @ rotMatOfExpMap(orig_tilt).T
        )

    # absolute and relative tvecs should not match
    # absolute rmat should not match, but relative rmat should
    for group in instr.detector_groups:
        for key, panel in instr.detectors_in_group(group).items():
            if key == comparee_keys[group]:
                continue

            assert not np.allclose(panel.tvec, orig_tvecs[key])
            assert not np.allclose(
                orig_relative_translations[group][key],
                new_relative_translations[group][key],
            )
            assert not np.allclose(orig_rmats[key], panel.rmat)
            assert np.allclose(
                orig_relative_rmats[group][key],
                new_relative_rmats[group][key],
                atol=relative_rmats_atol,
            )

            # The vectors from the detector tvecs to the center should match
            # if we apply the rotation.
            tilt_rmat_diff = tilt_rmat_diffs[group]
            orig_center = orig_centers[group]
            assert np.allclose(
                tilt_rmat_diff @ (orig_tvecs[key] - orig_center),
                panel.tvec - instr.mean_group_center(group),
            )

    # Verify that two subpanels from different groups do not match
    assert not np.allclose(
        orig_relative_translations['ff1']['ff1_0_0'],
        new_relative_translations['ff1']['ff2_0_0'],
    )
    assert not np.allclose(
        orig_relative_rmats['ff1']['ff1_0_0'],
        new_relative_rmats['ff1']['ff2_0_0'],
        atol=relative_rmats_atol,
    )

    #### VARY GROUP TRANSLATIONS AND TILTS ####
    # Now, constrain group center translations but allow relative tilt
    instr = copy.deepcopy(orig_instr)
    calibrator = make_calibrator(instr)

    calibrator.relative_constraints_type = RelativeConstraintsType.group

    orig_centers = {k: instr.mean_group_center(k) for k in group_names}
    orig_tvecs = {k: v.tvec for k, v in instr.detectors.items()}
    orig_rmats = {k: v.rmat for k, v in instr.detectors.items()}

    # Now allow all parameters to vary.
    for tvec_name in tvec_names:
        calibrator.params[tvec_name].vary = True

    for tilt_name in tilt_names:
        calibrator.params[tilt_name].vary = True

    relative_params = calibrator.relative_constraints.params
    orig_group_tilts = {
        'ff1': relative_params['ff1']['tilt'].copy(),
        'ff2': relative_params['ff2']['tilt'].copy(),
    }

    # Run the calibration
    calibrator.run_calibration(calibration_options)

    # The new centers should be different
    for group in group_names:
        new_center = instr.mean_group_center(group)
        assert not np.allclose(orig_centers[group], new_center)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    new_group_tilts = {
        'ff1': relative_params['ff1']['tilt'].copy(),
        'ff2': relative_params['ff2']['tilt'].copy(),
    }

    tilt_rmat_diffs = {}
    for group, tilt in new_group_tilts.items():
        orig_tilt = orig_group_tilts[group]
        tilt_rmat_diffs[group] = (
            rotMatOfExpMap(tilt) @ rotMatOfExpMap(orig_tilt).T
        )

    # absolute and relative tvecs should not match
    # absolute rmat should not match, but relative rmat should
    for group in instr.detector_groups:
        for key, panel in instr.detectors_in_group(group).items():
            if key == comparee_keys[group]:
                continue

            assert not np.allclose(panel.tvec, orig_tvecs[key])
            assert not np.allclose(
                orig_relative_translations[group][key],
                new_relative_translations[group][key],
            )
            assert not np.allclose(orig_rmats[key], panel.rmat)
            assert np.allclose(
                orig_relative_rmats[group][key],
                new_relative_rmats[group][key],
                atol=relative_rmats_atol,
            )

            # The vectors from the detector tvecs to the center should match
            # if we apply the rotation.
            tilt_rmat_diff = tilt_rmat_diffs[group]
            orig_center = orig_centers[group]
            assert np.allclose(
                tilt_rmat_diff @ (orig_tvecs[key] - orig_center),
                panel.tvec - instr.mean_group_center(group),
            )

    # Verify that two subpanels from different groups do not match
    assert not np.allclose(
        orig_relative_translations['ff1']['ff1_0_0'],
        new_relative_translations['ff1']['ff2_0_0'],
    )
    assert not np.allclose(
        orig_relative_rmats['ff1']['ff1_0_0'],
        new_relative_rmats['ff1']['ff2_0_0'],
        atol=relative_rmats_atol,
    )
