import copy
from pathlib import Path

import h5py
import numpy as np
import pytest

from hexrd.core import imageseries
from hexrd.core.fitting.calibration import (
    InstrumentCalibrator,
    PowderCalibrator,
)
from hexrd.core.fitting.calibration.relative_constraints import (
    RelativeConstraintsType,
)
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.material import load_materials_hdf5, Material
from hexrd.core.rotations import rotMatOfExpMap
from hexrd.core.utils.hdf5 import unwrap_h5_to_dict


@pytest.fixture
def eiger_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'eiger'


@pytest.fixture
def ceria_examples_path(eiger_examples_path: Path) -> Path:
    return eiger_examples_path / 'first_ceria'


@pytest.fixture
def ceria_example_data(ceria_examples_path: Path) -> np.ndarray:
    data_path = ceria_examples_path / 'ff_000_data_000001.h5'
    with h5py.File(data_path, 'r') as rf:
        # Just return the first frame
        return rf['/entry/data/data'][0]


@pytest.fixture
def ceria_composite_instrument(ceria_examples_path: Path) -> HEDMInstrument:
    instr_path = (
        ceria_examples_path / 'eiger_ceria_uncalibrated_composite.hexrd'
    )
    with h5py.File(instr_path, 'r') as rf:
        return HEDMInstrument(rf)


@pytest.fixture
def ceria_material(ceria_examples_path: Path) -> Material:
    path = ceria_examples_path / 'ceria.h5'
    materials = load_materials_hdf5(path)
    return materials['CeO2']


@pytest.fixture
def ceria_picks(ceria_examples_path: Path) -> dict:
    path = ceria_examples_path / 'eiger_ceria_uncalibrated_picks.h5'
    with h5py.File(path) as rf:
        output = {}
        # We need the picks in cartesian format
        unwrap_h5_to_dict(rf['/cartesian/CeO2 powder'], output)

    return output


def test_relative_constraints(
    ceria_composite_instrument: HEDMInstrument,
    ceria_example_data: np.ndarray,
    ceria_material: Material,
    ceria_picks: dict,
):
    # For this test, we will first perform a calibration without relative
    # constraints between the subpanels, to show that they do all move without
    # any relative constraints, and then we will add relative constraints, and
    # show that they only move relative to each other.
    instr = ceria_composite_instrument
    image_data = ceria_example_data
    material = ceria_material

    # Keep a deep copy of the original instrument around
    orig_instr = copy.deepcopy(instr)

    # Break up the image data into separate images for each detector
    # It's easiest to do this using hexrd's imageseries and
    # ProcessedImageSeries
    ims_dict = {}
    ims = imageseries.open(None, format='array', data=image_data)
    for det_key, panel in instr.detectors.items():
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
    comparee_key = 'eiger_0_0'

    def compute_relative_translations(instr: HEDMInstrument) -> dict:
        comparee_tvec = instr.detectors[comparee_key].tvec

        relative_translations = {}
        for det_key, panel in instr.detectors.items():
            relative_translations[det_key] = panel.tvec - comparee_tvec
        return relative_translations

    def compute_relative_rmats(instr: HEDMInstrument) -> dict:
        comparee_rmat = instr.detectors[comparee_key].rmat

        relative_rmats = {}
        for det_key, panel in instr.detectors.items():
            # Compute transformation required to get this rmat to the comparee
            relative_rmats[det_key] = panel.rmat @ comparee_rmat.T
        return relative_rmats

    orig_relative_translations = compute_relative_translations(instr)
    orig_relative_rmats = compute_relative_rmats(instr)

    # FIXME: can we reduce quality to make this test faster?
    calibration_options = {}
    calibrator.run_calibration(calibration_options)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    # Relative tilts and rmats should all be different. Verify this.
    for key in instr.detectors:
        if key == comparee_key:
            continue

        assert not np.allclose(
            orig_relative_translations[key], new_relative_translations[key]
        )
        assert not np.allclose(
            orig_relative_rmats[key], new_relative_rmats[key]
        )

    #### VARY SYSTEM TRANSLATION ####
    # Now, constrain relative tilts but allow relative translations
    instr = copy.deepcopy(orig_instr)
    calibrator = make_calibrator(instr)

    calibrator.relative_constraints_type = RelativeConstraintsType.system

    orig_center = instr.mean_detector_center
    orig_tvecs = {k: v.tvec for k, v in instr.detectors.items()}
    orig_rmats = {k: v.rmat for k, v in instr.detectors.items()}

    system_tvec_names = (
        'system_tvec_x',
        'system_tvec_y',
        'system_tvec_z',
    )
    system_tilt_names = (
        'system_euler_x',
        'system_euler_y',
        'system_euler_z',
    )
    for tvec_name in system_tvec_names:
        calibrator.params[tvec_name].vary = True

    for tilt_name in system_tilt_names:
        calibrator.params[tilt_name].vary = False

    # Run the calibration
    calibrator.run_calibration(calibration_options)

    # The new center should not match
    assert not np.allclose(orig_center, instr.mean_detector_center)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    # absolute tvecs should not match, but relative tvecs should
    # absolute and relative rmats should match
    for key, panel in instr.detectors.items():
        if key == comparee_key:
            continue

        assert not np.allclose(panel.tvec, orig_tvecs[key])
        assert np.allclose(
            orig_relative_translations[key], new_relative_translations[key]
        )
        assert np.allclose(orig_rmats[key], panel.rmat)
        assert np.allclose(orig_relative_rmats[key], new_relative_rmats[key])

    #### VARY SYSTEM TILTS ####
    # Now, constrain system center translations but allow relative tilt
    instr = copy.deepcopy(orig_instr)
    calibrator = make_calibrator(instr)

    calibrator.relative_constraints_type = RelativeConstraintsType.system

    orig_center = instr.mean_detector_center
    orig_tvecs = {k: v.tvec for k, v in instr.detectors.items()}
    orig_rmats = {k: v.rmat for k, v in instr.detectors.items()}

    # Now allow tilts to vary, but not the center. The center should
    # remain the same.
    for tvec_name in system_tvec_names:
        calibrator.params[tvec_name].vary = False

    for tilt_name in system_tilt_names:
        calibrator.params[tilt_name].vary = True

    orig_system_tilt = calibrator.relative_constraints.params['tilt'].copy()

    # Run the calibration
    calibrator.run_calibration(calibration_options)

    # The new center should match
    assert np.allclose(orig_center, instr.mean_detector_center)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    new_system_tilt = calibrator.relative_constraints.params['tilt'].copy()

    tilt_rmat_diff = (
        rotMatOfExpMap(new_system_tilt) @ rotMatOfExpMap(orig_system_tilt).T
    )

    # absolute and relative tvecs should not match
    # absolute rmat should not match, but relative rmat should
    for key, panel in instr.detectors.items():
        if key == comparee_key:
            continue

        assert not np.allclose(panel.tvec, orig_tvecs[key])
        assert not np.allclose(
            orig_relative_translations[key], new_relative_translations[key]
        )
        assert not np.allclose(orig_rmats[key], panel.rmat)
        assert np.allclose(orig_relative_rmats[key], new_relative_rmats[key])

        # The vectors from the detector tvecs to the center should match
        # if we apply the rotation.
        assert np.allclose(
            tilt_rmat_diff @ (orig_tvecs[key] - orig_center),
            panel.tvec - instr.mean_detector_center,
        )

    #### VARY SYSTEM TRANSLATIONS AND TILTS ####
    # Now, constrain system center translations but allow relative tilt
    instr = copy.deepcopy(orig_instr)
    calibrator = make_calibrator(instr)

    calibrator.relative_constraints_type = RelativeConstraintsType.system

    orig_center = instr.mean_detector_center
    orig_tvecs = {k: v.tvec for k, v in instr.detectors.items()}
    orig_rmats = {k: v.rmat for k, v in instr.detectors.items()}

    # Now allow all parameters to vary.
    for tvec_name in system_tvec_names:
        calibrator.params[tvec_name].vary = True

    for tilt_name in system_tilt_names:
        calibrator.params[tilt_name].vary = True

    orig_system_tilt = calibrator.relative_constraints.params['tilt'].copy()

    # Run the calibration
    calibrator.run_calibration(calibration_options)

    # The new center should be different
    assert not np.allclose(orig_center, instr.mean_detector_center)

    # Find new translations and rmats
    new_relative_translations = compute_relative_translations(instr)
    new_relative_rmats = compute_relative_rmats(instr)

    new_system_tilt = calibrator.relative_constraints.params['tilt'].copy()

    tilt_rmat_diff = (
        rotMatOfExpMap(new_system_tilt) @ rotMatOfExpMap(orig_system_tilt).T
    )

    # absolute and relative tvecs should not match
    # absolute rmat should not match, but relative rmat should
    for key, panel in instr.detectors.items():
        if key == comparee_key:
            continue

        assert not np.allclose(panel.tvec, orig_tvecs[key])
        assert not np.allclose(
            orig_relative_translations[key], new_relative_translations[key]
        )
        assert not np.allclose(orig_rmats[key], panel.rmat)
        assert np.allclose(orig_relative_rmats[key], new_relative_rmats[key])

        # The vectors from the detector tvecs to the center should match
        # if we apply the rotation.
        assert np.allclose(
            tilt_rmat_diff @ (orig_tvecs[key] - orig_center),
            panel.tvec - instr.mean_detector_center,
        )
