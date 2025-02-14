from pathlib import Path

import h5py
import numpy as np

import pytest

from hexrd import imageseries
from hexrd.fitting.calibration import PowderCalibrator
from hexrd.material.material import load_materials_hdf5, Material
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument.hedm_instrument import HEDMInstrument


@pytest.fixture
def eiger_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'eiger'


@pytest.fixture
def ceria_examples_path(eiger_examples_path: Path) -> Path:
    return eiger_examples_path / 'first_ceria'


@pytest.fixture
def eiger_instrument(ceria_examples_path: Path) -> HEDMInstrument:
    instr_path = ceria_examples_path / 'eiger_ceria_calibrated_composite.hexrd'
    with h5py.File(instr_path, 'r') as rf:
        return HEDMInstrument(rf)


@pytest.fixture
def ceria_example_data(ceria_examples_path: Path) -> np.ndarray:
    data_path = ceria_examples_path / 'ff_000_data_000001.h5'
    with h5py.File(data_path, 'r') as rf:
        # Just return the first frame
        return rf['/entry/data/data'][0]


@pytest.fixture
def ceria_material(ceria_examples_path: Path) -> Material:
    path = ceria_examples_path / 'ceria.h5'
    materials = load_materials_hdf5(path)
    return materials['CeO2']


def test_powder_auto_pick(
    eiger_instrument: HEDMInstrument,
    ceria_material: Material,
    ceria_example_data: np.ndarray,
):
    instr = eiger_instrument
    material = ceria_material
    image_data = ceria_example_data
    pd = material.planeData

    # Disable all exclusions on the plane data
    pd.exclusions = None
    hkls = pd.getHKLs()
    tth_values = pd.getTTh()

    def hkl_idx(hkl: tuple | list) -> int | None:
        hkl = list(hkl)
        for i, hkl_ref in enumerate(hkls.tolist()):
            if hkl == hkl_ref:
                return i

        return None

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

    calibrator = PowderCalibrator(
        instr,
        material,
        img_dict,
        tth_tol=0.25,
        eta_tol=1.0,
        pktype='gaussian',
    )

    calibrator.autopick_points(
        fit_tth_tol=1.0,
        int_cutoff=1e-4,
    )

    picks = calibrator.calibration_picks

    tth_tol = np.radians(calibrator.tth_tol)

    unique_hkls = []
    num_picks_per_det = {}
    for det_key, det_picks in picks.items():
        panel = instr.detectors[det_key]
        num_picks_per_det[det_key] = 0
        for hkl_str, hkl_picks in det_picks.items():
            num_picks_per_det[det_key] += len(hkl_picks)
            if hkl_str not in unique_hkls:
                unique_hkls.append(hkl_str)

            idx = hkl_idx(list(map(int, hkl_str.split())))
            assert idx is not None

            # Verify that all picks are within tolerance distance of
            # the two theta values.
            tth = tth_values[idx]
            angles = panel.cart_to_angles(hkl_picks)[0]
            assert np.allclose(angles[:, 0], tth, atol=tth_tol)

            # Verify eta values are different if there are enough picks
            if len(hkl_picks) > 5:
                assert not np.allclose(angles[0, 1], angles[-1, 1])

    # There should be at least 12 unique hkls and 2000 total picks
    total_num_picks = sum(num_picks_per_det.values())
    assert len(unique_hkls) >= 12
    assert total_num_picks > 2000

    # There should be at least 30 picks per detector
    assert all(v > 30 for v in num_picks_per_det.values())
