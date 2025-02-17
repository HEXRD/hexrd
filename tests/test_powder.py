import copy
from pathlib import Path

import h5py
import numpy as np

import pytest

from hexrd.instrument.hedm_instrument import HEDMInstrument
from hexrd.material.material import load_materials_hdf5, Material
from hexrd.projections.polar import PolarView


@pytest.fixture
def eiger_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'eiger'


@pytest.fixture
def ceria_examples_path(eiger_examples_path: Path) -> Path:
    return eiger_examples_path / 'first_ceria'


@pytest.fixture
def eiger_instrument(ceria_examples_path: Path) -> HEDMInstrument:
    instr_path = (
        ceria_examples_path / 'eiger_ceria_calibrated_composite.hexrd'
    )
    with h5py.File(instr_path, 'r') as rf:
        return HEDMInstrument(rf)


@pytest.fixture
def ceria_material(ceria_examples_path: Path) -> Material:
    path = ceria_examples_path / 'ceria.h5'
    materials = load_materials_hdf5(path)
    return materials['CeO2']


@pytest.fixture
def expected_simulated_powder_lines_results(
    test_data_dir: Path,
) -> dict[str, dict]:
    path = test_data_dir / 'expected_simulated_powder_lines_results.npy'
    return np.load(path, allow_pickle=True).item()


def test_simulate_powder_lines(
    eiger_instrument: HEDMInstrument,
    ceria_material: Material,
    expected_simulated_powder_lines_results: dict[str, dict],
):
    instr = eiger_instrument
    material = copy.deepcopy(ceria_material)
    ref = expected_simulated_powder_lines_results

    pd = material.planeData

    pd.exclusions = None

    pd.tThWidth = np.radians(0.25)

    hkls = pd.getHKLs()
    tth = pd.getTTh()

    # There should be exactly 22 two HKLs generated
    assert len(tth) == 22

    def hkl_idx(hkl):
        hkl = list(hkl)
        for i, hkl_ref in enumerate(hkls.tolist()):
            if hkl_ref == hkl:
                return i

        return None

    # Verify a few manual cases
    # The four corners should all have these HKLs
    corner_hkls = {
        (4, 2, 0): 9.583035640913781,
        # 333 and 511 are nearly identical.
        (3, 3, 3): 11.139041445220014,
        (5, 1, 1): 11.139041445220013,
        (6, 2, 0): 13.568350297337751,
    }
    corner_hkls = {k: np.radians(v) for k, v in corner_hkls.items()}

    # The four inner subpanels should *only* have these HKLs
    inner_hkls = {
        (1, 1, 1): 3.7078160949754677,
        (2, 0, 0): 4.281666411410814,
    }
    inner_hkls = {k: np.radians(v) for k, v in inner_hkls.items()}

    inner_dets = ['eiger_3_1', 'eiger_3_2', 'eiger_4_1', 'eiger_4_2']
    corner_dets = ['eiger_0_0', 'eiger_0_3', 'eiger_7_0', 'eiger_7_3']

    full_results = {}
    for det_key, panel in instr.detectors.items():
        angs, xys, ranges = panel.make_powder_rings(
            pd,
            # Have to make a smaller delta eta to get all intersections,
            # since the Eiger subpanels are small.
            delta_eta=0.25,
        )
        full_results[det_key] = {
            'angs': angs,
            'xys': xys,
            'ranges': ranges,
        }

        valid_angs = [x for x in angs if x.size > 0]
        if det_key in inner_dets:
            for hkl, value in inner_hkls.items():
                idx = hkl_idx(hkl)
                assert np.allclose(angs[idx][0][0], value)

            # Also verify that there are no other HKLs
            assert len(valid_angs) == len(inner_hkls)

        if det_key in corner_dets:
            for hkl, value in corner_hkls.items():
                idx = hkl_idx(hkl)
                assert np.allclose(angs[idx][0][0], value)

            # There should be at least 9 valid HKLs on every corner detector
            assert len(valid_angs) >= 9

    # Verify that a few of the HKLs were combined
    indices, ranges = pd.getMergedRanges()

    num_merged_hkls = sum([len(x) > 1 for x in indices])
    assert num_merged_hkls == 5

    # Verify that 3, 3, 3 and 5, 1, 1 were merged
    idx1 = hkl_idx([5, 1, 1])
    idx2 = hkl_idx([3, 3, 3])
    assert [idx1, idx2] in indices

    # Now just verify that everything matches what it was before...
    for det_key, det_results in full_results.items():
        for entry_key, results in det_results.items():
            ref_results = ref[det_key][entry_key]
            for i in range(len(results)):
                assert np.allclose(results[i], ref_results[i], equal_nan=True)


def test_simulate_powder_pattern_image(
    eiger_instrument: HEDMInstrument,
    ceria_material: Material,
):
    instr = eiger_instrument
    material = ceria_material

    img_dict = instr.simulate_powder_pattern(
        [material],
        noise='poisson',
    )

    # Now do a warp to the polar view, create the lineout, and verify there
    # is intensity in a few places we'd expect.
    tth_range = [0.1, 14]
    eta_min = -180
    eta_max = 180
    pixel_size = (0.1, 0.1)
    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size)
    img = pv.warp_image(img_dict, pad_with_nans=True,
                        do_interpolation=True)

    lineout = img.mean(axis=0).filled(np.nan)

    # Now verify there is intensity at the predicted two theta values
    # Sort by structure factor and check the 4 most intense lines
    sf_sorting = np.argsort(-material.planeData.structFact)
    for i in range(4):
        tth_idx = sf_sorting[i]
        tth_val = np.degrees(material.planeData.getTTh()[tth_idx])

        # Compute the index for this tth value in the lineout
        idx = int(np.floor((tth_val - tth_range[0]) / pixel_size[1]))

        # Verify we are at a maximum, and that the value is higher
        # than the background.
        assert lineout[idx] > lineout[0]
        assert lineout[idx] > lineout[idx - 1]
        assert lineout[idx] > lineout[idx + 1]
