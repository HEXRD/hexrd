import importlib.resources
from pathlib import Path

import numpy as np
import yaml

import pytest

import hexrd.core.resources
from hexrd.core.instrument.hedm_instrument import HEDMInstrument


@pytest.fixture
def tardis_instrument() -> HEDMInstrument:
    path = importlib.resources.files(hexrd.core.resources).joinpath(
        'tardis_reference_config.yml'
    )
    with open(path, 'r') as rf:
        conf = yaml.safe_load(rf)

    return HEDMInstrument(conf)


@pytest.fixture
def expected_pixel_solid_angles_results(
    test_data_dir: Path,
) -> dict[str, np.ndarray]:
    path = test_data_dir / 'expected_pixel_solid_angles_results.npz'
    return {k: v for k, v in np.load(path).items()}


def test_pixel_solid_angles(
    tardis_instrument: HEDMInstrument,
    expected_pixel_solid_angles_results: np.ndarray,
):
    instr = tardis_instrument
    ref = expected_pixel_solid_angles_results

    assert sorted(instr.detectors.keys()) == sorted(ref.keys())
    for det_key, panel in instr.detectors.items():
        assert np.allclose(panel.pixel_solid_angles, ref[det_key], rtol=1e-3)
