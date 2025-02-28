from pathlib import Path

import numpy as np
import yaml

import pytest

from hexrd import imageutil
from hexrd.instrument.hedm_instrument import HEDMInstrument
from hexrd.projections.polar import PolarView


@pytest.fixture
def simulated_tardis_path(example_repo_path: Path) -> Path:
    return example_repo_path / 'tardis' / 'simulated'


@pytest.fixture
def simulated_tardis_images(
    simulated_tardis_path: Path,
) -> dict[str, np.ndarray]:
    path = simulated_tardis_path / 'tardis_images.npz'
    npz = np.load(path)
    return {k: v for k, v in npz.items()}


@pytest.fixture
def tardis_instrument(simulated_tardis_path: Path) -> HEDMInstrument:
    path = simulated_tardis_path / 'ideal_tardis.yml'
    with open(path, 'r') as rf:
        conf = yaml.safe_load(rf)

    return HEDMInstrument(conf)


@pytest.fixture
def expected_snip1d_results(test_data_dir: Path) -> np.ndarray:
    path = test_data_dir / 'expected_snip1d_results.npy'
    return np.load(path)


def test_snip1d(
    tardis_instrument: HEDMInstrument,
    simulated_tardis_images: dict[str, np.ndarray],
    expected_snip1d_results: np.ndarray,
):
    instr = tardis_instrument
    img_dict = simulated_tardis_images
    ref = expected_snip1d_results

    # Create the PolarView
    tth_range = [10, 120]
    eta_min = -180.0
    eta_max = 180.0
    pixel_size = (0.1, 1.0)

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size)
    img = pv.warp_image(img_dict, pad_with_nans=True,
                        do_interpolation=True)

    snip_width = 100
    numiter = 2
    output = imageutil.snip1d(
        img,
        snip_width,
        numiter,
    )

    assert np.allclose(output.filled(np.nan), ref, equal_nan=True)
