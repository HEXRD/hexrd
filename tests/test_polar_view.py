from pathlib import Path

import h5py
import numpy as np
import pytest

from hexrd.core import imageseries
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.projections.polar import PolarView


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


def test_polar_view(
    ceria_composite_instrument: HEDMInstrument,
    ceria_example_data: np.ndarray,
    test_data_dir: Path,
):
    instr = ceria_composite_instrument
    image_data = ceria_example_data

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

    # Create the PolarView
    tth_range = [0, 14.0]
    eta_min = -180.0
    eta_max = 180.0
    pixel_size = (0.01, 5.0)

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size)
    img = pv.warp_image(img_dict, pad_with_nans=True,
                        do_interpolation=True)

    # This is a masked array. Just fill it with nans.
    img = img.filled(np.nan)

    # Verify that the image is identical to a reference image
    ref = np.load(test_data_dir / 'test_polar_view_expected.npy')
    assert np.allclose(img, ref, equal_nan=True)

    # Also generate it using the cache
    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size,
                   cache_coordinate_map=True)
    fast_img = pv.warp_image(img_dict, pad_with_nans=True,
                             do_interpolation=True)

    # This should also be identical
    fast_img = fast_img.filled(np.nan)
    assert np.allclose(fast_img, ref, equal_nan=True)
