from pathlib import Path

import numpy as np
import pytest

from hexrd.material import load_materials_hdf5
from hexrd.valunits import valWUnit
from hexrd.wppf import LeBail, Rietveld
from hexrd.wppf.amorphous import Amorphous


@pytest.fixture
def amorphous_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'tests/wppf/amorphous'


@pytest.fixture
def amorphous_expt_spectrum(amorphous_examples_path: Path) -> np.ndarray:
    return np.loadtxt(amorphous_examples_path / 'expt_spectrum.xy')


@pytest.fixture
def amorphous_materials_path(amorphous_examples_path: Path) -> Path:
    return amorphous_examples_path / 'materials.h5'


@pytest.fixture
def wppf_amorphous_object(amorphous_expt_spectrum: np.ndarray) -> Amorphous:
    tth_list = amorphous_expt_spectrum[:, 0]

    '''the amorphous peak can be either "split_pv"
    or split_gaussian. you can add more than one peak
    in the amorphous model. if you need to have more than
    one peak, you can pass the dictionary with more entries
    e.g. center={'c1':36, 'c2':65}
    fwhm is the full-width at half-max for the peaks. pseudo-voight
    peaks need 4 numbers per peak e.g.

    fwhm_g_l: fwhm gaussian left
    fwhm_l_l: fwhm lorentzian left
    fwhm_g_r: fwhm gaussian right
    fwhm_l_r: fwhm lorentzian right

    For split_gaussian we only have 2 per peak

    fwhm_l: fwhm left
    fwhm_r: fwhm right

    '''
    return Amorphous(
        tth_list,
        model_type="split_pv",
        center={'c1': 36},
        fwhm={'c1': np.array([3, 3, 5, 5])},
        scale={'c1': 55},
    )


@pytest.fixture
def wppf_amorphous_kwargs(
    amorphous_expt_spectrum: np.ndarray,
    wppf_amorphous_object: Amorphous,
    amorphous_materials_path: Path,
) -> dict:
    lam = _nm(0.123957521)
    dmin = _angstrom(0.5)
    kev = _kev(10.00215218)

    materials = load_materials_hdf5(
        amorphous_materials_path,
        dmin=dmin,
        kev=kev,
    )
    Ti_amb = materials['Ti_ambient']
    Ti_beta = materials['Ti_beta']

    return {
        "expt_spectrum": amorphous_expt_spectrum,
        "phases": [Ti_amb, Ti_beta],
        "wavelength": {
            "lle": [lam, 1.0],
        },
        "bkgmethod": {"chebyshev": 2},
        "peakshape": "pvtch",
        "amorphous_model": wppf_amorphous_object,
    }


def _nm(x):
    return valWUnit("lp", "LENGTH", x, "nm")


def _angstrom(x):
    return valWUnit("lp", "LENGTH", x, "angstrom")


def _kev(x):
    return valWUnit("accvoltage", "ENERGY", x, "keV")


def test_wppf_amorphous_rietveld(wppf_amorphous_kwargs: dict):
    obj = Rietveld(**wppf_amorphous_kwargs)
    obj.scale = 1e-8

    obj.params['scale'].vary = True

    obj.Refine()

    # After refining only the scale, the goodness of fit and chi squared
    # shouldn't be that great.
    assert obj.Rwp > 0.1
    assert obj.gofF > 0.5

    # we can refine some parameters here
    obj.params['c1_amorphous_scale'].vary = True
    obj.params['c1_amorphous_center'].vary = True

    obj.Refine()

    obj.params['scale'].vary = False

    obj.params['c1_amorphous_fwhm_g_l'].vary = True
    obj.params['c1_amorphous_fwhm_l_l'].vary = True
    obj.params['c1_amorphous_fwhm_g_r'].vary = True
    obj.params['c1_amorphous_fwhm_l_r'].vary = True

    obj.Refine()

    obj.params['bkg_0'].vary = True
    obj.params['bkg_1'].vary = True
    obj.params['bkg_2'].vary = True

    obj.Refine()

    # Verify that the goodness of fit and chi-squared are below expected values
    assert obj.Rwp < 0.057
    assert obj.gofF < 0.27


def test_wppf_amorphous_lebail(wppf_amorphous_kwargs: dict):
    obj = LeBail(**wppf_amorphous_kwargs)

    # Before any refinement, the goodness of fit and chi-squared shouldn't
    # be that great.
    assert obj.Rwp > 0.1
    assert obj.gofF > 0.5

    # we can refine some parameters here
    obj.params['c1_amorphous_scale'].vary = True
    obj.params['c1_amorphous_center'].vary = True

    obj.RefineCycle()

    obj.params['c1_amorphous_fwhm_g_l'].vary = True
    obj.params['c1_amorphous_fwhm_l_l'].vary = True
    obj.params['c1_amorphous_fwhm_g_r'].vary = True
    obj.params['c1_amorphous_fwhm_l_r'].vary = True

    obj.RefineCycle()

    obj.params['bkg_0'].vary = True
    obj.params['bkg_1'].vary = True
    obj.params['bkg_2'].vary = True

    obj.RefineCycle()

    # Verify that the goodness of fit and chi-squared are below expected values
    assert obj.Rwp < 0.06
    assert obj.gofF < 0.3
