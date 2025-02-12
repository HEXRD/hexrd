import json
from pathlib import Path

import numpy as np
import pytest

from hexrd.material import _angstroms, load_materials_hdf5, Material
from hexrd.wppf import LeBail, Rietveld
from hexrd.wppf.parameters import Parameter, Parameters


@pytest.fixture
def wppf_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'ge' / 'wppf'


@pytest.fixture
def expt_spectrum(wppf_examples_path: Path) -> np.array:
    path = wppf_examples_path / 'expt_spectrum.npy'
    return np.load(path)


@pytest.fixture
def spline_picks(wppf_examples_path: Path) -> np.array:
    path = wppf_examples_path / 'spline_picks.npy'
    return np.load(path)


@pytest.fixture
def ceo2_material(wppf_examples_path: Path) -> Material:
    path = wppf_examples_path / 'ceo2.h5'
    return load_materials_hdf5(path)['CeO2']


@pytest.fixture
def rietveld_params(wppf_examples_path: Path) -> dict[str, Parameter]:
    path = wppf_examples_path / 'rietveld_params.json'
    with open(path, 'r') as rf:
        params_json = json.load(rf)

    params = Parameters()
    for k, v in params_json.items():
        params[k] = Parameter(**v)

    return params


@pytest.fixture
def lebail_params(wppf_examples_path: Path) -> dict[str, Parameter]:
    path = wppf_examples_path / 'lebail_params.json'
    with open(path, 'r') as rf:
        params_json = json.load(rf)

    params = Parameters()
    for k, v in params_json.items():
        params[k] = Parameter(**v)

    return params


def _fix_all_params(parameters: Parameters):
    for param in parameters.param_dict.values():
        param.vary = False


def test_wppf_rietveld(expt_spectrum, spline_picks, ceo2_material,
                       rietveld_params):

    beam_wavelength = 0.15358835358711712
    params = rietveld_params

    kwargs = {
        'expt_spectrum': expt_spectrum,
        'params': params,
        'phases': [ceo2_material],
        'wavelength': {'synchrotron': [_angstroms(beam_wavelength), 1.0]},
        'bkgmethod': {'spline': spline_picks.tolist()},
        'peakshape': 'pvtch',
    }

    rietveld = Rietveld(**kwargs)

    # First, only vary the scale
    _fix_all_params(params)
    params['scale'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.08

    # Next, vary the lattice constant
    _fix_all_params(params)
    params['CeO2_a'].vary = True
    rietveld.Refine()

    # Next, U, V, and W
    _fix_all_params(params)
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    rietveld.Refine()

    # Next, X and Y
    _fix_all_params(params)
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    rietveld.Refine()

    # Next, U, V, W, X and Y
    _fix_all_params(params)
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.072

    # Next, the debye-waller constants
    _fix_all_params(params)
    params['CeO2_O1_dw'].vary = True
    params['CeO2_Ce1_dw'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.07

    # Finally, everything
    _fix_all_params(params)
    params['scale'].vary = True
    params['CeO2_a'].vary = True
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    params['CeO2_O1_dw'].vary = True
    params['CeO2_Ce1_dw'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.069

    # Verify expected final values to some tolerances
    assert round(params['CeO2_a'].value, 4) == 0.5412
    assert round(params['CeO2_O1_dw'].value, 5) == 0.00209
    assert round(params['CeO2_Ce1_dw'].value, 5) == 0.00088


def test_wppf_lebail(expt_spectrum, ceo2_material, lebail_params):
    beam_wavelength = 0.15358835358711712
    params = lebail_params

    kwargs = {
        'expt_spectrum': expt_spectrum,
        'params': params,
        'phases': [ceo2_material],
        'wavelength': {'synchrotron': [_angstroms(beam_wavelength), 1.0]},
        'bkgmethod': {'chebyshev': 3},
        'peakshape': 'pvfcj',
    }

    lebail = LeBail(**kwargs)

    # First, only vary the lattice constant
    _fix_all_params(params)
    params['CeO2_a'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.25

    # Next, U, V, and W
    _fix_all_params(params)
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.1

    # Next, X and Y
    _fix_all_params(params)
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.08

    # Next, U, V, W, X and Y
    _fix_all_params(params)
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.07

    # Finally, everything
    _fix_all_params(params)
    params['CeO2_a'].vary = True
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.066

    # Verify expected final values to some tolerances
    assert round(params['CeO2_a'].value, 5) == 0.54112
