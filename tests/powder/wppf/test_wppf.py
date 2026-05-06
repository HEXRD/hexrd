import json
from pathlib import Path

import lmfit
import numpy as np
import pytest

from hexrd.core.material import _angstroms, load_materials_hdf5, Material
from hexrd.powder.wppf import LeBail, Rietveld


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
def rietveld_params(wppf_examples_path: Path) -> dict[str, lmfit.Parameter]:
    path = wppf_examples_path / 'rietveld_params.json'
    with open(path, 'r') as rf:
        params_json = json.load(rf)

    params = lmfit.Parameters()
    for k, v in params_json.items():
        # For backward compatibility
        if 'lb' in v:
            v['min'] = v.pop('lb')

        if 'ub' in v:
            v['max'] = v.pop('ub')

        params[k] = lmfit.Parameter(**v)

    return params


@pytest.fixture
def lebail_params(wppf_examples_path: Path) -> dict[str, lmfit.Parameter]:
    path = wppf_examples_path / 'lebail_params.json'
    with open(path, 'r') as rf:
        params_json = json.load(rf)

    params = lmfit.Parameters()
    for k, v in params_json.items():
        # For backward compatibility
        if 'lb' in v:
            v['min'] = v.pop('lb')

        if 'ub' in v:
            v['max'] = v.pop('ub')

        params[k] = lmfit.Parameter(**v)

    return params


def test_wppf_rietveld(
    expt_spectrum, spline_picks, ceo2_material, rietveld_params
):

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

    # Just exercise this
    rietveld.params_vary_on()

    # First, only vary the scale
    rietveld.params_vary_off()
    params['scale'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.081

    # Next, vary the lattice constant
    rietveld.params_vary_off()
    params['CeO2_a'].vary = True
    rietveld.Refine()

    # Next, U, V, and W
    rietveld.params_vary_off()
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    rietveld.Refine()

    # Next, X and Y
    rietveld.params_vary_off()
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    rietveld.Refine()

    # Next, U, V, W, X and Y
    rietveld.params_vary_off()
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.072

    # Next, the debye-waller constants
    rietveld.params_vary_off()
    params['CeO2_O1_dw'].vary = True
    params['CeO2_Ce1_dw'].vary = True
    rietveld.Refine()
    assert rietveld.Rwp < 0.07

    # Finally, everything
    rietveld.params_vary_off()
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
    lebail.params_vary_off()
    params['CeO2_a'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.25

    # Next, U, V, and W
    lebail.params_vary_off()
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.1

    # Next, X and Y
    lebail.params_vary_off()
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.08

    # Next, U, V, W, X and Y
    lebail.params_vary_off()
    params['U'].vary = True
    params['V'].vary = True
    params['W'].vary = True
    params['CeO2_X'].vary = True
    params['CeO2_Y'].vary = True
    lebail.RefineCycle()
    assert lebail.Rwp < 0.07

    # Finally, everything
    lebail.params_vary_off()
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


def test_lebail_no_vary_preserves_edits(
    expt_spectrum, ceo2_material, lebail_params
):
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

    # Run once with a varying param to populate self.res
    params['CeO2_a'].vary = True
    lebail.RefineCycle()

    # Now turn off all vary flags and manually edit U
    lebail.params_vary_off()
    edited_value = params['U'].value + 1.0
    params['U'].value = edited_value

    lebail.RefineCycle()

    assert params['U'].value == edited_value
    assert lebail.U == edited_value


def test_lebail_no_vary_preserves_material_edits(
    expt_spectrum, ceo2_material, lebail_params
):
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

    # Run once with a varying param
    params['CeO2_a'].vary = True
    lebail.RefineCycle()

    # Manually edit the lattice parameter, but vary a different param
    lebail.params_vary_off()
    params['U'].vary = True
    edited_value = params['CeO2_a'].value + 0.001
    params['CeO2_a'].value = edited_value

    lebail.RefineCycle()

    mat = lebail.phases['CeO2']
    assert np.isclose(mat.lparms[0], edited_value)


def test_rietveld_no_vary_preserves_edits(
    expt_spectrum, spline_picks, ceo2_material, rietveld_params
):
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

    # Run once with a varying param to populate self.res
    params['scale'].vary = True
    rietveld.Refine()

    # Now turn off all vary flags and manually edit U
    rietveld.params_vary_off()
    edited_value = params['U'].value + 1.0
    params['U'].value = edited_value

    rietveld.Refine()

    assert params['U'].value == edited_value
    assert rietveld.U == edited_value


def test_rietveld_no_vary_preserves_material_edits(
    expt_spectrum, spline_picks, ceo2_material, rietveld_params
):
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

    # Run once with a varying param
    params['scale'].vary = True
    rietveld.Refine()

    # Manually edit the lattice parameter, but vary a different param
    rietveld.params_vary_off()
    params['scale'].vary = True
    edited_value = params['CeO2_a'].value + 0.001
    params['CeO2_a'].value = edited_value

    rietveld.Refine()

    # In Rietveld, phases are nested by wavelength type
    for lpi in rietveld.phases['CeO2']:
        mat = rietveld.phases['CeO2'][lpi]
        assert np.isclose(mat.lparms[0], edited_value)
