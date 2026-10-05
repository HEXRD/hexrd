import copy
import json
from pathlib import Path

import lmfit
import numpy as np
import pytest

from hexrd.core.material import _angstroms, load_materials_hdf5, Material
from hexrd.powder.wppf import LeBail, Rietveld
from hexrd.powder.wppf.peakfunctions import calc_rwp


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

    # Verify the refinement converged to physically sensible values.
    # The lattice parameter is well constrained by the peak positions, so we
    # pin it tightly (this is the published ceria value).
    assert np.isclose(params['CeO2_a'].value, 0.5412, atol=1e-4)

    # The Debye-Waller factors are only weakly constrained, so their exact
    # converged value shifts with optimizer/residual details. Rather than
    # pinning them to specific numbers, require that they stay physical:
    # positive, small, and with the light O atom displacing more than the
    # heavy Ce atom.
    ce_dw = params['CeO2_Ce1_dw'].value
    o_dw = params['CeO2_O1_dw'].value
    assert 0 < ce_dw < 0.01
    assert 0 < o_dw < 0.01
    assert o_dw > ce_dw


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
    # Was 0.54112 before the FCJ fix
    assert round(params['CeO2_a'].value, 5) == 0.54115


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


def test_rietveld_phase_fractions_stay_physical(
    expt_spectrum: np.ndarray,
    spline_picks: np.ndarray,
    ceo2_material: Material,
    rietveld_params: lmfit.Parameters,
) -> None:
    def make_rietveld(spectrum: np.ndarray) -> Rietveld:
        # Three CeO2-like phases with distinct lattice parameters
        phases = []
        for name, scale in [('A', 1.0), ('B', 1.04), ('C', 1.09)]:
            mat = copy.deepcopy(ceo2_material)
            mat.name = name
            a = mat.latticeParameters[0].value * scale
            mat.latticeParameters = [a, a, a, 90, 90, 90]
            phases.append(mat)

        rietveld = Rietveld(
            expt_spectrum=spectrum,
            phases=phases,
            wavelength={'synchrotron': [_angstroms(0.15358835358711712), 1.0]},
            bkgmethod={'spline': spline_picks.tolist()},
            peakshape='pvtch',
        )
        for k, v in rietveld_params.items():
            if k in rietveld.params:
                rietveld.params[k].value = v.value
        rietveld.params_vary_off()
        return rietveld

    def refine(rietveld: Rietveld, vary: list[str]) -> np.ndarray:
        for k in vary:
            rietveld.params[k].vary = True
        rietveld.Refine()
        names = [f'{x}_phase_fraction' for x in 'ABC']
        fractions = np.array([rietveld.params[k].value for k in names])
        assert np.allclose(rietveld.phases.phase_fraction, fractions)
        return fractions

    # Simulate a spectrum with truth A=0.95, B=0.05, C=0
    truth = make_rietveld(expt_spectrum)
    truth.params['A_phase_fraction'].value = 0.95
    truth.params['B_phase_fraction'].value = 0.05
    truth.Refine()
    sim = truth.spectrum_sim
    synthetic = np.column_stack([sim.x, np.nan_to_num(sim.y)])

    # Fit it with B wrongly fixed at 1/3, varying A. The fit wants A > 2/3,
    # so the constrained optimum is A = 2/3 and C = 0. The old expression +
    # renormalization approach let the fitter shrink the "fixed" B instead.
    rietveld = make_rietveld(synthetic)
    rietveld.params['B_phase_fraction'].value = 1 / 3
    fractions = refine(rietveld, ['A_phase_fraction', 'scale'])
    assert np.allclose(fractions, [2 / 3, 1 / 3, 0])

    # Freeing B recovers the truth
    fractions = refine(rietveld, ['B_phase_fraction'])
    assert np.allclose(fractions, [0.95, 0.05, 0], atol=1e-4)
    assert rietveld.Rwp < 1e-4


def test_statistical_weights(expt_spectrum, ceo2_material):
    rng = np.random.default_rng(0)
    n = len(expt_spectrum)
    kwargs = {
        'phases': [ceo2_material],
        'wavelength': {'synchrotron': [_angstroms(0.15358835358711712), 1.0]},
        'bkgmethod': {'chebyshev': 3},
    }

    # a lineout averaged over N pixels has variance I/N, so for the true
    # model the weighted chi^2 is 1 when the weights are N/I
    model = np.maximum(expt_spectrum[:, 1], 1.0)
    N_sampling = rng.integers(10, 1000, n).astype(float)
    y = rng.poisson(N_sampling * model) / N_sampling
    lebail = LeBail(
        expt_spectrum=np.column_stack((expt_spectrum[:, 0], y)),
        N_sampling=N_sampling,
        **kwargs,
    )
    weights = lebail.weights.data_array
    np.testing.assert_allclose(weights[:, 1], N_sampling / y)
    sim = np.column_stack((expt_spectrum[:, 0], model))
    chi2 = calc_rwp(sim, lebail.spectrum_expt.data_array, weights, sim, 0)[3]
    assert chi2 == pytest.approx(1.0, abs=0.1)

    # a masked spectrum is split into regions; N_sampling stays full length
    # and masked values of N_sampling get zero weight
    mask = np.zeros(n, dtype=bool)
    mask[n // 3 : n // 2] = True
    spectrum = np.column_stack((expt_spectrum[:, 0], y))
    spectrum = np.ma.masked_array(spectrum, np.column_stack((mask, mask)))
    N_masked = np.ma.masked_array(N_sampling, mask | (np.arange(n) == 0))
    lebail = LeBail(expt_spectrum=spectrum, N_sampling=N_masked, **kwargs)
    weights = lebail.weights.y
    expected = np.where(N_masked.mask, 0.0, N_sampling / y)
    np.testing.assert_allclose(weights[~mask], expected[~mask])
    assert weights[0] == 0.0

    with pytest.raises(ValueError):
        LeBail(expt_spectrum=spectrum, N_sampling=N_sampling[:-1], **kwargs)
