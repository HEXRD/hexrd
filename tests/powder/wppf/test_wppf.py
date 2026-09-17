import copy
import json
from pathlib import Path

import lmfit
import numpy as np
import pytest

from hexrd.core.material import _angstroms, load_materials_hdf5, Material
from hexrd.powder.wppf import LeBail, Rietveld
from hexrd.powder.wppf import wppfsupport


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


def _fraction_params(values: dict[str, float], vary: list[bool]) -> lmfit.Parameters:
    params = lmfit.Parameters()
    names = [f'{k}_phase_fraction' for k in values]
    for name, value, v in zip(names, values.values(), vary):
        params.add(name, value, min=0, max=1, vary=v)
    params[names[-1]].expr = '1 - ' + ' - '.join(names[:-1])
    return params


def _fractions(params: lmfit.Parameters) -> list[float]:
    return [v.value for k, v in params.items() if k.endswith('_phase_fraction')]


def test_stick_breaking_params() -> None:
    # A varies, B is fixed, C is the remainder. A + B is over budget, so the
    # start gets projected onto the simplex (C clamps to 0).
    params = _fraction_params({'A': 0.9, 'B': 0.333, 'C': 0.0}, [True, False, False])
    fit_params = wppfsupport.add_stick_breaking_params(params)
    assert params['A_phase_fraction'].value == 0.9  # The original is untouched
    assert set(fit_params) - set(params) == {'_stick_A'}
    assert np.allclose(_fractions(fit_params), [0.667, 0.333, 0.0])
    assert np.isinf(fit_params['A_phase_fraction'].min)  # Nothing can clamp

    # Every value of t must stay physical, with B fixed
    for t in (0.0, 0.25, 1.0):
        fit_params['_stick_A'].value = t
        f = _fractions(fit_params)
        assert np.isclose(sum(f), 1) and min(f) >= 0 and f[1] == 0.333

    # Mapping back restores the original structure with values and stderr
    fit_params['A_phase_fraction'].stderr = 0.01
    result = wppfsupport.strip_stick_breaking_params(fit_params, params)
    assert list(result) == list(params)
    assert result['A_phase_fraction'].vary and result['A_phase_fraction'].expr is None
    assert result['C_phase_fraction'].expr == params['C_phase_fraction'].expr
    assert np.allclose(_fractions(result), [0.667, 0.333, 0])
    assert result['A_phase_fraction'].stderr == 0.01

    # All varying: t values reproduce the current fractions
    params = _fraction_params({'A': 0.5, 'B': 0.3, 'C': 0.2}, [True, True, False])
    fit_params = wppfsupport.add_stick_breaking_params(params)
    assert np.allclose(_fractions(fit_params), [0.5, 0.3, 0.2])
    assert np.isclose(fit_params['_stick_B'].value, 0.6)

    # Fixed fractions summing to more than 1 are a user error
    params = _fraction_params({'A': 0.7, 'B': 0.6, 'C': 0.0}, [False, False, False])
    with pytest.raises(ValueError, match='sum to more than 1'):
        wppfsupport.add_stick_breaking_params(params)

    # Narrowed bounds on free fractions are reset; fixed ones are kept
    params = _fraction_params({'A': 0.5, 'B': 0.3, 'C': 0.2}, [True, False, False])
    params['A_phase_fraction'].max = 0.6
    params['B_phase_fraction'].max = 0.4
    wppfsupport.reset_phase_fraction_bounds(params)
    assert params['A_phase_fraction'].max == 1
    assert params['B_phase_fraction'].max == 0.4


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
        fractions = np.array(_fractions(rietveld.params))
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
    assert not any(k.startswith('_stick_') for k in rietveld.res.params)
    assert rietveld.res.params['A_phase_fraction'].stderr > 0

    # Freeing B recovers the truth
    fractions = refine(rietveld, ['B_phase_fraction'])
    assert np.allclose(fractions, [0.95, 0.05, 0], atol=1e-4)
    assert rietveld.Rwp < 1e-4

    # A narrowed bound on a varying fraction (delta boundaries produce these)
    # is reset rather than clamping the result to a value the model never used
    rietveld = make_rietveld(synthetic)
    rietveld.params['A_phase_fraction'].max = 2 / 3
    vary = ['A_phase_fraction', 'B_phase_fraction', 'scale']
    assert np.allclose(refine(rietveld, vary), [0.95, 0.05, 0], atol=1e-4)
    assert rietveld.params['A_phase_fraction'].max == 1
