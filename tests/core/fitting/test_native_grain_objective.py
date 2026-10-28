import copy

import numpy as np
import pytest

from hexrd.core import distortion
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.transforms import xfcapi
from hexrd.hedm.fitting.grains import _FitGrainData, objFuncFitGrain


@pytest.mark.parametrize('count', [1, 3, 40])
@pytest.mark.parametrize('distorted', [False, True])
@pytest.mark.parametrize('tilted_beam', [False, True])
@pytest.mark.parametrize('reverse_panels', [False, True])
def test_native_objective_preserves_residuals_and_parameter_updates(
    count, distorted, tilted_beam, reverse_panels
):
    instrument = HEDMInstrument()
    instrument.chi = 0.17
    if tilted_beam:
        beam = np.array([0.1, -0.2, -1.0])
        instrument.beam_vector = beam / np.linalg.norm(beam)
        instrument.active_beam['energy_correction'] = {
            'axis': 'y',
            'intercept': 0.0,
            'slope': 200.0,
        }
    panel = next(iter(instrument.detectors.values()))
    panel.saturation_level = 65535
    panel.tilt = [0.03, -0.02, 0.01]
    if distorted:
        panel.distortion = distortion.get_mapping(
            'GE_41RT', [0.01, -0.005, 0.001, 2.0, 2.0, 2.0]
        )
    rng = np.random.default_rng(91)
    bmat = np.diag([0.2, 0.3, 0.4])
    wavelength = 0.2
    full = np.r_[0.1, -0.2, 0.3, 0.2, -0.4, 0.1, 1.0, 1.0, 1.0, 0.0001, -0.0002, 0.0003]
    hkls = rng.integers(-5, 6, (200, 3)).astype(float)
    rc = xfcapi.make_rmat_of_expmap(full[:3])
    first, second = xfcapi.oscill_angles_of_hkls(
        hkls,
        instrument.chi,
        rc,
        bmat,
        wavelength,
        v_inv=full[6:],
        beam_vec=instrument.beam_vector,
        eta_vec=instrument.eta_vector,
    )
    valid = np.all(np.isfinite(first), axis=1)
    hkls = hkls[valid][:count]
    omes = np.where(
        np.arange(count) % 2, first[valid][:count, 2], second[valid][:count, 2]
    )
    measured = np.column_stack((rng.uniform(-5, 5, (count, 2)), omes + 0.001))
    reflections = {
        next(iter(instrument.detectors)): {'hkls': hkls.T, 'meas_xyo': measured}
    }
    if reverse_panels:
        second = copy.deepcopy(panel)
        second.tvec = panel.tvec + [0.2, -0.1, 0.0]
        instrument.detectors['second'] = second
        reflections = {
            'second': {'hkls': hkls[::-1].T, 'meas_xyo': measured[::-1]},
            **reflections,
        }
    data = _FitGrainData(reflections, instrument, bmat)
    native = data.native
    assert native.residual(full, wavelength) is not None
    for flags in (np.ones(12, dtype=bool), np.arange(12) % 2 == 0):
        for repeat in range(5):
            varied = full + rng.normal(size=12) * 1e-7
            if repeat < 3:
                varied = full.copy()
                varied[repeat + 3] += 1e-7
            fit = varied[flags]
            expected_full = full.copy()
            data.native = None
            expected = objFuncFitGrain(
                fit, expected_full, flags, instrument, data, bmat, wavelength, None
            )
            data.native = native
            actual_full = full.copy()
            actual = objFuncFitGrain(
                fit, actual_full, flags, instrument, data, bmat, wavelength, None
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_full, expected_full)
