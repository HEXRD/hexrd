from __future__ import annotations

import numpy as np
import pytest

from hexrd.core.transforms.xfcapi import oscill_angles_of_hkls


@pytest.fixture
def bragg_inputs():
    chi = 0.0
    rmat_c = np.eye(3)
    bmat = np.eye(3) * 2 * np.pi
    hkls = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    return hkls, chi, rmat_c, bmat


def test_scalar_wavelength(bragg_inputs):
    hkls, chi, rmat_c, bmat = bragg_inputs
    o0, o1 = oscill_angles_of_hkls(hkls, chi, rmat_c, bmat, 0.2)
    assert o0.shape == (3, 3)
    assert o1.shape == (3, 3)


def test_array_matches_scalar(bragg_inputs):
    hkls, chi, rmat_c, bmat = bragg_inputs
    wl = 0.2
    o0_s, o1_s = oscill_angles_of_hkls(hkls, chi, rmat_c, bmat, wl)
    o0_a, o1_a = oscill_angles_of_hkls(
        hkls, chi, rmat_c, bmat, np.full(len(hkls), wl)
    )
    np.testing.assert_allclose(o0_s, o0_a, equal_nan=True)
    np.testing.assert_allclose(o1_s, o1_a, equal_nan=True)


def test_per_reflection_wavelength(bragg_inputs):
    hkls, chi, rmat_c, bmat = bragg_inputs
    wl_a, wl_b = 0.2, 0.21
    wl_varied = np.array([wl_a, wl_a, wl_b])

    o0_v, o1_v = oscill_angles_of_hkls(hkls, chi, rmat_c, bmat, wl_varied)

    o0_a, o1_a = oscill_angles_of_hkls(hkls, chi, rmat_c, bmat, wl_a)
    np.testing.assert_allclose(o0_v[:2], o0_a[:2], equal_nan=True)
    np.testing.assert_allclose(o1_v[:2], o1_a[:2], equal_nan=True)

    o0_b, o1_b = oscill_angles_of_hkls(hkls[2:3], chi, rmat_c, bmat, wl_b)
    np.testing.assert_allclose(o0_v[2:3], o0_b, equal_nan=True)
    np.testing.assert_allclose(o1_v[2:3], o1_b, equal_nan=True)


def test_different_wavelengths_give_different_results(bragg_inputs):
    hkls, chi, rmat_c, bmat = bragg_inputs
    o0_a, _ = oscill_angles_of_hkls(hkls[:1], chi, rmat_c, bmat, 0.2)
    o0_b, _ = oscill_angles_of_hkls(hkls[:1], chi, rmat_c, bmat, 0.25)
    valid_a = ~np.isnan(o0_a[:, 0])
    valid_b = ~np.isnan(o0_b[:, 0])
    if valid_a.any() and valid_b.any():
        assert not np.allclose(o0_a[valid_a], o0_b[valid_b])
