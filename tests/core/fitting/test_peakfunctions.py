# tests/test_peakfunctions_simplified.py
import numpy as np
import pytest
from scipy.special import erfc as scipy_erfc
import hexrd.core.fitting.peakfunctions as pf

# ----------------- small utilities -----------------


def finite_diff_derivative(func, p, x, idx, rel_step=1e-6):
    p = p.astype(float)
    base = max(1.0, abs(p[idx]))
    delta = rel_step * base
    p_plus = p.copy()
    p_minus = p.copy()
    p_plus[idx] += delta
    p_minus[idx] -= delta
    return (func(p_plus, x) - func(p_minus, x)) / (2.0 * delta)


# ----------------- special functions -----------------


def test_erfc_and_exp1exp_smoke():
    vals = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 3.2], dtype=float)
    got = pf.erfc(vals)
    assert got.shape == vals.shape
    assert np.all((got >= 0.0) & (got <= 2.0))
    assert np.allclose(got, scipy_erfc(vals), rtol=1e-6, atol=1e-6)

    x_pos = np.array([0.1, 0.5, 1.0, 2.0], dtype=float)
    for fn in (pf.exp1exp_under1, pf.exp1exp_over1, pf.exp1exp):
        arr = fn(x_pos)
        assert arr.shape == x_pos.shape
        assert np.all(np.isfinite(arr))


# ----------------- 1D Gaussian -----------------


def test_unit_and_no_bg_gaussian_properties_and_symmetry():
    p_unit = np.array([0.0, 2.0])
    x = np.linspace(-5, 5, 101)
    u = pf._unit_gaussian(p_unit, x)
    assert u.shape == x.shape
    assert np.all((u >= 0.0) & (u <= 1.0))
    assert np.isclose(x[np.argmax(u)], p_unit[0], atol=1e-6)
    left = u[x < 0]
    right = u[x > 0][::-1]
    assert np.allclose(left, right, atol=1e-6)

    sigma = p_unit[1] / pf.gauss_width_fact
    expected = np.exp(-((x - p_unit[0]) ** 2) / (2.0 * sigma**2))
    np.testing.assert_allclose(u, expected, rtol=1e-12, atol=1e-12)

    # no-bg scales correctly
    p_no_bg = np.array([2.0, 0.0, 2.0])
    g = pf._gaussian1d_no_bg(p_no_bg, x)
    assert np.all(g >= 0.0) and np.all(g <= p_no_bg[0])
    np.testing.assert_allclose(g, p_no_bg[0] * expected, rtol=1e-12)


def test_gaussian1d_full_and_derivatives():
    p = np.array([2.0, 0.5, 1.5, 0.1, -0.2])
    x = np.linspace(-2, 3, 200)
    out = pf.gaussian1d(p, x)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))

    A, x0, FWHM, c0, c1 = p
    sigma = FWHM / pf.gauss_width_fact
    expected = A * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) + c0 + c1 * x
    np.testing.assert_allclose(out, expected, rtol=1e-12)

    dmat = pf.gaussian1d_deriv(p, x)
    assert dmat.shape == (5, x.size)
    np.testing.assert_allclose(dmat[3, :], 1.0)
    np.testing.assert_allclose(dmat[4, :], x)
    analytic_no_bg = pf._gaussian1d_no_bg_deriv(p[:3], x)
    np.testing.assert_allclose(dmat[0:3, :], analytic_no_bg)

    for i in range(3):
        fd = finite_diff_derivative(
            pf._gaussian1d_no_bg, p[:3], x, i, rel_step=1e-6
        )
        np.testing.assert_allclose(
            analytic_no_bg[i, :], fd, rtol=1e-5, atol=1e-8
        )


# ----------------- Lorentzian -----------------


def test_lorentzian_unit_and_full_and_deriv():
    A = 3.0
    x0 = 0.7
    FWHM = 1.2
    c0 = 0.05
    c1 = -0.01
    x = np.linspace(-5.0, 5.0, 501)

    lorentz_width_fact = __import__(
        'hexrd.core.fitting.peakfunctions', fromlist=['lorentz_width_fact']
    ).lorentz_width_fact
    gamma = FWHM / lorentz_width_fact
    expected_unit = (gamma**2) / ((x - x0) ** 2 + gamma**2)

    unit_out = pf._unit_lorentzian(np.array([x0, FWHM]), x)
    np.testing.assert_allclose(unit_out, expected_unit, rtol=1e-12)
    assert np.all(unit_out > 0.0) and np.all(unit_out <= 1.0 + 1e-12)

    no_bg_out = pf._lorentzian1d_no_bg(np.array([A, x0, FWHM]), x)
    np.testing.assert_allclose(no_bg_out, A * expected_unit, rtol=1e-12)

    full_out = pf.lorentzian1d(np.array([A, x0, FWHM, c0, c1]), x)
    np.testing.assert_allclose(
        full_out, A * expected_unit + c0 + c1 * x, rtol=1e-12
    )
    assert int(np.argmax(no_bg_out)) == int(np.argmin(np.abs(x - x0)))

    p = np.array([2.5, -0.3, 0.9], dtype=float)
    x_small = np.linspace(-4.0, 4.0, 301)
    analytic = pf._lorentzian1d_no_bg_deriv(p, x_small)
    for i in range(3):
        fd = finite_diff_derivative(
            pf._lorentzian1d_no_bg, p, x_small, i, rel_step=1e-7
        )
        np.testing.assert_allclose(analytic[i, :], fd, rtol=1e-5, atol=1e-9)

    p_full = np.array([2.5, -0.3, 0.9, 0.12, -0.02], dtype=float)
    dmat = pf.lorentzian1d_deriv(p_full, x_small)
    assert dmat.shape == (5, x_small.size)
    np.testing.assert_allclose(dmat[3, :], 1.0)
    np.testing.assert_allclose(dmat[4, :], x_small)
    np.testing.assert_allclose(
        dmat[0:3, :], pf._lorentzian1d_no_bg_deriv(p_full[:3], x_small)
    )


# ----------------- Pseudo-Voigt and split variants -----------------


def test_pvoigt_and_split_variants():
    A = 1.5
    x0 = 0.3
    FWHM = 1.0
    n = 0.4
    c0 = 0.07
    c1 = -0.02
    x = np.linspace(-5.0, 5.0, 501)
    unit_expected = n * pf._unit_gaussian(np.array([x0, FWHM]), x) + (
        1.0 - n
    ) * pf._unit_lorentzian(np.array([x0, FWHM]), x)
    np.testing.assert_allclose(
        pf._unit_pvoigt1d(np.array([x0, FWHM, n]), x), unit_expected
    )

    no_bg = pf._pvoigt1d_no_bg(np.array([A, x0, FWHM, n]), x)
    np.testing.assert_allclose(no_bg, A * unit_expected)

    full = pf.pvoigt1d(np.array([A, x0, FWHM, n, c0, c1]), x)
    np.testing.assert_allclose(full, A * unit_expected + c0 + c1 * x)
    assert int(np.argmax(no_bg)) == int(np.argmin(np.abs(x - x0)))

    p_g = np.array([A, x0, FWHM, 1.0, c0, c1], dtype=float)
    np.testing.assert_allclose(
        pf._unit_pvoigt1d(p_g[[1, 2, 3]], x),
        pf._unit_gaussian(np.array([x0, FWHM]), x),
    )
    np.testing.assert_allclose(
        pf.pvoigt1d(p_g, x),
        A * pf._unit_gaussian(np.array([x0, FWHM]), x) + c0 + c1 * x,
    )

    p_l = np.array([A, x0, FWHM, 0.0, c0, c1], dtype=float)
    np.testing.assert_allclose(
        pf._unit_pvoigt1d(p_l[[1, 2, 3]], x),
        pf._unit_lorentzian(np.array([x0, FWHM]), x),
    )
    np.testing.assert_allclose(
        pf.pvoigt1d(p_l, x),
        A * pf._unit_lorentzian(np.array([x0, FWHM]), x) + c0 + c1 * x,
    )

    p_zeroA = np.array([0.0, x0, FWHM, 0.5, c0, c1], dtype=float)
    np.testing.assert_allclose(pf.pvoigt1d(p_zeroA, x), c0 + c1 * x)

    A = 1.2
    x0 = 0.0
    FWHM_minus = 0.8
    FWHM_plus = 1.6
    n_minus = 0.2
    n_plus = 0.7
    p_no_bg = np.array(
        [A, x0, FWHM_minus, FWHM_plus, n_minus, n_plus], dtype=float
    )
    x_small = np.linspace(-2.0, 2.0, 401)
    out_no_bg = pf._split_pvoigt1d_no_bg(p_no_bg, x_small)
    expected = np.zeros_like(x_small)
    xr = x_small >= x0
    xl = x_small < x0
    expected[xr] = A * pf._unit_pvoigt1d(
        np.array([x0, FWHM_plus, n_plus]), x_small[xr]
    )
    expected[xl] = A * pf._unit_pvoigt1d(
        np.array([x0, FWHM_minus, n_minus]), x_small[xl]
    )
    np.testing.assert_allclose(out_no_bg, expected)
    np.testing.assert_allclose(
        pf.split_pvoigt1d(
            np.hstack([p_no_bg, np.array([0.05, -0.01])]), x_small
        ),
        expected + 0.05 - 0.01 * x_small,
    )

    p_sym = np.array([1.1, -0.5, 1.1, 1.1, 0.3, 0.3], dtype=float)
    x_sym = np.linspace(-2.0, 2.0, 201)
    np.testing.assert_allclose(
        pf._split_pvoigt1d_no_bg(p_sym, x_sym),
        p_sym[0]
        * pf._unit_pvoigt1d(np.array([p_sym[1], p_sym[2], p_sym[4]]), x_sym),
    )

    p_zeroA = np.array(
        [0.0, x0, FWHM_minus, FWHM_plus, 0.5, 0.5, 0.05, -0.01], dtype=float
    )
    np.testing.assert_allclose(
        pf.split_pvoigt1d(p_zeroA, x_small), 0.05 - 0.01 * x_small
    )


# ----------------- pink-beam DCS utilities & composition -----------------


def test_calc_alpha_beta_and_mixing_factor_behavior_and_wrappers():
    a0, a1 = 1.0, 0.5
    b0, b1 = -0.2, 2.0
    for x0 in [0.0, 30.0, 60.0, 120.0]:
        expected_alpha = a0 + a1 * np.tan(np.radians(0.5 * x0))
        expected_beta = b0 + b1 * np.tan(np.radians(0.5 * x0))
        got_alpha = pf._calc_alpha(np.array([a0, a1], dtype=float), x0)
        got_beta = pf._calc_beta(np.array([b0, b1], dtype=float), x0)
        assert np.isfinite(got_alpha) and np.isfinite(got_beta)
        assert np.allclose(got_alpha, expected_alpha, atol=1e-12)
        assert np.allclose(got_beta, expected_beta, atol=1e-12)

    for fwhm_g, fwhm_l in [(0.1, 0.2), (0.5, 0.5), (1.0, 2.0), (2.5, 0.3)]:
        eta, fwhm = pf._mixing_factor_pv(
            np.float64(fwhm_g), np.float64(fwhm_l)
        )
        assert np.isfinite(eta) and np.isfinite(fwhm)
        assert 0.0 <= eta <= 1.0
        poly = (
            fwhm_g**5
            + 2.69269 * fwhm_g**4 * fwhm_l
            + 2.42843 * fwhm_g**3 * fwhm_l**2
            + 4.47163 * fwhm_g**2 * fwhm_l**3
            + 0.07842 * fwhm_g * fwhm_l**4
            + fwhm_l**5
        )
        expected_fwhm = poly**0.20
        assert np.allclose(fwhm, expected_fwhm, rtol=1e-12)

    A = 1.5
    x0 = 0.2
    alpha0, alpha1 = 0.05, 0.001
    beta0, beta1 = 0.08, -0.0005
    fwhm_g = 0.45
    fwhm_l = 0.9
    b0, b1 = 0.03, -0.01
    p_full = np.array(
        [A, x0, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l, b0, b1],
        dtype=float,
    )
    x = np.linspace(-1.0, 1.0, 201)

    alpha = pf._calc_alpha(np.array([alpha0, alpha1], dtype=float), x0)
    beta = pf._calc_beta(np.array([beta0, beta1], dtype=float), x0)
    p_g = np.hstack(([A, x0], np.array([alpha, beta, fwhm_g], dtype=float)))
    p_l = np.hstack(([A, x0], np.array([alpha, beta, fwhm_l], dtype=float)))
    eta, _ = pf._mixing_factor_pv(fwhm_g, fwhm_l)
    G = pf._gaussian_pink_beam(p_g, x)
    L = pf._lorentzian_pink_beam(p_l, x)
    expected_full = eta * L + (1.0 - eta) * G + b0 + b1 * x
    out_wrapper = pf.pink_beam_dcs(p_full, x)
    np.testing.assert_allclose(
        out_wrapper, expected_full, rtol=1e-8, atol=1e-12
    )

    x_lm = np.linspace(-0.8, 0.8, 101)
    out_lm = pf.pink_beam_dcs_lmfit(
        x_lm, A, x0, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l
    )
    alpha_lm = pf._calc_alpha(np.array([alpha0, alpha1], dtype=float), x0)
    beta_lm = pf._calc_beta(np.array([beta0, beta1], dtype=float), x0)
    p_g_lm = np.hstack(
        ([A, x0], np.array([alpha_lm, beta_lm, fwhm_g], dtype=float))
    )
    p_l_lm = np.hstack(
        ([A, x0], np.array([alpha_lm, beta_lm, fwhm_l], dtype=float))
    )
    eta_lm, _ = pf._mixing_factor_pv(fwhm_g, fwhm_l)
    expected_lm = eta_lm * pf._lorentzian_pink_beam(p_l_lm, x_lm) + (
        1.0 - eta_lm
    ) * pf._gaussian_pink_beam(p_g_lm, x_lm)
    np.testing.assert_allclose(out_lm, expected_lm, rtol=1e-8, atol=1e-12)

    p = np.array(
        [10.0, 0.5, 0.01, 0.02, 0.03, 0.01, 0.15, 0.20, 1.0, 0.1], dtype=float
    )
    x_small = np.linspace(-1.0, 2.0, 7)
    y_no_bg = pf._pink_beam_dcs_no_bg(p[:-2], x_small)
    y_full = pf.pink_beam_dcs(p, x_small)
    assert y_no_bg.shape == x_small.shape and y_full.shape == x_small.shape
    assert np.all(np.isfinite(y_no_bg)) and np.all(np.isfinite(y_full))
    assert np.allclose(y_full - y_no_bg, p[-2] + p[-1] * x_small)
    y_lm = pf.pink_beam_dcs_lmfit(x_small, *p[:8])
    assert np.allclose(y_lm, y_no_bg, rtol=1e-12, atol=1e-12)


# ----------------- other utilities: tanh, 2d transforms, 2d/3d shapes -----------------


def test_tanh_and_2d_coord_transform_and_gauss2d_rot_and_3d():
    p = np.array([1.0, 0.0, 1.0])
    x = np.linspace(-2, 2, 5)
    out = pf.tanh_stepdown_nobg(p, x)
    assert out.shape == x.shape
    assert out[0] > out[-1]
    assert np.isclose(out[len(x) // 2], 0.5, atol=1e-2)

    xmat = np.array([[0.0, 1.0], [2.0, 3.0]])
    ymat = np.array([[5.0, 6.0], [7.0, 8.0]])
    _, _, xp, yp = pf._2d_coord_transform(0.0, 1.0, 2.0, xmat, ymat)
    assert np.allclose(xp, xmat) and np.allclose(yp, ymat)
    x1 = np.array([[1.0, 0.0]])
    y1 = np.array([[0.0, 1.0]])
    xp2, yp2 = pf._2d_coord_transform(np.pi / 2, 1.0, 0.0, x1, y1)[2:]
    assert np.allclose(xp2, y1) and np.allclose(yp2, -x1)

    p2 = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    xz = np.zeros((2, 2))
    yz = np.zeros((2, 2))
    out2 = pf._gaussian2d_no_bg(p2, xz, yz)
    assert out2.shape == (2, 2)
    assert np.allclose(out2, out2[0, 0])

    p_rot = np.array([1.0, 0.0, 0.0, 1.0, 1.0, np.pi / 3])
    out_rot = pf._gaussian2d_rot_no_bg(
        p_rot, np.zeros((3, 3)), np.zeros((3, 3))
    )
    assert out_rot.shape == (3, 3) and np.isfinite(out_rot).all()

    p_full = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.5, -0.5])
    out_g2 = pf.gaussian2d(p_full, np.zeros((2, 2)), np.zeros((2, 2)))
    base = pf._gaussian2d_no_bg(p_full[:5], np.zeros((2, 2)), np.zeros((2, 2)))
    bg = 2.0 + 0.5 * np.zeros((2, 2)) - 0.5 * np.zeros((2, 2))
    assert np.allclose(out_g2, base + bg)

    p_gr = np.array([1.0, 0.0, 0.0, 1.0, 1.0, np.pi / 4, 1.0, 0.3, 0.2])
    out_gr = pf.gaussian2d_rot(p_gr, np.zeros((3, 3)), np.zeros((3, 3)))
    base_r = pf._gaussian2d_rot_no_bg(
        p_gr[:6], np.zeros((3, 3)), np.zeros((3, 3))
    )
    assert np.allclose(out_gr, base_r + 1.0 + 0.3 * 0 - 0.2 * 0)

    xq = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    yq = np.array([[-1.0, -1.0], [1.0, 1.0]])
    p2d = np.array([1.0, 0.0, 0.0, 1.0, 2.0, 1.5, 2.5, 0.1, 0.2, 0.3, 0.4])
    out2d = pf._split_pvoigt2d_no_bg(p2d, xq, yq)
    assert (
        out2d.shape == (2, 2)
        and np.isfinite(out2d).all()
        and (out2d >= 0).all()
    )

    out2drot = pf._split_pvoigt2d_rot_no_bg(
        np.hstack((p2d, np.pi / 6)), np.zeros((3, 3)), np.zeros((3, 3))
    )
    assert out2drot.shape == (3, 3) and np.isfinite(out2drot).all()

    p_split_bg = np.hstack((p2d, np.pi / 7, 1.0, 0.2, -0.1))
    out_sp_bg = pf.split_pvoigt2d_rot(
        p_split_bg, np.zeros((4, 4)), np.zeros((4, 4))
    )
    base_sp = pf._split_pvoigt2d_rot_no_bg(
        p_split_bg[:12], np.zeros((4, 4)), np.zeros((4, 4))
    )
    assert np.allclose(out_sp_bg, base_sp + 1.0 + 0.2 * 0 - 0.1 * 0)

    x3 = np.zeros((2, 2, 2))
    y3 = np.zeros_like(x3)
    z3 = np.zeros_like(x3)
    p3_no = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    p3_full = np.hstack((p3_no, np.array([0.5, 0.1, -0.2, 0.05], dtype=float)))
    out3_no = pf._gaussian3d_no_bg(p3_no, x3, y3, z3)
    assert (
        out3_no.shape == x3.shape
        and np.isfinite(out3_no).all()
        and (out3_no >= 0).all()
    )
    out3_full = pf.gaussian3d(p3_full, x3, y3, z3)
    bg3 = p3_full[7] + p3_full[8] * x3 + p3_full[9] * y3 + p3_full[10] * z3
    np.testing.assert_allclose(out3_full, out3_no + bg3, rtol=1e-12)


# ----------------- mpeak_1d multi-peak wrapper -----------------


def _build_peak_row(pktype):
    if pktype == "gaussian":
        return np.array([1.5, 0.2, 0.8], dtype=float)
    if pktype == "lorentzian":
        return np.array([1.2, -0.1, 0.6], dtype=float)
    if pktype == "pvoigt":
        return np.array([1.0, 0.5, 0.9, 0.3], dtype=float)
    if pktype == "split_pvoigt":
        return np.array([0.9, 0.0, 0.7, 1.1, 0.25, 0.75], dtype=float)
    if pktype == "pink_beam_dcs":
        return np.array(
            [1.3, 0.1, 0.01, 0.002, 0.02, -0.0003, 0.2, 0.4], dtype=float
        )
    raise ValueError


@pytest.mark.parametrize(
    "pktype",
    ["gaussian", "lorentzian", "pvoigt", "split_pvoigt", "pink_beam_dcs"],
)
def test_mpeak_1d_no_bg_and_backgrounds(pktype):
    npp = pf.mpeak_nparams_dict[pktype]
    num_pks = 2
    row = _build_peak_row(pktype)
    if len(row) < npp:
        row = np.hstack((row, np.zeros(npp - len(row), dtype=float)))
    else:
        row = row[:npp].copy()

    p_flat = np.hstack([row.copy() for _ in range(num_pks)])
    x = np.linspace(-2.0, 2.0, 201)

    out = pf._mpeak_1d_no_bg(p_flat, x, pktype, num_pks)
    assert out.shape == x.shape and np.isfinite(out).all()

    expected = np.zeros_like(x)
    for ii in range(num_pks):
        p_row = p_flat[ii * npp : (ii + 1) * npp]
        if pktype == "gaussian":
            expected += pf._gaussian1d_no_bg(p_row, x)
        elif pktype == "lorentzian":
            expected += pf._lorentzian1d_no_bg(p_row, x)
        elif pktype == "pvoigt":
            expected += pf._pvoigt1d_no_bg(p_row, x)
        elif pktype == "split_pvoigt":
            expected += pf._split_pvoigt1d_no_bg(p_row, x)
        elif pktype == "pink_beam_dcs":
            expected += pf._pink_beam_dcs_no_bg(p_row, x)

    np.testing.assert_allclose(out, expected, rtol=1e-12)

    out_lin = pf.mpeak_1d(
        np.hstack([p_flat, np.array([0.2, 0.01], dtype=float)]),
        x,
        pktype,
        num_pks,
        bgtype="linear",
    )
    np.testing.assert_allclose(out_lin, expected + 0.2 + 0.01 * x, rtol=1e-12)
    out_const = pf.mpeak_1d(
        np.hstack([p_flat, np.array([0.3], dtype=float)]),
        x,
        pktype,
        num_pks,
        bgtype="constant",
    )
    np.testing.assert_allclose(out_const, expected + 0.3, rtol=1e-12)
    out_quad = pf.mpeak_1d(
        np.hstack([p_flat, np.array([0.4, 0.005, -0.0001], dtype=float)]),
        x,
        pktype,
        num_pks,
        bgtype="quadratic",
    )
    np.testing.assert_allclose(
        out_quad, expected + 0.4 + 0.005 * x - 0.0001 * x**2, rtol=1e-12
    )


@pytest.mark.parametrize(
    "fwhm_g,fwhm_l",
    [
        (0.1, -0.01),
        (1.0, -0.1),
        (10.0, -1.0),
    ],
)
def test_mixing_factor_clamps_negative_eta_to_zero(fwhm_g, fwhm_l):
    """When computed eta < 0.0 the function should clamp it to 0.0."""
    eta, fwhm = pf._mixing_factor_pv(np.float64(fwhm_g), np.float64(fwhm_l))
    assert np.isfinite(fwhm)
    assert eta == pytest.approx(0.0, abs=0.0)


@pytest.mark.parametrize(
    "fwhm_g,fwhm_l",
    [
        (-1.0, 100.0),
        (-0.1, 10.0),
        (-0.01, 1.0),
    ],
)
def test_mixing_factor_clamps_large_eta_to_one(fwhm_g, fwhm_l):
    """When computed eta > 1.0 the function should clamp it to 1.0."""
    eta, fwhm = pf._mixing_factor_pv(np.float64(fwhm_g), np.float64(fwhm_l))
    assert np.isfinite(fwhm)
    assert eta == pytest.approx(1.0, rel=0.0, abs=0.0)
