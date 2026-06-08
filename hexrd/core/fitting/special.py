import math
import numpy as np
from numba import njit, vectorize, complex128, float64
from hexrd.core.constants import c_erf

_EULER_GAMMA = 0.577215664901532860606512090082402431
_EPS = 2.220446049250313e-16
_TINY = 1.0e-300
_MAX_ITER = 1000

_SMALL_SERIES_CUTOFF = 4.0
_LEFT_SERIES_CUTOFF = 100.0
_MAX_LOG = 709.782712893384  # approximately log(max float64)


# Fixed N=11 modified trapezoidal-rule approximation.
# h = sqrt(pi / (N + 1)), H = pi / h
_NTRAP = 11
_H_STEP = math.sqrt(math.pi / (_NTRAP + 1.0))
_H_CAP = math.pi / _H_STEP

_MAX_EXP = 709.782712893384  # approx log(float64 max)
_MIN_EXP = -745.1332191019411  # exp(x) underflows to 0 below this


@njit(cache=True, nogil=True)
def erfc(x: np.ndarray) -> np.ndarray:
    # save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1, a2, a3, a4, a5, p = c_erf

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    erf = sign * y  # erf(-x) = -erf(x)
    return 1.0 - erf


@njit(cache=True, nogil=True)
def _complex_log_principal_numba(z: complex) -> complex:
    """
    Principal complex log, written manually so signed zero in imag
    is handled through atan2.
    """
    return complex(
        math.log(math.hypot(z.real, z.imag)),
        math.atan2(z.imag, z.real),
    )


@njit(cache=True, nogil=True)
def _exp_neg_complex_numba(z: complex) -> complex:
    """
    Compute exp(-z) with simple overflow/underflow guards.
    """
    x = z.real
    y = z.imag

    if x > 745.0:
        return complex(0.0, 0.0)

    if x < -_MAX_LOG:
        # The true result is generally outside float64 range.
        return complex(math.inf, math.nan)

    scale = math.exp(-x)
    return complex(scale * math.cos(y), -scale * math.sin(y))


@njit(cache=True, nogil=True)
def _exp1_series_numba(z: complex) -> complex:
    """
    Logarithmic power series:

        E1(z) = -gamma - log(z) - sum_{k=1}^inf (-z)^k / (k k!)

    Valid on the principal branch, away from z = 0.
    """
    term = -z
    s = term

    for k in range(2, _MAX_ITER + 1):
        term *= -z / k
        add = term / k
        s += add

        if abs(add) <= _EPS * max(1.0, abs(s)):
            break

    return complex(-_EULER_GAMMA, 0.0) - _complex_log_principal_numba(z) - s


@njit(cache=True, nogil=True)
def _exp1_contfrac_numba(z: complex) -> complex:
    """
    Continued fraction via modified Lentz iteration.
    Good away from the origin and away from the difficult part
    of the negative-real branch cut.
    """
    b = z + 1.0
    c = complex(1.0 / _TINY, 0.0)
    d = 1.0 / b
    h = d

    for i in range(1, _MAX_ITER + 1):
        a = -float(i * i)
        b += 2.0

        denom = a * d + b
        if abs(denom) < _TINY:
            denom = complex(_TINY, 0.0)
        d = 1.0 / denom

        c = b + a / c
        if abs(c) < _TINY:
            c = complex(_TINY, 0.0)

        delta = c * d
        h *= delta

        if abs(delta - 1.0) <= _EPS:
            break

    return h * _exp_neg_complex_numba(z)


@njit(cache=True, nogil=True)
def exp1_complex_scalar_numba(z: complex) -> complex:
    """
    Numba-compatible complex exponential integral E1(z).

    This is the complex analogue of scipy.special.exp1 for complex128 input.
    For exact negative-real complex inputs, the side of the branch cut is
    selected by the sign of the imaginary zero:
        complex(-x, +0.0) -> -i*pi side
        complex(-x, -0.0) -> +i*pi side
    """
    x = z.real
    y = z.imag

    if math.isnan(x) or math.isnan(y):
        return complex(math.nan, math.nan)

    if x == 0.0 and y == 0.0:
        return complex(math.inf, 0.0)

    if math.isinf(x) or math.isinf(y):
        if x == math.inf and not math.isinf(y):
            return complex(0.0, 0.0)
        return complex(math.nan, math.nan)

    az = abs(z)

    near_negative_real_axis = (
        x < 0.0 and az <= _LEFT_SERIES_CUTOFF and abs(y) <= max(2.0, 0.25 * abs(x))
    )

    if az <= _SMALL_SERIES_CUTOFF or near_negative_real_axis:
        return _exp1_series_numba(z)

    out = _exp1_contfrac_numba(z)

    # If the continued fraction path is used exactly on the negative-real
    # branch cut, insert the correct signed-zero branch jump.
    if x < 0.0 and y == 0.0:
        if math.copysign(1.0, y) < 0.0:
            out += complex(0.0, math.pi)
        else:
            out += complex(0.0, -math.pi)

    return out


@vectorize([complex128(complex128)], nopython=True)
def exp1_complex_numba(z: complex) -> complex:
    """
    Vectorized complex ufunc version.
    """
    return exp1_complex_scalar_numba(z)


@njit(cache=True, nogil=True)
def exp1exp(z: np.ndarray) -> np.ndarray:
    """
    E1(z) * exp(z), the combination used by the back-to-back-exponential
    pink-beam Lorentzian (Von Dreele et al., J. Appl. Cryst. (2021) 54, 3-6).
    The exp(z) factor cancels the exponential growth of E1, keeping the
    result well-behaved across the peak.
    """
    return np.exp(z) * exp1_complex_numba(z)


@njit(cache=True, nogil=True)
def exp1_real_scalar_numba(x: float) -> float:
    """
    Real-valued SciPy-like wrapper.

    For real negative x, scipy.special.exp1(x) returns nan.
    For complex negative x + 0j, use exp1_complex_scalar_numba instead.
    """
    if math.isnan(x):
        return math.nan

    if x < 0.0:
        return math.nan

    if x == 0.0:
        return math.inf

    if math.isinf(x):
        return 0.0

    return exp1_complex_scalar_numba(complex(x, 0.0)).real


@vectorize([float64(float64)], nopython=True)
def exp1_real_numba(x: float) -> float:
    """
    Vectorized real ufunc version.
    """
    return exp1_real_scalar_numba(x)


# def exp1exp(x):
#     if x.dtype == float:
#         return exp1_real_numba(x)
#     elif x.dtype == complex:
#         return exp1_complex_numba(x)


@njit(cache=True, nogil=True)
def _signed_inf_from_trig(v: float) -> float:
    if v > 0.0:
        return math.inf
    if v < 0.0:
        return -math.inf
    return 0.0


@njit(cache=True, nogil=True)
def _cexp_parts(a: float, b: float) -> complex:
    """
    Compute exp(a + i b) with simple overflow/underflow guards.
    """
    if a > _MAX_EXP:
        c = math.cos(b)
        s = math.sin(b)
        return complex(_signed_inf_from_trig(c), _signed_inf_from_trig(s))

    if a < _MIN_EXP:
        return complex(0.0, 0.0)

    ea = math.exp(a)
    return complex(ea * math.cos(b), ea * math.sin(b))


@njit(cache=True, nogil=True)
def _exp_neg_z2(z: complex) -> complex:
    """
    Compute exp(-z*z), where z = x + i y.
    -z^2 = (y^2 - x^2) - 2 i x y
    """
    x = z.real
    y = z.imag
    return _cexp_parts(y * y - x * x, -2.0 * x * y)


@njit(cache=True, nogil=True)
def _trap_correction(z: complex, plus: bool) -> complex:
    """
    Correction term:

        2 exp(-z^2) / (1 + exp(-2 i H z))   if plus=True
        2 exp(-z^2) / (1 - exp(-2 i H z))   if plus=False

    Uses a reciprocal form when exp(-2 i H z) would be very large.
    """
    x = z.real
    y = z.imag

    log_abs_b = 2.0 * _H_CAP * y

    if log_abs_b > 40.0:
        # B = exp(-2 i H z), inv_b = 1/B = exp(2 i H z)
        inv_b = _cexp_parts(
            -2.0 * _H_CAP * y,
            2.0 * _H_CAP * x,
        )

        # 2 exp(-z^2) / B = 2 exp(-z^2 + 2 i H z)
        a_over_b = 2.0 * _cexp_parts(
            y * y - x * x - 2.0 * _H_CAP * y,
            -2.0 * x * y + 2.0 * _H_CAP * x,
        )

        if plus:
            return a_over_b / (1.0 + inv_b)
        else:
            return a_over_b / (inv_b - 1.0)

    else:
        a = 2.0 * _exp_neg_z2(z)
        b = _cexp_parts(
            log_abs_b,
            -2.0 * _H_CAP * x,
        )

        if plus:
            return a / (1.0 + b)
        else:
            return a / (1.0 - b)


@njit(cache=True, nogil=True)
def _wofz_midpoint(z: complex) -> complex:
    """
    Midpoint-rule approximation used in part of the first quadrant.
    """
    z2 = z * z
    az = complex(0.0, 2.0 / _H_CAP) * z

    h0 = 0.5 * _H_STEP
    t2 = h0 * h0
    s = math.exp(-t2) / (z2 - t2)

    for k in range(1, _NTRAP + 1):
        t = _H_STEP * (k + 0.5)
        t2 = t * t
        s += math.exp(-t2) / (z2 - t2)

    return az * s


@njit(cache=True, nogil=True)
def _wofz_modified_midpoint(z: complex) -> complex:
    """
    Modified midpoint-rule approximation.
    """
    return _wofz_midpoint(z) + _trap_correction(z, True)


@njit(cache=True, nogil=True)
def _wofz_modified_trapezoid(z: complex) -> complex:
    """
    Modified trapezoidal-rule approximation.
    """
    z2 = z * z
    az = complex(0.0, 2.0 / _H_CAP) * z

    t = _H_STEP * _NTRAP
    t2 = t * t
    s = math.exp(-t2) / (z2 - t2)

    for k in range(1, _NTRAP):
        t = _H_STEP * k
        t2 = t * t
        s += math.exp(-t2) / (z2 - t2)

    return complex(0.0, 1.0 / _H_CAP) / z + az * s + _trap_correction(z, False)


@njit(cache=True, nogil=True)
def _wofz_first_quadrant(z: complex) -> complex:
    """
    Evaluate w(z) for Re(z) >= 0, Im(z) >= 0.
    """
    x = z.real
    y = z.imag

    rzh = x / _H_STEP
    buff = abs(rzh - math.floor(rzh) - 0.5)

    if y >= max(x, _H_CAP):
        return _wofz_midpoint(z)

    elif y < x and buff <= 0.25:
        return _wofz_modified_trapezoid(z)

    else:
        return _wofz_modified_midpoint(z)


@njit(cache=True, nogil=True)
def wofz_complex_scalar_numba(z: complex) -> complex:
    """
    Numba-compatible scalar complex Faddeeva function.

    Approximate replacement for:

        scipy.special.wofz(z)

    for complex128-like finite inputs.
    """
    x = z.real
    y = z.imag

    if math.isnan(x) or math.isnan(y):
        return complex(math.nan, math.nan)

    # Basic SciPy-like limiting behavior for common infinite cases.
    # For ordinary finite inputs, the main algorithm below is used.
    if math.isinf(x) or math.isinf(y):
        if y == -math.inf and x == 0.0:
            return complex(math.inf, 0.0)
        if y == -math.inf:
            return complex(math.nan, math.nan)
        return complex(0.0, 0.0)

    # Map arbitrary z to the first quadrant using symmetries.
    xneg = x < 0.0
    yneg = y < 0.0
    not_both = xneg != yneg

    zw = z

    if xneg:
        zw = -zw

    if not_both:
        zw = complex(zw.real, -zw.imag)

    w = _wofz_first_quadrant(zw)

    if not_both:
        w = complex(w.real, -w.imag)

    if yneg:
        # w(z) = 2 exp(-z^2) - w(-z)
        w = 2.0 * _exp_neg_z2(z) - w

    return w


@njit(cache=True, nogil=True)
def wofz_real_scalar_numba(x: float) -> complex:
    """
    Convenience wrapper for real x, returning complex w(x).
    """
    return wofz_complex_scalar_numba(complex(x, 0.0))


@vectorize(
    [complex128(complex128), complex128(float64)],
    nopython=True,
)
def wofz(z: complex) -> complex:
    """
    Vectorized ufunc version.

    Works on complex128 arrays and float64 arrays.
    """
    return wofz_complex_scalar_numba(z)
