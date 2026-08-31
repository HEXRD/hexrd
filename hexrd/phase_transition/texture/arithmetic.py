"""
Arithmetic on orientation distribution functions.

Provides the operator support shared by every ODF class in this package:
adding and subtracting ODFs, and adding or subtracting a constant offset in
MRD.
"""

import numbers
from typing import Any

import numpy as np


def _is_constant(value: Any) -> bool:
    """Return True for a real scalar usable as a constant MRD offset."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_odf(value: Any) -> bool:
    """Return True for anything that has `eval`."""
    return callable(getattr(value, 'eval', None))


class ODFArithmetic:
    """
    Mixin giving an ODF class `+` and `-` operators.

    Mix into any class with an `eval(orientations)` method to support:

    - `odf1 + odf2` and `odf1 - odf2` between ODFs
    - `odf + c`, `c + odf`, `odf - c` and `c - odf` for a real
      scalar `c`, which acts as a constant offset in MRD

    Every operation returns a
    :class:`~hexrd.phase_transition.texture.composite_odf.CompositeODF`,
    which evaluates as the linear combination of its terms. Sums are not
    renormalized.
    """

    def __add__(self, other: Any) -> Any:
        """`self + other`, where other is an ODF or a constant in MRD."""
        return self._combine(other, sign=1.0, reflected=False)

    def __radd__(self, other: Any) -> Any:
        """`other + self`, where other is an ODF or a constant in MRD."""
        return self._combine(other, sign=1.0, reflected=True)

    def __sub__(self, other: Any) -> Any:
        """`self - other`, where other is an ODF or a constant in MRD."""
        return self._combine(other, sign=-1.0, reflected=False)

    def __rsub__(self, other: Any) -> Any:
        """`other - self`, where other is an ODF or a constant in MRD."""
        return self._combine(other, sign=-1.0, reflected=True)

    def _combine(self, other: Any, sign: float, reflected: bool) -> Any:
        """
        Build the composite for one binary operation.

        `sign` is the coefficient carried by the right-hand operand of the
        written expression; `reflected` marks the Python-reflected forms
        (`__radd__`/`__rsub__`), where `self` is that right-hand
        operand and so takes the sign instead.
        """
        from .composite_odf import CompositeODF

        if reflected:
            self_coefficient, other_coefficient = sign, 1.0
        else:
            self_coefficient, other_coefficient = 1.0, sign

        if _is_constant(other):
            return CompositeODF(
                [self],
                [self_coefficient],
                constant=other_coefficient * float(other),
            )

        if _is_odf(other):
            return CompositeODF(
                [self, other],
                [self_coefficient, other_coefficient],
            )

        if isinstance(other, np.ndarray):
            raise TypeError(
                f"Cannot combine an ODF with an array: an ODF is a "
                f"function, not a value. Got an array of shape "
                f"{other.shape}. Add or subtract another ODF, or a real "
                f"scalar acting as a constant offset in MRD."
            )

        return NotImplemented
