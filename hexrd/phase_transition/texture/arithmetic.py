"""
Arithmetic on orientation distribution functions.

Provides the operator support shared by every ODF class in this package:
adding and subtracting ODFs, and adding or subtracting a constant offset in
MRD.
"""

import numbers
from typing import Any


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

    # Opt out of numpy's ufunc machinery (NEP 13) so `array + odf` defers to
    # __radd__, which reports the operation as unsupported, instead of
    # broadcasting into an object array holding one composite per element.
    __array_ufunc__ = None

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

        if reflected:
            # The left operand already declined, so this is the last handler
            # Python will consult. Returning NotImplemented would let CPython
            # fall through to the sequence-concatenation slot, and a numpy
            # left operand reports that as a failed concatenation - an error
            # about np.concatenate for an expression containing no sequences.
            raise TypeError(
                f"unsupported operand type(s) for {'+' if sign > 0 else '-'}: "
                f"{type(other).__name__!r} and {type(self).__name__!r}. "
                f"An ODF can be combined with another ODF or with a real "
                f"scalar, which acts as a constant offset in MRD."
            )

        # Forward operation: defer to the right operand's reflected method,
        # which may know how to combine with an ODF.
        return NotImplemented
