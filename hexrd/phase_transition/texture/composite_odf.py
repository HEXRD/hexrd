"""
Composite Orientation Distribution Function (ODF)

Implements the ODF produced by adding or subtracting other ODFs: a linear
combination of component ODFs plus a constant offset in MRD,

    f(g) = c + Sum_i a_i * f_i(g).

Functions on SO(3) form a vector space, so this class is closed under the
operations that build it: components are stored as given (they may be
different ODF types), nested combinations are flattened by distributing
coefficients, and uniform components collapse into the constant term. The
result is NOT renormalized, so f need not integrate to 1 MRD and may be
negative wherever a subtracted term dominates.
"""

from typing import Optional, Sequence, Union

import numpy as np

from hexrd.phase_transition.texture.arithmetic import (
    ODFArithmetic,
    _is_odf,
)
from hexrd.phase_transition.texture.uniform_odf import UniformODF


def _shared_symmetry_label(
    components: Sequence[object],
    attribute: str,
) -> Optional[str]:
    """
    Return the one symmetry label shared by all components, or None.

    Raises
    ------
    ValueError
        If two components carry different non-None labels.
    """
    labels: list[str] = []
    for component in components:
        label = getattr(component, attribute, None)
        if label is not None and label not in labels:
            labels.append(label)

    if len(labels) > 1:
        raise ValueError(
            f"Cannot combine ODFs with incompatible {attribute}: "
            f"{labels!r}. Addition and subtraction require all components "
            f"to share the same symmetry."
        )

    return labels[0] if labels else None


class CompositeODF(ODFArithmetic):
    """
    Linear combination of ODFs with a constant offset.

    Produced by `+` and `-` on ODF objects; it can also be constructed
    directly. Evaluates as

        f(g) = constant + Sum_i coefficients[i] * components[i].eval(g)

    with no renormalization, so a sum of two proper ODFs has mean 2 MRD and
    a difference may be negative.

    Two algebraic simplifications are applied at construction. Both rewrite
    the representation without changing f(g):

    - a component that is itself a `CompositeODF` is flattened into this
      one, its coefficients scaled accordingly;
    - a :class:`~hexrd.phase_transition.texture.uniform_odf.UniformODF`
      component is folded into `constant` (it is constant at 1 MRD), so it
      is not retained as a separate term.

    Parameters
    ----------
    components : sequence of ODF
        ODF objects to combine. Each must provide an `eval(orientations)`
        method.
    coefficients : array_like, optional
        Multiplier for each component, same length as `components`.
        Defaults to 1.0 for every component. Addition and subtraction only
        ever produce +1 and -1.
    constant : float, optional
        Constant offset in MRD, default 0.0.

    Attributes
    ----------
    components : tuple
        Component ODFs, after flattening and folding out uniform terms
    coefficients : numpy.ndarray
        Coefficient of each component, shape (n_components,)
    constant : float
        Constant offset in MRD
    n_components : int
        Number of component ODFs (excluding the constant)
    crystal_symmetry : str or None
        Crystal symmetry label shared by the components
    sample_symmetry : str or None
        Sample symmetry label shared by the components

    Raises
    ------
    ValueError
        If `coefficients` has the wrong length, or the components carry
        conflicting crystal or sample symmetries.
    TypeError
        If a component has no `eval` method.
    """

    def __init__(
        self,
        components: Sequence[object],
        coefficients: Optional[Sequence[float]] = None,
        constant: float = 0.0,
    ) -> None:
        components = list(components)

        if coefficients is None:
            coefficients = [1.0] * len(components)
        else:
            coefficients = [float(c) for c in np.atleast_1d(coefficients)]
            if len(coefficients) != len(components):
                raise ValueError(
                    f"coefficients must have one entry per component: got "
                    f"{len(coefficients)} for {len(components)} components"
                )

        flat_components: list[object] = []
        flat_coefficients: list[float] = []
        total_constant = float(constant)

        for component, coefficient in zip(components, coefficients):
            if isinstance(component, CompositeODF):
                total_constant += coefficient * component.constant
                for sub, sub_coefficient in zip(
                    component.components, component.coefficients
                ):
                    flat_components.append(sub)
                    flat_coefficients.append(coefficient * sub_coefficient)
            elif isinstance(component, UniformODF):
                # A uniform ODF is constant at 1 MRD, so it contributes only
                # to the offset.
                total_constant += coefficient * component.value
            elif _is_odf(component):
                flat_components.append(component)
                flat_coefficients.append(coefficient)
            else:
                raise TypeError(
                    f"Components must be ODFs with an eval() method, got "
                    f"{type(component).__name__}"
                )

        self._components = tuple(flat_components)
        self._coefficients = np.asarray(flat_coefficients, dtype=float)
        self._constant = total_constant

        # Reject mismatched symmetries up front
        self._crystal_symmetry = _shared_symmetry_label(
            self._components, 'crystal_symmetry'
        )
        self._sample_symmetry = _shared_symmetry_label(
            self._components, 'sample_symmetry'
        )

    @property
    def components(self) -> tuple:
        """tuple: Component ODFs (constant offset excluded)."""
        return self._components

    @property
    def coefficients(self) -> np.ndarray:
        """numpy.ndarray: Coefficient of each component, shape (N,)."""
        return self._coefficients.copy()

    @property
    def constant(self) -> float:
        """float: Constant offset in MRD."""
        return self._constant

    @property
    def n_components(self) -> int:
        """int: Number of component ODFs, excluding the constant offset."""
        return len(self._components)

    @property
    def crystal_symmetry(self) -> Optional[str]:
        """str or None: Crystal symmetry label shared by the components."""
        return self._crystal_symmetry

    @property
    def sample_symmetry(self) -> Optional[str]:
        """str or None: Sample symmetry label shared by the components."""
        return self._sample_symmetry

    def eval(
        self, orientations: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Evaluate the composite ODF at given orientations.

        Computes f(g) = constant + Sum_i a_i * f_i(g), in MRD.

        Parameters
        ----------
        orientations : array_like
            Orientation matrices of shape (..., 3, 3)

        Returns
        -------
        float or numpy.ndarray
            ODF values in MRD. A scalar float for a single (3, 3)
            orientation; otherwise an array whose shape matches the leading
            dimensions of the input. Values may be negative when the
            composite contains a subtraction.
        """
        orientations = np.asarray(orientations)

        if orientations.shape[-2:] != (3, 3):
            raise ValueError(
                f"Orientation matrices must have shape (..., 3, 3), "
                f"got {orientations.shape}"
            )

        output_shape = orientations.shape[:-2]
        results = np.full(output_shape, self._constant, dtype=float)

        for coefficient, component in zip(
            self._coefficients, self._components
        ):
            results = results + coefficient * np.asarray(
                component.eval(orientations), dtype=float
            )

        if output_shape == ():
            return float(results)
        return results

    def analytic_texture_index(self) -> Optional[float]:
        """
        Exact texture index J = <f^2> when a closed form is available.

        A composite with no component ODFs is the constant function
        f = `constant`, whose index <f^2> / <f>^2 is exactly 1 for any
        non-zero constant (it is undefined for f = 0). Any other linear
        combination has no closed form here, so this returns None and
        callers fall back to Monte Carlo estimation.

        Returns
        -------
        float or None
            Exact texture index, or None if no closed form applies.
        """
        if self._components or self._constant == 0.0:
            return None
        return 1.0

    def texture_index(self, n_orientations: int = 100000, seed=None) -> float:
        """
        Texture index J = <f^2> / <f>^2 of the composite, in MRD^2.

        Uses the exact value when available; otherwise estimates J by Monte
        Carlo over Haar-uniform orientations.

        Because a composite is not renormalized, this is only meaningful
        when the composite is a genuine density: non-negative everywhere and
        with a non-zero mean. A difference of ODFs generally satisfies
        neither, and a composite whose mean is zero yields nan.

        Parameters
        ----------
        n_orientations : int, optional
            Number of Haar-uniform samples for the Monte Carlo estimate,
            default 100000
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        float
            Texture index J
        """
        from .evaluation import texture_index as _texture_index

        return _texture_index(self, n_orientations=n_orientations, seed=seed)

    def norm(self, n_orientations: int = 100000, seed=None) -> float:
        """
        L2 norm ||f|| = sqrt(J) of the composite, in MRD.

        Carries the same caveats as `texture_index`: the composite must be
        a genuine density for the value to be meaningful.

        Parameters
        ----------
        n_orientations : int, optional
            Number of Haar-uniform samples for the Monte Carlo estimate,
            default 100000
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        float
            L2 norm ||f|| = sqrt(J)
        """
        from .evaluation import texture_norm as _texture_norm

        return _texture_norm(self, n_orientations=n_orientations, seed=seed)

    def __repr__(self) -> str:
        return (
            f"CompositeODF(n_components={self.n_components}, "
            f"constant={self.constant:.6f}, "
            f"crystal_symmetry={self.crystal_symmetry!r}, "
            f"sample_symmetry={self.sample_symmetry!r})"
        )

    def __str__(self) -> str:
        desc = (
            f"Composite ODF with {self.n_components} component(s)\n"
            f"Crystal symmetry: {self.crystal_symmetry or 'none'}\n"
            f"Sample symmetry: {self.sample_symmetry or 'none'}\n"
            f"Constant offset: {self.constant:.6f} MRD\n"
        )

        if self.n_components:
            desc += "Components:\n"
            desc += "\n".join(
                f"  {coefficient:+g} * {component!r}"
                for coefficient, component in zip(
                    self._coefficients, self._components
                )
            )

        return desc
