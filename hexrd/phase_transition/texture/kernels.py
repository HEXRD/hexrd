"""
SO(3) Kernel Functions for Texture Analysis

Implements radial basis functions on the rotation group SO(3) used for 
constructing smooth orientation distribution functions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import warnings

import numpy as np
from scipy.special import beta as betafn

from hexrd.core import rotations


_IDENTITY_SYMMETRY = np.c_[1.0, 0.0, 0.0, 0.0].T
_SAMPLE_SYMMETRY_TO_LAUE = {
    'triclinic': 'ci',
    'monoclinic': 'c2h',
    'orthorhombic': 'd2h',
}
_AngleResult = Union[float, np.ndarray]
_Symmetry = Optional[Union[str, np.ndarray]]


def _symmetry_quaternions(
    symmetry: _Symmetry,
    *,
    symtype: str,
) -> Optional[np.ndarray]:
    if symmetry is None:
        return None
    if isinstance(symmetry, np.ndarray):
        if len(symmetry.shape) != 2 or symmetry.shape[0] != 4:
            raise ValueError(f"{symtype} symmetry must have shape (4, n)")
        return symmetry
    if not isinstance(symmetry, str):
        raise ValueError(f"{symtype} symmetry must be a string or ndarray")
    if symtype == 'sample':
        symmetry = _SAMPLE_SYMMETRY_TO_LAUE.get(symmetry.lower(), symmetry)
    try:
        return rotations.quatOfLaueGroup(symmetry)
    except RuntimeError as error:
        raise ValueError(
            f"Invalid {symtype} symmetry: {symmetry!r}"
        ) from error


class SO3Kernel(ABC):
    """
    Abstract base class for kernels on the SO(3) rotation group.

    All SO(3) kernels should inherit from this class and implement
    the eval() method for kernel evaluation.
    """

    @abstractmethod
    def eval(
        self, R1: np.ndarray, R2: np.ndarray
    ) -> _AngleResult:
        """
        Evaluate kernel between two rotations.

        Parameters
        ----------
        R1, R2 : array_like
            Rotation matrices of shape (..., 3, 3)

        Returns
        -------
        float or numpy.ndarray
            Kernel values
        """
        pass


class DeLaValleePoussinKernel(SO3Kernel):
    """
    De la Vallée Poussin kernel on SO(3).

    A radially symmetric kernel on SO(3) defined by:

        K(ω) = C · cos(ω/2)^(2κ)

    where ω is the misorientation angle, κ is the shape parameter, and
    C = B(3/2, 1/2) / B(3/2, κ + 1/2) is the normalization constant
    (B denotes the Beta function).

    The halfwidth h is the angle at which the kernel drops to half its
    peak value. It is related to κ analytically by:

        κ = ln(0.5) / (2 · ln(cos(h/2)))

    Parameters
    ----------
    halfwidth : float
        Half-width parameter in radians — the angle at which K drops
        to half its maximum. Must be > 0, typically in [π/180, π/2].
    crystal_symmetry : str or numpy.ndarray, optional
        Crystal symmetry used to compute symmetry-reduced misorientation
        angles. Strings are interpreted as Laue group labels.
    sample_symmetry : str or numpy.ndarray, optional
        Sample symmetry used to compute symmetry-reduced misorientation
        angles. Strings are interpreted as Laue group labels, with
        triclinic, monoclinic, and orthorhombic sample labels mapped to
        their corresponding finite Laue groups.

    Attributes
    ----------
    halfwidth : float
        Half-width parameter in radians
    kappa : float
        Shape parameter κ derived from half-width
    norm_constant : float
        Normalization constant from the Beta function

    Examples
    --------
    >>> kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
    >>> R1 = np.eye(3)
    >>> R2 = np.eye(3)  # Same orientation
    >>> value = kernel.eval(R1, R2)  # Maximum value = C
    """

    def __init__(
        self,
        halfwidth: float,
        crystal_symmetry: _Symmetry = None,
        sample_symmetry: _Symmetry = None,
    ) -> None:
        """
        Initialize de la Vallée Poussin kernel.

        Parameters
        ----------
        halfwidth : float
            Half-width parameter in radians
        crystal_symmetry : str or numpy.ndarray, optional
            Crystal symmetry for symmetry-reduced misorientation angles
        sample_symmetry : str or numpy.ndarray, optional
            Sample symmetry for symmetry-reduced misorientation angles
        """
        if halfwidth <= 0:
            raise ValueError("Half-width must be positive")
        if halfwidth > np.pi / 2:
            warnings.warn(
                f"Large half-width "
                f"{np.degrees(halfwidth):.1f}° may produce "
                f"very broad distributions",
                UserWarning,
            )

        self._halfwidth = float(halfwidth)

        # Shape parameter κ from the half-maximum condition K(h) = K(0)/2:
        #   cos(h/2)^(2κ) = 1/2
        #   κ = ln(0.5) / (2·ln(cos(h/2)))
        self._kappa = 0.5 * np.log(0.5) / np.log(np.cos(halfwidth / 2.0))

        # Normalization constant: C = B(3/2, 1/2) / B(3/2, κ + 1/2)
        self._C = betafn(1.5, 0.5) / betafn(1.5, self._kappa + 0.5)
        self._crystal_symmetry = _symmetry_quaternions(
            crystal_symmetry,
            symtype='crystal',
        )
        self._sample_symmetry = _symmetry_quaternions(
            sample_symmetry,
            symtype='sample',
        )

    @property
    def has_symmetry(self) -> bool:
        """bool: Whether symmetry reduction is enabled."""
        return (
            self._crystal_symmetry is not None
            or self._sample_symmetry is not None
        )

    @property
    def halfwidth(self) -> float:
        """float: Half-width in radians (angle where K = K_max / 2)."""
        return self._halfwidth

    @property
    def kappa(self) -> float:
        """float: Shape parameter κ."""
        return self._kappa

    @property
    def norm_constant(self) -> float:
        """float: Normalization constant from Beta function."""
        return self._C

    def misorientation_angle(
        self, R1: np.ndarray, R2: np.ndarray
    ) -> _AngleResult:
        """
        Calculate misorientation angle between rotation matrices.

        Without symmetry, uses the formula:
        cos(ω) = (trace(R1^T @ R2) - 1) / 2.
        With crystal or sample symmetry, delegates to
        hexrd.core.rotations.misorientation for symmetry-reduced angles.

        Parameters
        ----------
        R1, R2 : array_like
            Rotation matrices of shape (..., 3, 3)

        Returns
        -------
        float or numpy.ndarray
            Misorientation angles in radians, shape matches input broadcasting
        """
        R1 = np.asarray(R1)
        R2 = np.asarray(R2)

        if R1.shape[-2:] != (3, 3) or R2.shape[-2:] != (3, 3):
            raise ValueError(
                "Input matrices must have shape (..., 3, 3)"
            )

        if self.has_symmetry:
            return self._symmetry_reduced_misorientation_angle(R1, R2)

        # Relative rotation: R1^T @ R2
        R1_T = np.swapaxes(R1, -2, -1)
        relative = np.matmul(R1_T, R2)
        trace = np.trace(relative, axis1=-2, axis2=-1)

        # cos(ω) = (tr(R) - 1) / 2, clamped for numerical safety
        cos_omega = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        return np.arccos(cos_omega)

    def _symmetry_reduced_misorientation_angle(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
    ) -> _AngleResult:
        output_shape = np.broadcast_shapes(R1.shape[:-2], R2.shape[:-2])
        R1_flat = np.broadcast_to(
            R1,
            output_shape + (3, 3),
        ).reshape(-1, 3, 3)
        R2_flat = np.broadcast_to(
            R2,
            output_shape + (3, 3),
        ).reshape(-1, 3, 3)

        angles = np.empty(R1_flat.shape[0], dtype=float)
        symmetries = self._misorientation_symmetries()
        for i, (r1, r2) in enumerate(zip(R1_flat, R2_flat)):
            q1 = rotations.quatOfRotMat(r1).reshape(4, 1)
            q2 = rotations.quatOfRotMat(r2).reshape(4, 1)
            angle, _ = rotations.misorientation(
                q1,
                q2,
                symmetries=symmetries,
            )
            angles[i] = angle[0]

        if output_shape == ():
            return angles[0]
        return angles.reshape(output_shape)

    def _misorientation_symmetries(self) -> tuple[np.ndarray, ...]:
        crystal_symmetry = self._crystal_symmetry
        if crystal_symmetry is None and self._sample_symmetry is not None:
            crystal_symmetry = _IDENTITY_SYMMETRY
        symmetries = (crystal_symmetry,) if crystal_symmetry is not None else ()
        if self._sample_symmetry is not None:
            symmetries = symmetries + (self._sample_symmetry,)
        return symmetries

    def eval(
        self, R1: np.ndarray, R2: np.ndarray
    ) -> _AngleResult:
        """
        Evaluate de la Vallée Poussin kernel between rotations.

        K(ω) = C · cos(ω/2)^(2κ)

        Parameters
        ----------
        R1, R2 : array_like
            Rotation matrices of shape (..., 3, 3)

        Returns
        -------
        float or numpy.ndarray
            Kernel values, shape matches broadcasting of R1 and R2

        Examples
        --------
        >>> kernel = DeLaValleePoussinKernel(
        ...     halfwidth=np.radians(10)
        ... )
        >>> value = kernel.eval(np.eye(3), np.eye(3))
        """
        omega = self.misorientation_angle(R1, R2)
        co2 = np.cos(omega / 2.0)
        return self._C * co2 ** (2.0 * self._kappa)

    def eval_centered(
        self, R: np.ndarray, center: np.ndarray
    ) -> _AngleResult:
        """
        Evaluate kernel centered at a specific orientation.

        Equivalent to eval(center, R) with clearer semantics
        for unimodal distributions.

        Parameters
        ----------
        R : array_like
            Rotation matrices to evaluate at, shape (..., 3, 3)
        center : array_like
            Central (modal) orientation, shape (3, 3)

        Returns
        -------
        float or numpy.ndarray
            Kernel values relative to center orientation
        """
        return self.eval(center, R)

    def __repr__(self) -> str:
        """String representation of kernel."""
        hw_deg = np.degrees(self.halfwidth)
        return (
            f"DeLaValleePoussinKernel("
            f"halfwidth={self.halfwidth:.6f} rad "
            f"= {hw_deg:.2f}°, "
            f"kappa={self.kappa:.3f})"
        )

    def __str__(self) -> str:
        """Human-readable description."""
        hw_deg = np.degrees(self.halfwidth)
        return (
            f"de la Vallée Poussin kernel with "
            f"{hw_deg:.1f}° half-width\n"
            f"kappa = {self.kappa:.3f}, "
            f"C = {self.norm_constant:.6e}\n"
            f"K(ω) = C · cos(ω/2)^(2κ)"
        )
