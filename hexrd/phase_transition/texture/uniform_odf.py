"""
Uniform Orientation Distribution Function (ODF)

Implements a constant ODF representing completely random texture where all
orientations are equally likely. The uniform ODF has a constant value of
1 MRD (multiples of a random distribution), the standard normalization
for texture analysis on SO(3).
"""

from typing import Optional, Union

import numpy as np

from hexrd.phase_transition.texture.kernels import (
    _symmetry_quaternions,
    _Symmetry,
)


class UniformODF:
    """
    Uniform (random) orientation distribution function.

    Represents a completely isotropic texture where all crystal orientations
    are equally likely. The value is constant at 1 MRD (multiples of a
    random distribution), the standard normalization where the uniform
    distribution serves as the reference density.

    Parameters
    ----------
    crystal_symmetry : str or numpy.ndarray, optional
        Crystal symmetry as a Laue group label ('ci', 'c2h', 'd2h', 'c4h',
        'd4h', 's6', 'd3d', 'c6h', 'd6h', 'th', 'oh') or a quaternion
        symmetry array. Validated but inert (the value is 1 MRD regardless);
        default None.
    sample_symmetry : str or numpy.ndarray, optional
        Sample symmetry as a label ('triclinic', 'monoclinic',
        'orthorhombic') or a quaternion array. Validated but inert;
        default None.

    Attributes
    ----------
    value : float
        Constant ODF value = 1.0 (MRD)
    crystal_symmetry : str or None
        Crystal symmetry label, or None if unset or given as an array
    sample_symmetry : str or None
        Sample symmetry label, or None if unset or given as an array

    Examples
    --------
    >>> odf = UniformODF('d6h', 'triclinic')  # hexagonal crystal
    >>> orientations = np.eye(3).reshape(1, 3, 3)
    >>> values = odf.eval(orientations)
    >>> print(values[0])  # 1.0
    """

    # Constant value for uniform ODF: 1 MRD (standard normalization)
    _UNIFORM_VALUE = 1.0

    def __init__(
        self,
        crystal_symmetry: _Symmetry = None,
        sample_symmetry: _Symmetry = None,
    ) -> None:
        """
        Initialize uniform ODF with optional symmetries.

        Parameters
        ----------
        crystal_symmetry : str or numpy.ndarray, optional
            Crystal symmetry notation, default None
        sample_symmetry : str or numpy.ndarray, optional
            Sample symmetry notation, default None
        """
        # Validate through the same path the kernel uses, so the texture
        # package shares one accepted symmetry set and one error behavior.
        # Symmetry is inert for a uniform ODF (the value is 1 MRD for every
        # orientation); it is kept as metadata and for parity with the
        # kernel-backed ODFs (e.g. UnimodalODF).
        _symmetry_quaternions(crystal_symmetry, symtype='crystal')
        _symmetry_quaternions(sample_symmetry, symtype='sample')

        # Retain the string label; a symmetry given as a quaternion array
        # has no recoverable label (mirrors the kernel's behavior).
        self._crystal_symmetry = (
            crystal_symmetry if isinstance(crystal_symmetry, str) else None
        )
        self._sample_symmetry = (
            sample_symmetry if isinstance(sample_symmetry, str) else None
        )

    @property
    def crystal_symmetry(self) -> Optional[str]:
        """Crystal symmetry label, or None if unset or given as an array."""
        return self._crystal_symmetry

    @property
    def sample_symmetry(self) -> Optional[str]:
        """Sample symmetry label, or None if unset or given as an array."""
        return self._sample_symmetry

    @property
    def value(self) -> float:
        """Constant ODF value in MRD."""
        return self._UNIFORM_VALUE

    def analytic_texture_index(self) -> float:
        """
        Exact texture index J = <f^2> of the uniform ODF.

        The uniform ODF is constant at 1 MRD, so J = 1 exactly.

        Returns
        -------
        float
            Texture index, always 1.0.
        """
        return self._UNIFORM_VALUE ** 2

    def texture_index(self, n_orientations: int = 100000, seed=None) -> float:
        """
        Texture index J = <f^2> (always 1.0 for a uniform ODF).

        Parameters
        ----------
        n_orientations : int, optional
            Unused; accepted for API parity with sampled ODFs.
        seed : int, optional
            Unused; accepted for API parity with sampled ODFs.

        Returns
        -------
        float
            Texture index, always 1.0.
        """
        from .evaluation import texture_index as _texture_index

        return _texture_index(self, n_orientations=n_orientations, seed=seed)

    def norm(self, n_orientations: int = 100000, seed=None) -> float:
        """
        L2 norm ||f|| = sqrt(J) (always 1.0 for a uniform ODF).

        Parameters
        ----------
        n_orientations : int, optional
            Unused; accepted for API parity with sampled ODFs.
        seed : int, optional
            Unused; accepted for API parity with sampled ODFs.

        Returns
        -------
        float
            L2 norm, always 1.0.
        """
        from .evaluation import texture_norm as _texture_norm

        return _texture_norm(self, n_orientations=n_orientations, seed=seed)

    def eval(
        self, orientations: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """
        Evaluate uniform ODF at given orientations.

        For a uniform ODF, all orientations return 1.0 MRD
        (multiples of a random distribution).

        Parameters
        ----------
        orientations : array_like
            Orientation matrices. Can be:
            - Single 3x3 rotation matrix
            - Array of shape (N, 3, 3) for N orientations
            - Any shape ending in (3, 3) for rotation matrices

        Returns
        -------
        float or numpy.ndarray
            ODF values, all equal to 1.0 (MRD). A scalar float for a single
            (3, 3) orientation; otherwise an array whose shape matches the
            leading dimensions of the input.

        Examples
        --------
        >>> odf = UniformODF('oh', 'triclinic')
        >>>
        >>> # Single orientation
        >>> R = np.eye(3)
        >>> value = odf.eval(R)  # scalar
        >>>
        >>> # Multiple orientations
        >>> Rs = np.array([np.eye(3), np.eye(3)])  # shape (2, 3, 3)
        >>> values = odf.eval(Rs)  # shape (2,)
        """
        orientations = np.asarray(orientations)

        # Validate input shape - must end with (3, 3)
        if orientations.shape[-2:] != (3, 3):
            raise ValueError(
                f"Orientation matrices must have shape (..., 3, 3), "
                f"got {orientations.shape}"
            )

        # Return array of uniform values with shape matching input leading dims
        output_shape = orientations.shape[:-2]

        if output_shape == ():
            # Single orientation - return scalar
            return self._UNIFORM_VALUE
        else:
            # Multiple orientations - return array
            return np.full(output_shape, self._UNIFORM_VALUE)

    def __repr__(self) -> str:
        """String representation of UniformODF."""
        return (
            f"UniformODF(crystal_symmetry={self.crystal_symmetry!r}, "
            f"sample_symmetry={self.sample_symmetry!r}, "
            f"value={self.value:.6f})"
        )

    def __str__(self) -> str:
        """String representation of UniformODF."""
        return (
            f"Uniform ODF with {self.crystal_symmetry or 'none'} crystal "
            f"symmetry and {self.sample_symmetry or 'none'} sample symmetry\n"
            f"Constant value: {self.value:.6f} (random texture)"
        )
