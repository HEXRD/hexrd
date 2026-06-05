"""
Uniform Orientation Distribution Function (ODF)

Implements a constant ODF representing completely random texture where all
orientations are equally likely. The uniform ODF has a constant value of
1 MRD (multiples of a random distribution), the standard normalization
for texture analysis on SO(3).
"""

from typing import Union

import numpy as np

# Valid symmetry labels. Crystal symmetries mirror the crystal portion of
# hexrd.powder.wppf.texture.SYMLIST. Sample symmetries are the subset
# currently supported by DeLaValleePoussinKernel; 'axial' is intentionally
# excluded to stay consistent with the kernel's sample symmetry handling.
_CRYSTAL_SYMMETRIES = frozenset([
    'ci', 'c2h', 'd2h', 'c4h', 'd4h',
    's6', 'd3d', 'c6h', 'd6h', 'th', 'oh',
])
_SAMPLE_SYMMETRIES = frozenset([
    'triclinic', 'monoclinic', 'orthorhombic',
])


class UniformODF:
    """
    Uniform (random) orientation distribution function.

    Represents a completely isotropic texture where all crystal orientations
    are equally likely. The value is constant at 1 MRD (multiples of a
    random distribution), the standard normalization where the uniform
    distribution serves as the reference density.

    Parameters
    ----------
    crystal_symmetry : str
        Crystal symmetry using standard crystallographic notation.
        Supported: 'ci', 'c2h', 'd2h', 'c4h', 'd4h', 's6', 'd3d',
                  'c6h', 'd6h', 'th', 'oh'
    sample_symmetry : str
        Sample symmetry. Supported: 'triclinic', 'monoclinic',
                       'orthorhombic'

    Attributes
    ----------
    value : float
        Constant ODF value = 1.0 (MRD)
    crystal_symmetry : str
        Crystal symmetry string
    sample_symmetry : str
        Sample symmetry string

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
        crystal_symmetry: str,
        sample_symmetry: str = 'triclinic',
    ) -> None:
        """
        Initialize uniform ODF with specified symmetries.

        Parameters
        ----------
        crystal_symmetry : str
            Crystal symmetry notation
        sample_symmetry : str, optional
            Sample symmetry notation, default 'triclinic'
        """
        if crystal_symmetry not in _CRYSTAL_SYMMETRIES:
            raise ValueError(
                f"Invalid crystal symmetry '{crystal_symmetry}'. "
                f"Must be one of: "
                f"{', '.join(sorted(_CRYSTAL_SYMMETRIES))}"
            )

        if sample_symmetry not in _SAMPLE_SYMMETRIES:
            raise ValueError(
                f"Invalid sample symmetry '{sample_symmetry}'. "
                f"Must be one of: "
                f"{', '.join(sorted(_SAMPLE_SYMMETRIES))}"
            )

        # Symmetry is validated and stored to satisfy the common ODF
        # interface used by the upcoming UnimodalODF work. It is
        # intentionally inert here: the uniform ODF value is constant at
        # 1 MRD for all orientations regardless of crystal or sample
        # symmetry.
        self._crystal_symmetry = crystal_symmetry
        self._sample_symmetry = sample_symmetry

    @property
    def crystal_symmetry(self) -> str:
        """Crystal symmetry notation."""
        return self._crystal_symmetry

    @property
    def sample_symmetry(self) -> str:
        """Sample symmetry notation."""
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
        numpy.ndarray
            ODF values with shape matching the leading dimensions of input.
            All values equal to 1.0 (MRD).

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
            f"UniformODF(crystal_symmetry='{self.crystal_symmetry}', "
            f"sample_symmetry='{self.sample_symmetry}', "
            f"value={self.value:.6f})"
        )

    def __str__(self) -> str:
        """String representation of UniformODF."""
        return (
            f"Uniform ODF with {self.crystal_symmetry} crystal symmetry "
            f"and {self.sample_symmetry} sample symmetry\n"
            f"Constant value: {self.value:.6f} (random texture)"
        )
