"""
Unimodal Orientation Distribution Function (ODF)

Implements a kernel-based ODF with a single or multiple preferred orientations.
Uses a combination of modal orientations and kernel functions to create
smooth, localized texture distributions.
"""

from typing import Optional

import numpy as np
from scipy.special import beta as beta_fn

from .kernels import DeLaValleePoussinKernel


class UnimodalODF:
    """
    Unimodal orientation distribution function.

    Represents a texture with one or more preferred orientations around which
    crystal orientations are concentrated. Uses kernel functions to create
    smooth distributions around modal orientations.

    Mathematical form: f(g) = Σᵢ wᵢ K(g, g₀ᵢ),  Σᵢ wᵢ = 1
    Because the de la Vallée Poussin kernel is normalized so its mean over
    SO(3) is 1 (MRD), this weighted sum is itself a valid ODF in MRD units
    (uniform texture = 1).

    Parameters
    ----------
    modal_orientations : array_like
        Modal (preferred) orientation(s). Can be:
        - Single 3x3 rotation matrix
        - Array of shape (N, 3, 3) for N modal orientations
    kernel : DeLaValleePoussinKernel
        Kernel function defining the shape of the distribution. The kernel
        is the single source of truth for symmetry: any crystal/sample
        symmetry must be set on the kernel, and the ODF exposes it via the
        ``crystal_symmetry``/``sample_symmetry`` properties.
    weights : array_like, optional
        Weights for multiple modal orientations. Must sum to 1.
        If None, equal weights are used for multiple orientations.

    Attributes
    ----------
    modal_orientations : numpy.ndarray
        Array of modal orientations, shape (N, 3, 3)
    kernel : DeLaValleePoussinKernel
        Kernel function used for the distribution
    weights : numpy.ndarray
        Component weights, shape (N,)
    crystal_symmetry : str or None
        Crystal symmetry label, delegated from the kernel
    sample_symmetry : str or None
        Sample symmetry label, delegated from the kernel
    n_components : int
        Number of modal orientations

    Examples
    --------
    >>> from hexrd.phase_transition.texture import UnimodalODF
    >>> from hexrd.phase_transition.texture import DeLaValleePoussinKernel
    >>>
    >>> # Single modal orientation (cubic crystal symmetry on the kernel)
    >>> kernel = DeLaValleePoussinKernel(
    ...     halfwidth=np.radians(15), crystal_symmetry='oh'
    ... )
    >>> modal = np.eye(3)  # Identity orientation
    >>> odf = UnimodalODF(modal, kernel)
    >>>
    >>> # Evaluate at modal orientation (should give maximum value)
    >>> value = odf.eval(modal)
    """

    def __init__(self, modal_orientations, kernel, weights=None):
        """
        Initialize unimodal ODF.

        Parameters
        ----------
        modal_orientations : array_like
            Modal orientation(s) as rotation matrices
        kernel : DeLaValleePoussinKernel
            Kernel function for the distribution. Symmetry, if any, must be
            configured on the kernel; the ODF delegates to it.
        weights : array_like, optional
            Component weights for multiple modal orientations
        """
        # Validate and store kernel. The kernel is the single source of
        # truth for symmetry (validated when the kernel was constructed).
        if not isinstance(kernel, DeLaValleePoussinKernel):
            raise TypeError("kernel must be a DeLaValleePoussinKernel instance")

        self._kernel = kernel

        # Process modal orientations
        modal_orientations = np.asarray(modal_orientations)

        # Handle single vs multiple modal orientations
        if modal_orientations.ndim == 2 and modal_orientations.shape == (3, 3):
            # Single modal orientation
            self._modal_orientations = modal_orientations.reshape(1, 3, 3)
            self._n_components = 1
        elif modal_orientations.ndim == 3 and modal_orientations.shape[-2:] == (3, 3):
            # Multiple modal orientations
            self._modal_orientations = modal_orientations
            self._n_components = modal_orientations.shape[0]
        else:
            raise ValueError(
                f"Modal orientations must have shape (3, 3) or (N, 3, 3), "
                f"got {modal_orientations.shape}"
            )

        # Process weights
        if weights is None:
            # Equal weights for all components
            self._weights = np.full(self._n_components, 1.0 / self._n_components)
        else:
            weights = np.asarray(weights)
            if weights.shape != (self._n_components,):
                raise ValueError(
                    f"Weights must have shape ({self._n_components},), "
                    f"got {weights.shape}"
                )
            if not np.allclose(np.sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative")
            self._weights = weights.copy()

    @property
    def modal_orientations(self):
        """numpy.ndarray: Modal orientations, shape (N, 3, 3)."""
        return self._modal_orientations.copy()

    @property
    def kernel(self):
        """DeLaValleePoussinKernel: Kernel function."""
        return self._kernel

    @property
    def weights(self):
        """numpy.ndarray: Component weights, shape (N,)."""
        return self._weights.copy()

    @property
    def crystal_symmetry(self):
        """str or None: Crystal symmetry label, delegated from the kernel."""
        return self._kernel.crystal_symmetry

    @property
    def sample_symmetry(self):
        """str or None: Sample symmetry label, delegated from the kernel."""
        return self._kernel.sample_symmetry

    @property
    def n_components(self):
        """int: Number of modal orientations."""
        return self._n_components

    def eval(self, orientations):
        """
        Evaluate unimodal ODF at given orientations.

        Computes f(g) = Σᵢ wᵢ K(g, g₀ᵢ) for all input orientations, in MRD.

        Parameters
        ----------
        orientations : array_like
            Orientation matrices of shape (..., 3, 3)

        Returns
        -------
        numpy.ndarray
            ODF values with shape matching leading dimensions of input

        Examples
        --------
        >>> modal = np.eye(3)
        >>> kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10))
        >>> odf = UnimodalODF(modal, kernel)
        >>>
        >>> # Single evaluation
        >>> value = odf.eval(modal)  # Should give maximum
        >>>
        >>> # Batch evaluation
        >>> Rs = np.array([np.eye(3), rotation_matrix_z(np.pi/4)])
        >>> values = odf.eval(Rs)  # shape (2,)
        """
        orientations = np.asarray(orientations)

        # Validate input shape
        if orientations.shape[-2:] != (3, 3):
            raise ValueError(
                f"Orientation matrices must have shape (..., 3, 3), "
                f"got {orientations.shape}"
            )

        # Determine output shape
        output_shape = orientations.shape[:-2]
        if output_shape == ():
            # Single orientation
            output_shape = ()
            orientations = orientations.reshape(1, 3, 3)
            squeeze_output = True
        else:
            # Multiple orientations - flatten for processing
            n_orientations = int(np.prod(output_shape))
            orientations = orientations.reshape(n_orientations, 3, 3)
            squeeze_output = False

        # Initialize results
        n_query = orientations.shape[0]
        results = np.zeros(n_query)

        # Evaluate each component and sum with weights
        for i in range(self.n_components):
            modal_i = self._modal_orientations[i]  # Shape (3, 3)
            weight_i = self._weights[i]

            # Evaluate kernel between all query orientations and this modal orientation
            # We need to broadcast: kernel.eval expects both args to have same leading dims
            modal_broadcast = np.broadcast_to(
                modal_i.reshape(1, 3, 3),
                (n_query, 3, 3)
            )

            kernel_values = self.kernel.eval(orientations, modal_broadcast)

            # Add weighted contribution to results
            results += weight_i * kernel_values

        # The de la Vallée Poussin kernel is normalized so its mean over
        # SO(3) is 1 (MRD). With weights that sum to 1, this weighted sum is
        # already a valid ODF in MRD units (uniform texture = 1), so no
        # additional normalization is required.

        # Reshape to match input
        if squeeze_output:
            return float(results[0])
        else:
            return results.reshape(output_shape)

    def estimated_max_value(self):
        """
        Estimate the maximum ODF value, in MRD.

        The ODF maxima occur at (or very near) the modal orientations, so
        this evaluates the full ODF at each mode and returns the largest
        value.

        Returns
        -------
        float
            Maximum ODF value in MRD (multiples of a random distribution)
        """
        # ODF maxima occur at (or very near) the modal orientations.
        # Evaluating the full ODF at each mode accounts for overlap between
        # nearby modes.
        modal_values = np.atleast_1d(self.eval(self._modal_orientations))
        return float(np.max(modal_values))

    def analytic_texture_index(self) -> Optional[float]:
        """
        Exact texture index J = <f^2> when a closed form is available.

        For a single mode with no kernel symmetry, the de la Vallee Poussin
        ODF depends only on the misorientation angle from the mode, whose
        Haar density on SO(3) is p(omega) = (1 - cos omega) / pi. Integrating
        f^2 against it gives the closed form

            J = (2 * C^2 / pi) * B(2*kappa + 1/2, 3/2),

        where C and kappa are the kernel normalization constant and shape
        parameter. Multi-modal ODFs (cross terms between modes) and
        symmetry-reduced kernels have no simple closed form here, so this
        returns None and callers fall back to Monte Carlo estimation.

        Returns
        -------
        float or None
            Exact texture index, or None if no closed form applies.
        """
        if self._n_components != 1 or self._kernel.has_symmetry:
            return None

        kappa = self._kernel.kappa
        norm_c = self._kernel.norm_constant
        return float(
            (2.0 * norm_c ** 2 / np.pi) * beta_fn(2.0 * kappa + 0.5, 1.5)
        )

    def texture_index(self, n_orientations=100000, seed=None):
        """
        Texture index J = <f^2> of the ODF, in MRD^2.

        Uses the exact closed form when available (see
        ``analytic_texture_index``); otherwise estimates J by Monte Carlo
        over Haar-uniform orientations.

        Parameters
        ----------
        n_orientations : int, optional
            Number of Haar-uniform samples for the Monte Carlo fallback,
            default 100000. Ignored when the closed form applies.
        seed : int, optional
            Random seed for the Monte Carlo fallback.

        Returns
        -------
        float
            Texture index J = <f^2> (>= 1 in MRD units).
        """
        from .evaluation import texture_index as _texture_index

        return _texture_index(self, n_orientations=n_orientations, seed=seed)

    def norm(self, n_orientations=100000, seed=None):
        """
        L2 norm ||f|| = sqrt(J) of the ODF (MTEX ``norm``), in MRD.

        Uses the exact closed form when available; otherwise estimates the
        norm by Monte Carlo over Haar-uniform orientations.

        Parameters
        ----------
        n_orientations : int, optional
            Number of Haar-uniform samples for the Monte Carlo fallback,
            default 100000. Ignored when the closed form applies.
        seed : int, optional
            Random seed for the Monte Carlo fallback.

        Returns
        -------
        float
            L2 norm ||f|| = sqrt(J) (>= 1 in MRD units).
        """
        from .evaluation import texture_norm as _texture_norm

        return _texture_norm(self, n_orientations=n_orientations, seed=seed)

    def __repr__(self):
        """String representation of UnimodalODF."""
        return (
            f"UnimodalODF(n_components={self.n_components}, "
            f"kernel_halfwidth={np.degrees(self.kernel.halfwidth):.1f}°, "
            f"crystal_symmetry={self.crystal_symmetry!r}, "
            f"sample_symmetry={self.sample_symmetry!r})"
        )

    def __str__(self):
        """Human-readable description."""
        desc = (
            f"Unimodal ODF with {self.n_components} component(s)\n"
            f"Crystal symmetry: {self.crystal_symmetry or 'none'}\n"
            f"Sample symmetry: {self.sample_symmetry or 'none'}\n"
            f"Kernel: {self.kernel}\n"
        )

        if self.n_components == 1:
            desc += "Modal orientations: 1 component\n"
        else:
            desc += f"Modal orientations: {self.n_components} components\n"
            desc += f"Weights: {self.weights}\n"

        desc += f"Estimated max value: {self.estimated_max_value():.1f} MRD"

        return desc
