"""
ODF Evaluation Functions

Provides common evaluation functionality for orientation distribution functions.
This module serves as a foundation for more complex ODF implementations.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def eval_odf_batch(odf, orientations, chunk_size=10000):
    """
    Evaluate ODF on large batches of orientations with memory management.

    For very large orientation datasets, this function processes orientations
    in chunks to avoid memory issues while maintaining vectorization benefits.

    Parameters
    ----------
    odf : ODF object
        ODF object with eval() method
    orientations : array_like
        Large array of orientation matrices, shape (N, 3, 3). A single
        (3, 3) matrix is also accepted.
    chunk_size : int, optional
        Number of orientations to process per chunk, default 10000

    Returns
    -------
    numpy.ndarray or float
        ODF values, shape (N,) for an (N, 3, 3) input, or a scalar float
        for a single (3, 3) orientation.

    Examples
    --------
    >>> odf = UniformODF('oh', 'triclinic')
    >>> # Large batch of orientations
    >>> Rs = np.random.normal(size=(50000, 3, 3))  # Not actual rotations!
    >>> # In practice, use proper rotation matrices
    >>> values = eval_odf_batch(odf, Rs, chunk_size=5000)
    """
    orientations = np.asarray(orientations)

    # Handle single orientation or small batches normally
    if orientations.ndim == 2 or orientations.shape[0] <= chunk_size:
        return odf.eval(orientations)

    # Process large batches in chunks
    n_orientations = orientations.shape[0]
    results = []

    for start_idx in range(0, n_orientations, chunk_size):
        end_idx = min(start_idx + chunk_size, n_orientations)
        chunk = orientations[start_idx:end_idx]
        chunk_results = odf.eval(chunk)
        results.append(chunk_results)

    return np.concatenate(results)


def eval_random_orientations(odf, n_orientations=1000, seed=None):
    """
    Evaluate ODF at random orientations for statistical analysis.

    Generates random rotation matrices and evaluates the ODF at these
    orientations. Useful for normalization checks and Monte Carlo integration.

    Parameters
    ----------
    odf : ODF object
        ODF to evaluate
    n_orientations : int, optional
        Number of random orientations to generate, default 1000
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        orientations : Random rotation matrices, shape (n_orientations, 3, 3)
        values : ODF values at these orientations, shape (n_orientations,)

    Examples
    --------
    >>> odf = UniformODF('oh', 'triclinic')
    >>> orientations, values = eval_random_orientations(odf, n_orientations=100)
    >>> print(f"Mean ODF value: {np.mean(values)}")  # Should be ~1.0 (MRD)
    """
    # Haar-uniform random rotations on SO(3).
    orientations = Rotation.random(n_orientations, random_state=seed).as_matrix()

    # Evaluate ODF at these orientations
    values = odf.eval(orientations)

    return orientations, values


def texture_index(odf, n_orientations=100000, seed=None):
    """
    Monte Carlo estimate of the texture index J of an ODF.

    The texture index is the mean of the squared ODF over SO(3) with the
    normalized Haar measure (so vol(SO(3)) = 1):

        J = <f^2> = (1 / 8pi^2) * integral_{SO(3)} f(R)^2 dR

    It is estimated here as <f^2> / <f>^2 over Haar-uniform random
    orientations. The <f>^2 denominator makes the estimate invariant to
    the absolute scale of the ODF, so J = 1 for the uniform (random)
    texture and J >= 1 for any texture, even when the ODF is not
    pre-normalized to mean 1 MRD (e.g. symmetry-reduced kernels, whose
    mean over SO(3) is the symmetry-group order rather than 1). For a
    properly mean-1 ODF this reduces to the usual J = <f^2>.

    Parameters
    ----------
    odf : ODF object
        ODF object with an eval() method (e.g., UniformODF, UnimodalODF)
    n_orientations : int, optional
        Number of Haar-uniform samples used for the estimate, default
        100000. Sharp textures need more samples for a given accuracy.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    float
        Estimated texture index J = <f^2> (>= 1 in MRD units)

    Notes
    -----
    This is a stochastic estimate; its standard error decreases like
    1/sqrt(n_orientations). For a single-mode de la Vallee Poussin ODF
    without symmetry it converges to the analytic value
    J = (2 C^2 / pi) * B(2*kappa + 1/2, 3/2), where C and kappa are the
    kernel normalization constant and shape parameter.

    If the ODF exposes an ``analytic_texture_index()`` method that returns
    a value (not None), that exact result is used instead of sampling.
    """
    analytic = getattr(odf, 'analytic_texture_index', None)
    if callable(analytic):
        value = analytic()
        if value is not None:
            return float(value)

    _, values = eval_random_orientations(
        odf, n_orientations=n_orientations, seed=seed
    )
    # Normalize by <f>^2 so the index is independent of the ODF's absolute
    # scale (it equals <f^2> exactly when the ODF is already mean-1 MRD).
    return float(np.mean(values ** 2) / np.mean(values) ** 2)


def texture_norm(odf, n_orientations=100000, seed=None):
    """
    Monte Carlo estimate of the L2 norm of an ODF (MTEX ``norm``).

    The L2 norm is the square root of the texture index:

        ||f|| = sqrt(<f^2>) = sqrt( (1 / 8pi^2) * integral f(R)^2 dR )

    In MRD units the uniform (random) texture has norm 1; sharper
    textures have larger norms.

    Parameters
    ----------
    odf : ODF object
        ODF object with an eval() method
    n_orientations : int, optional
        Number of Haar-uniform samples used for the estimate, default
        100000
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    float
        Estimated L2 norm ||f|| = sqrt(J) (>= 1 in MRD units)

    See Also
    --------
    texture_index : the squared L2 norm, J = <f^2>
    """
    return float(
        np.sqrt(texture_index(odf, n_orientations=n_orientations, seed=seed))
    )
