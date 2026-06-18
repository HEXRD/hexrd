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
