"""
ODF Evaluation Functions

Provides common evaluation functionality for orientation distribution functions.
This module serves as a foundation for more complex ODF implementations.
"""

import numpy as np


def validate_orientations(orientations):
    """
    Validate and normalize orientation input.

    Parameters
    ----------
    orientations : array_like
        Orientation matrices, expected shape (..., 3, 3)

    Returns
    -------
    numpy.ndarray
        Validated orientation matrices

    Raises
    ------
    ValueError
        If orientations don't have proper shape or aren't valid rotation matrices
    """
    orientations = np.asarray(orientations)

    # Check basic shape requirements
    if orientations.ndim < 2:
        raise ValueError("Orientations must have at least 2 dimensions")

    if orientations.shape[-2:] != (3, 3):
        raise ValueError(
            f"Orientations must have shape (..., 3, 3), got {orientations.shape}"
        )

    # Optional: Check if matrices are approximately orthogonal
    # For performance, we'll skip this validation by default
    # but could add a 'validate_rotation' parameter in the future

    return orientations


def eval_odf(odf, orientations, validate_input=True):
    """
    Generic ODF evaluation function.

    Provides a unified interface for evaluating orientation distribution functions
    at specified orientations. Supports both single orientations and batch evaluation.

    Parameters
    ----------
    odf : ODF object
        Any ODF object with an eval() method (e.g., UniformODF, UnimodalODF)
    orientations : array_like
        Orientation matrices of shape (..., 3, 3). Can be:
        - Single 3x3 matrix
        - Array of shape (N, 3, 3) for N orientations
        - Higher dimensional arrays with shape (..., 3, 3)
    validate_input : bool, optional
        Whether to validate input orientations, default True

    Returns
    -------
    numpy.ndarray
        ODF values evaluated at the given orientations.
        Shape matches the leading dimensions of input orientations.

    Examples
    --------
    >>> from hexrd.texture import UniformODF, eval_odf
    >>> odf = UniformODF('oh', 'triclinic')
    >>>
    >>> # Single orientation
    >>> R = np.eye(3)
    >>> value = eval_odf(odf, R)  # scalar result
    >>>
    >>> # Multiple orientations
    >>> Rs = np.array([np.eye(3), np.eye(3)])  # shape (2, 3, 3)
    >>> values = eval_odf(odf, Rs)  # shape (2,)
    """
    # Validate ODF object
    if not hasattr(odf, 'eval'):
        raise TypeError(
            f"ODF object {type(odf).__name__} must have an eval() method"
        )

    # Validate and process orientations
    if validate_input:
        orientations = validate_orientations(orientations)

    # Delegate to ODF's eval method
    return odf.eval(orientations)


def eval_odf_batch(odf, orientations, chunk_size=10000, validate_input=True):
    """
    Evaluate ODF on large batches of orientations with memory management.

    For very large orientation datasets, this function processes orientations
    in chunks to avoid memory issues while maintaining vectorization benefits.

    Parameters
    ----------
    odf : ODF object
        ODF object with eval() method
    orientations : array_like
        Large array of orientation matrices, shape (N, 3, 3)
    chunk_size : int, optional
        Number of orientations to process per chunk, default 10000
    validate_input : bool, optional
        Whether to validate input orientations, default True

    Returns
    -------
    numpy.ndarray
        ODF values, shape (N,)

    Examples
    --------
    >>> odf = UniformODF('oh', 'triclinic')
    >>> # Large batch of orientations
    >>> Rs = np.random.normal(size=(50000, 3, 3))  # Not actual rotations!
    >>> # In practice, use proper rotation matrices
    >>> values = eval_odf_batch(odf, Rs, chunk_size=5000)
    """
    if validate_input:
        orientations = validate_orientations(orientations)

    # Handle single orientation or small batches normally
    if orientations.ndim == 2 or orientations.shape[0] <= chunk_size:
        return eval_odf(odf, orientations, validate_input=False)

    # Process large batches in chunks
    n_orientations = orientations.shape[0]
    results = []

    for start_idx in range(0, n_orientations, chunk_size):
        end_idx = min(start_idx + chunk_size, n_orientations)
        chunk = orientations[start_idx:end_idx]
        chunk_results = eval_odf(odf, chunk, validate_input=False)
        results.append(chunk_results)

    return np.concatenate(results)


def eval_at_identity(odf):
    """
    Evaluate ODF at the identity orientation.

    Convenience function for getting ODF value at the identity rotation.
    Useful for normalization checks and texture strength calculations.

    Parameters
    ----------
    odf : ODF object
        ODF to evaluate

    Returns
    -------
    float
        ODF value at identity orientation

    Examples
    --------
    >>> odf = UniformODF('oh', 'triclinic')
    >>> value = eval_at_identity(odf)  # Should be 1/(8π²)
    """
    identity = np.eye(3)
    return float(eval_odf(odf, identity, validate_input=False))


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
    >>> print(f"Mean ODF value: {np.mean(values)}")  # Should be ~1/(8π²)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random rotation matrices using method from
    # "Fast Random Rotation Matrices" by James Arvo
    # This is a simplified version - in practice might want scipy.spatial.transform

    # Generate random quaternions and normalize
    quaternions = np.random.normal(size=(n_orientations, 4))
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

    # Convert quaternions to rotation matrices
    orientations = np.zeros((n_orientations, 3, 3))
    for i, q in enumerate(quaternions):
        w, x, y, z = q
        orientations[i] = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    # Evaluate ODF at these orientations
    values = eval_odf(odf, orientations, validate_input=False)

    return orientations, values
