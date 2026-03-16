"""
ODF Norm and Texture Index Calculations

Provides functions for calculating texture strength and orientation distribution
characteristics through L2 norms and related texture measures.
"""

import numpy as np
import warnings
from .evaluation import eval_random_orientations, eval_odf


def norm_l2(odf, n_samples=10000, seed=None, convergence_check=True):
    """
    Calculate L2 norm of an ODF using Monte Carlo integration.

    Computes ||f||₂ = √(∫[f(g)]² dg) over SO(3) using random sampling.

    Parameters
    ----------
    odf : ODF object
        ODF to calculate norm for (must have eval() method)
    n_samples : int, optional
        Number of random orientations for Monte Carlo integration, default 10000
    seed : int, optional
        Random seed for reproducibility
    convergence_check : bool, optional
        Whether to check convergence and warn if insufficient, default True

    Returns
    -------
    float
        L2 norm of the ODF

    Examples
    --------
    >>> from hexrd.phase_transition.texture import UniformODF
    >>> odf = UniformODF('oh', 'triclinic')
    >>> norm_val = norm_l2(odf, n_samples=5000)
    >>> print(f"L2 norm: {norm_val:.6f}")  # Should be ~1.0 for uniform
    """
    if not hasattr(odf, 'eval'):
        raise TypeError(f"ODF object {type(odf).__name__} must have an eval() method")

    # Generate random orientations and evaluate ODF
    orientations, values = eval_random_orientations(odf, n_orientations=n_samples, seed=seed)

    # Calculate squared values for L2 norm
    values_squared = values ** 2

    # Monte Carlo integration: ||f||₂² = V(SO(3)) * mean([f(g)]²)
    volume_SO3 = 8.0 * np.pi**2
    l2_norm_squared = volume_SO3 * np.mean(values_squared)

    # Take square root to get L2 norm
    l2_norm = np.sqrt(l2_norm_squared)

    # Optional convergence check
    if convergence_check:
        _check_convergence(values_squared, n_samples)

    return float(l2_norm)


def texture_index(odf, n_samples=10000, seed=None):
    """
    Calculate texture index J = ||f||₂².

    The texture index quantifies texture strength:
    - J = 1: Random texture (uniform ODF)
    - J > 1: Textured material (higher = stronger texture)

    Parameters
    ----------
    odf : ODF object
        ODF to analyze
    n_samples : int, optional
        Number of random orientations for integration, default 10000
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    float
        Texture index J

    Examples
    --------
    >>> odf_uniform = UniformODF('oh', 'triclinic')
    >>> j_uniform = texture_index(odf_uniform)  # Should be ~1.0
    >>>
    >>> odf_textured = UnimodalODF(np.eye(3), kernel, 'oh', 'triclinic')
    >>> j_textured = texture_index(odf_textured)  # Should be >1.0
    """
    # J = ||f||₂², so we can compute directly without square root
    orientations, values = eval_random_orientations(odf, n_orientations=n_samples, seed=seed)
    values_squared = values ** 2

    volume_SO3 = 8.0 * np.pi**2
    texture_idx = volume_SO3 * np.mean(values_squared)

    return float(texture_idx)


def texture_strength(odf, n_samples=10000, seed=None):
    """
    Alias for texture_index() - same calculation, more intuitive name.

    Parameters
    ----------
    odf : ODF object
        ODF to analyze
    n_samples : int, optional
        Number of random orientations, default 10000
    seed : int, optional
        Random seed

    Returns
    -------
    float
        Texture strength (same as texture index)
    """
    return texture_index(odf, n_samples=n_samples, seed=seed)


def max_intensity_ratio(odf, n_samples_max=1000, n_samples_baseline=5000, seed=None):
    """
    Calculate ratio of maximum ODF intensity to uniform baseline.

    Alternative texture measure: max(f) / f_uniform where f_uniform = 1/(8π²).

    Parameters
    ----------
    odf : ODF object
        ODF to analyze
    n_samples_max : int, optional
        Number of samples to estimate maximum, default 1000
    n_samples_baseline : int, optional
        Number of samples for baseline estimation, default 5000
    seed : int, optional
        Random seed

    Returns
    -------
    float
        Maximum intensity ratio
    """
    if seed is not None:
        np.random.seed(seed)

    # Estimate maximum by sampling
    orientations, values = eval_random_orientations(odf, n_orientations=n_samples_max, seed=seed)
    estimated_max = np.max(values)

    # Uniform baseline
    uniform_baseline = 1.0 / (8.0 * np.pi**2)

    return float(estimated_max / uniform_baseline)


def peak_to_mean_ratio(odf, n_samples=10000, seed=None):
    """
    Calculate ratio of peak intensity to mean intensity.

    Measures texture sharpness: higher ratios indicate more concentrated textures.

    Parameters
    ----------
    odf : ODF object
        ODF to analyze
    n_samples : int, optional
        Number of samples for estimation, default 10000
    seed : int, optional
        Random seed

    Returns
    -------
    float
        Peak to mean ratio
    """
    orientations, values = eval_random_orientations(odf, n_orientations=n_samples, seed=seed)

    peak_value = np.max(values)
    mean_value = np.mean(values)

    if mean_value == 0:
        return float('inf')

    return float(peak_value / mean_value)


def texture_summary(odf, n_samples=10000, seed=None):
    """
    Comprehensive texture analysis summary.

    Calculates multiple texture measures and provides interpretation.

    Parameters
    ----------
    odf : ODF object
        ODF to analyze
    n_samples : int, optional
        Number of samples for calculations, default 10000
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with texture analysis results:
        - 'l2_norm': L2 norm value
        - 'texture_index': Texture index J
        - 'max_intensity_ratio': Peak/uniform ratio
        - 'peak_to_mean_ratio': Peak/mean ratio
        - 'interpretation': Text description of texture strength
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate all measures using same random orientations for consistency
    orientations, values = eval_random_orientations(odf, n_orientations=n_samples, seed=seed)

    # L2 norm and texture index
    values_squared = values ** 2
    volume_SO3 = 8.0 * np.pi**2
    j_value = volume_SO3 * np.mean(values_squared)
    l2_norm_val = np.sqrt(j_value)

    # Maximum intensity measures
    peak_value = np.max(values)
    mean_value = np.mean(values)
    uniform_baseline = 1.0 / volume_SO3

    max_intensity_ratio_val = peak_value / uniform_baseline
    peak_to_mean_ratio_val = peak_value / mean_value if mean_value > 0 else float('inf')

    # Interpretation based on texture index
    if j_value <= 1.1:
        interpretation = "Random texture (very weak)"
    elif j_value <= 2.0:
        interpretation = "Weak texture"
    elif j_value <= 5.0:
        interpretation = "Moderate texture"
    elif j_value <= 20.0:
        interpretation = "Strong texture"
    else:
        interpretation = "Very strong texture"

    return {
        'l2_norm': float(l2_norm_val),
        'texture_index': float(j_value),
        'max_intensity_ratio': float(max_intensity_ratio_val),
        'peak_to_mean_ratio': float(peak_to_mean_ratio_val),
        'interpretation': interpretation,
        'n_samples_used': n_samples
    }


def compare_texture_strength(*odfs, labels=None, n_samples=10000, seed=None):
    """
    Compare texture strength between multiple ODFs.

    Parameters
    ----------
    *odfs : ODF objects
        Variable number of ODFs to compare
    labels : list of str, optional
        Labels for each ODF, default None uses indices
    n_samples : int, optional
        Number of samples for calculations, default 10000
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with comparison results

    Examples
    --------
    >>> odf1 = UniformODF('oh', 'triclinic')
    >>> odf2 = UnimodalODF(np.eye(3), kernel, 'oh', 'triclinic')
    >>> comparison = compare_texture_strength(odf1, odf2,
    ...                                      labels=['Uniform', 'Textured'])
    """
    if len(odfs) < 2:
        raise ValueError("Need at least 2 ODFs to compare")

    if labels is None:
        labels = [f"ODF_{i}" for i in range(len(odfs))]
    elif len(labels) != len(odfs):
        raise ValueError("Number of labels must match number of ODFs")

    results = {}
    j_values = []

    for i, (odf, label) in enumerate(zip(odfs, labels)):
        j_val = texture_index(odf, n_samples=n_samples, seed=seed)
        j_values.append(j_val)
        results[label] = {
            'texture_index': float(j_val),
            'rank': 0  # Will be filled in below
        }

    # Add rankings (1 = strongest texture)
    sorted_indices = np.argsort(j_values)[::-1]  # Descending order
    for rank, idx in enumerate(sorted_indices):
        label = labels[idx]
        results[label]['rank'] = rank + 1

    results['summary'] = {
        'strongest_texture': labels[sorted_indices[0]],
        'weakest_texture': labels[sorted_indices[-1]],
        'texture_index_range': [float(np.min(j_values)), float(np.max(j_values))]
    }

    return results


def _check_convergence(values_squared, n_samples, min_samples=1000):
    """
    Check if Monte Carlo integration has likely converged.

    Issues warning if integration may be unreliable.

    Parameters
    ----------
    values_squared : numpy.ndarray
        Squared ODF values from sampling
    n_samples : int
        Number of samples used
    min_samples : int, optional
        Minimum samples recommended, default 1000
    """
    if n_samples < min_samples:
        warnings.warn(
            f"Monte Carlo integration used only {n_samples} samples "
            f"(recommended: >{min_samples}). Results may be unreliable.",
            UserWarning
        )

    # Check for high variance (may indicate need for more samples)
    mean_val = np.mean(values_squared)
    std_val = np.std(values_squared)

    if mean_val > 0:
        coefficient_of_variation = std_val / mean_val
        if coefficient_of_variation > 1.0:  # High relative variance
            warnings.warn(
                f"High variance in Monte Carlo integration (CV={coefficient_of_variation:.2f}). "
                f"Consider increasing n_samples for more reliable results.",
                UserWarning
            )
