"""
Validate calculated pole figures against reference values.

The reference values below are transcribed constants, not a data file:
they come from an independent calculation of the same two model ODFs,
run outside this repository. Only the numbers needed to pin the result
are kept here - each reflection's peak density, its solid-angle-weighted
mean, and its density at eight fixed grid points - so the test has no
external dependency.

The two phases are 4H-SiC (hexagonal, 6/mmm) and B1-SiC (cubic, m-3m),
each given a single-mode de la Vallee Poussin ODF about the same
orientation. The B1 model uses one stubbed orientation rather than a set
of crystallographic variants, which is why its densities are milder.

Tolerances are set by the reference's own approximation, not by ours.
It expands the pole density in spherical harmonics and truncates the
series, which rings around sharp peaks: its values dip to -0.06 (4H)
and -0.009 (B1) where a density cannot be negative at all. Agreement is
therefore asserted to 0.25% of each figure's peak, and the sign of that
residual is asserted separately - our closed form stays non-negative
where the reference does not.
"""

import numpy as np
import pytest

from hexrd.phase_transition.texture import (
    DeLaValleePoussinKernel,
    UnimodalODF,
    calc_pole_figure,
    directions_from_angles,
    regular_s2_grid,
)

# --------------------------------------------------------------- the model

N_POLAR = 30
N_AZIMUTH = 72

# Orientation of the single mode: a tilt about x, then a clocking about z.
TILT_DEG = 27.48
CLOCKING_DEG = 15.0

# 4H-SiC: hexagonal, a = 2.5812 A, c = 8.45 A, Laue group 6/mmm.
LPARMS_4H = (2.5812, 8.45)
HALFWIDTH_4H_DEG = 4.0
HKLS_4H = (
    (0, 0, 0, 2),
    (1, 0, -1, 0),
    (1, 0, -1, 1),
    (0, 0, 0, 4),
    (1, 0, -1, 2),
    (1, 0, -1, 3),
)

# B1-SiC: cubic (rock salt), a = 3.675 A, Laue group m-3m.
LPARMS_B1 = 3.675
HALFWIDTH_B1_DEG = 6.0
HKLS_B1 = (
    (1, 1, 1),
    (2, 0, 0),
    (2, 2, 0),
    (3, 1, 1),
    (2, 2, 2),
)

# ---------------------------------------------------------- the references

#: Grid points the sampled densities below are taken at.
SAMPLE_INDICES = (0, 271, 542, 813, 1084, 1355, 1626, 1897)

PEAK_4H = (278.9593, 94.5822, 47.0684, 278.9593, 46.9771, 46.4879)
MEAN_4H = (0.9603, 1.0772, 0.9700, 0.9603, 1.1101, 0.9965)
SAMPLES_4H = (
    (0.02816, -0.02575, 0.02039, -0.02581, 0.02553, -0.01593, 4.33251,
     0.08345),
    (-0.01154, 0.00361, 0.00026, 0.00808, -0.00537, 0.00142, 0.00246,
     -0.00110),
    (0.00306, 0.00151, -0.00099, -0.00429, -0.00642, 0.00019, -0.00250,
     0.00142),
    (0.02816, -0.02575, 0.02039, -0.02581, 0.02553, -0.01593, 4.33251,
     0.08345),
    (0.00301, 0.00179, -0.00237, -0.00114, -0.01297, 0.00268, -0.00187,
     0.00109),
    (-0.00407, 0.00020, 0.00471, 0.00329, 0.00825, -0.00313, -0.00270,
     0.00245),
)

PEAK_B1 = (30.6910, 42.2082, 20.8134, 10.5117, 30.6910)
MEAN_B1 = (0.9627, 1.0394, 0.9767, 0.9974, 0.9627)
SAMPLES_B1 = (
    (0.00326, -0.00252, 0.00269, -0.00094, -0.00403, -0.00212, -0.00045,
     0.00094),
    (0.00111, 0.00163, -0.00079, -0.00143, -0.00415, 0.00526, 6.51495,
     0.49090),
    (0.05123, 0.13547, 1.59341, 2.25999, 0.01567, -0.00059, -0.00135,
     0.00057),
    (0.01608, 0.01778, 0.00136, 0.00456, 0.49848, 7.00820, 0.05080,
     1.31688),
    (0.00326, -0.00252, 0.00269, -0.00094, -0.00403, -0.00212, -0.00045,
     0.00094),
)

#: Fraction of a figure's peak the reference's truncation accounts for.
#:
#: The residual scales with each figure's own peak rather than with the
#: phase's strongest figure: across both phases it runs 0.03% to 0.20% of
#: the figure's own maximum, so 0.25% covers every reflection with a
#: little margin.
TRUNCATION_TOLERANCE = 2.5e-3

# ----------------------------------------------------------------- helpers


def _rotation_matrix(axis, degrees):
    """Rotation matrix about an axis, by the right-hand rule."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def _crystal_directions_hexagonal(hkils, lparms):
    """
    Cartesian reciprocal lattice vectors for 4-index hexagonal indices.

    Built in the frame the reference used: a* along x with length
    2 / (a sqrt(3)), b* at 60 degrees to it, and c* along z with length
    1 / c. The pole figure normalizes each direction, but the basis has
    to be the reference's or the figures come out rotated.
    """
    a, c = lparms
    a_star = 2.0 / (a * np.sqrt(3.0))
    basis = (
        np.array([a_star, 0.0, 0.0]),
        np.array([a_star * 0.5, a_star * np.sqrt(3.0) / 2.0, 0.0]),
        np.array([0.0, 0.0, 1.0 / c]),
    )
    return np.array([
        h * basis[0] + k * basis[1] + ell * basis[2]
        for h, k, _, ell in hkils
    ])


def _crystal_directions_cubic(hkls, lparm):
    """Reciprocal lattice vectors for cubic indices: g = (h, k, l) / a."""
    return np.asarray(hkls, dtype=float) / lparm


@pytest.fixture(scope='module')
def specimen_grid():
    """The 30 x 72 specimen grid the reference values are sampled on."""
    polar, azimuth = regular_s2_grid(n_polar=N_POLAR, n_azimuth=N_AZIMUTH)
    return polar, directions_from_angles(polar, azimuth)


@pytest.fixture(scope='module')
def orientation():
    """The single modal orientation shared by both phases."""
    return (
        _rotation_matrix([0, 0, 1], CLOCKING_DEG)
        @ _rotation_matrix([1, 0, 0], TILT_DEG)
    )


def _pole_figures(orientation, specimen, laue, halfwidth_deg, directions):
    """Pole figures of a one-mode ODF, as (n_points, n_figures)."""
    kernel = DeLaValleePoussinKernel(
        halfwidth=np.radians(halfwidth_deg), crystal_symmetry=laue
    )
    figures = calc_pole_figure(
        UnimodalODF(orientation, kernel),
        directions,
        specimen_directions=specimen,
    )
    return figures.values.T


@pytest.fixture(scope='module')
def phases(orientation, specimen_grid):
    """Both phases' pole figures, keyed by name."""
    _, specimen = specimen_grid
    return {
        '4H': _pole_figures(
            orientation, specimen, 'd6h', HALFWIDTH_4H_DEG,
            _crystal_directions_hexagonal(HKLS_4H, LPARMS_4H),
        ),
        'B1': _pole_figures(
            orientation, specimen, 'oh', HALFWIDTH_B1_DEG,
            _crystal_directions_cubic(HKLS_B1, LPARMS_B1),
        ),
    }


PHASE_CASES = (
    ('4H', PEAK_4H, MEAN_4H, SAMPLES_4H),
    ('B1', PEAK_B1, MEAN_B1, SAMPLES_B1),
)


# ------------------------------------------------------------------- tests


@pytest.mark.parametrize('name, peaks, means, samples', PHASE_CASES)
class TestAgainstReference:
    """Calculated densities against the transcribed reference values."""

    def test_peak_density(self, phases, name, peaks, means, samples):
        """Each reflection peaks at the reference's value."""
        values = phases[name]

        for index, expected in enumerate(peaks):
            assert values[:, index].max() == pytest.approx(
                expected, abs=TRUNCATION_TOLERANCE * expected
            )

    def test_sampled_densities(self, phases, name, peaks, means, samples):
        """Density at eight fixed grid points matches the reference."""
        values = phases[name]

        for index, expected in enumerate(samples):
            np.testing.assert_allclose(
                values[SAMPLE_INDICES, index],
                expected,
                atol=TRUNCATION_TOLERANCE * peaks[index],
            )

    def test_mean_density(
        self, phases, specimen_grid, name, peaks, means, samples
    ):
        """
        The solid-angle-weighted mean matches the reference.

        A plain mean over this grid is not 1 MRD: a regular polar/azimuth
        grid oversamples the pole, so the points have to be weighted by
        sin(polar) to sample the sphere evenly.
        """
        polar, _ = specimen_grid
        weights = np.sin(polar)
        values = phases[name]

        for index, expected in enumerate(means):
            weighted = (
                (values[:, index] * weights).sum() / weights.sum()
            )
            assert weighted == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize('name, peaks, means, samples', PHASE_CASES)
class TestNormalization:
    """Properties a pole density must have, independent of the reference."""

    def test_weighted_mean_is_one_mrd(
        self, phases, specimen_grid, name, peaks, means, samples
    ):
        """
        Every figure averages to 1 MRD over the sphere.

        The tolerance is quadrature error, not modelling error: a 4
        degree kernel is narrow against this grid's ~5 degree spacing, so
        a coarse sum over it recovers the mean only to about 15%.
        """
        polar, _ = specimen_grid
        weights = np.sin(polar)
        values = phases[name]

        for index in range(values.shape[1]):
            weighted = (
                (values[:, index] * weights).sum() / weights.sum()
            )
            assert weighted == pytest.approx(1.0, abs=0.15)

    def test_densities_are_non_negative(
        self, phases, name, peaks, means, samples
    ):
        """
        Ours never goes negative, where the reference's does.

        Evaluating the Radon transform in closed form cannot ring, so the
        density stays non-negative by construction. The reference's
        truncated series does not: its sampled values above include
        negatives, which is the residual the tolerances allow for.
        """
        assert phases[name].min() >= 0.0
        assert min(min(row) for row in samples) < 0.0


@pytest.mark.parametrize(
    'name, first, second',
    [
        ('4H', 0, 3),  # (0 0 0 2) and (0 0 0 4)
        ('B1', 0, 4),  # (1 1 1) and (2 2 2)
    ],
)
def test_parallel_poles_give_identical_figures(phases, name, first, second):
    """
    Reflections along the same direction share a pole figure.

    (0 0 0 2) is parallel to (0 0 0 4), and (1 1 1) to (2 2 2); a pole
    figure depends only on the direction, not the order of the
    reflection. The reference's values agree on this too - its peaks for
    each pair are equal - so this checks our figures against each other
    and against that.
    """
    values = phases[name]

    np.testing.assert_allclose(
        values[:, first], values[:, second], rtol=1e-12
    )
