"""Tests for the ODF pole figure renderer."""

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from hexrd.phase_transition.texture import plotting as pf_plotting  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Do not leak figures between tests."""
    yield
    plt.close('all')


@pytest.fixture
def scattered_pole_figure():
    """Azimuth, radius and intensity for one small pole figure."""
    rng = np.random.default_rng(0)
    polar = rng.uniform(0.0, np.pi / 2, 200)
    azimuth = rng.uniform(-np.pi, np.pi, 200)
    radius = pf_plotting.stereographic_radius(polar)
    return azimuth, radius, np.cos(polar) + 1.0


class TestStereographicRadius:
    """The projection used for the radial coordinate."""

    def test_pole_maps_to_centre(self):
        """A direction along the sample normal projects to r = 0."""
        assert pf_plotting.stereographic_radius(0.0) == pytest.approx(0.0)

    def test_equator_maps_to_rim(self):
        """A direction in the sample plane projects to r = 1."""
        assert pf_plotting.stereographic_radius(
            np.pi / 2
        ) == pytest.approx(1.0)

    def test_monotonic_over_the_hemisphere(self):
        """Radius grows monotonically from pole to equator."""
        polar = np.linspace(0.0, np.pi / 2, 50)
        radius = pf_plotting.stereographic_radius(polar)

        assert np.all(np.diff(radius) > 0)
        assert np.all((radius >= 0.0) & (radius <= 1.0))

    def test_symmetric_about_the_equator(self):
        """The lower hemisphere folds onto the upper one."""
        polar = np.linspace(0.0, np.pi / 2, 20)

        np.testing.assert_allclose(
            pf_plotting.stereographic_radius(polar),
            pf_plotting.stereographic_radius(np.pi - polar),
            atol=1e-15,
        )


class TestSubplotShape:
    """Grid layout for n pole figures."""

    @pytest.mark.parametrize(
        'n_figures, expected',
        [
            (1, (1, 1)),
            (2, (1, 2)),
            (3, (2, 3)),
            (4, (2, 3)),
            (6, (3, 3)),
            (7, (3, 3)),
        ],
    )
    def test_shape(self, n_figures, expected):
        """Rows fill up to three columns, matching HarmonicModel."""
        assert pf_plotting.subplot_shape(n_figures) == expected

    @pytest.mark.parametrize('n_figures', [3, 6, 9])
    def test_trailing_blank_row_is_preserved(self, n_figures):
        """A multiple of three gets a trailing row of blank axes."""
        n_rows, n_columns = pf_plotting.subplot_shape(n_figures)

        assert n_rows == n_figures // n_columns + 1
        assert n_rows * n_columns - n_figures == n_columns

    def test_every_figure_gets_an_axes(self):
        """The grid is always large enough for all the pole figures."""
        for n_figures in range(1, 13):
            n_rows, n_columns = pf_plotting.subplot_shape(n_figures)
            assert n_rows * n_columns >= n_figures


class TestHklLabel:
    """Reflection labels used as plot titles."""

    def test_three_index(self):
        assert pf_plotting.hkl_label([1, 0, 0]) == '(1, 0, 0)'

    def test_four_index_with_negative(self):
        assert pf_plotting.hkl_label([1, 0, -1, 2]) == '(1, 0, -1, 2)'

    def test_accepts_floats(self):
        """hkls arriving as floats still render as integers."""
        assert pf_plotting.hkl_label(np.array([2.0, 0.0, 0.0])) == '(2, 0, 0)'


class TestCreateAxes:
    """The blank grid of polar axes."""

    def test_axes_shape_is_two_dimensional(self):
        """Axes are always 2-D, so [row][col] indexing is safe."""
        fig, axes = pf_plotting.create_axes(5)

        assert axes.ndim == 2
        assert axes.shape == (2, 3)
        assert fig is axes[0][0].figure

    def test_single_figure(self):
        """One pole figure still yields a 2-D array."""
        _, axes = pf_plotting.create_axes(1)

        assert axes.shape == (1, 1)

    def test_axes_are_polar_and_bare(self):
        """Axes start with no frame, ticks or grid."""
        _, axes = pf_plotting.create_axes(2)

        for ax in axes.flatten():
            assert ax.name == 'polar'
            assert not ax.axison

    def test_axes_for_indexes_row_major(self):
        """axes_for walks the grid left to right, top to bottom."""
        _, axes = pf_plotting.create_axes(6)

        assert pf_plotting.axes_for(axes, 0) is axes[0][0]
        assert pf_plotting.axes_for(axes, 2) is axes[0][2]
        assert pf_plotting.axes_for(axes, 3) is axes[1][0]


class TestDraw:
    """Drawing a single pole figure."""

    def test_draws_contours_and_title(self, scattered_pole_figure):
        """A contour set and colorbar are attached to the axes."""
        azimuth, radius, intensities = scattered_pole_figure
        _, axes = pf_plotting.create_axes(1)
        ax = axes[0][0]

        pf_plotting.draw(ax, azimuth, radius, intensities, label='(1, 1, 1)')

        assert ax.get_title() == '(1, 1, 1)'
        assert ax._hexrd_pf_contours is not None
        assert ax._hexrd_pf_colorbar is not None

    def test_redraw_replaces_artists(self, scattered_pole_figure):
        """Drawing twice does not stack contour sets or colorbars."""
        azimuth, radius, intensities = scattered_pole_figure
        fig, axes = pf_plotting.create_axes(1)
        ax = axes[0][0]

        pf_plotting.draw(ax, azimuth, radius, intensities)
        first = ax._hexrd_pf_contours
        n_axes_after_first = len(fig.axes)

        pf_plotting.draw(ax, azimuth, radius, 2.0 * intensities)

        assert ax._hexrd_pf_contours is not first
        # A stacked colorbar would add another axes to the figure.
        assert len(fig.axes) == n_axes_after_first

    def test_colorbar_can_be_suppressed(self, scattered_pole_figure):
        """colorbar=False leaves the figure with just the pole figure axes."""
        azimuth, radius, intensities = scattered_pole_figure
        fig, axes = pf_plotting.create_axes(1)

        pf_plotting.draw(
            axes[0][0], azimuth, radius, intensities, colorbar=False
        )

        assert len(fig.axes) == 1

    def test_filled_and_line_contours(self, scattered_pole_figure):
        """Both contour styles draw without error."""
        azimuth, radius, intensities = scattered_pole_figure
        _, axes = pf_plotting.create_axes(2)

        for i, filled in enumerate((False, True)):
            contours = pf_plotting.draw(
                pf_plotting.axes_for(axes, i),
                azimuth, radius, intensities, filled=filled,
            )
            assert contours is not None


class TestSeamClosure:
    """The +/-pi azimuth discontinuity."""

    def test_points_on_the_seam_are_mirrored(self):
        """A point at +pi is copied to -pi, carrying its radius and value."""
        azimuth = np.array([-np.pi, 0.0, np.pi])
        radius = np.array([0.1, 0.2, 0.3])
        intensities = np.array([1.0, 2.0, 3.0])

        out_azimuth, out_radius, out_intensities = (
            pf_plotting.close_azimuth_seam(azimuth, radius, intensities)
        )

        assert len(out_azimuth) == 5
        np.testing.assert_allclose(out_azimuth[3:], [np.pi, -np.pi])
        np.testing.assert_allclose(out_radius[3:], [0.1, 0.3])
        np.testing.assert_allclose(out_intensities[3:], [1.0, 3.0])

    def test_interior_points_are_left_alone(self):
        """Data away from the seam is passed straight through."""
        azimuth = np.array([0.0, 1.0, -1.0])
        radius = np.array([0.1, 0.2, 0.3])
        intensities = np.array([1.0, 2.0, 3.0])

        out = pf_plotting.close_azimuth_seam(azimuth, radius, intensities)

        for original, returned in zip((azimuth, radius, intensities), out):
            np.testing.assert_array_equal(original, returned)

    def test_grid_missing_the_seam_is_left_open(self):
        """FIXME: a grid with no point exactly at +/-pi is not closed."""
        azimuth = np.linspace(0.0, 2 * np.pi, 72)
        azimuth = np.arctan2(np.sin(azimuth), np.cos(azimuth))
        radius = np.full_like(azimuth, 0.5)

        assert not np.any(np.isclose(np.abs(azimuth), np.pi))

        out_azimuth, _, _ = pf_plotting.close_azimuth_seam(
            azimuth, radius, radius
        )

        assert len(out_azimuth) == len(azimuth)

    def test_seam_is_closed_when_the_grid_lands_on_it(self):
        """
        End to end: a grid that does place points at +/-pi triangulates
        across the branch cut.
        """
        polar = np.repeat(np.linspace(0.05, np.pi / 2, 20), 72)
        azimuth = np.tile(
            np.linspace(-np.pi, np.pi, 72), 20
        )
        radius = pf_plotting.stereographic_radius(polar)

        _, axes = pf_plotting.create_axes(1)
        contours = pf_plotting.draw(
            axes[0][0], azimuth, radius, np.cos(polar), filled=True
        )

        assert len(contours.get_paths()) > 0


class TestPlotPoleFigures:
    """The whole-figure convenience wrapper."""

    def test_one_axes_per_reflection(self, scattered_pole_figure):
        """Each label gets its own titled panel."""
        azimuth, radius, intensities = scattered_pole_figure
        labels = ['(1, 1, 1)', '(2, 0, 0)']

        fig, axes = pf_plotting.plot_pole_figures(
            labels, [azimuth] * 2, [radius] * 2, [intensities] * 2
        )

        assert fig is not None
        titles = [pf_plotting.axes_for(axes, i).get_title() for i in range(2)]
        assert titles == labels

    def test_mismatched_lengths_rejected(self, scattered_pole_figure):
        """All four sequences must describe the same set of figures."""
        azimuth, radius, intensities = scattered_pole_figure

        with pytest.raises(ValueError):
            pf_plotting.plot_pole_figures(
                ['a', 'b'], [azimuth], [radius], [intensities]
            )

    def test_window_title(self, scattered_pole_figure):
        """A window title is applied when the backend supports one."""
        azimuth, radius, intensities = scattered_pole_figure

        fig, _ = pf_plotting.plot_pole_figures(
            ['(1, 1, 1)'], [azimuth], [radius], [intensities],
            window_title='Pole Figures for Ruby',
        )

        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__])
