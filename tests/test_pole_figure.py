"""Tests for pole figure calculation from ODFs."""

import matplotlib
matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from hexrd.phase_transition.texture.kernels import (  # noqa: E402
    DeLaValleePoussinKernel,
    SO3Kernel,
)
from hexrd.phase_transition.texture.pole_figure import (  # noqa: E402
    PoleFigures,
    angles_from_directions,
    calc_pole_figure,
    directions_from_angles,
    equivalent_directions,
    pole_density,
    regular_s2_grid,
)
from hexrd.phase_transition.texture.unimodal_odf import UnimodalODF  # noqa: E402,E501
from hexrd.phase_transition.texture.uniform_odf import UniformODF  # noqa: E402,E501


IDENTITY_OPERATION = np.eye(3).reshape(1, 3, 3)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def grid():
    """The default specimen direction grid, as unit vectors."""
    polar, azimuth = regular_s2_grid()
    return directions_from_angles(polar, azimuth)


@pytest.fixture
def cubic_odf():
    """A single-mode cubic ODF tilted off the specimen axes."""
    kernel = DeLaValleePoussinKernel(
        halfwidth=np.radians(10.0), crystal_symmetry='oh'
    )
    orientation = Rotation.from_euler(
        'ZXZ', [30.0, 20.0, 10.0], degrees=True
    ).as_matrix()
    return UnimodalODF(orientation, kernel)


def _sphere_mean(values, directions):
    """Area-weighted mean over the grid (weights go as sin(polar))."""
    polar, _ = angles_from_directions(directions)
    weights = np.sin(polar)
    return float((values * weights).sum() / weights.sum())


class TestGrid:
    """Specimen direction grids and angle conversions."""

    def test_default_grid_size(self):
        """The default grid is 30 polar by 72 azimuthal points."""
        polar, azimuth = regular_s2_grid()

        assert polar.shape == azimuth.shape == (2160,)
        assert len(np.unique(polar)) == 30
        assert len(np.unique(azimuth)) == 72

    def test_grid_covers_the_upper_hemisphere(self):
        """Polar spans 0 to pi/2 and azimuth 0 to 2*pi, endpoints included."""
        polar, azimuth = regular_s2_grid()

        assert polar.min() == pytest.approx(0.0)
        assert polar.max() == pytest.approx(np.pi / 2)
        assert azimuth.min() == pytest.approx(0.0)
        assert azimuth.max() == pytest.approx(2 * np.pi)

    def test_polar_varies_fastest(self):
        """Polar varies fastest, so the grid ravels column-major."""
        polar, azimuth = regular_s2_grid(n_polar=4, n_azimuth=3)

        np.testing.assert_allclose(azimuth[:4], 0.0)
        assert len(np.unique(polar[:4])) == 4

    def test_directions_are_unit_vectors(self, grid):
        """Every generated specimen direction is normalized."""
        np.testing.assert_allclose(np.linalg.norm(grid, axis=1), 1.0)

    def test_angles_round_trip(self):
        """directions_from_angles and angles_from_directions invert."""
        polar_in = np.array([0.1, 0.7, 1.4])
        azimuth_in = np.array([0.0, 1.0, -2.0])

        polar, azimuth = angles_from_directions(
            directions_from_angles(polar_in, azimuth_in)
        )

        np.testing.assert_allclose(polar, polar_in)
        np.testing.assert_allclose(azimuth, azimuth_in)

    def test_pole_direction(self):
        """Polar angle 0 is the sample normal."""
        np.testing.assert_allclose(
            directions_from_angles(0.0, 0.0), [0.0, 0.0, 1.0]
        )


class TestEquivalentDirections:
    """Symmetry equivalents of a crystal direction."""

    def test_antipodal_doubles_a_trivial_group(self):
        """With no symmetry, antipodal gives h and -h."""
        with_antipodal = equivalent_directions(
            [0.0, 0.0, 1.0], IDENTITY_OPERATION, antipodal=True
        )
        without = equivalent_directions(
            [0.0, 0.0, 1.0], IDENTITY_OPERATION, antipodal=False
        )

        assert len(with_antipodal) == 2
        assert len(without) == 1

    def test_cubic_multiplicities(self):
        """<100>, <110> and <111> have the expected cubic multiplicities."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10.0), crystal_symmetry='oh'
        )
        operations = kernel.crystal_symmetry_operations

        multiplicity = {
            (0, 0, 1): 6,
            (1, 1, 0): 12,
            (1, 1, 1): 8,
        }
        for direction, expected in multiplicity.items():
            unit = np.asarray(direction, float)
            unit = unit / np.linalg.norm(unit)
            assert len(
                equivalent_directions(unit, operations)
            ) == expected

    def test_hexagonal_c_axis(self):
        """The hexagonal c axis has only itself and its opposite."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10.0), crystal_symmetry='d6h'
        )

        equivalents = equivalent_directions(
            [0.0, 0.0, 1.0], kernel.crystal_symmetry_operations
        )

        assert len(equivalents) == 2

    def test_equivalents_are_distinct(self):
        """Duplicates produced by the group are removed."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10.0), crystal_symmetry='oh'
        )

        equivalents = equivalent_directions(
            [0.0, 0.0, 1.0], kernel.crystal_symmetry_operations
        )

        for i, first in enumerate(equivalents):
            for second in equivalents[i + 1:]:
                assert np.linalg.norm(first - second) > 1e-6


class TestRadonTransform:
    """The kernel's Radon transform, which pole figures are built from."""

    def test_peak_value(self):
        """At zero angle the S2 kernel peaks at 1 + kappa."""
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10.0))

        assert kernel.radon(1.0) == pytest.approx(1.0 + kernel.kappa)

    def test_vanishes_at_the_opposite_pole(self):
        """The kernel is zero 180 degrees away."""
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10.0))

        assert kernel.radon(-1.0) == pytest.approx(0.0)

    def test_mean_over_the_sphere_is_one(self):
        """
        The Radon transform integrates to 1 MRD, which is what keeps a
        pole figure on the same scale as the ODF.
        """
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10.0))

        # <RK> = (1/2) * integral over cos from -1 to 1
        cosines = np.linspace(-1.0, 1.0, 200001)
        mean = 0.5 * np.trapz(kernel.radon(cosines), cosines)

        assert mean == pytest.approx(1.0, rel=1e-6)

    def test_halfwidth_relation(self):
        """
        The S2 kernel falls to half its peak at the angle implied by
        kappa, which is where the ODF halfwidth puts it.
        """
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10.0))

        half_angle = np.arccos(2.0 * 0.5 ** (1.0 / kernel.kappa) - 1.0)

        assert kernel.radon(np.cos(half_angle)) == pytest.approx(
            0.5 * kernel.radon(1.0)
        )

    def test_kernel_without_radon_is_rejected(self, grid):
        """A kernel with no Radon transform cannot make a pole figure."""

        class PlainKernel(SO3Kernel):
            def eval(self, R1, R2):
                return np.zeros(np.shape(R1)[:-2])

        odf = UnimodalODF(np.eye(3), PlainKernel())

        with pytest.raises(NotImplementedError):
            pole_density(odf, [0.0, 0.0, 1.0], grid)


class TestUniformODF:
    """Pole figures of a random texture."""

    def test_density_is_one_everywhere(self, grid):
        """An isotropic ODF projects to a flat 1 MRD pole figure."""
        values = pole_density(UniformODF(), [0.0, 0.0, 1.0], grid)

        np.testing.assert_allclose(values, 1.0)

    def test_independent_of_crystal_direction(self, grid):
        """Every reflection gives the same flat pole figure."""
        uniform = UniformODF()

        for direction in ([0, 0, 1], [1, 1, 0], [1, 2, 3]):
            np.testing.assert_allclose(
                pole_density(uniform, direction, grid), 1.0
            )

    def test_single_direction_returns_scalar(self):
        """A single (3,) specimen direction evaluates to a scalar."""
        value = UniformODF().pole_density(
            [0.0, 0.0, 1.0], np.array([0.0, 0.0, 1.0])
        )

        assert value == 1.0


class TestUnimodalODF:
    """Pole figures of a kernel-based texture."""

    def test_normalized_to_mrd(self, cubic_odf, grid):
        """The pole figure averages to 1 MRD over the sphere."""
        values = pole_density(cubic_odf, [0.0, 0.0, 1.0], grid)

        assert _sphere_mean(values, grid) == pytest.approx(1.0, abs=0.05)

    def test_strictly_non_negative(self, cubic_odf, grid):
        """A pole density is a density; the closed form never rings."""
        for direction in ([0, 0, 1], [1, 1, 0], [1, 1, 1]):
            assert np.all(pole_density(cubic_odf, direction, grid) >= 0.0)

    def test_peaks_at_the_rotated_crystal_direction(self, cubic_odf):
        """
        The maximum sits where the mode sends the crystal direction, which
        is what makes a pole figure interpretable.
        """
        direction = np.array([0.0, 0.0, 1.0])
        expected = cubic_odf.modal_orientations[0] @ direction
        if expected[2] < 0:
            expected = -expected

        polar, azimuth = regular_s2_grid(n_polar=180, n_azimuth=360)
        specimen = directions_from_angles(polar, azimuth)
        values = pole_density(cubic_odf, direction, specimen)

        peak = specimen[np.argmax(values)]
        assert np.degrees(
            np.arccos(np.clip(abs(peak @ expected), -1.0, 1.0))
        ) < 1.0

    def test_sharper_kernel_gives_a_higher_peak(self, grid):
        """Halving the halfwidth concentrates the pole density."""
        orientation = np.eye(3)
        peaks = []
        for halfwidth in (20.0, 5.0):
            kernel = DeLaValleePoussinKernel(
                halfwidth=np.radians(halfwidth), crystal_symmetry='oh'
            )
            odf = UnimodalODF(orientation, kernel)
            peaks.append(pole_density(odf, [0, 0, 1], grid).max())

        assert peaks[1] > peaks[0]

    def test_antipodal_flag_changes_an_unsymmetric_result(self, grid):
        """
        Without crystal symmetry, h and -h are distinct, so the antipodal
        flag is observable.
        """
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10.0))
        odf = UnimodalODF(np.eye(3), kernel)

        with_antipodal = pole_density(odf, [0, 0, 1], grid, antipodal=True)
        without = pole_density(odf, [0, 0, 1], grid, antipodal=False)

        assert not np.allclose(with_antipodal, without)


class TestCompositeODF:
    """Pole figures are linear in the ODF."""

    def test_sum_of_odfs_is_sum_of_pole_figures(self, cubic_odf, grid):
        """The Radon transform is linear, so + carries through."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(15.0), crystal_symmetry='oh'
        )
        other = UnimodalODF(
            Rotation.from_euler('ZXZ', [5, 40, 15], degrees=True).as_matrix(),
            kernel,
        )
        direction = [1.0, 1.0, 1.0]

        combined = pole_density(cubic_odf + other, direction, grid)
        separate = (
            pole_density(cubic_odf, direction, grid)
            + pole_density(other, direction, grid)
        )

        np.testing.assert_allclose(combined, separate)

    def test_uniform_background_adds_one(self, cubic_odf, grid):
        """Adding a uniform ODF raises the pole figure by 1 MRD."""
        direction = [1.0, 1.0, 0.0]

        with_background = pole_density(
            cubic_odf + UniformODF(), direction, grid
        )

        np.testing.assert_allclose(
            with_background, pole_density(cubic_odf, direction, grid) + 1.0
        )

    def test_self_difference_vanishes(self, cubic_odf, grid):
        """An ODF minus itself has no pole density anywhere."""
        values = pole_density(cubic_odf - cubic_odf, [0, 0, 1], grid)

        np.testing.assert_allclose(values, 0.0, atol=1e-12)

    def test_constant_offset(self, cubic_odf, grid):
        """A scalar offset shifts the pole figure."""
        shifted = pole_density(cubic_odf + 0.5, [0, 0, 1], grid)

        np.testing.assert_allclose(
            shifted, pole_density(cubic_odf, [0, 0, 1], grid) + 0.5
        )


class TestCalcPoleFigure:
    """The calc_pole_figure entry point."""

    def test_one_figure_per_direction(self, cubic_odf):
        """Each crystal direction yields its own pole figure."""
        pfs = calc_pole_figure(cubic_odf, [[1, 1, 1], [2, 0, 0], [2, 2, 0]])

        assert pfs.num_pfs == 3
        assert pfs.values.shape == (3, 2160)

    def test_default_grid_is_used(self, cubic_odf):
        """Omitting specimen directions falls back to the 30 x 72 grid."""
        pfs = calc_pole_figure(cubic_odf, [1, 1, 1])

        assert len(pfs.specimen_directions) == 2160

    def test_explicit_specimen_directions(self, cubic_odf):
        """A caller-supplied grid is used as given."""
        specimen = directions_from_angles(*regular_s2_grid(10, 12))

        pfs = calc_pole_figure(
            cubic_odf, [1, 1, 1], specimen_directions=specimen
        )

        assert pfs.values.shape == (1, 120)
        np.testing.assert_allclose(pfs.specimen_directions, specimen)

    def test_matches_pole_density(self, cubic_odf, grid):
        """calc_pole_figure is a wrapper over pole_density."""
        pfs = calc_pole_figure(
            cubic_odf, [1, 1, 1], specimen_directions=grid
        )

        np.testing.assert_allclose(
            pfs.values[0], pole_density(cubic_odf, [1, 1, 1], grid)
        )

    def test_hkls_convert_through_the_material(self, cubic_odf):
        """A material supplies the crystal frame via TransSpace."""

        class FakeCubicMaterial:
            """Cubic: a reciprocal vector is parallel to (h, k, l)."""

            def TransSpace(self, hkl, inspace, outspace):
                assert (inspace, outspace) == ('r', 'c')
                return np.asarray(hkl, dtype=float)

        hkls = np.array([[1, 1, 1], [2, 0, 0]])

        pfs = calc_pole_figure(
            cubic_odf, hkls=hkls, material=FakeCubicMaterial()
        )

        assert pfs.num_pfs == 2
        assert pfs.keys == ((1, 1, 1), (2, 0, 0))
        np.testing.assert_allclose(
            pfs.crystal_directions[0], np.full(3, 1 / np.sqrt(3))
        )

    def test_hkls_without_material_rejected(self, cubic_odf):
        """hkls cannot be converted without knowing the lattice."""
        with pytest.raises(ValueError, match='requires a material'):
            calc_pole_figure(cubic_odf, hkls=[[1, 1, 1]])

    def test_four_index_hkls_without_material_rejected(self, cubic_odf):
        """Miller-Bravais indices need a material too."""
        with pytest.raises(ValueError, match='4-index'):
            calc_pole_figure(cubic_odf, hkls=[[1, 0, -1, 0]])

    def test_no_directions_at_all_rejected(self, cubic_odf):
        """One of crystal_directions or hkls is required."""
        with pytest.raises(ValueError, match='crystal_directions or hkls'):
            calc_pole_figure(cubic_odf)

    def test_hkls_label_explicit_directions(self, cubic_odf):
        """hkls may label directions that were passed directly."""
        pfs = calc_pole_figure(
            cubic_odf, [[0.0, 0.0, 1.0]], hkls=[[0, 0, 0, 2]]
        )

        assert pfs.keys == ((0, 0, 0, 2),)
        assert pfs.label(0) == '(0, 0, 0, 2)'

    def test_bad_direction_shape_rejected(self, cubic_odf):
        """Directions must be 3-vectors."""
        with pytest.raises(ValueError):
            calc_pole_figure(cubic_odf, [[1.0, 0.0]])

    def test_unsupported_odf_rejected(self, grid):
        """An object with no pole_density gets a clear error."""

        class NotAnODF:
            pass

        with pytest.raises(TypeError, match='pole_density'):
            pole_density(NotAnODF(), [0, 0, 1], grid)


class TestPoleFiguresContainer:
    """Accessors on the returned container."""

    @pytest.fixture
    def pfs(self, cubic_odf):
        return calc_pole_figure(
            cubic_odf,
            [[1, 1, 1], [2, 0, 0]],
            specimen_directions=directions_from_angles(
                *regular_s2_grid(10, 12)
            ),
            hkls=[[1, 1, 1], [2, 0, 0]],
        )

    def test_keys_and_counts(self, pfs):
        assert pfs.num_pfs == 2
        assert pfs.keys == ((1, 1, 1), (2, 0, 0))

    def test_intensities_dict(self, pfs):
        """intensities mirrors the WPPF accessor: hkl -> values."""
        intensities = pfs.intensities

        assert set(intensities) == set(pfs.keys)
        np.testing.assert_allclose(intensities[(1, 1, 1)], pfs.values[0])

    def test_angs_dict(self, pfs):
        """angs gives (polar, azimuth) per figure, as WPPF does."""
        angs = pfs.angs[(1, 1, 1)]

        assert angs.shape == (120, 2)
        np.testing.assert_allclose(
            angs[:, 0], angles_from_directions(pfs.specimen_directions)[0]
        )

    def test_stereo_radius_dict(self, pfs):
        radius = pfs.stereo_radius[(1, 1, 1)]

        assert radius.shape == (120,)
        assert np.all((radius >= 0.0) & (radius <= 1.0))

    def test_pfdata_layout(self, pfs):
        """pfdata is (x, y, z, intensity), matching WPPF."""
        data = pfs.pfdata[(2, 0, 0)]

        assert data.shape == (120, 4)
        np.testing.assert_allclose(data[:, :3], pfs.specimen_directions)
        np.testing.assert_allclose(data[:, 3], pfs.values[1])

    def test_labels_without_hkls(self, cubic_odf):
        """Unlabelled pole figures fall back to an index."""
        pfs = calc_pole_figure(cubic_odf, [[1, 1, 1]])

        assert pfs.keys == (0,)
        assert pfs.label(0) == 'Pole figure 1'

    def test_intensity_shape_validated(self):
        """Values must match the directions they were computed on."""
        with pytest.raises(ValueError, match='intensities must have shape'):
            PoleFigures(
                [[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], np.zeros((1, 5))
            )

    def test_hkl_count_validated(self):
        """One hkl label per pole figure."""
        with pytest.raises(ValueError, match='one row per pole figure'):
            PoleFigures(
                [[0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0]],
                np.ones((1, 1)),
                hkls=[[1, 1, 1], [2, 0, 0]],
            )

    def test_repr_and_str(self, pfs):
        assert 'PoleFigures(num_pfs=2' in repr(pfs)
        assert '(1, 1, 1)' in str(pfs)
        assert 'MRD' in str(pfs)


class TestPlotting:
    """PoleFigures renders through the texture package renderer."""

    @pytest.fixture
    def pfs(self, cubic_odf):
        return calc_pole_figure(
            cubic_odf,
            [[1, 1, 1], [2, 0, 0]],
            specimen_directions=directions_from_angles(
                *regular_s2_grid(20, 36)
            ),
            hkls=[[1, 1, 1], [2, 0, 0]],
        )

    def test_returns_a_figure(self, pfs):
        """plot_pf produces a matplotlib figure, as HarmonicModel does."""
        fig = pfs.plot_pf(show=False)

        assert fig is pfs.fig
        assert isinstance(fig, plt.Figure)

    def test_titles_come_from_hkls(self, pfs):
        """Each panel is titled with its reflection."""
        from hexrd.phase_transition.texture import plotting as pf_plotting

        pfs.plot_pf(show=False)

        titles = [
            pf_plotting.axes_for(pfs.ax, i).get_title() for i in range(2)
        ]
        assert titles == ['(1, 1, 1)', '(2, 0, 0)']

    def test_filled_contours(self, pfs):
        """The filled style renders too."""
        assert pfs.plot_pf(show=False, filled=True) is not None

    def test_stereographic_radii_reach_the_renderer(self, pfs, monkeypatch):
        """
        The radii handed to the renderer are the stereographic ones.

        The projection is applied on the way to the plot, not stored, so
        this checks the wiring rather than the accessor.
        """
        from hexrd.phase_transition.texture import plotting as pf_plotting

        recorded = {}
        original = pf_plotting.draw

        def record(ax, azimuth, radius, *args, **kwargs):
            recorded.setdefault('radius', radius)
            return original(ax, azimuth, radius, *args, **kwargs)

        monkeypatch.setattr(pf_plotting, 'draw', record)

        polar = angles_from_directions(pfs.specimen_directions)[0]
        pfs.plot_pf(show=False)

        np.testing.assert_allclose(
            recorded['radius'], pf_plotting.stereographic_radius(polar)
        )


if __name__ == '__main__':
    pytest.main([__file__])
