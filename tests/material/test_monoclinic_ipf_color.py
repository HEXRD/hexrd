"""Tests that monoclinic IPF coloring respects the true crystal symmetry.

The monoclinic system has its 2-fold axis along b (Cartesian y). The IPF
coloring code rotates symmetry operations by 90 deg about x so the 2-fold
aligns with z, matching the zone definitions in sphere_sector.py. The
direction vectors must be rotated into the same frame, and the PG/supergroup
operations must be mutually consistent, otherwise symmetry-equivalent
directions receive different colors.
"""

from pathlib import Path

import numpy as np
import pytest

from hexrd.material.material import Material


@pytest.fixture
def monoclinic_c2h():
    cif = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "materials"
        / "Ag(TeMo)6.cif"
    )
    return Material(name="AgTeMo6", material_file=str(cif))


def _random_unit_directions(n, seed=42):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _check_symmetry_equivalence(uc, dirs, equiv_dirs, laueswitch):
    """Assert that dirs and equiv_dirs produce the same IPF colors."""
    rgb1 = uc.color_directions(dirs, laueswitch)
    rgb2 = uc.color_directions(equiv_dirs, laueswitch)
    np.testing.assert_allclose(rgb1, rgb2, atol=1e-5)


class TestMonoclinicIPFColor:
    """Verify that directions related by monoclinic symmetry get the same
    IPF color for both Laue and non-Laue modes."""

    def test_c2y_equivalence_laue(self, monoclinic_c2h):
        """C2 about b (y): (x,y,z) -> (-x,y,-z) must give same color."""
        uc = monoclinic_c2h.unitcell
        dirs = _random_unit_directions(50)
        equiv = dirs.copy()
        equiv[:, 0] *= -1
        equiv[:, 2] *= -1
        _check_symmetry_equivalence(uc, dirs, equiv, laueswitch=True)

    def test_inversion_equivalence_laue(self, monoclinic_c2h):
        """Inversion: (x,y,z) -> (-x,-y,-z) must give same color."""
        uc = monoclinic_c2h.unitcell
        dirs = _random_unit_directions(50)
        _check_symmetry_equivalence(uc, dirs, -dirs, laueswitch=True)

    def test_mirror_y_equivalence_laue(self, monoclinic_c2h):
        """Mirror perp b (y): (x,y,z) -> (x,-y,z) must give same color."""
        uc = monoclinic_c2h.unitcell
        dirs = _random_unit_directions(50)
        equiv = dirs.copy()
        equiv[:, 1] *= -1
        _check_symmetry_equivalence(uc, dirs, equiv, laueswitch=True)

    def test_c2y_equivalence_nonlaue(self, monoclinic_c2h):
        """Non-Laue mode: C2_y equivalents must still match."""
        uc = monoclinic_c2h.unitcell
        dirs = _random_unit_directions(50)
        equiv = dirs.copy()
        equiv[:, 0] *= -1
        equiv[:, 2] *= -1
        _check_symmetry_equivalence(uc, dirs, equiv, laueswitch=False)

    def test_rgb_values_in_range(self, monoclinic_c2h):
        """All RGB values must be in [0, 1] with no NaN."""
        uc = monoclinic_c2h.unitcell
        dirs = _random_unit_directions(200)
        rgb = uc.color_directions(dirs, laueswitch=True)
        assert not np.any(np.isnan(rgb))
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_color_orientations_consistency(self, monoclinic_c2h):
        """color_orientations should give same result one-at-a-time or
        batched, matching the existing test pattern for cubic."""
        uc = monoclinic_c2h.unitcell
        rng = np.random.default_rng(123)
        rmats = np.zeros((3, 3, 3))
        for i in range(3):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            rmats[i] = q

        rgb_batch = uc.color_orientations(rmats)
        rgb_single = np.vstack(
            [uc.color_orientations(rmats[i]) for i in range(3)]
        )
        np.testing.assert_allclose(rgb_batch, rgb_single, atol=1e-6)

    def test_nonequivalent_directions_differ(self, monoclinic_c2h):
        """Directions NOT related by any c2h operation should generally
        produce different colors (smoke test, not all will differ)."""
        uc = monoclinic_c2h.unitcell
        d1 = np.array([[1.0, 0.0, 0.0]])
        d2 = np.array([[0.0, 1.0, 0.0]])
        rgb1 = uc.color_directions(d1, laueswitch=True)
        rgb2 = uc.color_directions(d2, laueswitch=True)
        assert not np.allclose(rgb1, rgb2, atol=1e-3)
