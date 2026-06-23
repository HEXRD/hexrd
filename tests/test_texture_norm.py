"""Tests for the ODF texture index and L2 norm helpers."""

import numpy as np
import unittest
from scipy.integrate import quad

from hexrd.phase_transition.texture import (
    UniformODF,
    UnimodalODF,
    DeLaValleePoussinKernel,
    eval_random_orientations,
    texture_index,
    texture_norm,
)


def _quad_texture_index(kernel):
    """
    Independent numerical reference for the single-mode texture index.

    Integrates f^2 against the Haar misorientation-angle density
    p(omega) = (1 - cos omega) / pi on [0, pi] using adaptive quadrature.

    NOTE: This intentionally re-derives the de la Vallee Poussin kernel
    formula f(omega) = C * cos(omega/2)^(2*kappa) rather than calling into
    production code, so it remains a truly independent reference. If the
    kernel parameterization changes, this function must be updated to match.
    """
    kappa = kernel.kappa
    norm_c = kernel.norm_constant

    def integrand(omega):
        kernel_value = norm_c * np.cos(omega / 2.0) ** (2.0 * kappa)
        density = (1.0 - np.cos(omega)) / np.pi
        return kernel_value ** 2 * density

    value, _ = quad(integrand, 0.0, np.pi)
    return value


class TestUniformNorm(unittest.TestCase):
    """The uniform (random) texture is the MRD reference: J = norm = 1."""

    def setUp(self):
        self.odf = UniformODF('oh', 'triclinic')

    def test_uniform_texture_index_is_one(self):
        # f == 1 everywhere, so <f^2> = 1 exactly regardless of sampling.
        self.assertAlmostEqual(
            texture_index(self.odf, n_orientations=1000, seed=0), 1.0, places=12
        )

    def test_uniform_norm_is_one(self):
        self.assertAlmostEqual(
            texture_norm(self.odf, n_orientations=1000, seed=0), 1.0, places=12
        )


class TestSingleModeAnalytic(unittest.TestCase):
    """Exact (closed-form) texture index for single-mode, no-symmetry ODFs."""

    # Hard-coded ground-truth texture indices computed via independent
    # numerical quadrature (scipy.integrate.quad, limit=200) for regression
    # protection. These values are invariant to formula changes in the
    # production code.
    EXPECTED_TEXTURE_INDEX = {
        5: 4371.3941914154,
        10: 553.24119966486,
        25: 38.519848689319,
        35: 15.379295015085,
    }

    def test_analytic_matches_hardcoded_ground_truth(self):
        for hw_deg, expected_j in self.EXPECTED_TEXTURE_INDEX.items():
            with self.subTest(halfwidth_deg=hw_deg):
                kernel = DeLaValleePoussinKernel(halfwidth=np.radians(hw_deg))
                odf = UnimodalODF(np.eye(3), kernel)
                self.assertAlmostEqual(
                    odf.analytic_texture_index(), expected_j, places=5
                )

    def test_analytic_matches_independent_quadrature(self):
        for hw_deg in (15, 35, 60):
            with self.subTest(halfwidth_deg=hw_deg):
                kernel = DeLaValleePoussinKernel(halfwidth=np.radians(hw_deg))
                odf = UnimodalODF(np.eye(3), kernel)

                analytic = odf.analytic_texture_index()
                reference = _quad_texture_index(kernel)
                self.assertAlmostEqual(analytic / reference, 1.0, places=6)

    def test_free_function_uses_analytic_and_ignores_seed(self):
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(35))
        odf = UnimodalODF(np.eye(3), kernel)

        # The analytic path is exact, so results are seed-independent and
        # match the closed form to machine precision.
        a = texture_index(odf, seed=1)
        b = texture_index(odf, seed=99999)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, odf.analytic_texture_index(), places=12)

    def test_analytic_matches_monte_carlo(self):
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(35))
        odf = UnimodalODF(np.eye(3), kernel)

        analytic = odf.analytic_texture_index()
        _, values = eval_random_orientations(odf, n_orientations=500000, seed=1)
        monte_carlo = float(np.mean(values ** 2))
        self.assertAlmostEqual(monte_carlo / analytic, 1.0, delta=0.03)

    def test_sharper_texture_has_larger_norm(self):
        sharp = UnimodalODF(
            np.eye(3), DeLaValleePoussinKernel(halfwidth=np.radians(10))
        )
        broad = UnimodalODF(
            np.eye(3), DeLaValleePoussinKernel(halfwidth=np.radians(40))
        )
        # Exact for single-mode ODFs, so no sampling needed.
        self.assertGreater(sharp.norm(), broad.norm())


class TestNormMethodWrappers(unittest.TestCase):
    """ODF method wrappers mirror the free functions (MTEX-style API)."""

    def test_unimodal_methods_match_free_functions(self):
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(30))
        odf = UnimodalODF(np.eye(3), kernel)

        self.assertEqual(odf.texture_index(), texture_index(odf))
        self.assertEqual(odf.norm(), texture_norm(odf))
        self.assertAlmostEqual(odf.norm(), np.sqrt(odf.texture_index()), places=12)
        self.assertGreater(odf.norm(), 1.0)

    def test_uniform_methods_are_one(self):
        odf = UniformODF('oh', 'triclinic')
        self.assertEqual(odf.analytic_texture_index(), 1.0)
        self.assertAlmostEqual(odf.texture_index(), 1.0, places=12)
        self.assertAlmostEqual(odf.norm(), 1.0, places=12)


class TestMonteCarloFallback(unittest.TestCase):
    """Multi-mode / symmetry ODFs have no closed form and fall back to MC."""

    def test_multimode_has_no_closed_form(self):
        # 180° rotation about x-axis (proper rotation, det=+1)
        modes = np.array([np.eye(3), np.diag([1, -1, -1]).astype(float)])
        odf = UnimodalODF(modes, DeLaValleePoussinKernel(halfwidth=np.radians(20)))
        self.assertIsNone(odf.analytic_texture_index())

    def test_symmetry_has_no_closed_form(self):
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(20), crystal_symmetry='oh'
        )
        odf = UnimodalODF(np.eye(3), kernel)
        self.assertIsNone(odf.analytic_texture_index())

    def test_symmetric_index_is_scale_invariant(self):
        # A symmetry-reduced kernel is not mean-1 over SO(3) (its mean is
        # the symmetry-group order), so a raw <f^2> would be hugely inflated.
        # texture_index normalizes by <f>^2, so it must still report a
        # sane O(1-10) index that matches an independent <f^2>/<f>^2 estimate.
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(20), crystal_symmetry='oh'
        )
        odf = UnimodalODF(np.eye(3), kernel)
        self.assertIsNone(odf.analytic_texture_index())  # MC path is taken

        j = texture_index(odf, n_orientations=400000, seed=1)

        orientations, values = eval_random_orientations(
            odf, n_orientations=400000, seed=1
        )
        reference = float(np.mean(values ** 2) / np.mean(values) ** 2)
        self.assertAlmostEqual(j / reference, 1.0, places=6)
        # Sanity: a real texture, but nowhere near the un-normalized ~1700.
        self.assertGreater(j, 1.0)
        self.assertLess(j, 50.0)

    def test_multimode_fallback_is_reasonable(self):
        # 180° rotation about x-axis (proper rotation, det=+1)
        modes = np.array([np.eye(3), np.diag([1, -1, -1]).astype(float)])
        odf = UnimodalODF(modes, DeLaValleePoussinKernel(halfwidth=np.radians(25)))

        # The 2-mode ODF with 25° halfwidth has norm ~ 3.1 (empirically
        # stable across seeds). Assert a conservative lower bound that
        # catches regressions where MC silently returns near-uniform values.
        norm_value = odf.norm(n_orientations=300000, seed=5)
        self.assertGreater(norm_value, 2.5)
        # Monte Carlo path: a fixed seed is reproducible.
        self.assertEqual(
            odf.texture_index(n_orientations=50000, seed=5),
            odf.texture_index(n_orientations=50000, seed=5),
        )


if __name__ == '__main__':
    unittest.main()
