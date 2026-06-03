"""Tests for DeLaValleePoussinKernel."""

import numpy as np
import unittest
from scipy.special import beta as betafn

from hexrd.phase_transition.texture.kernels import DeLaValleePoussinKernel


MTEX_KERNEL_KAPPA = 90.0
MTEX_KERNEL_OMEGA = np.array([
    0.0000000000,
    0.5235987756,
    0.7853981634,
    1.0471975512,
    1.5707963268,
    3.1415926536,
])
MTEX_KERNEL_VALUES = np.array([
    1532.2892935006,
    2.9869009813,
    0.0009910670,
    0.0000000087,
    0.0000000000,
    0.0000000000,
])
MTEX_HALFWIDTH_RAD = 0.1745329252
MTEX_HALFWIDTH_KAPPA = 90.9031059932


def _rotation_about_z(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0],
    ])


class TestDeLaValleePoussinKernel(unittest.TestCase):
    """Test DeLaValleePoussinKernel."""

    def test_kernel_matches_mtex_ground_truth_values(self):
        """Test kernel evaluation against values generated from MTEX."""
        halfwidth = 2.0 * np.arccos(
            0.5 ** (1.0 / (2.0 * MTEX_KERNEL_KAPPA))
        )
        kernel = DeLaValleePoussinKernel(halfwidth=halfwidth)

        identity = np.eye(3)
        rotations = np.array([
            _rotation_about_z(angle)
            for angle in MTEX_KERNEL_OMEGA
        ])

        values = kernel.eval(identity, rotations)

        np.testing.assert_allclose(
            kernel.kappa, MTEX_KERNEL_KAPPA, rtol=0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            values, MTEX_KERNEL_VALUES, rtol=1e-8, atol=1e-10
        )

    def test_halfwidth_conversion_matches_mtex_ground_truth(self):
        """Test half-width to kappa conversion against MTEX output."""
        kernel = DeLaValleePoussinKernel(halfwidth=MTEX_HALFWIDTH_RAD)

        np.testing.assert_allclose(
            kernel.kappa, MTEX_HALFWIDTH_KAPPA, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            kernel.halfwidth, MTEX_HALFWIDTH_RAD, rtol=0.0, atol=1e-10
        )

    def test_kappa_from_halfwidth(self):
        """Test that κ is derived analytically from half-width."""
        halfwidth_rad = np.radians(4.0)
        kernel = DeLaValleePoussinKernel(halfwidth=halfwidth_rad)

        # Regression guard: verify against the formula directly
        expected_kappa = (
            0.5 * np.log(0.5)
            / np.log(np.cos(halfwidth_rad / 2.0))
        )
        self.assertAlmostEqual(
            kernel.kappa, expected_kappa, places=10
        )
    
    def test_kernel_evaluation_at_identity(self):
        """Test that K(0) equals the normalization constant.

        At ω = 0, cos(0/2) = 1, so K(0) = C · 1^(2κ) = C.
        """
        halfwidth_rad = np.radians(4.0)
        kernel = DeLaValleePoussinKernel(halfwidth=halfwidth_rad)

        identity = np.eye(3)
        value = kernel.eval(identity, identity)

        # Verify against independently computed norm constant
        expected_C = (
            betafn(1.5, 0.5)
            / betafn(1.5, kernel.kappa + 0.5)
        )
        self.assertAlmostEqual(value, expected_C, places=5)
        self.assertAlmostEqual(
            value, kernel.norm_constant, places=10
        )

    def test_string_representations(self):
        """Test string representation methods."""
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(4.0))

        repr_str = repr(kernel)
        self.assertIn('DeLaValleePoussinKernel', repr_str)
        self.assertIn('halfwidth=', repr_str)
        self.assertIn('kappa=', repr_str)

        str_repr = str(kernel)
        self.assertIn('de la Vallée Poussin kernel', str_repr)
        self.assertIn('half-width', str_repr)
        self.assertIn('K(ω)', str_repr)
        
    def test_kernel_decreases_with_misorientation(self):
        """Test that kernel value decreases as misorientation grows."""
        halfwidth_rad = np.radians(4.0)
        kernel = DeLaValleePoussinKernel(halfwidth=halfwidth_rad)

        identity = np.eye(3)

        # 22.5° rotation about x-axis
        rotated = np.array([
            [1.0000, 0.0000,  0.0000],
            [0.0000, 0.9239, -0.3827],
            [0.0000, 0.3827,  0.9239],
        ])

        value_identity = kernel.eval(identity, identity)
        value_rotated = kernel.eval(identity, rotated)

        # 22.5° is well beyond the 4° halfwidth, so kernel ≈ 0
        self.assertGreater(value_identity, value_rotated)
        self.assertAlmostEqual(value_rotated, 0.0, places=3)

    def test_halfwidth_is_half_maximum(self):
        """Test that K(halfwidth) = K(0) / 2."""
        halfwidth_rad = np.radians(4.0)
        kernel = DeLaValleePoussinKernel(halfwidth=halfwidth_rad)

        # K(0) = norm_constant
        peak = kernel.norm_constant

        # Evaluate at exactly the halfwidth angle
        co2 = np.cos(halfwidth_rad / 2.0)
        value_at_hw = (
            kernel.norm_constant * co2 ** (2.0 * kernel.kappa)
        )

        self.assertAlmostEqual(value_at_hw, peak / 2.0, places=5)

    # --- Invalid input tests ---

    def test_negative_halfwidth_raises(self):
        """Test that negative half-width raises ValueError."""
        with self.assertRaises(ValueError):
            DeLaValleePoussinKernel(halfwidth=-0.1)

    def test_zero_halfwidth_raises(self):
        """Test that zero half-width raises ValueError."""
        with self.assertRaises(ValueError):
            DeLaValleePoussinKernel(halfwidth=0.0)

    def test_large_halfwidth_warns(self):
        """Test that half-width > π/2 emits a warning."""
        with self.assertWarns(UserWarning):
            DeLaValleePoussinKernel(halfwidth=np.radians(100))

    def test_invalid_symmetry_type_raises_value_error(self):
        """Test that invalid symmetry types raise ValueError."""
        with self.assertRaises(ValueError):
            DeLaValleePoussinKernel(
                halfwidth=np.radians(10),
                crystal_symmetry=123,
            )

    def test_invalid_symmetry_string_raises_value_error(self):
        """Test that invalid symmetry strings raise ValueError."""
        with self.assertRaises(ValueError):
            DeLaValleePoussinKernel(
                halfwidth=np.radians(10),
                crystal_symmetry='invalid',
            )

    def test_invalid_symmetry_array_shape_raises_value_error(self):
        """Test that invalid symmetry array shapes raise ValueError."""
        with self.assertRaises(ValueError):
            DeLaValleePoussinKernel(
                halfwidth=np.radians(10),
                crystal_symmetry=np.eye(3),
            )

    # --- misorientation_angle tests ---

    def test_misorientation_angle_identity(self):
        """Test misorientation angle between identical rotations is 0."""
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10))
        angle = kernel.misorientation_angle(np.eye(3), np.eye(3))
        self.assertAlmostEqual(float(angle), 0.0, places=10)

    def test_misorientation_angle_known_rotation(self):
        """Test misorientation angle for a known rotation."""
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10))

        # 90° rotation about z-axis
        R90z = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)

        angle = kernel.misorientation_angle(np.eye(3), R90z)
        self.assertAlmostEqual(
            float(angle), np.radians(90), places=8
        )

    def test_misorientation_angle_uses_crystal_symmetry(self):
        """Test symmetry-reduced misorientation angle."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10),
            crystal_symmetry='d4h',
        )

        R90z = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)

        angle = kernel.misorientation_angle(np.eye(3), R90z)
        self.assertAlmostEqual(float(angle), 0.0, places=8)

    def test_symmetric_kernel_value_is_peak_for_equivalent_rotations(self):
        """Test symmetry-equivalent rotations evaluate to the peak value."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10),
            crystal_symmetry='d4h',
        )

        R90z = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)

        value = kernel.eval(np.eye(3), R90z)
        self.assertAlmostEqual(value, kernel.norm_constant, places=8)

    def test_batch_misorientation_angle_uses_crystal_symmetry(self):
        """Test symmetry-reduced misorientation for batches."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10),
            crystal_symmetry='d4h',
        )

        R45z = _rotation_about_z(np.radians(45.0))
        R90z = _rotation_about_z(np.radians(90.0))
        batch = np.stack([np.eye(3), R45z, R90z])

        angles = kernel.misorientation_angle(np.eye(3), batch)

        self.assertEqual(angles.shape, (3,))
        np.testing.assert_allclose(
            angles,
            [0.0, np.radians(45.0), 0.0],
            atol=1e-8,
        )

    def test_misorientation_angle_bad_shape(self):
        """Test that non-(3,3) inputs raise ValueError."""
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(10))
        with self.assertRaises(ValueError):
            kernel.misorientation_angle(
                np.eye(2), np.eye(3)
            )

    # --- eval_centered test ---

    def test_eval_centered_matches_eval(self):
        """Test that eval_centered(R, center) == eval(center, R)."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10)
        )
        center = np.eye(3)
        R = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)

        val_eval = kernel.eval(center, R)
        val_centered = kernel.eval_centered(R, center)
        self.assertAlmostEqual(
            float(val_eval), float(val_centered), places=10
        )

    # --- Batch evaluation test ---

    def test_batch_evaluation(self):
        """Test eval with a batch of rotation matrices (N, 3, 3)."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(10)
        )
        identity = np.eye(3)

        # Build batch: identity + 90° rotations about x, y, z
        R90x = np.array([
            [1, 0,  0],
            [0, 0, -1],
            [0, 1,  0],
        ], dtype=float)
        R90y = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0],
        ], dtype=float)
        R90z = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        batch = np.stack([identity, R90x, R90y, R90z])

        values = kernel.eval(identity, batch)

        self.assertEqual(values.shape, (4,))
        # First entry (identity vs identity) should be the peak
        self.assertAlmostEqual(
            values[0], kernel.norm_constant, places=5
        )
        # Remaining entries (90° away) should be near zero
        for val in values[1:]:
            self.assertAlmostEqual(float(val), 0.0, places=5)

