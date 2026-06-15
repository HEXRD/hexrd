"""Tests for UniformODF."""

import numpy as np
import unittest

from hexrd.phase_transition.texture.uniform_odf import UniformODF


class TestUniformODF(unittest.TestCase):
    """Test UniformODF."""

    def test_package_export(self):
        """Test that UniformODF is importable from the texture package."""
        from hexrd.phase_transition.texture import UniformODF as Cls
        self.assertIs(Cls, UniformODF)

    def test_uniform_value_is_one_mrd(self):
        """Test that uniform ODF value is 1 MRD."""
        odf = UniformODF('oh', 'triclinic')
        self.assertEqual(odf.value, 1.0)

    def test_symmetry_validation(self):
        """Test crystal and sample symmetry validation."""
        # Valid symmetries should work
        odf = UniformODF('oh', 'triclinic')
        self.assertEqual(odf.crystal_symmetry, 'oh')
        self.assertEqual(odf.sample_symmetry, 'triclinic')

        # Test different valid combinations
        odf_hex = UniformODF('d6h', 'orthorhombic')
        self.assertEqual(odf_hex.crystal_symmetry, 'd6h')
        self.assertEqual(odf_hex.sample_symmetry, 'orthorhombic')

    def test_single_orientation_evaluation(self):
        """Test ODF evaluation at a single orientation."""
        odf = UniformODF('oh', 'triclinic')

        identity = np.eye(3)
        value = odf.eval(identity)

        self.assertEqual(value, 1.0)
        self.assertTrue(np.isscalar(value))

    def test_multiple_orientations_evaluation(self):
        """Test ODF evaluation at multiple orientations."""
        odf = UniformODF('oh', 'triclinic')

        orientations = np.array([
            np.eye(3),
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        ])

        values = odf.eval(orientations)

        np.testing.assert_array_equal(values, np.ones(3))
        self.assertEqual(values.shape, (3,))

    def test_batch_orientations(self):
        """Test evaluation with different batch shapes."""
        odf = UniformODF('d6h', 'triclinic')

        # 2D batch: (2, 2, 3, 3)
        batch_2d = np.tile(np.eye(3), (2, 2, 1, 1))
        values_2d = odf.eval(batch_2d)
        self.assertEqual(values_2d.shape, (2, 2))
        np.testing.assert_array_equal(values_2d, np.ones((2, 2)))

        # 1D batch: (5, 3, 3)
        batch_1d = np.tile(np.eye(3), (5, 1, 1))
        values_1d = odf.eval(batch_1d)
        self.assertEqual(values_1d.shape, (5,))
        np.testing.assert_array_equal(values_1d, np.ones(5))

    def test_invalid_orientation_shapes(self):
        """Test that invalid orientation shapes raise errors."""
        odf = UniformODF('oh', 'triclinic')

        with self.assertRaises(ValueError):
            odf.eval(np.zeros((3, 2)))

        with self.assertRaises(ValueError):
            odf.eval(np.zeros((2, 3, 2)))

        with self.assertRaises(ValueError):
            odf.eval(np.zeros((2, 2, 3)))

    def test_mrd_convention(self):
        """Test MRD normalization: uniform ODF = 1 everywhere.

        In MRD (multiples of a random distribution), the uniform
        ODF is the reference density at 1.0. Textured ODFs have
        values > 1 near preferred orientations and < 1 elsewhere.
        """
        odf = UniformODF('oh', 'triclinic')
        self.assertEqual(odf.value, 1.0)

    def test_all_orientations_equal(self):
        """Test that all orientations return the same value."""
        odf = UniformODF('d6h', 'triclinic')

        identity = np.eye(3)
        rot_90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                           dtype=float)

        self.assertEqual(odf.eval(identity), odf.eval(rot_90z))
        self.assertEqual(odf.eval(identity), odf.value)

    def test_string_representations(self):
        """Test string representation methods."""
        odf = UniformODF('oh', 'triclinic')

        # Test __repr__
        repr_str = repr(odf)
        self.assertIn('UniformODF', repr_str)
        self.assertIn('oh', repr_str)
        self.assertIn('triclinic', repr_str)

        # Test __str__
        str_repr = str(odf)
        self.assertIn('Uniform ODF', str_repr)
        self.assertIn('oh', str_repr)
        self.assertIn('random texture', str_repr)

    def test_different_crystal_symmetries(self):
        """Test that value is 1 MRD regardless of symmetry."""
        for sym in ['oh', 'd6h', 'c6h', 'd4h', 'th']:
            with self.subTest(crystal_symmetry=sym):
                odf = UniformODF(sym, 'triclinic')
                self.assertEqual(odf.value, 1.0)
                self.assertEqual(odf.eval(np.eye(3)), 1.0)

    def test_all_crystal_symmetries_accepted(self):
        """Every supported crystal symmetry is accepted and stored."""
        crystal_symmetries = [
            'ci', 'c2h', 'd2h', 'c4h', 'd4h',
            's6', 'd3d', 'c6h', 'd6h', 'th', 'oh',
        ]
        for sym in crystal_symmetries:
            with self.subTest(crystal_symmetry=sym):
                odf = UniformODF(sym, 'triclinic')
                self.assertEqual(odf.crystal_symmetry, sym)

    def test_all_sample_symmetries_accepted(self):
        """Every supported sample symmetry is accepted and stored."""
        for sym in ['triclinic', 'monoclinic', 'orthorhombic']:
            with self.subTest(sample_symmetry=sym):
                odf = UniformODF('oh', sym)
                self.assertEqual(odf.sample_symmetry, sym)

    def test_invalid_crystal_symmetry(self):
        """Test that invalid crystal symmetry raises ValueError."""
        with self.assertRaises(ValueError):
            UniformODF('invalid', 'triclinic')
        with self.assertRaises(ValueError):
            UniformODF('6/mmm', 'triclinic')

    def test_invalid_sample_symmetry(self):
        """Test that invalid sample symmetry raises ValueError."""
        with self.assertRaises(ValueError):
            UniformODF('oh', 'invalid')
        with self.assertRaises(ValueError):
            UniformODF('oh', 'cubic')
        # 'axial' is intentionally excluded to match the kernel.
        with self.assertRaises(ValueError):
            UniformODF('oh', 'axial')

    def test_no_symmetry_defaults_to_none(self):
        """UniformODF can be built without symmetry; labels are None."""
        odf = UniformODF()
        self.assertIsNone(odf.crystal_symmetry)
        self.assertIsNone(odf.sample_symmetry)
        self.assertEqual(odf.value, 1.0)
        self.assertEqual(odf.eval(np.eye(3)), 1.0)

    def test_array_symmetry_has_no_label(self):
        """Symmetry given as a quaternion array is accepted; label is None."""
        from hexrd.core import rotations
        odf = UniformODF(rotations.quatOfLaueGroup('d4h'))
        self.assertIsNone(odf.crystal_symmetry)
        self.assertEqual(odf.eval(np.eye(3)), 1.0)

    def test_string_representations_without_symmetry(self):
        """repr/str render gracefully when no symmetry is set."""
        odf = UniformODF()
        self.assertIn('crystal_symmetry=None', repr(odf))
        self.assertIn('none', str(odf))
