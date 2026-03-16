"""Tests for the generic ODF evaluation helpers."""

import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from hexrd.phase_transition.texture import (
    UniformODF,
    UnimodalODF,
    DeLaValleePoussinKernel,
    validate_orientations,
    eval_odf,
    eval_odf_batch,
    eval_at_identity,
    eval_random_orientations,
)


class TestValidateOrientations(unittest.TestCase):
    """Test validate_orientations."""

    def test_valid_single(self):
        result = validate_orientations(np.eye(3))
        self.assertEqual(result.shape, (3, 3))

    def test_valid_batch(self):
        batch = np.tile(np.eye(3), (5, 1, 1))
        result = validate_orientations(batch)
        self.assertEqual(result.shape, (5, 3, 3))

    def test_too_few_dimensions(self):
        with self.assertRaises(ValueError):
            validate_orientations(np.array([1.0, 2.0, 3.0]))

    def test_wrong_trailing_shape(self):
        with self.assertRaises(ValueError):
            validate_orientations(np.zeros((4, 3, 2)))


class TestEvalOdf(unittest.TestCase):
    """Test eval_odf against the real ODF classes."""

    def setUp(self):
        self.uniform = UniformODF('oh', 'triclinic')
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
        self.unimodal = UnimodalODF(np.eye(3), kernel)

    def test_single_uniform_is_one_mrd(self):
        self.assertEqual(eval_odf(self.uniform, np.eye(3)), 1.0)

    def test_batch_uniform_is_ones(self):
        batch = np.tile(np.eye(3), (4, 1, 1))
        np.testing.assert_array_equal(eval_odf(self.uniform, batch), np.ones(4))

    def test_matches_odf_eval(self):
        orientations = Rotation.random(50, random_state=1).as_matrix()
        np.testing.assert_allclose(
            eval_odf(self.unimodal, orientations),
            self.unimodal.eval(orientations),
        )

    def test_requires_eval_method(self):
        with self.assertRaises(TypeError):
            eval_odf(object(), np.eye(3))


class TestEvalOdfBatch(unittest.TestCase):
    """Test eval_odf_batch chunking."""

    def setUp(self):
        self.uniform = UniformODF('oh', 'triclinic')
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
        self.unimodal = UnimodalODF(np.eye(3), kernel)

    def test_small_batch_matches_eval_odf(self):
        batch = np.tile(np.eye(3), (3, 1, 1))
        np.testing.assert_array_equal(
            eval_odf_batch(self.uniform, batch),
            eval_odf(self.uniform, batch),
        )

    def test_chunking_matches_direct_eval(self):
        orientations = Rotation.random(2500, random_state=2).as_matrix()
        chunked = eval_odf_batch(self.unimodal, orientations, chunk_size=1000)
        self.assertEqual(chunked.shape, (2500,))
        np.testing.assert_allclose(chunked, self.unimodal.eval(orientations))

    def test_chunking_preserves_uniform_value(self):
        batch = np.tile(np.eye(3), (2500, 1, 1))
        chunked = eval_odf_batch(self.uniform, batch, chunk_size=500)
        np.testing.assert_array_equal(chunked, np.ones(2500))


class TestEvalAtIdentity(unittest.TestCase):
    """Test eval_at_identity (and MRD convention)."""

    def test_uniform_identity_is_one_mrd(self):
        odf = UniformODF('oh', 'triclinic')
        self.assertEqual(eval_at_identity(odf), 1.0)

    def test_unimodal_identity_is_kernel_peak(self):
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
        odf = UnimodalODF(np.eye(3), kernel)
        self.assertAlmostEqual(
            eval_at_identity(odf), kernel.norm_constant, places=6
        )


class TestEvalRandomOrientations(unittest.TestCase):
    """Test eval_random_orientations."""

    def setUp(self):
        self.uniform = UniformODF('oh', 'triclinic')

    def test_output_shapes(self):
        orientations, values = eval_random_orientations(
            self.uniform, n_orientations=200, seed=0
        )
        self.assertEqual(orientations.shape, (200, 3, 3))
        self.assertEqual(values.shape, (200,))

    def test_returns_valid_rotations(self):
        orientations, _ = eval_random_orientations(
            self.uniform, n_orientations=50, seed=0
        )
        identity = np.tile(np.eye(3), (50, 1, 1))
        products = np.matmul(orientations, np.swapaxes(orientations, -2, -1))
        np.testing.assert_allclose(products, identity, atol=1e-10)
        dets = np.linalg.det(orientations)
        np.testing.assert_allclose(dets, np.ones(50), atol=1e-10)

    def test_reproducible_with_seed(self):
        o1, _ = eval_random_orientations(self.uniform, n_orientations=20, seed=42)
        o2, _ = eval_random_orientations(self.uniform, n_orientations=20, seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_uniform_mean_is_one_mrd(self):
        _, values = eval_random_orientations(
            self.uniform, n_orientations=1000, seed=3
        )
        self.assertAlmostEqual(np.mean(values), 1.0)


if __name__ == '__main__':
    unittest.main()
