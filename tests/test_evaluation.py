"""Tests for the generic ODF evaluation helpers."""

import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from hexrd.phase_transition.texture import (
    UniformODF,
    UnimodalODF,
    DeLaValleePoussinKernel,
    eval_odf_batch,
    eval_random_orientations,
)


class TestEvalOdfBatch(unittest.TestCase):
    """Test eval_odf_batch chunking."""

    def setUp(self):
        self.uniform = UniformODF('oh', 'triclinic')
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
        self.unimodal = UnimodalODF(np.eye(3), kernel)

    def test_single_orientation_returns_scalar(self):
        # A single (3, 3) goes through the no-chunk branch and returns
        # whatever odf.eval returns for a single orientation: a scalar.
        result = eval_odf_batch(self.uniform, np.eye(3))
        self.assertEqual(result, 1.0)
        self.assertEqual(np.ndim(result), 0)

    def test_small_batch_matches_direct_eval(self):
        batch = np.tile(np.eye(3), (3, 1, 1))
        np.testing.assert_array_equal(
            eval_odf_batch(self.uniform, batch),
            self.uniform.eval(batch),
        )

    def test_chunking_matches_direct_eval(self):
        orientations = Rotation.random(2500, random_state=2).as_matrix()
        chunked = eval_odf_batch(self.unimodal, orientations, chunk_size=1000)
        self.assertEqual(chunked.shape, (2500,))
        np.testing.assert_allclose(chunked, self.unimodal.eval(orientations))

    def test_chunk_boundary_is_exact(self):
        # n exactly divisible by chunk_size exercises the loop's final chunk
        # ending precisely at n_orientations.
        orientations = Rotation.random(2000, random_state=5).as_matrix()
        chunked = eval_odf_batch(self.unimodal, orientations, chunk_size=1000)
        self.assertEqual(chunked.shape, (2000,))
        np.testing.assert_allclose(chunked, self.unimodal.eval(orientations))

    def test_chunking_preserves_uniform_value(self):
        batch = np.tile(np.eye(3), (2500, 1, 1))
        chunked = eval_odf_batch(self.uniform, batch, chunk_size=500)
        np.testing.assert_array_equal(chunked, np.ones(2500))

    def test_accepts_list_input(self):
        # array_like (not just ndarray) is coerced via np.asarray.
        batch = [np.eye(3) for _ in range(3)]
        np.testing.assert_array_equal(
            eval_odf_batch(self.uniform, batch),
            np.ones(3),
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

    def test_values_match_direct_eval(self):
        orientations, values = eval_random_orientations(
            self.uniform, n_orientations=100, seed=7
        )
        np.testing.assert_array_equal(values, self.uniform.eval(orientations))


if __name__ == '__main__':
    unittest.main()
