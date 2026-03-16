"""
Test ODF Evaluation Functions
Tests the common evaluation functionality for orientation distribution functions
"""

import numpy as np
import unittest
import sys
from pathlib import Path

# Add texture module path directly to avoid complex imports
texture_path = Path(__file__).parent.parent / 'hexrd' / 'texture'
sys.path.insert(0, str(texture_path))

# Import evaluation functions
from evaluation import (
    validate_orientations,
    eval_odf,
    eval_odf_batch, 
    eval_at_identity,
    eval_random_orientations
)

class MockODF:
    """Mock ODF class for testing evaluation functions."""
    
    def __init__(self, constant_value=1.0):
        self.constant_value = constant_value
    
    def eval(self, orientations):
        """Mock evaluation - return constant value(s) based on input shape."""
        orientations = np.asarray(orientations)
        output_shape = orientations.shape[:-2]
        
        if output_shape == ():
            return self.constant_value
        else:
            return np.full(output_shape, self.constant_value)

class InvalidODF:
    """Mock ODF without eval method for error testing."""
    pass

class TestValidateOrientations(unittest.TestCase):
    """Test orientation validation function."""
    
    def test_valid_single_orientation(self):
        """Test validation of single 3x3 orientation matrix."""
        orientation = np.eye(3)
        result = validate_orientations(orientation)
        
        np.testing.assert_array_equal(result, orientation)
        self.assertEqual(result.shape, (3, 3))
    
    def test_valid_multiple_orientations(self):
        """Test validation of multiple orientation matrices."""
        orientations = np.array([np.eye(3), -np.eye(3)])  # shape (2, 3, 3)
        result = validate_orientations(orientations)
        
        np.testing.assert_array_equal(result, orientations)
        self.assertEqual(result.shape, (2, 3, 3))
    
    def test_valid_batch_orientations(self):
        """Test validation of batch orientation arrays."""
        orientations = np.tile(np.eye(3), (3, 2, 1, 1))  # shape (3, 2, 3, 3)
        result = validate_orientations(orientations)
        
        np.testing.assert_array_equal(result, orientations)
        self.assertEqual(result.shape, (3, 2, 3, 3))
    
    def test_invalid_dimensions(self):
        """Test validation errors for invalid dimensions."""
        # Too few dimensions
        with self.assertRaises(ValueError) as cm:
            validate_orientations(np.array([1, 2, 3]))
        self.assertIn("at least 2 dimensions", str(cm.exception))
        
        # Wrong last dimensions
        with self.assertRaises(ValueError) as cm:
            validate_orientations(np.random.random((3, 2)))
        self.assertIn("(..., 3, 3)", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_orientations(np.random.random((2, 3, 2)))
        self.assertIn("(..., 3, 3)", str(cm.exception))
    
    def test_list_input_conversion(self):
        """Test that list inputs are properly converted."""
        orientation_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = validate_orientations(orientation_list)
        
        expected = np.eye(3)
        np.testing.assert_array_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)

class TestEvalODF(unittest.TestCase):
    """Test generic ODF evaluation function."""
    
    def test_single_orientation(self):
        """Test evaluation at single orientation."""
        odf = MockODF(constant_value=2.5)
        orientation = np.eye(3)
        
        result = eval_odf(odf, orientation)
        
        self.assertEqual(result, 2.5)
        self.assertTrue(np.isscalar(result))
    
    def test_multiple_orientations(self):
        """Test evaluation at multiple orientations."""
        odf = MockODF(constant_value=1.5)
        orientations = np.array([np.eye(3), -np.eye(3)])
        
        result = eval_odf(odf, orientations)
        
        expected = np.array([1.5, 1.5])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2,))
    
    def test_batch_orientations(self):
        """Test evaluation with batch dimensions."""
        odf = MockODF(constant_value=0.8)
        orientations = np.tile(np.eye(3), (2, 3, 1, 1))  # shape (2, 3, 3, 3)
        
        result = eval_odf(odf, orientations)
        
        expected = np.full((2, 3), 0.8)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2, 3))
    
    def test_validation_disabled(self):
        """Test evaluation with validation disabled."""
        odf = MockODF(constant_value=3.0)
        orientation = np.eye(3)
        
        result = eval_odf(odf, orientation, validate_input=False)
        
        self.assertEqual(result, 3.0)
    
    def test_invalid_odf_object(self):
        """Test error handling for invalid ODF objects."""
        invalid_odf = InvalidODF()
        orientation = np.eye(3)
        
        with self.assertRaises(TypeError) as cm:
            eval_odf(invalid_odf, orientation)
        self.assertIn("must have an eval() method", str(cm.exception))
        self.assertIn("InvalidODF", str(cm.exception))
    
    def test_invalid_orientation_shape(self):
        """Test error handling for invalid orientation shapes."""
        odf = MockODF(constant_value=1.0)
        
        with self.assertRaises(ValueError):
            eval_odf(odf, np.array([1, 2, 3]))

class TestEvalODFBatch(unittest.TestCase):
    """Test batch ODF evaluation function."""
    
    def test_small_batch(self):
        """Test that small batches are processed normally."""
        odf = MockODF(constant_value=2.0)
        orientations = np.tile(np.eye(3), (5, 1, 1))
        
        result = eval_odf_batch(odf, orientations, chunk_size=10)
        
        expected = np.full(5, 2.0)
        np.testing.assert_array_equal(result, expected)
    
    def test_large_batch_chunking(self):
        """Test chunked processing of large batches."""
        odf = MockODF(constant_value=1.5)
        n_orientations = 25
        orientations = np.tile(np.eye(3), (n_orientations, 1, 1))
        
        result = eval_odf_batch(odf, orientations, chunk_size=10)
        
        expected = np.full(n_orientations, 1.5)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (n_orientations,))
    
    def test_exact_chunk_boundary(self):
        """Test processing when batch size exactly matches chunk size."""
        odf = MockODF(constant_value=3.5)
        orientations = np.tile(np.eye(3), (20, 1, 1))
        
        result = eval_odf_batch(odf, orientations, chunk_size=20)
        
        expected = np.full(20, 3.5)
        np.testing.assert_array_equal(result, expected)
    
    def test_single_orientation_batch(self):
        """Test batch processing with single orientation."""
        odf = MockODF(constant_value=0.5)
        orientation = np.eye(3)
        
        result = eval_odf_batch(odf, orientation)
        
        self.assertEqual(result, 0.5)
    
    def test_validation_in_batch(self):
        """Test that validation works in batch processing."""
        odf = MockODF(constant_value=1.0)
        
        with self.assertRaises(ValueError):
            eval_odf_batch(odf, np.array([[1, 2]]))  # Invalid shape

class TestEvalAtIdentity(unittest.TestCase):
    """Test identity orientation evaluation."""
    
    def test_identity_evaluation(self):
        """Test evaluation at identity orientation."""
        odf = MockODF(constant_value=4.2)
        
        result = eval_at_identity(odf)
        
        self.assertEqual(result, 4.2)
        self.assertIsInstance(result, float)
    
    def test_identity_with_different_values(self):
        """Test identity evaluation with various ODF values."""
        test_values = [0.0, 1.0, 1.5, 10.0, 1e-6, 1e6]
        
        for value in test_values:
            with self.subTest(value=value):
                odf = MockODF(constant_value=value)
                result = eval_at_identity(odf)
                self.assertEqual(result, float(value))

class TestEvalRandomOrientations(unittest.TestCase):
    """Test random orientation generation and evaluation."""
    
    def test_random_orientations_generation(self):
        """Test generation of random orientations."""
        odf = MockODF(constant_value=1.0)
        n_orientations = 50
        
        orientations, values = eval_random_orientations(odf, n_orientations=n_orientations)
        
        # Check shapes
        self.assertEqual(orientations.shape, (n_orientations, 3, 3))
        self.assertEqual(values.shape, (n_orientations,))
        
        # Check that all values are the expected constant
        expected_values = np.full(n_orientations, 1.0)
        np.testing.assert_array_equal(values, expected_values)
    
    def test_random_orientations_are_rotations(self):
        """Test that generated orientations are proper rotation matrices."""
        odf = MockODF(constant_value=1.0)
        
        orientations, _ = eval_random_orientations(odf, n_orientations=10, seed=42)
        
        # Check that matrices are orthogonal (R @ R.T ≈ I)
        for i, R in enumerate(orientations):
            with self.subTest(matrix=i):
                product = R @ R.T
                identity = np.eye(3)
                np.testing.assert_array_almost_equal(product, identity, decimal=10)
                
                # Check determinant is +1 (proper rotation)
                det = np.linalg.det(R)
                self.assertAlmostEqual(det, 1.0, places=10)
    
    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results."""
        odf = MockODF(constant_value=2.0)
        
        orientations1, values1 = eval_random_orientations(odf, n_orientations=20, seed=123)
        orientations2, values2 = eval_random_orientations(odf, n_orientations=20, seed=123)
        
        np.testing.assert_array_equal(orientations1, orientations2)
        np.testing.assert_array_equal(values1, values2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        odf = MockODF(constant_value=1.0)
        
        orientations1, _ = eval_random_orientations(odf, n_orientations=10, seed=42)
        orientations2, _ = eval_random_orientations(odf, n_orientations=10, seed=43)
        
        # Should not be identical (with very high probability)
        self.assertFalse(np.array_equal(orientations1, orientations2))
    
    def test_small_batch_random_orientations(self):
        """Test random orientation generation with small batch."""
        odf = MockODF(constant_value=0.75)
        
        orientations, values = eval_random_orientations(odf, n_orientations=3)
        
        self.assertEqual(orientations.shape, (3, 3, 3))
        expected_values = np.array([0.75, 0.75, 0.75])
        np.testing.assert_array_equal(values, expected_values)
    
    def test_no_seed_variability(self):
        """Test that without seed, results vary between calls."""
        odf = MockODF(constant_value=1.0)
        
        orientations1, _ = eval_random_orientations(odf, n_orientations=5)
        orientations2, _ = eval_random_orientations(odf, n_orientations=5)
        
        # Should be different (with very high probability)
        self.assertFalse(np.allclose(orientations1, orientations2))

class TestEvaluationIntegration(unittest.TestCase):
    """Test integration between evaluation functions."""
    
    def test_function_compatibility(self):
        """Test that functions work together correctly."""
        odf = MockODF(constant_value=1.5)
        
        # Test that eval_odf and eval_at_identity are consistent
        identity_direct = eval_at_identity(odf)
        identity_generic = eval_odf(odf, np.eye(3))
        
        self.assertEqual(identity_direct, identity_generic)
    
    def test_validation_consistency(self):
        """Test that validation behaves consistently across functions."""
        odf = MockODF(constant_value=2.0)
        invalid_orientation = np.array([[1, 2]])  # Wrong shape
        
        # Both should raise ValueError for invalid shapes
        with self.assertRaises(ValueError):
            eval_odf(odf, invalid_orientation)
        
        with self.assertRaises(ValueError):
            eval_odf_batch(odf, invalid_orientation)
    
    def test_batch_vs_single_consistency(self):
        """Test that batch and single evaluations are consistent."""
        odf = MockODF(constant_value=3.0)
        orientation = np.eye(3)
        orientations_batch = np.array([orientation])
        
        single_result = eval_odf(odf, orientation)
        batch_result = eval_odf_batch(odf, orientations_batch)
        
        self.assertEqual(single_result, batch_result[0])

if __name__ == '__main__':
    unittest.main()
