"""Tests for UnimodalODF."""

import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from hexrd.phase_transition.texture.kernels import DeLaValleePoussinKernel
from hexrd.phase_transition.texture.unimodal_odf import UnimodalODF


def _rotation_about_z(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _cubic_kernel(halfwidth_deg: float = 15.0) -> DeLaValleePoussinKernel:
    """A kernel carrying cubic crystal symmetry (single source of truth)."""
    return DeLaValleePoussinKernel(
        halfwidth=np.radians(halfwidth_deg),
        crystal_symmetry='oh',
        sample_symmetry='triclinic',
    )


class TestUnimodalODF(unittest.TestCase):
    """Test the real UnimodalODF against the real DeLaValleePoussinKernel."""

    def setUp(self):
        # Default kernel carries NO symmetry, so modes stay distinct.
        self.kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
        self.modal = np.eye(3)

    # --- construction ---

    def test_package_export(self):
        """UnimodalODF is importable from the texture package."""
        from hexrd.phase_transition.texture import UnimodalODF as Cls
        self.assertIs(Cls, UnimodalODF)

    def test_initialization_single_modal(self):
        """A single (3, 3) modal orientation is stored as one component."""
        odf = UnimodalODF(self.modal, self.kernel)
        self.assertEqual(odf.n_components, 1)
        self.assertEqual(odf.modal_orientations.shape, (1, 3, 3))
        np.testing.assert_array_equal(odf.modal_orientations[0], self.modal)
        np.testing.assert_array_almost_equal(odf.weights, [1.0])
        self.assertIs(odf.kernel, self.kernel)

    def test_initialization_multiple_modals(self):
        """Multiple modal orientations and explicit weights are stored."""
        modals = np.array([np.eye(3), _rotation_about_z(np.radians(90))])
        odf = UnimodalODF(modals, self.kernel, weights=[0.6, 0.4])
        self.assertEqual(odf.n_components, 2)
        np.testing.assert_array_equal(odf.modal_orientations, modals)
        np.testing.assert_array_almost_equal(odf.weights, [0.6, 0.4])

    def test_equal_weights_default(self):
        """Equal weights are assigned when none are provided."""
        modals = np.array([
            np.eye(3),
            _rotation_about_z(np.radians(45)),
            _rotation_about_z(np.radians(90)),
        ])
        odf = UnimodalODF(modals, self.kernel)
        np.testing.assert_array_almost_equal(odf.weights, np.full(3, 1.0 / 3))

    # --- symmetry delegation (Option A: kernel is the single source) ---

    def test_symmetry_delegates_to_kernel(self):
        """crystal/sample symmetry are read straight from the kernel."""
        kernel = _cubic_kernel()
        odf = UnimodalODF(self.modal, kernel)
        self.assertEqual(odf.crystal_symmetry, 'oh')
        self.assertEqual(odf.sample_symmetry, 'triclinic')
        self.assertEqual(odf.crystal_symmetry, kernel.crystal_symmetry)
        self.assertEqual(odf.sample_symmetry, kernel.sample_symmetry)

    def test_no_symmetry_kernel_reports_none(self):
        """A kernel built without symmetry reports None for both labels."""
        odf = UnimodalODF(self.modal, self.kernel)
        self.assertIsNone(odf.crystal_symmetry)
        self.assertIsNone(odf.sample_symmetry)

    def test_symmetry_makes_equivalent_modes_equal(self):
        """Symmetry now flows through the kernel into eval.

        Under cubic symmetry, a 90 deg rotation about z is equivalent to the
        identity, so the ODF evaluates equally at both. Without symmetry the
        two orientations are distinct.
        """
        cubic = UnimodalODF(self.modal, _cubic_kernel())
        nosym = UnimodalODF(self.modal, self.kernel)
        r90 = _rotation_about_z(np.radians(90))
        self.assertAlmostEqual(cubic.eval(r90), cubic.eval(np.eye(3)), places=6)
        self.assertLess(nosym.eval(r90), nosym.eval(np.eye(3)))

    # --- validation ---

    def test_invalid_kernel_type(self):
        """A kernel that is not an SO3Kernel raises TypeError."""
        with self.assertRaises(TypeError):
            UnimodalODF(self.modal, object())

    def test_accepts_any_so3_kernel_subclass(self):
        """Any SO3Kernel subclass is accepted, not just DeLaValleePoussin."""
        from hexrd.phase_transition.texture.kernels import SO3Kernel

        class _DummyKernel(SO3Kernel):
            crystal_symmetry = None
            sample_symmetry = None

            def eval(self, orientations, reference=None):
                return np.zeros(np.asarray(orientations).shape[:-2])

        # Construction must succeed (does not raise TypeError).
        odf = UnimodalODF(self.modal, _DummyKernel())
        self.assertIsInstance(odf.kernel, SO3Kernel)

    def test_invalid_modal_orientation_shape(self):
        """Modal orientations must have shape (3, 3) or (N, 3, 3)."""
        with self.assertRaises(ValueError):
            UnimodalODF(np.array([1, 2, 3]), self.kernel)
        with self.assertRaises(ValueError):
            UnimodalODF(np.random.random((3, 2)), self.kernel)

    def test_invalid_weights(self):
        """Weights must match component count, be non-negative, and sum to 1."""
        modals = np.array([np.eye(3), _rotation_about_z(np.radians(90))])
        with self.assertRaises(ValueError):
            UnimodalODF(modals, self.kernel, weights=[1.0])
        with self.assertRaises(ValueError):
            UnimodalODF(modals, self.kernel, weights=[0.3, 0.3])
        with self.assertRaises(ValueError):
            UnimodalODF(modals, self.kernel, weights=[1.2, -0.2])

    # --- evaluation ---

    def test_single_orientation_evaluation(self):
        """A single orientation evaluates to a positive scalar, peaked at the mode."""
        odf = UnimodalODF(self.modal, self.kernel)
        value_modal = odf.eval(self.modal)
        value_far = odf.eval(_rotation_about_z(np.radians(90)))
        self.assertIsInstance(value_modal, float)
        self.assertIsInstance(value_far, float)
        self.assertGreater(value_modal, value_far)
        self.assertGreater(value_far, 0.0)

    def test_eval_at_mode_equals_kernel_peak(self):
        """For a single mode (weight 1), eval at the mode equals the kernel peak C."""
        odf = UnimodalODF(self.modal, self.kernel)
        self.assertAlmostEqual(
            odf.eval(self.modal), self.kernel.norm_constant, places=6
        )

    def test_multiple_orientations_evaluation(self):
        """A batch of orientations returns a 1-D array peaked at the mode."""
        odf = UnimodalODF(self.modal, self.kernel)
        orientations = np.array([
            np.eye(3),
            _rotation_about_z(np.radians(45)),
            _rotation_about_z(np.radians(90)),
        ])
        values = odf.eval(orientations)
        self.assertEqual(values.shape, (3,))
        self.assertEqual(np.argmax(values), 0)
        self.assertTrue(np.all(values > 0))

    def test_batch_orientation_evaluation(self):
        """Multi-dimensional batches preserve leading dimensions."""
        odf = UnimodalODF(self.modal, self.kernel)
        batch = np.tile(self.modal, (2, 3, 1, 1))  # (2, 3, 3, 3)
        values = odf.eval(batch)
        self.assertEqual(values.shape, (2, 3))
        np.testing.assert_allclose(values, self.kernel.norm_constant)

    def test_invalid_evaluation_shapes(self):
        """Evaluation rejects inputs that are not (..., 3, 3)."""
        odf = UnimodalODF(self.modal, self.kernel)
        with self.assertRaises(ValueError):
            odf.eval(np.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            odf.eval(np.random.random((3, 2)))

    def test_weighted_components(self):
        """At a well-separated mode the value is approximately w_i * C.

        The default kernel carries no crystal symmetry, so the two modes
        (identity and 90 deg about z) are treated as distinct orientations.
        """
        modals = np.array([np.eye(3), _rotation_about_z(np.radians(90))])
        odf = UnimodalODF(modals, self.kernel, weights=[0.7, 0.3])
        peak = self.kernel.norm_constant
        value_0 = odf.eval(np.eye(3))
        value_1 = odf.eval(_rotation_about_z(np.radians(90)))
        self.assertGreater(value_0, value_1)
        np.testing.assert_allclose(value_0, 0.7 * peak, rtol=1e-2)
        np.testing.assert_allclose(value_1, 0.3 * peak, rtol=1e-2)

    def test_mrd_normalization(self):
        """The ODF integrates to mean 1 (MRD) over SO(3).

        A weighted de la Vallee Poussin kernel sum is already normalized,
        so the Monte Carlo mean over uniformly random orientations is ~1.
        """
        kernel = DeLaValleePoussinKernel(halfwidth=np.radians(40))
        odf = UnimodalODF(np.eye(3), kernel)
        samples = Rotation.random(60000, random_state=7).as_matrix()
        mean_mrd = np.mean(odf.eval(samples))
        self.assertAlmostEqual(mean_mrd, 1.0, delta=0.03)

    # --- derived quantities ---

    def test_estimated_max_value(self):
        """estimated_max_value equals the peak (kernel C) for a single mode."""
        odf = UnimodalODF(self.modal, self.kernel)
        max_val = odf.estimated_max_value()
        self.assertIsInstance(max_val, float)
        self.assertAlmostEqual(max_val, self.kernel.norm_constant, places=6)
        self.assertAlmostEqual(max_val, odf.eval(self.modal), places=6)

    # --- properties / representations ---

    def test_property_getters_return_copies(self):
        """modal_orientations and weights getters return independent copies."""
        modals = np.array([np.eye(3), _rotation_about_z(np.radians(90))])
        odf = UnimodalODF(modals, self.kernel, weights=[0.6, 0.4])

        modals_view = odf.modal_orientations
        modals_view[0] *= 2
        np.testing.assert_array_equal(odf.modal_orientations[0], np.eye(3))

        weights_view = odf.weights
        weights_view[0] = 0.9
        self.assertAlmostEqual(odf.weights[0], 0.6)

    def test_string_representations(self):
        """__repr__ and __str__ contain the key descriptors."""
        odf = UnimodalODF(self.modal, _cubic_kernel())

        repr_str = repr(odf)
        self.assertIn('UnimodalODF', repr_str)
        self.assertIn('oh', repr_str)
        self.assertIn('triclinic', repr_str)

        str_repr = str(odf)
        self.assertIn('Unimodal ODF', str_repr)
        self.assertIn('MRD', str_repr)

    def test_string_representations_without_symmetry(self):
        """repr/str render gracefully when the kernel has no symmetry."""
        odf = UnimodalODF(self.modal, self.kernel)
        self.assertIn('crystal_symmetry=None', repr(odf))
        self.assertIn('none', str(odf))


if __name__ == '__main__':
    unittest.main()
