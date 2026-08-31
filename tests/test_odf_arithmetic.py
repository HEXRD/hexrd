"""Tests for ODF addition and subtraction (CompositeODF)."""

import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from hexrd.phase_transition.texture.composite_odf import CompositeODF
from hexrd.phase_transition.texture.kernels import DeLaValleePoussinKernel
from hexrd.phase_transition.texture.unimodal_odf import UnimodalODF
from hexrd.phase_transition.texture.uniform_odf import UniformODF


def _rotation_about_z(angle: float) -> np.ndarray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0],
    ])


class TestODFArithmetic(unittest.TestCase):
    """Addition and subtraction between ODFs."""

    def setUp(self):
        # Kernels carrying NO symmetry, so modes stay distinct.
        self.kernel = DeLaValleePoussinKernel(halfwidth=np.radians(15))
        self.odf_a = UnimodalODF(np.eye(3), self.kernel)
        self.odf_b = UnimodalODF(
            _rotation_about_z(np.radians(60)), self.kernel
        )
        self.uniform = UniformODF()
        self.orientations = Rotation.random(25, random_state=7).as_matrix()

    def test_package_export(self):
        """CompositeODF is importable from the texture package."""
        from hexrd.phase_transition.texture import CompositeODF as Cls
        self.assertIs(Cls, CompositeODF)

    def test_addition_returns_composite(self):
        """odf + odf produces a CompositeODF holding both operands."""
        total = self.odf_a + self.odf_b

        self.assertIsInstance(total, CompositeODF)
        self.assertEqual(total.n_components, 2)
        self.assertIs(total.components[0], self.odf_a)
        self.assertIs(total.components[1], self.odf_b)
        np.testing.assert_array_equal(total.coefficients, [1.0, 1.0])
        self.assertEqual(total.constant, 0.0)

    def test_subtraction_negates_second_operand(self):
        """odf - odf keeps both operands with coefficients +1 and -1."""
        difference = self.odf_a - self.odf_b

        self.assertEqual(difference.n_components, 2)
        np.testing.assert_array_equal(difference.coefficients, [1.0, -1.0])

    def test_coefficient_length_validated(self):
        """A coefficient per component is required."""
        with self.assertRaises(ValueError):
            CompositeODF([self.odf_a, self.odf_b], [1.0])

    def test_non_odf_component_rejected(self):
        """Components must expose an eval() method."""
        with self.assertRaises(TypeError):
            CompositeODF([self.odf_a, 'not an odf'])

    def test_sum_evaluates_pointwise(self):
        """(a + b)(g) == a(g) + b(g), with no renormalization."""
        total = self.odf_a + self.odf_b

        expected = (
            self.odf_a.eval(self.orientations)
            + self.odf_b.eval(self.orientations)
        )
        np.testing.assert_allclose(total.eval(self.orientations), expected)

    def test_difference_evaluates_pointwise(self):
        """(a - b)(g) == a(g) - b(g)."""
        difference = self.odf_a - self.odf_b

        expected = (
            self.odf_a.eval(self.orientations)
            - self.odf_b.eval(self.orientations)
        )
        np.testing.assert_allclose(
            difference.eval(self.orientations), expected
        )

    def test_self_difference_is_zero(self):
        """a - a vanishes everywhere."""
        difference = self.odf_a - self.odf_a

        np.testing.assert_allclose(
            difference.eval(self.orientations),
            np.zeros(len(self.orientations)),
            atol=1e-10,
        )

    def test_difference_can_be_negative(self):
        """A difference of ODFs is not constrained to be non-negative."""
        difference = self.odf_a - self.odf_b

        # At b's mode, b dominates a (60 deg away, 15 deg halfwidth).
        value = difference.eval(_rotation_about_z(np.radians(60)))
        self.assertLess(value, 0.0)

    def test_sum_is_not_renormalized(self):
        """A sum of two mean-1 MRD ODFs has mean 2 MRD."""
        total = self.odf_a + self.odf_b

        samples = Rotation.random(200000, random_state=3).as_matrix()
        self.assertAlmostEqual(np.mean(total.eval(samples)), 2.0, places=1)

    def test_single_orientation_returns_scalar(self):
        """A single (3, 3) input evaluates to a scalar float."""
        total = self.odf_a + self.odf_b

        value = total.eval(np.eye(3))
        self.assertIsInstance(value, float)

    def test_output_shape_matches_leading_dimensions(self):
        """Output shape follows the leading dimensions of the input."""
        total = self.odf_a + self.odf_b

        orientations = np.broadcast_to(np.eye(3), (2, 4, 3, 3))
        self.assertEqual(total.eval(orientations).shape, (2, 4))

    def test_invalid_orientation_shape(self):
        """Orientations must have shape (..., 3, 3)."""
        total = self.odf_a + self.odf_b

        with self.assertRaises(ValueError):
            total.eval(np.zeros((5, 2, 2)))

    def test_add_constant(self):
        """odf + c and c + odf add a constant MRD offset."""
        for shifted in (self.odf_a + 0.5, 0.5 + self.odf_a):
            self.assertEqual(shifted.constant, 0.5)
            self.assertEqual(shifted.n_components, 1)
            np.testing.assert_allclose(
                shifted.eval(self.orientations),
                self.odf_a.eval(self.orientations) + 0.5,
            )

    def test_subtract_constant(self):
        """odf - c subtracts a constant MRD offset."""
        shifted = self.odf_a - 0.25

        self.assertEqual(shifted.constant, -0.25)
        np.testing.assert_allclose(
            shifted.eval(self.orientations),
            self.odf_a.eval(self.orientations) - 0.25,
        )

    def test_constant_minus_odf(self):
        """c - odf negates the ODF and keeps c as the offset."""
        flipped = 2.0 - self.odf_a

        self.assertEqual(flipped.constant, 2.0)
        np.testing.assert_array_equal(flipped.coefficients, [-1.0])
        np.testing.assert_allclose(
            flipped.eval(self.orientations),
            2.0 - self.odf_a.eval(self.orientations),
        )

    def test_numpy_scalars_accepted_as_constants(self):
        """numpy real scalars work as constant offsets."""
        self.assertEqual((self.odf_a + np.float64(1.5)).constant, 1.5)
        self.assertEqual((np.int64(2) + self.odf_a).constant, 2.0)

    def test_unsupported_operands_raise_type_error(self):
        """Non-ODF, non-scalar operands are rejected."""
        for other in ('text', None, [1.0, 2.0], np.array([1.0, 2.0]), True):
            with self.assertRaises(TypeError):
                self.odf_a + other

    def test_unsupported_left_operand_reports_the_operand_types(self):
        """An unsupported left operand names both types and the operator.

        The reflected methods are the last handler Python consults, so they
        raise rather than return NotImplemented. Otherwise CPython falls
        through to the sequence-concatenation slot, and a numpy left operand
        reports the failure as a suggestion to use np.concatenate.
        """
        for other in ('text', None, [1.0, 2.0], np.array([1.0, 2.0]), True):
            for symbol in ('+', '-'):
                with self.subTest(other=type(other).__name__, op=symbol):
                    with self.assertRaises(TypeError) as caught:
                        if symbol == '+':
                            other + self.odf_a
                        else:
                            other - self.odf_a

                    message = str(caught.exception)
                    self.assertIn('unsupported operand type(s)', message)
                    self.assertIn(f'for {symbol}:', message)
                    self.assertIn(type(other).__name__, message)
                    self.assertIn('UnimodalODF', message)
                    self.assertNotIn('concatenat', message.lower())

    def test_forward_operations_still_defer_to_the_right_operand(self):
        """__add__ returns NotImplemented so other types can interoperate.

        Only the reflected methods raise; the forward ones must keep
        deferring, or a class defining __radd__ against an ODF could never
        be reached.
        """
        class HandlesODF:
            def __radd__(self, other):
                return 'handled'

        self.assertEqual(self.odf_a + HandlesODF(), 'handled')

        # sum() relies on the same deferral: it seeds the total with int 0.
        total = sum([self.odf_a, self.odf_b])
        self.assertEqual(total.n_components, 2)
        self.assertEqual(total.constant, 0.0)

    def test_uniform_folds_into_constant(self):
        """A UniformODF operand becomes a 1 MRD constant, not a component."""
        total = self.odf_a + self.uniform

        self.assertEqual(total.n_components, 1)
        self.assertEqual(total.constant, 1.0)
        np.testing.assert_allclose(
            total.eval(self.orientations),
            self.odf_a.eval(self.orientations) + 1.0,
        )

    def test_uniform_plus_uniform_is_pure_constant(self):
        """uniform + uniform is the constant 2 MRD function."""
        total = self.uniform + self.uniform

        self.assertEqual(total.n_components, 0)
        self.assertEqual(total.constant, 2.0)
        self.assertEqual(total.eval(np.eye(3)), 2.0)

    def test_subtracted_uniform_folds_negatively(self):
        """odf - uniform subtracts a 1 MRD background."""
        difference = self.odf_a - self.uniform

        self.assertEqual(difference.constant, -1.0)
        np.testing.assert_allclose(
            difference.eval(self.orientations),
            self.odf_a.eval(self.orientations) - 1.0,
        )

    def test_nested_sums_are_flattened(self):
        """Chained operations keep a single flat list of components."""
        total = (self.odf_a + self.odf_b) - self.odf_a

        self.assertEqual(total.n_components, 3)
        np.testing.assert_array_equal(total.coefficients, [1.0, 1.0, -1.0])
        for component in total.components:
            self.assertNotIsInstance(component, CompositeODF)

    def test_subtracting_a_composite_propagates_the_sign(self):
        """a - (b - a) flattens to +a, -b, +a."""
        total = self.odf_a - (self.odf_b - self.odf_a)

        np.testing.assert_array_equal(total.coefficients, [1.0, -1.0, 1.0])
        np.testing.assert_allclose(
            total.eval(self.orientations),
            2.0 * self.odf_a.eval(self.orientations)
            - self.odf_b.eval(self.orientations),
        )

    def test_constants_accumulate_through_flattening(self):
        """Constant offsets of nested composites add up."""
        total = (self.odf_a + 1.0) - 0.25

        self.assertEqual(total.n_components, 1)
        self.assertAlmostEqual(total.constant, 0.75)

    # --- symmetry ---

    def test_matching_symmetries_are_inherited(self):
        """A composite reports the symmetry shared by its components."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(15),
            crystal_symmetry='oh',
            sample_symmetry='triclinic',
        )
        first = UnimodalODF(np.eye(3), kernel)
        second = UnimodalODF(_rotation_about_z(np.radians(30)), kernel)

        total = first + second
        self.assertEqual(total.crystal_symmetry, 'oh')
        self.assertEqual(total.sample_symmetry, 'triclinic')

    def test_conflicting_crystal_symmetry_rejected(self):
        """Components with different crystal symmetries cannot be combined."""
        cubic = UnimodalODF(
            np.eye(3),
            DeLaValleePoussinKernel(
                halfwidth=np.radians(15), crystal_symmetry='oh'
            ),
        )
        hexagonal = UnimodalODF(
            np.eye(3),
            DeLaValleePoussinKernel(
                halfwidth=np.radians(15), crystal_symmetry='d6h'
            ),
        )

        with self.assertRaises(ValueError):
            cubic + hexagonal

    def test_conflicting_sample_symmetry_rejected(self):
        """Components with different sample symmetries cannot be combined."""
        triclinic = UnimodalODF(
            np.eye(3),
            DeLaValleePoussinKernel(
                halfwidth=np.radians(15), sample_symmetry='triclinic'
            ),
        )
        orthorhombic = UnimodalODF(
            np.eye(3),
            DeLaValleePoussinKernel(
                halfwidth=np.radians(15), sample_symmetry='orthorhombic'
            ),
        )

        with self.assertRaises(ValueError):
            triclinic - orthorhombic

    def test_uniform_is_exempt_from_symmetry_check(self):
        """A uniform operand folds into a constant, so its label is inert."""
        cubic = UnimodalODF(
            np.eye(3),
            DeLaValleePoussinKernel(
                halfwidth=np.radians(15), crystal_symmetry='oh'
            ),
        )

        total = UniformODF('d6h', 'monoclinic') + cubic
        self.assertEqual(total.crystal_symmetry, 'oh')
        self.assertEqual(total.constant, 1.0)

    def test_unlabeled_symmetry_does_not_conflict(self):
        """A component with no symmetry label is not checked against others."""
        cubic = UnimodalODF(
            np.eye(3),
            DeLaValleePoussinKernel(
                halfwidth=np.radians(15), crystal_symmetry='oh'
            ),
        )

        total = cubic + self.odf_a
        self.assertEqual(total.crystal_symmetry, 'oh')

    def test_symmetric_sum_is_not_renormalized(self):
        """With symmetry too, a sum of two ODFs has mean 2 MRD."""
        kernel = DeLaValleePoussinKernel(
            halfwidth=np.radians(15), crystal_symmetry='oh'
        )
        first = UnimodalODF(np.eye(3), kernel)
        second = UnimodalODF(_rotation_about_z(np.radians(30)), kernel)

        samples = Rotation.random(50000, random_state=5).as_matrix()
        values = (first + second).eval(samples)
        self.assertAlmostEqual(np.mean(values), 2.0, places=1)

    def test_constant_composite_texture_index_is_exact(self):
        """A pure-constant composite has texture index 1 with no sampling."""
        constant = self.uniform + self.uniform

        self.assertEqual(constant.analytic_texture_index(), 1.0)
        self.assertEqual(constant.texture_index(), 1.0)
        self.assertEqual(constant.norm(), 1.0)

    def test_no_closed_form_with_components(self):
        """A composite with ODF terms falls back to Monte Carlo."""
        total = self.odf_a + self.odf_b

        self.assertIsNone(total.analytic_texture_index())
        self.assertGreater(total.texture_index(n_orientations=5000, seed=0), 1.0)

    def test_batch_evaluation_helper(self):
        """eval_odf_batch works on a composite."""
        from hexrd.phase_transition.texture import eval_odf_batch

        total = self.odf_a + self.odf_b
        values = eval_odf_batch(total, self.orientations, chunk_size=10)

        np.testing.assert_allclose(values, total.eval(self.orientations))

    def test_repr_and_str(self):
        """repr and str summarize the composite."""
        total = self.odf_a + self.odf_b + 0.5

        self.assertIn('CompositeODF(n_components=2', repr(total))
        self.assertIn('Constant offset: 0.500000 MRD', str(total))
        self.assertIn('+1 * UnimodalODF', str(total))


if __name__ == '__main__':
    unittest.main()
