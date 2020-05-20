from .common import InstrumentTest, np

from hexrd.instrument import beam

class TestBeam(InstrumentTest):

    def test_beam_values(self):
        b = beam.Beam(83.2, np.array((1., 2.0, 3.0)))
        self.assertEqual(83.2, b.energy)
        self.assertEqual(1.0, b.vector[0])
        self.assertEqual(2.0, b.vector[1])
        self.assertEqual(3.0, b.vector[2])
