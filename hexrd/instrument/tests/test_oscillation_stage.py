from .common import InstrumentTest, np

from hexrd.instrument import oscillation_stage as ostage

class TestOscillationStage(InstrumentTest):

    def test_ostage_values(self):
        stage = ostage.OscillationStage(np.array((1., 2.0, 3.0)), 0.5)
        self.assertEqual(0.5, stage.chi)
        self.assertEqual(1.0, stage.tvec[0])
        self.assertEqual(2.0, stage.tvec[1])
        self.assertEqual(3.0, stage.tvec[2])
