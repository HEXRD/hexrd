from .common import InstrumentTest, np

from hexrd.instrument import instrument

class TestInstrument(InstrumentTest):

    def test_instantiation(self):
        # make sure it instantiates successfully
        self.assertTrue(isinstance(instrument.HEDMInstrument(),
                                   instrument.HEDMInstrument)
        )
