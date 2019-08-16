from .common import InstrumentTest, np

from hexrd.instrument import io

class TestIO(InstrumentTest):

    fname = '/tmp/filename'

    def test_io_instantiation(self):
        self.assertTrue(isinstance(io.PatchDataWriter(self.fname),
                                   io.PatchDataWriter)
        )
        self.assertTrue(isinstance(io.GrainDataWriter(self.fname),
                                   io.GrainDataWriter)
        )
