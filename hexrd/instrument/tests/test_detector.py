from .common import InstrumentTest, np

from hexrd.instrument import detector

class TestDetector(InstrumentTest):

    def
    def test_instantiation(self):
        # make sure it instantiates successfully
        self.assertTrue(isinstance(detector.PlanarDetector(),
                                   detector.PlanarDetector)
        )

    def test_pixel_size(self):
