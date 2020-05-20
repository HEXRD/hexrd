from .common import InstrumentTest, np

from hexrd.instrument import detector

class TestDetector(InstrumentTest):

    def test_instantiation(self):
        # make sure it instantiates successfully
        self.assertTrue(isinstance(detector.PlanarDetector(),
                                   detector.PlanarDetector)
        )

    def test_rows_cols(self):
        det = detector.PlanarDetector()
        self.assertTrue(det.rows == detector.rows_DFLT)
        self.assertTrue(det.cols == detector.cols_DFLT)
        det.rows = 2
        self.assertTrue(det.rows == 2)
        det.cols = 3
        self.assertTrue(det.cols == 3)


    def test_pixel_size(self):
        det = detector.PlanarDetector()
        dflt = detector.pixel_size_DFLT
        self.assertTrue(det.pixel_size_row == dflt[0])
        self.assertTrue(det.pixel_size_col == dflt[1])
        self.assertTrue(det.pixel_area == dflt[0]*dflt[1])
        det.pixel_size_row = 10.
        self.assertTrue(det.pixel_size_row == 10)
        det.pixel_size_col = 11
        self.assertTrue(det.pixel_size_col == 11)

    def test_distortion(self):
        det = detector.PlanarDetector()
        det.distortion = None
