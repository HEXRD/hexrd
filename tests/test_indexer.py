import io
import logging
import unittest
from types import SimpleNamespace

import numpy as np

from hexrd.hedm.indexer import _effective_map_tolerances


class TestEffectiveMapTolerances(unittest.TestCase):
    def setUp(self):
        self.eta_ome_maps = SimpleNamespace(
            etaEdges=np.radians(np.array([-1.0, 0.0, 1.0])),
            omeEdges=np.radians(np.array([-2.0, 0.0, 2.0])),
        )

        self.logger = logging.getLogger('hexrd.hedm.indexer')
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.old_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
        self.logger.setLevel(self.old_level)

    def test_warns_and_floors(self):
        eta_tol, ome_tol = _effective_map_tolerances(
            self.eta_ome_maps,
            np.radians(0.25),
            np.radians(0.5),
        )

        self.assertTrue(np.isclose(eta_tol, np.radians(1.0)))
        self.assertTrue(np.isclose(ome_tol, np.radians(2.0)))

        output = self.stream.getvalue()
        self.assertIn('eta map bin width', output)
        self.assertIn('omega map bin width', output)

    def test_keeps_requested_values(self):
        eta_tol, ome_tol = _effective_map_tolerances(
            self.eta_ome_maps,
            np.radians(1.5),
            np.radians(2.5),
        )

        self.assertTrue(np.isclose(eta_tol, np.radians(1.5)))
        self.assertTrue(np.isclose(ome_tol, np.radians(2.5)))
        self.assertEqual(self.stream.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
