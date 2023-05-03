import os

from .common import TestConfig, test_data


reference_data = \
"""
analysis_name: analysis
---
fit_grains:
  do_fit: false
  estimate: %(nonexistent_file)s
  npdiv: 1
  panel_buffer: 10
  threshold: 1850
  tolerance:
    eta: 1
    omega: 2
    tth: 3
  tth_max: false
---
fit_grains:
  estimate: %(existing_file)s
  panel_buffer: [20, 30]
  tolerance:
    eta: [1, 2]
    omega: [3, 4]
    tth: [5, 6]
  tth_max: 15
---
fit_grains:
  tth_max: -1
""" % test_data


class TestFitGrainsConfig(TestConfig):


    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_do_fit(self):
        self.assertTrue(self.cfgs[0].fit_grains.do_fit)
        self.assertFalse(self.cfgs[1].fit_grains.do_fit)

    def test_estimate(self):
        self.assertEqual(self.cfgs[0].fit_grains.estimate, None)
        # nonexistent file needs to return None
        self.assertEqual(
            self.cfgs[1].fit_grains.estimate,
            None
            )
        self.assertEqual(
            self.cfgs[2].fit_grains.estimate,
            test_data['existing_file']
            )

    def test_npdiv(self):
        self.assertEqual(self.cfgs[0].fit_grains.npdiv, 2)
        self.assertEqual(self.cfgs[1].fit_grains.npdiv, 1)

    def test_threshold(self):
        self.assertRaises(
            RuntimeError,
            getattr, self.cfgs[0].fit_grains, 'threshold'
            )
        self.assertEqual(self.cfgs[1].fit_grains.threshold, 1850)

    def test_tth_max(self):
        self.assertTrue(self.cfgs[0].fit_grains.tth_max)
        self.assertFalse(self.cfgs[1].fit_grains.tth_max)
        self.assertEqual(self.cfgs[2].fit_grains.tth_max, 15)
        self.assertRaises(
            RuntimeError,
            getattr, self.cfgs[3].fit_grains, 'tth_max'
            )


class TestToleranceConfig(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_eta(self):
        self.assertRaises(
            RuntimeError,
            getattr, self.cfgs[0].fit_grains.tolerance, 'eta'
            )
        self.assertEqual(
            self.cfgs[1].fit_grains.tolerance.eta,
            [1, 1]
            )
        self.assertEqual(
            self.cfgs[2].fit_grains.tolerance.eta,
            [1, 2]
            )

    def test_omega(self):
        self.assertRaises(
            RuntimeError,
            getattr, self.cfgs[0].fit_grains.tolerance, 'omega'
            )
        self.assertEqual(
            self.cfgs[1].fit_grains.tolerance.omega,
            [2, 2]
            )
        self.assertEqual(
            self.cfgs[2].fit_grains.tolerance.omega,
            [3, 4]
            )

    def test_tth(self):
        self.assertRaises(
            RuntimeError,
            getattr, self.cfgs[0].fit_grains.tolerance, 'tth'
            )
        self.assertEqual(
            self.cfgs[1].fit_grains.tolerance.tth,
            [3, 3]
            )
        self.assertEqual(
            self.cfgs[2].fit_grains.tolerance.tth,
            [5, 6]
            )
