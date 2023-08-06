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
  output_format: ["a", "b"]
---
# cgf #4
fit_grains:
  reset_exclusions: false
---
# cfg #5
fit_grains:
  dmin: 0.1
  tthmin: 0.2
  sfacmin: 0.3
  pintmin: 0.4
---
# cfg #6
fit_grains:
  dmax: 1.1
  tthmax: 1.2
  sfacmax: 1.3
  pintmax: 1.4
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

    def test_output_format(self):
        self.assertTrue(self.cfgs[0].fit_grains.output_format[0] == 'summary')
        self.assertTrue(self.cfgs[3].fit_grains.output_format[0] == 'a')

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


class TestExclusions(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_reset_exclusions(self):
        for i in range(4):
            self.assertTrue(self.cfgs[i].fit_grains.reset_exclusions)
        for i in range(4, 7):
            self.assertFalse(self.cfgs[i].fit_grains.reset_exclusions)

    def test_exclusion_parameters(self):
        ep = self.cfgs[5].fit_grains.exclusion_parameters
        self.assertEqual(ep.dmin, 0.1)
        self.assertEqual(ep.tthmin, 0.2)
        self.assertEqual(ep.sfacmin, 0.3)
        self.assertEqual(ep.pintmin, 0.4)

        self.assertEqual(ep.dmax, None)
        self.assertEqual(ep.tthmax, None)
        self.assertEqual(ep.sfacmax, None)
        self.assertEqual(ep.pintmax, None)

        ep = self.cfgs[6].fit_grains.exclusion_parameters
        self.assertEqual(ep.dmin, 0.1)
        self.assertEqual(ep.tthmin, 0.2)
        self.assertEqual(ep.sfacmin, 0.3)
        self.assertEqual(ep.pintmin, 0.4)

        self.assertEqual(ep.dmax, 1.1)
        self.assertEqual(ep.tthmax, 1.2)
        self.assertEqual(ep.sfacmax, 1.3)
        self.assertEqual(ep.pintmax, 1.4)
