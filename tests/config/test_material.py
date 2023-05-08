from .common import TestConfig, test_data
from hexrd.config.material import TTHW_DFLT, DMIN_DFLT


reference_data = \
"""
material:
  definitions: %(existing_file)s
#  active: # not set to test error
#  instrument: %(instrument-cfg)s
---
material:
  definitions: %(nonexistent_file)s
  active: ruby
  tth_width: 0.2
---
material:
#  definitions: # not set to test inheritance
  active: CeO2
use_saved_hkls: true
---
# cfg: 3
material:
  reset_exclusions: false
---
# cfg 4
material:
  dmin: 1.0
  dmax: 3.0
---
# cfg 5
material:
  tthmin: 1.5
  tthmax: 5.0
  min_sfac_ratio: 0.02
---
# cfg 6
material:
  sfacmin: 0.01
  sfacmax: 0.99
---
# cfg 7
material:
  pintmin: 0.05
  pintmax: 0.95
"""


class TestMaterialConfig(TestConfig):


    @classmethod
    def get_reference_data(cls):
        return reference_data % test_data


    def test_definitions(self):
        self.assertEqual(
            self.cfgs[0].material.definitions,
            test_data['existing_file']
            )
        self.assertRaises(
            IOError,
            getattr, self.cfgs[1].material, 'definitions'
            )


    def test_active(self):
        self.assertRaises(
            RuntimeError,
            getattr, self.cfgs[0].material, 'active'
            )
        self.assertEqual(self.cfgs[1].material.active, 'ruby')
        self.assertEqual(self.cfgs[2].material.active, 'CeO2')

    def test_two_theta_width(self):
        self.assertEqual(self.cfgs[0].material.tthw, TTHW_DFLT)
        self.assertEqual(self.cfgs[1].material.tthw, 0.2)
        self.assertEqual(self.cfgs[2].material.tthw, 0.2)

    def test_reset_exclusions(self):
        self.assertEqual(self.cfgs[0].material.reset_exclusions, True)
        self.assertEqual(self.cfgs[1].material.reset_exclusions, True)
        self.assertEqual(self.cfgs[2].material.reset_exclusions, True)
        self.assertEqual(self.cfgs[3].material.reset_exclusions, False)
        self.assertEqual(self.cfgs[4].material.reset_exclusions, False)

    def test_exclusion_parameters_d(self):
        self.assertEqual(self.cfgs[3].material.exclusion_parameters.dmin, None)
        self.assertEqual(self.cfgs[3].material.exclusion_parameters.dmax, None)
        self.assertEqual(self.cfgs[4].material.exclusion_parameters.dmin, 1.0)
        self.assertEqual(self.cfgs[4].material.exclusion_parameters.dmax, 3.0)

    def test_exclusion_parameters_tth(self):
        self.assertEqual(
            self.cfgs[4].material.exclusion_parameters.tthmin, None
        )
        self.assertEqual(
            self.cfgs[4].material.exclusion_parameters.tthmax, None
        )
        self.assertEqual(
            self.cfgs[5].material.exclusion_parameters.tthmin, 1.5
        )
        self.assertEqual(
            self.cfgs[5].material.exclusion_parameters.tthmax, 5.0
        )

    def test_exclusion_parameters_sfac(self):
        self.assertEqual(
            self.cfgs[4].material.exclusion_parameters.sfacmin, None
        )
        self.assertEqual(
            self.cfgs[5].material.exclusion_parameters.sfacmax, None
        )
        self.assertEqual(
            self.cfgs[5].material.exclusion_parameters.sfacmin, 0.02
        )
        self.assertEqual(
            self.cfgs[6].material.exclusion_parameters.sfacmin, 0.01
        )
        self.assertEqual(
            self.cfgs[6].material.exclusion_parameters.sfacmax, 0.99
        )

    def test_exclusion_parameters_pint(self):
        self.assertEqual(
            self.cfgs[6].material.exclusion_parameters.pintmin, None
        )
        self.assertEqual(
            self.cfgs[6].material.exclusion_parameters.pintmax, None
        )
        self.assertEqual(
            self.cfgs[7].material.exclusion_parameters.pintmin, 0.05
        )
        self.assertEqual(
            self.cfgs[7].material.exclusion_parameters.pintmax, 0.95
        )
