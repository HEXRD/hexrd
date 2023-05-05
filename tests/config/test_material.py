from .common import TestConfig, test_data
from hexrd.config.material import TTHW_DFLT, DMIN_DFLT


reference_data = \
"""
material:
  definitions: %(existing_file)s
#  active: # not set to test error
# instrument: %(instrument-cfg)s
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
reset_exclusions: True
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
        self.assertEqual(self.cfgs[0].material.reset_exclusions, False)
        self.assertEqual(self.cfgs[1].material.reset_exclusions, False)
        self.assertEqual(self.cfgs[2].material.reset_exclusions, False)
