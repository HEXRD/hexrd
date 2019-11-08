from .common import TestConfig, test_data


reference_data = \
"""
material:
  definitions: %(existing_file)s
#  active: # not set to test error
---
material:
  definitions: %(nonexistent_file)s
  active: ruby
---
material:
#  definitions: # not set to test inheritance
  active: CeO2
""" % test_data


class TestMaterialConfig(TestConfig):


    @classmethod
    def get_reference_data(cls):
        return reference_data


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
