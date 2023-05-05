import os
import shutil
import tempfile
import logging
import unittest

from hexrd import config


test_data = {
    'existing_path': os.path.abspath('..'),
    'nonexistent_path': 'an_unlikely_name_for_a_directory',
    'existing_file': os.path.abspath(__file__),
    'nonexistent_file': 'an_unlikely_name_for_a_file.dat',
    'file_stem': 'test_%%05d.dat',
    'tempdir': tempfile.gettempdir(),
    'pathsep': os.path.sep,
    }

test_instrument_cfg =r"""beam:
  energy: 80.725
  vector:
    azimuth: 90.0
    polar_angle: 90.0
detectors:
  GE:
    buffer:
    - 2
    - 2
    distortion:
      function_name: GE_41RT
      parameters:
      - -2.0
      - -3.0
      - -4.0
      - 0.0
      - 1.0
      - 2.0
    pixels:
      columns: 2048
      rows: 1024
      size:
      - 0.1
      - 0.2
    saturation_level: 14000.0
    transform:
      tilt:
      - 0.0
      - -0.1
      - 0.2
      translation:
      - -1.
      - -3.
      - -1050
id: instrument
oscillation_stage:
  chi: 0
  translation:
  - 0.0
  - 0.0
  - 0.0
"""


class TestConfig(unittest.TestCase):


    file_name = None


    @classmethod
    def setUpClass(cls):
        logging.disable()
        with tempfile.NamedTemporaryFile(
                delete=False, mode="w+", suffix=".yml") as f:
            test_data["instrument-cfg"] = f.name
        with tempfile.NamedTemporaryFile(delete=False, mode="w+") as f:
            cls.file_name = f.name
            f.file.write(cls.get_reference_data())


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.file_name):
            os.remove(cls.file_name)
        instr_cfg = test_data["instrument-cfg"]
        if os.path.exists(instr_cfg):
            os.remove(instr_cfg)


    def setUp(self):
        self.cfgs = config.open(self.file_name)


    def tearDown(self):
        del(self.cfgs)


    @classmethod
    def get_reference_data(cls):
        raise NotImplementedError
