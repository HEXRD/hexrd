import os
import shutil
import tempfile
import logging

from hexrd import config
from hexrd import testing


test_data = {
    'existing_path': os.path.abspath('..'),
    'nonexistent_path': 'an_unlikely_name_for_a_directory',
    'existing_file': os.path.abspath(__file__),
    'nonexistent_file': 'an_unlikely_name_for_a_file.dat',
    'file_stem': 'test_%%05d.dat',
    'tempdir': tempfile.gettempdir(),
    'pathsep': os.path.sep
    }


class TestConfig(testing.TestCase):


    file_name = None


    @classmethod
    def setUpClass(cls):
        logging.disable()
        with tempfile.NamedTemporaryFile(delete=False, mode="w+") as f:
            cls.file_name = f.name
            f.file.write(cls.get_reference_data())


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.file_name):
            os.remove(cls.file_name)


    def setUp(self):
        self.cfgs = config.open(self.file_name)


    def tearDown(self):
        del(self.cfgs)


    @classmethod
    def get_reference_data(cls):
        raise NotImplementedError
