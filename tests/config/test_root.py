import multiprocessing as mp
import os
import tempfile
from unittest import skipIf

from .common import TestConfig, test_data
from hexrd.hedm import config


reference_data = \
"""
analysis_name: analysis
#working_dir: # not set to test defaulting to cwd
---
analysis_name: analysis_2
working_dir: %(existing_path)s
multiprocessing: -1
---
#analysis_name: # not set to test inheritance
working_dir: %(nonexistent_path)s
multiprocessing: all
---
multiprocessing: half
---
multiprocessing: 2
---
multiprocessing: 1000
---
multiprocessing: -1000
---
multiprocessing: foo
""" % test_data


class TestRootConfig(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_analysis_dir(self):
        self.assertEqual(
            str(self.cfgs[0].analysis_dir),
            os.path.join(os.getcwd(), 'analysis')
            )

    def test_analysis_name(self):
        self.assertEqual(self.cfgs[0].analysis_name, 'analysis')
        self.assertEqual(self.cfgs[1].analysis_name, 'analysis_2')
        self.cfgs[3].analysis_name = 'analysis_3'
        self.assertEqual(self.cfgs[3].analysis_name, 'analysis_3')

    def test_section_inheritance(self):
        self.assertEqual(self.cfgs[0].analysis_name, 'analysis')
        self.assertEqual(self.cfgs[1].analysis_name, 'analysis_2')
        self.assertEqual(self.cfgs[2].analysis_name, 'analysis_2')

    def test_working_dir(self):
        self.assertEqual(str(self.cfgs[0].working_dir), os.getcwd())
        self.assertEqual(
            str(self.cfgs[1].working_dir), test_data['existing_path']
        )
        self.assertRaises(IOError, getattr, self.cfgs[2], 'working_dir')
        self.cfgs[7].working_dir = test_data['existing_path']
        self.assertEqual(
            str(self.cfgs[7].working_dir), test_data['existing_path']
        )
        self.assertRaises(
            IOError, setattr, self.cfgs[7], 'working_dir',
            test_data['nonexistent_path']
            )

    @skipIf(mp.cpu_count() < 2, 'test requires at least two cores')
    def test_multiprocessing(self):
        ncpus = mp.cpu_count()
        self.assertEqual(self.cfgs[0].multiprocessing, ncpus - 1)
        self.assertEqual(self.cfgs[1].multiprocessing, ncpus - 1)
        self.assertEqual(self.cfgs[2].multiprocessing, ncpus)
        self.assertEqual(self.cfgs[3].multiprocessing, ncpus//2)
        self.assertEqual(self.cfgs[4].multiprocessing, 2)
        self.assertEqual(self.cfgs[5].multiprocessing, ncpus)
        self.assertEqual(self.cfgs[6].multiprocessing, 1)
        self.assertEqual(self.cfgs[7].multiprocessing, ncpus-1)
        self.cfgs[7].multiprocessing = 1
        self.assertEqual(self.cfgs[7].multiprocessing, 1)
        self.cfgs[7].multiprocessing = 'all'
        self.assertEqual(self.cfgs[7].multiprocessing, ncpus)
        self.cfgs[7].multiprocessing = 2
        self.assertEqual(self.cfgs[7].multiprocessing, 2)
        self.assertRaises(
            RuntimeError, setattr, self.cfgs[7], 'multiprocessing', 'foo'
            )
        self.assertRaises(
            RuntimeError, setattr, self.cfgs[7], 'multiprocessing', -2
            )



class TestSingleConfig(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return "analysis_name: foo"

    def test_analysis_name(self):
        self.assertEqual(self.cfgs[0].analysis_name, 'foo')

    def test_dirty(self):
        self.assertEqual(self.cfgs[0].dirty, False)
        self.cfgs[0].analysis_name = 'bar'
        self.assertEqual(self.cfgs[0].analysis_name, 'bar')
        self.assertEqual(self.cfgs[0].dirty, True)

    def test_dump(self):
        self.assertEqual(self.cfgs[0].dirty, False)
        self.cfgs[0].analysis_name = 'baz'
        self.assertEqual(self.cfgs[0].dirty, True)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pass
        self.cfgs[0].dump(f.name)
        self.assertEqual(self.cfgs[0].dirty, False)
        cfg = config.open(f.name)[0]
        self.assertEqual(self.cfgs[0].analysis_name, 'baz')
