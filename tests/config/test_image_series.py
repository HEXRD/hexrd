import os
import tempfile

from .common import TestConfig, test_data

reference_data = (
    """
image_series:
  format: array
  data:
    - filename: f1
      args: a1
    - filename: f2
      args: a2
"""
    % test_data
)


class TestImageSeries(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_format(self):

        self.assertEqual('array', self.cfgs[0].get('image_series:format'))

    def test_data(self):

        d = self.cfgs[0].get('image_series:data')
        self.assertEqual(len(d), 2)

    def test_data_filename(self):

        d = self.cfgs[0].get('image_series:data')
        self.assertEqual(d[0]['filename'], 'f1')

    def test_data_args(self):

        d = self.cfgs[0].get('image_series:data')
        self.assertEqual(d[1]['args'], 'a2')
