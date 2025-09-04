"""Trivial adapter: just for testing"""

from . import ImageSeriesAdapter


class TrivialAdapter(ImageSeriesAdapter):

    def __init__(self, fname):
        pass

    def __len__(self):
        return 0

    def __getitem__(self):
        return None
