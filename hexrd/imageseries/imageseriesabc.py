"""Abstract Base Class"""
import collections.abc

# Type for extracting regions
RegionType = tuple[tuple[int, int], tuple[int, int]]


class ImageSeriesABC(collections.abc.Sequence):
    pass
