"""imageseries iterator

For use by adapter classes.
"""

import collections.abc


class ImageSeriesIterator(collections.abc.Iterator):

    def __init__(self, iterable):
        self._iterable = iterable
        self._remaining = list(range(len(iterable)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._iterable[self._remaining.pop(0)]
        except IndexError:
            raise StopIteration
