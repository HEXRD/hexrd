"""Byte-bounded reuse of exact source pixels across changed grain windows.

Keys describe detector frame and integer source coordinates, never a centroid
or a fitted grain. New windows can reuse only pixels already sampled at those
coordinates; every missing pixel still comes from the original image reader.
"""

from collections import OrderedDict

import numpy as np


class PixelCache:
    def __init__(self):
        self.max_bytes = self.bytes = self.peak_bytes = 0
        self.hits = self.queries = 0
        self.frames = OrderedDict()

    def set_limit(self, limit):
        self.max_bytes = max(0, int(limit))
        while self.bytes > self.max_bytes:
            _, (_, _, size) = self.frames.popitem(last=False)
            self.bytes -= size

    def clear(self):
        self.frames.clear()
        self.bytes = 0

    def query(self, reader, frame, rows, cols):
        from hexrd.core.extensions import transforms as native

        make = getattr(native, '_makePixelQuery', None)
        if make is None or any(n > np.iinfo('i8').max for n in reader.shape):
            return None
        key = reader, frame
        old = self.frames.get(key)
        query = make(
            np.asarray(rows).ravel(),
            np.asarray(cols).ravel(),
            *reader.shape,
            reader.dtype,
            old
        )
        if query is not None:
            if old is not None:
                self.frames.move_to_end(key)
            self.queries += query.unique_count
            self.hits += query.unique_count - query.missing_count
        return query

    def keys(self, reader, rows, cols):
        rows, cols = np.asarray(rows).ravel(), np.asarray(cols).ravel()
        # Callers supply clipped source coordinates. Reject anything else
        # rather than allowing packed-key collisions across different rows.
        if (
            np.any(rows < 0)
            or np.any(rows >= reader.shape[0])
            or np.any(cols < 0)
            or np.any(cols >= reader.shape[1])
        ):
            return None
        cells = reader.shape[0] * reader.shape[1]
        if cells > 2**64:
            return None
        dtype = 'u4' if cells <= 2**32 else 'u8'
        return (rows.astype('u8') * reader.shape[1] + cols.astype('u8')).astype(dtype)

    def lookup(self, reader, frame, keys):
        self.queries += len(keys)
        found = np.zeros(len(keys), dtype=bool)
        values = np.empty(len(keys), dtype=reader.dtype)
        item = self.frames.pop((reader, frame), None)
        if item is not None:
            self.frames[reader, frame] = item
            stored_keys, stored_values, _ = item
            indices = np.searchsorted(stored_keys, keys)
            found = indices < len(stored_keys)
            found[found] &= stored_keys[indices[found]] == keys[found]
            values[found] = stored_values[indices[found]]
            self.hits += int(np.count_nonzero(found))
        return found, values

    def remember(self, reader, frame, keys, values):
        if not len(keys) or not self.max_bytes:
            return
        key = reader, frame
        unique, indices = np.unique(keys, return_index=True)
        incoming = values[indices]
        old = self.frames.pop(key, None)
        if old is not None:
            self.bytes -= old[2]
            joined = np.union1d(old[0], unique)
        else:
            joined = unique
        size = 1024 + joined.nbytes + len(joined) * reader.dtype.itemsize
        if size > self.max_bytes:
            if old is not None:
                self.frames[key] = old
                self.bytes += old[2]
            return
        # Keep a useful subset of source frames when a scan exceeds capacity.
        # Cycling every frame through an LRU would evict the entire previous
        # pass just before it could be reused on a large scan.
        if old is None and self.bytes + size > self.max_bytes:
            return
        output = np.empty(len(joined), dtype=reader.dtype)
        if old is not None:
            output[np.searchsorted(joined, old[0])] = old[1]
        output[np.searchsorted(joined, unique)] = incoming
        joined.flags.writeable = output.flags.writeable = False
        while self.bytes + size > self.max_bytes:
            _, (_, _, removed) = self.frames.popitem(last=False)
            self.bytes -= removed
        self.frames[key] = joined, output, size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
