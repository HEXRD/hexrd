"""Byte-bounded reuse of exact source pixels across changed grain windows.

Keys describe detector frame and integer source coordinates, never a centroid
or a fitted grain. New windows can reuse only pixels already sampled at those
coordinates; every missing pixel still comes from the original image reader.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class _Entry:
    keytype: np.dtype
    valuetype: np.dtype
    count: int
    keys: bytes
    values: bytes

    @property
    def size(self):
        return 1024 + len(self.keys) + len(self.values)


def _codec():
    from hexrd.core.extensions import transforms as native
    from hexrd.hedm._chunk_io import _blosc

    kind = getattr(native, '_BloscReader', None)
    try:
        lib = _blosc() if kind is not None else None
    except (ImportError, OSError):
        return None
    return kind(lib) if lib is not None and hasattr(lib, 'blosc_compress_ctx') else None


class PixelCache:
    def __init__(self, *, compressed=True):
        self.max_bytes = self.bytes = self.peak_bytes = 0
        self.hits = self.queries = 0
        self.frames = OrderedDict()
        self.codec = _codec() if compressed else None
        self._decoded_source = self._decoded = None

    def _encode(self, array):
        return (
            self.codec.encode(np.ascontiguousarray(array))
            if self.codec is not None
            else array.tobytes()
        )

    def _decode(self, blob, dtype, count):
        return (
            self.codec.decode(blob, dtype, count)
            if self.codec is not None
            else np.frombuffer(blob, dtype=dtype, count=count)
        )

    def _arrays(self, entry):
        if self._decoded_source is not entry:
            arrays = (
                self._decode(entry.keys, entry.keytype, entry.count),
                self._decode(entry.values, entry.valuetype, entry.count),
                entry.size,
            )
            self._decoded_source, self._decoded = entry, arrays
        return self._decoded

    def set_limit(self, limit):
        self._decoded_source = self._decoded = None
        self.max_bytes = max(0, int(limit))
        while self.bytes > self.max_bytes:
            self.bytes -= self.frames.popitem(last=False)[1].size

    def clear(self):
        self._decoded_source = self._decoded = None
        self.frames.clear()
        self.bytes = 0

    def query(self, reader, frame, rows, cols):
        from hexrd.core.extensions import transforms as native

        if any(n > np.iinfo('i8').max for n in reader.shape):
            return None
        key = reader, frame
        stored = self.frames.get(key)
        old = self._arrays(stored) if stored is not None else None
        query = native._makePixelQuery(
            np.asarray(rows).ravel(),
            np.asarray(cols).ravel(),
            *reader.shape,
            reader.dtype,
            old,
            False
        )
        if query is not None:
            if old is not None:
                self.frames.move_to_end(key)
            self.queries += query.unique_count
            self.hits += query.unique_count - query.missing_count
        return query

    def remember(self, reader, frame, keys, values):
        if not len(keys) or not self.max_bytes:
            return
        key = reader, frame
        # Entries come from the native query's sorted, unique missing keys.
        unique, incoming = keys, values
        if incoming.dtype != reader.dtype:
            incoming = incoming.astype(reader.dtype)
        old = self.frames.pop(key, None)
        if old is not None:
            self.bytes -= old.size
            old_keys, old_values, _ = self._arrays(old)
            from hexrd.core.extensions import transforms as native

            joined, output = native._mergePixelEntries(
                old_keys, old_values, unique, incoming
            )
        else:
            joined, output = unique, incoming
        keys_blob, values_blob = self._encode(joined), self._encode(output)
        item = (
            _Entry(joined.dtype, output.dtype, len(joined), keys_blob, values_blob)
            if keys_blob is not None and values_blob is not None
            else None
        )
        self._decoded_source = self._decoded = None
        if item is None or item.size > self.max_bytes:
            if old is not None:
                self.frames[key] = old
                self.bytes += old.size
            return
        size = item.size
        # Keep a useful subset of source frames when a scan exceeds capacity.
        # Cycling every frame through an LRU would evict the entire previous
        # pass just before it could be reused on a large scan.
        if old is None and self.bytes + size > self.max_bytes:
            return
        while self.bytes + size > self.max_bytes:
            self.bytes -= self.frames.popitem(last=False)[1].size
        self.frames[key] = item
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
