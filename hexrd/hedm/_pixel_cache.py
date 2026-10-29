"""Byte-bounded reuse of exact source pixels across changed grain windows.

Keys describe detector frame and integer source coordinates, never a centroid
or a fitted grain. New windows can reuse only pixels already sampled at those
coordinates; every missing pixel still comes from the original image reader.
"""

from collections import OrderedDict

import numpy as np


class PixelCache:
    def __init__(self, *, compress=True, compression_min_bytes=32 * 1024**2):
        self.max_bytes = self.bytes = self.peak_bytes = 0
        self.hits = self.queries = 0
        self.frames = OrderedDict()
        self.compress = compress
        self.compression_min_bytes = compression_min_bytes
        self.packing = False
        self._codec = None
        self._codec_unavailable = False
        self._decoded_source = self._decoded = None

    def _drop_decoded(self):
        self._decoded_source = self._decoded = None

    def _arrays(self, item):
        if item[1] is not None:
            return item
        if self._decoded_source is not item[0]:
            self._drop_decoded()
            keys, values = item[0].restore()
            self._decoded_source = item[0]
            self._decoded = keys, values, 1024 + keys.nbytes + values.nbytes
        return self._decoded

    def _enable_packing(self):
        if (
            self.packing
            or not self.compress
            or self.max_bytes < self.compression_min_bytes
        ):
            return
        from hexrd.core.extensions import transforms as native
        from hexrd.hedm._chunk_io import _blosc
        from hexrd.hedm._progress import check_cancelled

        kind = getattr(native, '_BloscReader', None)
        if self._codec_unavailable:
            return
        if kind is None:
            self._codec_unavailable = True
            return
        library = _blosc()
        if library is None or not hasattr(library, 'blosc_compress_ctx'):
            self._codec_unavailable = True
            return
        self._codec = kind(library)
        self.packing = True
        # Snapshot keys only: retaining all old tuples would keep the entire
        # raw catalogue alive while constructing its compressed replacement.
        for key in list(self.frames):
            check_cancelled()
            item = self.frames[key]
            if item[1] is not None:
                packed = self._codec.pack_pixels(item[0], item[1])
                if packed is not None and packed.storage_bytes < item[2]:
                    self.frames[key] = packed, None, packed.storage_bytes
                    self.bytes -= item[2] - packed.storage_bytes

    def set_limit(self, limit):
        self._drop_decoded()
        self.max_bytes = max(0, int(limit))
        if self.bytes > self.max_bytes:
            self._enable_packing()
        while self.bytes > self.max_bytes:
            _, (_, _, size) = self.frames.popitem(last=False)
            self.bytes -= size

    def clear(self):
        self._drop_decoded()
        self.frames.clear()
        self.bytes = 0
        self._codec = None
        self.packing = False

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
            stored_keys, stored_values, _ = self._arrays(item)
            indices = np.searchsorted(stored_keys, keys)
            found = indices < len(stored_keys)
            found[found] &= stored_keys[indices[found]] == keys[found]
            values[found] = stored_values[indices[found]]
            self.hits += int(np.count_nonzero(found))
        return found, values

    def remember(self, reader, frame, keys, values, *, sorted_unique=False):
        if not len(keys) or not self.max_bytes:
            return
        key = reader, frame
        if sorted_unique:
            # Native queries and the generic lookup already produce sorted,
            # unique keys. Keep private ownership without sorting them again.
            unique, incoming = keys, values
        else:
            unique, indices = np.unique(keys, return_index=True)
            incoming = values[indices]
        if incoming.dtype != reader.dtype:
            incoming = incoming.astype(reader.dtype)
        if self.bytes + unique.nbytes + incoming.nbytes + 1024 > self.max_bytes:
            self._enable_packing()
        old = self.frames.pop(key, None)
        if old is not None:
            self.bytes -= old[2]
            old_keys, old_values, _ = self._arrays(old)
            from hexrd.core.extensions import transforms as native

            merged = native._mergePixelEntries(old_keys, old_values, unique, incoming)
            if merged is not None:
                joined, output = merged
            else:
                joined = np.union1d(old_keys, unique)
                output = np.empty(len(joined), dtype=reader.dtype)
                output[np.searchsorted(joined, old_keys)] = old_values
                output[np.searchsorted(joined, unique)] = incoming
        else:
            joined, output = (
                (unique.copy(), incoming.copy())
                if sorted_unique
                else (unique, incoming)
            )
        joined.flags.writeable = output.flags.writeable = False
        size = 1024 + joined.nbytes + len(joined) * reader.dtype.itemsize
        item = joined, output, size
        if self.packing:
            packed = self._codec.pack_pixels(joined, output)
            if packed is not None and packed.storage_bytes < size:
                size = packed.storage_bytes
                item = packed, None, size
        self._drop_decoded()
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
        while self.bytes + size > self.max_bytes:
            _, (_, _, removed) = self.frames.popitem(last=False)
            self.bytes -= removed
        self.frames[key] = item
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
