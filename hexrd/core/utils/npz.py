"""Lazy numeric NPZ reads and invalidated, atomic intermediate caches."""
import hashlib
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import mmap
import math
import re
import struct
import zipfile

import numpy as np

version = lru_cache(maxsize=None)(version)

try:
    from isal import isal_zlib as zlib
except ImportError:
    import zlib


_NUMERIC_HEADER = re.compile(
    rb"\{'descr': '([^']+)', 'fortran_order': (False|True), "
    rb"'shape': \(((?:[0-9]+(?:, [0-9]+)*,?)?)\),? \}")


def _read_array(raw, *, copy=True):
    """Read standard numeric NPY members without literal-evaluating every header.

    Frame caches contain thousands of small, ordinary np.save members. The
    strict common-header fast path still validates lengths and object dtypes;
    unusual or structured headers use NumPy's complete format reader.
    """
    if raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
        start = 10 if raw[6] == 1 else 12
        length = int.from_bytes(raw[8:start], 'little')
        header = bytes(raw[start:start + length])
        match = _NUMERIC_HEADER.fullmatch(header.strip()) if length <= 10000 else None
        if match is not None:
            descr, fortran, dimensions = match.groups()
            dtype = np.dtype(descr.decode('ascii'))
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be loaded when allow_pickle=False')
            if dtype.itemsize:
                shape = tuple(int(n) for n in dimensions.split(b',') if n.strip())
                count = math.prod(shape)
                array = np.frombuffer(raw, dtype=dtype, count=count, offset=start + length)
                return (array.copy() if copy else array).reshape(
                    shape, order='F' if fortran == b'True' else 'C')
    return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)


class _NpzReader:
    """Index the ZIP directory, then decompress only requested numeric members.

    Independent member reads release the GIL during decompression. Mapping
    the compressed archive avoids reading every detector frame for metadata.
    """

    def __init__(self, filename, *, metadata_only=False, copy=True):
        self._filename, self._metadata_only = filename, metadata_only
        self._copy = copy
        with open(filename, 'rb') as stream:
            self._buf = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._index()

    def _index(self, metadata_only=None):
        from hexrd.core.extensions import transforms

        if metadata_only is None:
            metadata_only = self._metadata_only
        members = transforms._npzIndex(self._buf, metadata_only)
        if members is None:
            with zipfile.ZipFile(self._filename) as archive:
                members = {i.filename: (i.header_offset, i.compress_size, i.compress_type, i.CRC)
                           for i in archive.infolist()}
            metadata_only = False
        # Native indexing releases the GIL. Publish the complete directory
        # before telling concurrent frame-loader threads that it is ready.
        self._members = members
        self._metadata_only = metadata_only

    def _complete(self, name=None):
        if self._metadata_only and (name is None or name + '.npy' not in self._members):
            self._index(metadata_only=False)

    def __getitem__(self, name):
        self._complete(name)
        offset, size, method, crc = self._members[name + '.npy']
        name_len, extra_len = struct.unpack_from('<HH', self._buf, offset + 26)
        start = offset + 30 + name_len + extra_len
        raw = self._buf[start:start + size] if self._copy else memoryview(self._buf)[start:start + size]
        if method == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(raw, wbits=-15)
        elif method != zipfile.ZIP_STORED:
            # Preserve np.load compatibility for less common ZIP codecs.
            with zipfile.ZipFile(io.BytesIO(self._buf)) as archive:
                raw = archive.read(name + '.npy')
        if zlib.crc32(raw) & 0xffffffff != crc:
            raise zipfile.BadZipFile(f'Bad CRC-32 for {name}.npy')
        return _read_array(raw, copy=self._copy)

    def __contains__(self, name):
        self._complete(name)
        return name + '.npy' in self._members

    def __iter__(self):
        self._complete()
        return (name[:-4] for name in self._members if name.endswith('.npy'))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self._buf.close()
        except BufferError:
            pass  # Array views own the mapping until their final reference is released.


logger = logging.getLogger(__name__)


def savez_compressed(path, **arrays):
    """Write standard deflated NPZ members using the available zlib backend."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                member._compressor = zlib.compressobj(wbits=-15)
                np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)


def fingerprint(settings, *, arrays=(), files=(), packages=()):
    """Hash inputs and their implementation; unavailable sources disable reuse."""
    try:
        context = (settings, {name: np.__version__ if name == 'numpy' else version(name)
                              for name in packages})
        digest = hashlib.sha256(json.dumps(
            context, sort_keys=True, default=lambda a: a.tolist()).encode())
        for array in arrays:
            array = np.ascontiguousarray(array)
            digest.update(str((array.shape, array.dtype.str)).encode())
            if array.size:
                digest.update(memoryview(array).cast('B'))
        for path in files:
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        return digest.hexdigest()
    except (OSError, PackageNotFoundError):
        return None


def load_array_cache(path, fingerprint, *, mmap=False):
    if fingerprint is None:
        return None
    try:
        with _NpzReader(path, copy=not mmap) as saved:
            if str(saved['fingerprint']) == fingerprint:
                return {name: saved[name] for name in saved if name != 'fingerprint'}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
        pass
    return None


def save_array_cache(path, fingerprint, *, compressed=False, **arrays):
    if fingerprint is None:
        return
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / 'arrays.npz'
            save = savez_compressed if compressed else np.savez
            save(temporary, fingerprint=fingerprint, **arrays)
            os.replace(temporary, path)
    except OSError:
        logger.debug('Could not save derived data in %s', path, exc_info=True)
