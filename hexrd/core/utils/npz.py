"""Lazy numeric NPZ reads and invalidated, atomic intermediate caches."""
import hashlib
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

try:
    from isal import isal_zlib as zlib
except ImportError:
    import zlib


_NUMERIC_HEADER = re.compile(
    rb"\{'descr': '([^']+)', 'fortran_order': (False|True), "
    rb"'shape': \(((?:[0-9]+(?:, [0-9]+)*,?)?)\),? \}")


def _read_array(raw):
    """Read standard numeric NPY members without literal-evaluating every header.

    Frame caches contain thousands of small, ordinary np.save members. The
    strict common-header fast path still validates lengths and object dtypes;
    unusual or structured headers use NumPy's complete format reader.
    """
    if raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
        start = 10 if raw[6] == 1 else 12
        length = int.from_bytes(raw[8:start], 'little')
        header = raw[start:start + length]
        match = _NUMERIC_HEADER.fullmatch(header.strip()) if length <= 10000 else None
        if match is not None:
            descr, fortran, dimensions = match.groups()
            dtype = np.dtype(descr.decode('ascii'))
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be loaded when allow_pickle=False')
            if dtype.itemsize:
                shape = tuple(int(n) for n in dimensions.split(b',') if n.strip())
                count = math.prod(shape)
                return np.frombuffer(raw, dtype=dtype, count=count,
                                     offset=start + length).copy().reshape(
                                         shape, order='F' if fortran == b'True' else 'C')
    return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)


class _NpzReader:
    """Index the ZIP directory, then decompress only requested numeric members.

    Independent member reads release the GIL during decompression. Mapping
    the compressed archive avoids reading every detector frame for metadata.
    """

    def __init__(self, filename):
        with open(filename, 'rb') as stream:
            with zipfile.ZipFile(stream) as archive:
                self._members = {i.filename: i for i in archive.infolist()}
            self._buf = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

    def __getitem__(self, name):
        info = self._members[name + '.npy']
        name_len, extra_len = struct.unpack_from('<HH', self._buf, info.header_offset + 26)
        start = info.header_offset + 30 + name_len + extra_len
        raw = self._buf[start:start + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(raw, wbits=-15)
        elif info.compress_type != zipfile.ZIP_STORED:
            # Preserve np.load compatibility for less common ZIP codecs.
            with zipfile.ZipFile(io.BytesIO(self._buf)) as archive:
                raw = archive.read(name + '.npy')
        if zlib.crc32(raw) & 0xffffffff != info.CRC:
            raise zipfile.BadZipFile(f'Bad CRC-32 for {name}.npy')
        return _read_array(raw)

    def __contains__(self, name):
        return name + '.npy' in self._members

    def __iter__(self):
        return (name[:-4] for name in self._members if name.endswith('.npy'))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._buf.close()


logger = logging.getLogger(__name__)


def fingerprint(settings, *, arrays=(), files=(), packages=()):
    """Hash inputs and their implementation; unavailable sources disable reuse."""
    try:
        context = (settings, {name: version(name) for name in packages})
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


def load_array_cache(path, fingerprint):
    if fingerprint is None:
        return None
    try:
        with _NpzReader(path) as saved:
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
            save = np.savez_compressed if compressed else np.savez
            save(temporary, fingerprint=fingerprint, **arrays)
            os.replace(temporary, path)
    except OSError:
        logger.debug('Could not save derived data in %s', path, exc_info=True)
