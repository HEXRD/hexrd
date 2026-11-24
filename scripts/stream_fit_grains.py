#!/usr/bin/env python3
"""Experimental Eiger Stream V1 replay and live HEDM sandbox (Linux only).

No scientific algorithms are forked here. Immutable received prefixes run the
ordinary orientation discovery and fitting functions in one separate process.
"""

import argparse
import binascii
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing
from contextvars import copy_context
from copy import copy, deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

MIB = 1024**2
MAX_MESSAGE = 256 * MIB
ANALYSIS_MIB = 2560
FIT_WORKER_MIB = 896


class AnalysisCancelled(Exception):
    pass


def allocation_limit(mebibytes):
    """Guard allocations before imports; all live roles total at most 7.75 GiB."""
    if sys.platform != 'linux':
        raise RuntimeError('This sandbox requires Linux RLIMIT_AS memory guards')
    import resource

    size = mebibytes * MIB
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        size = min(size, hard)
    # Leave the receiver permission to exec its single 4 GiB analysis child.
    ceiling = 4096 * MIB if hard == resource.RLIM_INFINITY else min(4096 * MIB, hard)
    resource.setrlimit(resource.RLIMIT_AS, (size, ceiling))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'


def json_bytes(value):
    return json.dumps(value, allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def read_wire(root, index):
    with zipfile.ZipFile(Path(root) / 'frames' / f'{index:06d}.zip') as saved:
        return [saved.read(str(i)) for i in range(len(saved.infolist()))]


def load_config(filename):
    """Resolve static calibration paths without opening any image or estimate file."""
    import yaml

    filename = Path(filename).resolve()
    with filename.open() as stream:
        config, *studies = yaml.safe_load_all(stream)
    if studies:
        raise ValueError('The sandbox accepts a single-study YAML')
    root = Path(config.get('working_dir', filename.parent)).resolve()
    config['working_dir'] = str(root)
    config['instrument'] = str(root / config['instrument'])
    config['material']['definitions'] = str(root / config['material']['definitions'])
    return config


def encode_image(frame, series, index, omega, frame_time, source_scale=1.):
    """Lossless conversion only: Eiger V1 supports unsigned integer counts."""
    import numpy as np
    import lz4.block
    from scipy import sparse

    values = frame.data if sparse.issparse(frame) else np.asarray(frame)
    if source_scale not in (1., .5) or (source_scale != 1. and values.dtype.kind != 'f'):
        raise ValueError('Replay scaling supports only exact floating half counts')
    counts = values if source_scale == 1. else values / source_scale
    if (not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or (values.dtype.kind == 'f' and np.any(np.signbit(values)))
            or np.any(counts > np.iinfo('uint32').max)
            or not np.array_equal(counts, counts.astype('uint32'))):
        raise ValueError('Eiger replay requires exact uint32 counts; refusing to '
                         'round, clip, or threshold processed input pixels')
    if (sparse.issparse(frame) and frame.format == 'csr' and frame.has_canonical_format):
        dense = sparse.csr_array((counts.astype('<u4'), frame.indices, frame.indptr),
                                 shape=frame.shape).toarray()
    else:
        dense = frame.toarray() if sparse.issparse(frame) else frame
        if source_scale != 1.:
            dense = dense / source_scale
    dense = np.ascontiguousarray(dense, dtype='<u4')
    if dense.nbytes > MAX_MESSAGE:
        raise ValueError('Frame exceeds the sandbox 256 MiB message limit')
    payload = lz4.block.compress(dense, store_size=False)
    return [
        json_bytes(dict(htype='dimage-1.0', series=series, frame=index,
                        hash=hashlib.md5(payload).hexdigest())),
        json_bytes(dict(htype='dimage_d-1.0', shape=list(dense.shape[::-1]),
                        type='uint32', encoding='lz4<', size=len(payload))),
        payload,
        json_bytes(dict(htype='dconfig-1.0', start_time=round(index*frame_time*1e9),
                        stop_time=round((index+1)*frame_time*1e9),
                        real_time=round(frame_time*1e9))),
        json_bytes(dict(omega=list(map(float, omega)), source_dtype=frame.dtype.str,
                        source_scale=source_scale)),
    ]


class DecodeBuffer:
    """Thread-owned decode destination; public codec fallbacks preserve portability."""

    def __init__(self):
        import ctypes
        import lz4.block._block
        import dectris.compression

        self.data = None
        self.decode = None
        self.bitshuffle = None
        try:
            self.library = ctypes.CDLL(lz4.block._block.__file__)
            self.decode = self.library.LZ4_decompress_safe
            self.decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int]
            self.decode.restype = ctypes.c_int
        except (OSError, AttributeError):
            pass
        try:
            self.dectris = ctypes.CDLL(dectris.compression.__file__)
            self.bitshuffle = self.dectris.compression_decompress_buffer
            self.bitshuffle.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            self.bitshuffle.restype = ctypes.c_size_t
        except (OSError, AttributeError):
            pass


def verify_image_checksum(parts):
    digest = json.loads(parts[0]).get('hash')
    if digest and hashlib.md5(parts[2]).hexdigest() != digest:
        raise ValueError('Eiger image checksum mismatch')


def image_description(parts, *, verify=True):
    import numpy as np

    if len(parts) not in (4, 5):
        raise ValueError('Expected four Eiger image parts and an optional appendix')
    header, spec = map(json.loads, parts[:2])
    if spec['htype'] != 'dimage_d-1.0' or spec['type'] not in ('uint8', 'uint16', 'uint32'):
        raise ValueError('Unsupported Eiger image description')
    shape = spec['shape'][::-1]
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Expected a two-dimensional detector frame')
    encoding = spec['encoding']
    if encoding[-1:] not in ('<', '>'):
        raise ValueError('Missing Eiger byte order')
    dtype = np.dtype(spec['type']).newbyteorder(encoding[-1])
    size = math.prod(shape) * dtype.itemsize
    if size > MAX_MESSAGE or spec['size'] != len(parts[2]):
        raise ValueError('Invalid or oversized Eiger payload')
    if verify:
        verify_image_checksum(parts)
    codec = encoding[:-1]
    bitshuffle = codec == f'bs{dtype.itemsize*8}-lz4'
    if bitshuffle and (len(parts[2]) < 12 or int.from_bytes(parts[2][:8], 'big') != size):
        raise ValueError('Invalid bitshuffle frame size')
    return shape, dtype, size, codec, bitshuffle


def decode_image(parts, buffer=None):
    """Decode one verified frame; a buffer is borrowed until its next decode."""
    return decode_pixels(parts, image_description(parts), buffer)


def decode_pixels(parts, description, buffer=None):
    import numpy as np
    import lz4.block
    from dectris.compression import decompress

    shape, dtype, size, codec, bitshuffle = description
    reuse = (buffer is not None and ((codec == 'lz4' and buffer.decode is not None)
                                    or (bitshuffle and buffer.bitshuffle is not None)))
    if reuse:
        if buffer.data is None or buffer.data.nbytes != size:
            buffer.data = np.empty(size, dtype='u1')
        source = np.frombuffer(parts[2], dtype='u1')
        decoded = (buffer.bitshuffle(0, buffer.data.ctypes.data, size, source.ctypes.data,
                                    len(parts[2]), dtype.itemsize) if bitshuffle else
                   buffer.decode(source.ctypes.data, buffer.data.ctypes.data, len(parts[2]), size))
        if decoded != size:
            raise ValueError('Invalid compressed payload or decoded frame size')
        raw = buffer.data
    elif codec == 'lz4':
        raw = lz4.block.decompress(parts[2], uncompressed_size=size)
    elif codec == '':
        raw = parts[2]
    elif bitshuffle:
        # DECTRIS's established decoder handles the bitshuffle block headers.
        raw = decompress(bytes(parts[2]), 'bslz4', elem_size=dtype.itemsize)
    else:
        raise ValueError(f'Unsupported Eiger codec: {codec}')
    if len(raw) != size:
        raise ValueError('Decoded frame size does not match its description')
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def restore_counts(values, dtype, scale=1.):
    """Restore emulator metadata without silently narrowing detector counts."""
    import numpy as np

    if (dtype.kind not in 'uif' or dtype.itemsize > 8 or scale not in (1., .5)
            or (scale != 1. and dtype.kind != 'f')):
        raise ValueError('Unsupported source dtype or scale')
    restored = values.astype(dtype, copy=False)
    if not np.can_cast(values.dtype, dtype, casting='safe') and not np.array_equal(values, restored):
        raise ValueError('Source dtype cannot retain received pixel values')
    if scale != 1.:
        restored *= scale
    return restored


class Capture:
    """Append-only sparse pixel support; published snapshots never change."""

    def __init__(self, directory, *, wire_spool=False, fit_threshold=None, raw_archive=None,
                 defer_checksums=False):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'frames').mkdir()
        (self.root / 'snapshots').mkdir()
        self.header = None
        self.omega = []
        self.complete = False
        self.map_plan = self.maps = None
        self._decoders = threading.local()
        self.raw_archive = Path(raw_archive).resolve() if raw_archive is not None else None
        self.raw_checksums = []
        self.wire_spool = wire_spool or self.raw_archive is not None
        if fit_threshold is not None and (not math.isfinite(fit_threshold) or fit_threshold < 0):
            raise ValueError('Fitting support requires a finite nonnegative threshold')
        self.fit_threshold, self.fit_records = fit_threshold, []
        self._support = ((self.root / 'fit-support.bin').open('xb')
                         if fit_threshold is not None else None)
        self._support_lock = threading.Lock()
        self.block_workers = 1
        self.defer_checksums = defer_checksums
        self._hash_pool = ThreadPoolExecutor(2) if defer_checksums else None
        self._checksums = deque()
        self._hash_submitted = self._hash_verified = 0
        self._checksum_error = None

    def _finish_checksum(self, future):
        try:
            future.result()
        except BaseException as error:
            self._checksum_error = error
            raise
        self._hash_verified += 1

    def queue_checksum(self, parts):
        if not self.defer_checksums:
            return
        self.check_checksums()
        if len(self._checksums) >= 4:
            self._finish_checksum(self._checksums.popleft())
        self._checksums.append(self._hash_pool.submit(verify_image_checksum, parts))
        self._hash_submitted += 1

    def check_checksums(self, *, wait=False):
        if self._checksum_error is not None:
            raise self._checksum_error
        while self._checksums and (wait or self._checksums[0].done()):
            self._finish_checksum(self._checksums.popleft())

    @property
    def checksum_status(self):
        if self._checksum_error is not None:
            return 'failed'
        return ('pending' if self.defer_checksums and self._hash_verified < len(self.omega)
                else 'verified')

    def close(self):
        if self._hash_pool is not None:
            self._hash_pool.shutdown(wait=True, cancel_futures=True)
        if self._support is not None:
            self._support.close()
        self._decoders = threading.local()

    def warm_decoder(self):
        from hexrd.core.extensions import transforms as native

        if not hasattr(self._decoders, 'buffer'):
            self._decoders.buffer = DecodeBuffer()
        decoder = self._decoders.buffer
        if (self._support is not None and self.wire_spool and decoder.decode is not None
                and hasattr(native, '_EigerSupport') and not hasattr(decoder, 'support')):
            decoder.support = native._EigerSupport(decoder.library, self.block_workers)

    def native_support(self, parts, description, dtype, scale):
        """Retain source pixels; use exact sparse map sums only in their safe range."""
        import numpy as np
        from hexrd.core.extensions import transforms as native

        shape, wire_dtype, _, codec, _ = description
        decoder = self._decoders.buffer
        if (self._support is None or not self.wire_spool or codec != 'bs32-lz4'
                or wire_dtype != np.dtype('u4') or sys.byteorder != 'little'
                or not ((dtype == np.dtype('f8') and scale in (1., .5))
                        or (dtype == np.dtype('u4') and scale == 1.))
                or decoder.decode is None or not hasattr(native, '_EigerSupport')):
            return None
        cut = min(self.fit_threshold/scale, float(np.iinfo('u4').max))
        map_cut = cut
        if self.maps is not None:
            threshold = float(self.map_plan.threshold)
            if not math.isfinite(threshold):
                return None
            map_cut = min(cut, max(0, math.ceil(threshold/scale)-1))
        values, pixels = decoder.support.support(np.frombuffer(parts[2], dtype='u1'), map_cut)
        histograms = {}
        if self.maps is not None:
            for panel, (plan, coverage) in self.map_plan.panels.items():
                histogram = plan.accumulate_sparse(pixels, values, threshold, scale)
                if histogram is None:
                    return None
                histogram[~coverage] = np.nan
                histograms[panel] = histogram
        if map_cut < cut:
            keep = values > cut
            values, pixels = values[keep], pixels[keep]
        offsets = np.r_[0, np.cumsum(np.bincount(pixels//shape[1], minlength=shape[0]))].astype('i4')
        return (values, (pixels % shape[1]).astype('i4'), offsets), histograms, cut

    def prepare_image(self, parts):
        """Independent bounded image work; no acquisition state is mutated here."""
        import numpy as np
        from scipy import sparse
        from hexrd.core.experiment import _as_csr

        self.warm_decoder()
        description = image_description(parts, verify=not self.defer_checksums)
        shape, wire_dtype, _, _, _ = description
        if tuple(shape) != self.shape:
            raise ValueError('Detector shape changed within a series')
        appendix = json.loads(parts[4]) if len(parts) == 5 else {}
        dtype = np.dtype(appendix.get('source_dtype', wire_dtype.str))
        scale = appendix.get('source_scale', 1.)
        compact = self.native_support(parts, description, dtype, scale)
        frame = None
        if compact is None:
            frame = decode_pixels(parts, description, self._decoders.buffer)
            integer_range = (dtype.kind in 'iu' and scale == 1.
                             and frame.max(initial=0) <= np.iinfo(dtype).max)
            if not (integer_range or (dtype == np.dtype('f8') and scale in (1., .5))):
                restore_counts(frame, dtype, scale)
        start, step = self.config.get('omega_start'), self.config.get('omega_increment')
        index = json.loads(parts[0])['frame']
        omega = appendix.get('omega')
        if omega is None and start is not None and step is not None:
            omega = [start + index*step, start + (index+1)*step]
        if (omega is None or np.shape(omega) != (2,) or not np.isfinite(omega).all()
                or omega[1] <= omega[0]):
            raise ValueError('Need measured omega edges or calibrated positive omega increments')
        histograms = (compact[1] if compact is not None else
                      {panel: self.map_plan.histogram(panel, frame, source_dtype=dtype,
                                                     source_scale=scale)
                       for panel in self.map_plan.panels} if self.maps is not None else {})
        # Keep every nonzero, including saturated pixels, for exact later windows.
        path = self.root / 'frames' / f'{index:06d}{".zip" if self.wire_spool else ".npz"}'
        temporary = path.with_suffix('.tmp')
        raw_checksums = None
        if self.raw_archive is not None:
            # Link only an arrived frame, validating all headers and the payload CRC.
            raw_checksums = [binascii.crc32(part) for part in parts]
            with zipfile.ZipFile(self.raw_archive / 'frames' / f'{index:06d}.zip') as archive:
                if len(archive.infolist()) != len(parts):
                    raise ValueError('Raw archive does not match the received frame')
                for i, part in enumerate(parts):
                    if (archive.getinfo(str(i)).CRC != raw_checksums[i]
                            or (i != 2 and archive.read(str(i)) != part)):
                        raise ValueError('Raw archive does not match the received frame')
        elif self.wire_spool:
            # Stored members retain the original codec and avoid array-copy buffers.
            with zipfile.ZipFile(temporary, 'w') as archive:
                for i, part in enumerate(parts):
                    archive.writestr(str(i), part)
        else:
            with temporary.open('wb') as stream:
                pixels = _as_csr(frame)
                pixels.data = restore_counts(pixels.data, dtype, scale)
                sparse.save_npz(stream, pixels)
        if self.raw_archive is None:
            temporary.replace(path)
        record = None
        if self._support is not None:
            from hexrd.core.extensions import transforms as native
            threshold = np.asarray(self.fit_threshold, dtype=np.result_type(
                np.empty(0, dtype=dtype), self.fit_threshold)).item()
            cut = min(threshold / scale, float(np.iinfo(wire_dtype).max))
            if compact is None:
                aligned = np.require(frame, dtype=wire_dtype.newbyteorder('='), requirements=['C', 'A'])
                data, columns, offsets = native._denseToCSR(aligned, cut)
            else:
                (data, columns, offsets), _, cut = compact
            if dtype == np.dtype('f8') and data.max(initial=0) <= 2**24 and scale in (1., .5):
                # Every wire count in this range and its half are exact in float32.
                data = data.astype('f4')
                if scale != 1.:
                    data *= scale
            else:
                data = restore_counts(data, dtype, scale)
                if dtype == np.dtype('f8'):
                    narrow = data.astype('f4')
                    if np.array_equal(narrow, data):
                        data = narrow
            if shape[1] <= 65536:
                columns = columns.astype('u2')
            records, digest = [], hashlib.sha256()
            with self._support_lock:
                for array in (data, columns, offsets):
                    padding = -self._support.tell() % 64
                    self._support.write(b'\0' * padding)
                    records.append((self._support.tell(), array.dtype.str, array.size))
                    raw = memoryview(array).cast('B')
                    digest.update(raw)
                    self._support.write(raw)
                self._support.flush()
            record = dict(arrays=records, sha256=digest.hexdigest(), threshold=cut * scale)
        return dtype, list(map(float, omega)), histograms, record, raw_checksums

    def accept(self, parts, prepared=None):
        if sum(map(len, parts)) > MAX_MESSAGE:
            raise ValueError('Oversized multipart message')
        message = json.loads(parts[0])
        kind = message['htype']
        if self.complete:
            raise ValueError('Message after end of series')
        if kind == 'dheader-1.0':
            if self.header is not None or len(parts) < 2:
                raise ValueError('Expected one basic/all Eiger series header')
            config = json.loads(parts[1])
            self.expected = int(config['nimages']) * int(config.get('ntrigger', 1))
            if self.expected < 1:
                raise ValueError('Empty acquisition')
            self.header, self.config = message, config
            self.shape = (int(config['y_pixels_in_detector']), int(config['x_pixels_in_detector']))
            self.dtype = None
            if self.map_plan is not None:
                from hexrd.hedm.find_orientations import EtaOmegaMapAccumulator
                self.maps = EtaOmegaMapAccumulator(
                    self.map_plan, self.expected, filename=self.root / 'maps.npy')
            atomic_json(self.root / 'header.json', dict(header=message, config=config))
        elif self.header is None or message['series'] != self.header['series']:
            raise ValueError('Missing header or inconsistent series ID')
        elif kind == 'dimage-1.0':
            if message['frame'] != len(self.omega) or len(self.omega) >= self.expected:
                raise ValueError('Missing, duplicate, out-of-order, or excess frame')
            if self.defer_checksums and message['frame'] >= self._hash_submitted:
                self.queue_checksum(parts)
            self.check_checksums()
            dtype, omega, histograms, record, raw_checksums = (
                self.prepare_image(parts) if prepared is None else prepared)
            if self.dtype is not None and dtype != self.dtype:
                raise ValueError('Unsupported or changing source dtype')
            if self.maps is not None:
                self.maps.append_histograms(histograms, omega)
            self.dtype = dtype
            self.omega.append(omega)
            if self.raw_archive is not None:
                self.raw_checksums.append(raw_checksums)
            if record is not None:
                self.fit_records.append(record)
        elif kind == 'dseries_end-1.0':
            if len(self.omega) != self.expected:
                raise ValueError(f'Incomplete acquisition: '
                                 f'{len(self.omega)}/{self.expected} frames')
            self.check_checksums(wait=True)
            self.complete = True
        else:
            raise ValueError(f'Unexpected Eiger message: {kind}')
        return kind

    def snapshot(self):
        self.check_checksums()
        suffix = '-final' if self.complete else ''
        path = self.root / 'snapshots' / f'{len(self.omega):06d}{suffix}.json'
        if path.exists():
            raise FileExistsError(path)
        metadata = dict(root=str(self.root), count=len(self.omega),
                        expected=self.expected, complete=self.complete,
                        shape=self.shape, dtype=self.dtype.str, omega=self.omega)
        metadata['wire_spool'] = self.wire_spool
        metadata['checksum_status'] = self.checksum_status
        if self.raw_archive is not None:
            metadata.update(raw_root=str(self.raw_archive), raw_checksums=self.raw_checksums)
        if self.fit_threshold is not None:
            metadata.update(fit_threshold=self.fit_records[0]['threshold'], fit_records=self.fit_records)
        if self.maps is not None:
            import numpy as np
            snapshot = self.maps.snapshot()
            if np.shares_memory(snapshot.ring_maps, self.maps._data):
                names = ('eta_edges', 'omega_edges', 'omegas', 'omega_period',
                         'two_theta_ranges', 'ring_ids', 'eta_step')
                metadata['map_axes'] = {name: np.asarray(getattr(snapshot, name)).tolist()
                                        for name in names}
                metadata['map_file'] = str(self.root / 'maps.npy')
        atomic_json(path, metadata)
        return path


def register_spool_adapter():
    import numpy as np
    from scipy.sparse import load_npz
    from hexrd.core.imageseries.load import ImageSeriesAdapter

    class SpoolAdapter(ImageSeriesAdapter):
        format = 'eiger-live-spool'

        def __init__(self, filename, *, fit_threshold=None, **kwargs):
            self.snapshot = json.loads(Path(filename).read_text())
            self.dtype = np.dtype(self.snapshot['dtype'])
            stored = self.snapshot.get('fit_threshold')
            if fit_threshold is not None:
                fit_threshold = np.asarray(fit_threshold, dtype=np.result_type(
                    np.empty(0, dtype=self.dtype), fit_threshold)).item()
            self.support = (fit_threshold is not None and stored is not None
                            and math.isfinite(fit_threshold) and fit_threshold >= stored)

        @property
        def shape(self):
            return tuple(self.snapshot['shape'])

        @property
        def metadata(self):
            return {'omega': np.asarray(self.snapshot['omega'])}

        def __len__(self):
            return self.snapshot['count']

        def _support_arrays(self, index, mapping=None):
            record = self.snapshot['fit_records'][index]
            arrays, digest = [], hashlib.sha256()
            if mapping is None:
                with (Path(self.snapshot['root']) / 'fit-support.bin').open('rb') as source:
                    for offset, dtype, count in record['arrays']:
                        source.seek(offset)
                        arrays.append(np.fromfile(source, dtype=dtype, count=count))
            else:
                arrays = [np.ndarray((n,), dtype=dtype, buffer=mapping, offset=offset)
                          for offset, dtype, n in record['arrays']]
            for array in arrays:
                digest.update(memoryview(array).cast('B'))
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Fitting support checksum mismatch')
            return arrays

        def native_view(self):
            """Eager fitting can share the immutable bank without inflating frames."""
            from types import SimpleNamespace
            from hexrd.core.extensions import transforms as native

            if not self.support or self.dtype not in tuple(map(np.dtype, ('u1','i2','u2','i4','u4','f4','f8'))):
                return None
            extent = max(off + n*np.dtype(dtype).itemsize
                         for record in self.snapshot['fit_records']
                         for off, dtype, n in record['arrays'])
            mapping = np.memmap(Path(self.snapshot['root']) / 'fit-support.bin',
                                mode='r', dtype='u1', shape=(extent,))
            frames = []
            for index in range(len(self)):
                data, columns, offsets = self._support_arrays(index, mapping)
                frames.append(SimpleNamespace(format='csr', has_sorted_indices=True,
                    shape=self.shape, data=data, indices=columns, indptr=offsets))
            return native._SparseFrames(frames, dtype=self.dtype)

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            if self.support:
                from scipy.sparse import csr_array
                arrays = self._support_arrays(index)
                return csr_array((arrays[0].astype(self.dtype, copy=False),
                                  arrays[1].astype('i4', copy=False), arrays[2]), shape=self.shape)
            if self.snapshot.get('wire_spool', False):
                from hexrd.core.experiment import _as_csr
                parts = read_wire(self.snapshot.get('raw_root', self.snapshot['root']), index)
                if ('raw_checksums' in self.snapshot and
                        [binascii.crc32(part) for part in parts] != self.snapshot['raw_checksums'][index]):
                    raise ValueError('Raw archive changed after frame commitment')
                appendix = json.loads(parts[4]) if len(parts) == 5 else {}
                pixels = _as_csr(decode_image(parts))
                pixels.data = restore_counts(pixels.data, self.dtype,
                                             appendix.get('source_scale', 1.))
                return pixels
            return load_npz(Path(self.snapshot['root']) / 'frames' / f'{index:06d}.npz')


class AnalysisSession:
    """Own static crystallography, bounded fitting processes and one archive writer."""

    def __init__(self, config_file, root, workers=2, discovery_workers=4, *, static=False):
        import multiprocessing as mp
        import yaml
        from hexrd.hedm.experiment import HedmExperiment

        self.started = time.monotonic()
        if workers not in (1, 2) or not 1 <= discovery_workers <= 8:
            raise ValueError('analysis worker counts exceed the sandbox allocation budget')
        self.config = load_config(config_file)
        self.workers, self.discovery_workers = workers, discovery_workers
        if static:
            filename = Path(config_file)
        else:
            config = deepcopy(self.config)
            config['image_series']['data'] = []  # Static setup never reads future images.
            config.update(multiprocessing=discovery_workers,
                          analysis_name=str(Path(root).resolve() / 'static'))
            filename = Path(root) / 'static-config.yml'
            filename.write_text(yaml.safe_dump(config))
        self.base = HedmExperiment(str(filename), new_file_placement=True)
        self.base.max_workers = discovery_workers
        self.material = self.base.get_active_material()
        self.base.instrument  # Retain one immutable calibration per process.
        register_spool_adapter()
        self.cancelled = None
        self.pool = self.writer = None
        if not static:
            self.context = mp.get_context('spawn')
            self.cancelled = self.context.Event()
            self.filename = str(filename.resolve())
            self.pool_workers = 0
            self.writer = ThreadPoolExecutor(1)
        self.setup_seconds = time.monotonic() - self.started

    def experiment(self, config):
        """New immutable source/output view sharing only static session assets."""
        exp = copy(self.base)
        exp.config = config
        exp.analysis_name = config['analysis_name']
        exp.analysis_dir = exp._path(exp.analysis_name)
        exp._configure_hedm()
        exp._load_sources()
        return exp

    def check(self):
        if self.cancelled.is_set():
            raise AnalysisCancelled()

    def fit(self, experiment, initial, *, provisional=False):
        import numpy as np
        from hexrd.hedm._progress import check_cancelled

        sources = experiment.config['image_series']
        large_fch5 = (sources.get('format', 'frame-cache') == 'frame-cache'
            and any(str(d.get('args', {}).get('style')).lower() == 'fch5' for d in sources['data'])
            and sum(len(s) * int(np.prod(s.shape)) * np.dtype(s.dtype).itemsize
                    for s in experiment.image_series_list) > 512 * MIB)
        grouped = large_fch5 or experiment.config.get('fit_grains', {}).get('measurement_mode') == 'staged'
        workers = 1 if grouped else min(self.workers, len(initial))
        if self.pool is None or self.pool_workers != workers:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
            self.pool = ProcessPoolExecutor(
                workers, mp_context=self.context, initializer=init_fit_worker,
                initargs=(self.filename, self.cancelled, 2 * FIT_WORKER_MIB // workers))
            self.pool_workers = workers
        # Preserve the 512 MiB grouped path for staged and large FCH5 work.
        groups = (len(initial) + 127) // 128 if grouped else min(len(initial), workers * 4)
        batches = iter(np.array_split(initial, groups))
        pending, results, pids = set(), {}, set()
        logging.info('Fitting %d grains with %d persistent workers (%d MiB each)',
                     len(initial), workers, 2 * FIT_WORKER_MIB // workers)
        try:
            while True:
                while len(pending) < workers:
                    batch = next(batches, None)
                    if batch is None:
                        break
                    pending.add(self.pool.submit(fit_job, experiment.config, batch, provisional))
                if not pending:
                    break
                check_cancelled()
                done, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    pid, fitted = future.result()
                    pids.add(pid)
                    results.update((row[0], row) for row in fitted)
                    logging.info('Fitted %d/%d grains', len(results), len(initial))
            return [results[int(row[0])] for row in initial], sorted(pids)
        except BaseException:
            self.cancelled.set()
            for future in pending:
                future.cancel()
            wait(pending)  # Drain old work before a new generation clears cancellation.
            raise

    def run(self, output, snapshot=None):
        import gc
        import numpy as np
        import yaml
        import resource
        from hexrd.hedm.find_orientations import (
            find_orientations, write_results, write_orientation_archive, EtaOmegaMaps)
        from hexrd.hedm.cli.fit_grains import write_results as write_fits
        from hexrd.hedm._progress import _cancellation_check, check_cancelled

        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        config = deepcopy(self.config)
        if snapshot is not None:
            panels = [entry['panel'] for entry in config['image_series']['data']]
            if len(panels) != 1 or not isinstance(panels[0], str):
                raise ValueError('The sandbox supports one physical detector panel')
            config['image_series'] = dict(format='eiger-live-spool', data=[
                dict(panel=panels[0], file=str(Path(snapshot).resolve()), args={})])
        else:
            for entry in config['image_series']['data']:
                entry['file'] = str(Path(config['working_dir']) / entry['file'])
        config.update(analysis_name=str(output), multiprocessing=self.discovery_workers)
        config['fit_grains'].update(estimate=None,
            image_cache=dict(mode='lazy', max_size_mb=128, read_workers=1))
        config['find_orientations']['orientation_maps']['file'] = None
        (output / 'config.yml').write_text(yaml.safe_dump(config))
        experiment = self.experiment(config)
        token = _cancellation_check.set(self.check)
        archive = None
        started = time.monotonic()
        try:
            maps = None
            metadata = json.loads(Path(snapshot).read_text()) if snapshot else {}
            if 'map_file' in metadata:
                axes = {key: np.asarray(value, dtype='i8' if key == 'ring_ids' else 'f8')
                        for key, value in metadata['map_axes'].items() if key != 'eta_step'}
                maps = EtaOmegaMaps(
                    np.load(metadata['map_file'], mmap_mode='r')[:, :metadata['count']],
                    **axes, eta_step=metadata['map_axes']['eta_step'])
            logging.info('Analyzing %s in persistent session %d', output.name, os.getpid())
            orientations = find_orientations(experiment, self.material, clean=True, maps=maps)
            check_cancelled()
            del maps
            gc.collect()
            discovered = time.monotonic()
            # Preserve the canonical text round trip used to initialize master fits.
            write_results(orientations, experiment, write_archive=False)
            archive = self.writer.submit(copy_context().run, write_orientation_archive,
                                         orientations, experiment)
            grains, pids = [], []
            fit_started = time.monotonic()
            if orientations.num_grains:
                initial = np.loadtxt(output / 'grains.out', ndmin=2)
                grains, pids = self.fit(experiment, initial,
                                        provisional=bool(snapshot and not metadata['complete']))
            fit_done = time.monotonic()
            check_cancelled()
            archive.result()  # Writer errors prevent final completion.
            archive_done = time.monotonic()
            write_fits(grains, experiment)
            np.savez(output / 'scientific.npz', orientations=orientations.grain_orientations,
                     trials=orientations.test_orientations, completeness=orientations.completeness,
                     fits=np.array([[*g[:3], *g[3]] for g in grains]).reshape(-1, 15))
            result = dict(frames=len(experiment.image_series_list[0]), grains=len(grains),
                awaiting_reflections=sum(int(np.isnan(g[1])) for g in grains),
                elapsed_seconds=time.monotonic()-started, session_setup_seconds=self.setup_seconds,
                discovery_seconds=discovered-started, fit_seconds=fit_done-fit_started,
                archive_wait_seconds=archive_done-fit_done, session_pid=os.getpid(), fit_worker_pids=pids,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                complete=snapshot is None or metadata['complete'],
                checksum_status=metadata.get('checksum_status', 'verified'))
            check_cancelled()
            atomic_json(output / 'result.json', result)
            logging.info('Analysis complete: %s', result)
            return result
        except BaseException:
            self.cancelled.set()
            if archive is not None:
                try:
                    archive.result()
                except BaseException:
                    pass
            raise
        finally:
            _cancellation_check.reset(token)

    def close(self):
        self.cancelled.set()
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.writer.shutdown(wait=True, cancel_futures=True)


def init_fit_worker(filename, cancelled, allowance):
    global fit_session, fit_cancelled
    allocation_limit(allowance)
    fit_session = AnalysisSession(filename, Path(filename).parent, 1, 1, static=True)
    fit_session.cache_mb = 512 if allowance >= 2 * FIT_WORKER_MIB else 128
    fit_session.read_workers = fit_session.base.fit_grains.image_cache.read_workers if fit_session.cache_mb == 512 else 1
    fit_cancelled = cancelled
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('Persistent fit worker %d ready (%d MiB)', os.getpid(), allowance)


def fit_job(config, table, provisional=False):
    import numpy as np
    import shutil
    from hexrd.hedm.fitgrains import fit_grains
    from hexrd.hedm._progress import _cancellation_check

    def check():
        if fit_cancelled.is_set():
            raise AnalysisCancelled()
    token = _cancellation_check.set(check)
    ids = table[:, 0].astype(int)
    root = Path(config['analysis_name'])
    directory = root / f'.fit-{int(ids[0])}'
    local = deepcopy(config)
    local.update(analysis_name=str(directory), multiprocessing=1)
    local['fit_grains']['image_cache'] = dict(mode='lazy', max_size_mb=fit_session.cache_mb,
                                            read_workers=fit_session.read_workers)
    if local['image_series'].get('format') == 'eiger-live-spool':
        for source in local['image_series']['data']:
            source.setdefault('args', {})['fit_threshold'] = local['fit_grains']['threshold']
    experiment = fit_session.experiment(local)
    experiment.max_workers = 1
    table[:, 0] = np.arange(len(table))
    try:
        check()
        results = fit_grains(experiment, table, material=fit_session.material,
                             write_spots_files=True, provisional=provisional)
        for panel in experiment.instrument.detectors:
            target = root / panel
            target.mkdir(exist_ok=True)
            for index, grain in enumerate(ids):
                source = directory / panel / f'spots_{index:05d}.out'
                if source.exists():
                    source.replace(target / f'spots_{grain:05d}.out')
        return os.getpid(), [(int(ids[row[0]]), *row[1:]) for row in results]
    finally:
        _cancellation_check.reset(token)
        shutil.rmtree(directory, ignore_errors=True)


def analyze(config_file, output, snapshot=None, workers=2, discovery_workers=4):
    root = Path(output).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # Keep session setup outside scientific output, which must start absent.
    session_root = root.with_name(root.name + '-session')
    session_root.mkdir(exist_ok=False)
    session = AnalysisSession(config_file, session_root, workers, discovery_workers)
    try:
        return session.run(output, snapshot)
    finally:
        session.close()


def serve_analysis(args):
    """Coalesce requests in one slot and cancel stale work without killing the pool."""
    condition = threading.Condition()
    pending, closed = None, False
    session = AnalysisSession(args.config, args.output, args.analysis_workers,
                              args.discovery_workers)

    def requests():
        nonlocal pending, closed
        for line in sys.stdin:
            job = json.loads(line)
            with condition:
                pending = job
                session.cancelled.set()
                condition.notify()
            if job['final']:
                return
        with condition:
            closed = True
            session.cancelled.set()
            condition.notify()
    threading.Thread(target=requests, daemon=True).start()
    try:
        while True:
            with condition:
                condition.wait_for(lambda: pending is not None or closed)
                if closed:
                    return
                job, pending = pending, None
                session.cancelled.clear()
            try:
                session.run(job['output'], job['snapshot'])
            except AnalysisCancelled:
                logging.info('Superseded analysis: %s', job['output'])
                continue
            if job['final']:
                return
    finally:
        session.close()


def stop_worker(worker):
    if worker is not None:
        status = worker.poll()
        owned_group = getattr(worker, '_hexrd_group', False)
        if status is None or (owned_group and status != 0):
            import signal
            grouped = owned_group
            try:
                grouped = grouped or os.getpgid(worker.pid) == worker.pid
                if grouped:
                    os.killpg(worker.pid, signal.SIGTERM)
                else:
                    worker.terminate()
            except ProcessLookupError:
                pass
            try:
                worker.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if grouped:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    worker.kill()
        worker.wait()


def start_worker(args):
    log = Path(args.output).resolve() / 'analysis.log'
    with log.open('wb') as stream:
        worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            'session', args.config, str(Path(args.output).resolve()),
            '--analysis-workers', str(args.analysis_workers),
            '--discovery-workers', str(args.discovery_workers)], stdin=subprocess.PIPE,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    worker._hexrd_group = True
    return worker


def submit_analysis(worker, args, snapshot, final=False):
    label = 'final' if final else f'prefix-{int(snapshot.stem):06d}'
    output = Path(args.output).resolve() / label
    worker.stdin.write(json_bytes(dict(output=str(output), snapshot=str(snapshot), final=final)) + b'\n')
    worker.stdin.flush()
    logging.info('Submitted %s analysis to session %d', label, worker.pid)
    return output


def prepared_messages(socket, capture, workers, timeout):
    """Keep at most one job per worker; publish only the ordered completed prefix."""
    pending, submitted, ending = deque(), 0, False
    last_message = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=workers, initializer=capture.warm_decoder)
    try:
        for task in [pool.submit(lambda: None) for _ in range(workers)]:
            task.result()  # Prepare libraries and native workers before receiving images.
        while True:
            capture.check_checksums()
            if pending and (pending[0][1] is None or pending[0][1].done()
                            or len(pending) >= workers or ending):
                parts, future = pending.popleft()
                yield parts, future.result() if future is not None else None
                if ending and not pending:
                    return
                continue
            if not socket.poll(1 if pending else 100):
                if not pending and time.monotonic() - last_message > timeout:
                    raise TimeoutError('No stream data; acquisition remains incomplete')
                if not pending:
                    yield None, None  # Let the owner check analysis failure during stream pauses.
                continue
            parts = [memoryview(part) if i == 2 else bytes(part)
                     for i, part in enumerate(socket.recv_multipart(copy=False))]
            if sum(map(len, parts)) > MAX_MESSAGE:
                raise ValueError('Oversized multipart message')
            message = json.loads(parts[0])
            last_message = time.monotonic()
            future = None
            if message['htype'] == 'dimage-1.0':
                if capture.header is None or message['series'] != capture.header['series']:
                    raise ValueError('Missing header or inconsistent series ID')
                if message['frame'] != submitted or submitted >= capture.expected:
                    raise ValueError('Missing, duplicate, out-of-order, or excess frame')
                submitted += 1
                capture.queue_checksum(parts)
                future = pool.submit(capture.prepare_image, parts)
            ending = message['htype'] == 'dseries_end-1.0'
            pending.append((parts, future))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def receive(args):
    import zmq

    threshold = (load_config(args.config)['fit_grains']['threshold']
                 if getattr(args, 'fit_support', False) else None)
    capture = Capture(args.output, wire_spool=getattr(args, 'wire_spool', False),
                      fit_threshold=threshold, raw_archive=getattr(args, 'raw_archive', None),
                      defer_checksums=getattr(args, 'defer_checksums', False))
    capture.block_workers = max(1, 8 // args.decode_workers)
    if getattr(args, 'incremental_maps', False):
        allocation_limit(3072)  # Analysis/fitting processes do not exist during static setup.
        import gc
        import yaml
        from hexrd.hedm.experiment import HedmExperiment
        from hexrd.hedm.find_orientations import EtaOmegaMapPlan, load_plane_data
        config = load_config(args.config)
        panels = [item['panel'] for item in config['image_series']['data']]
        if len(panels) != 1 or not isinstance(panels[0], str):
            raise ValueError('The sandbox supports a single physical panel')
        frozen = capture.root / 'analysis-config.yml'
        frozen.write_text(yaml.safe_dump(config))
        args.config = str(frozen)  # Workers use the same settings as the native map plan.
        config['image_series']['data'] = []  # Static setup never opens source images.
        config.update(multiprocessing=1, analysis_name=str(capture.root / 'map-setup'))
        filename = capture.root / 'map-config.yml'
        filename.write_text(yaml.safe_dump(config))
        experiment = HedmExperiment(str(filename))
        capture.map_plan = EtaOmegaMapPlan(experiment, load_plane_data(experiment), panels)
        del experiment
        from hexrd.core.instrument.planar_detector import _pixel_angles
        _pixel_angles.cache_clear()  # The native plan owns the required geometry.
        gc.collect()
        if threshold is not None and capture.wire_spool:
            for plan, _ in capture.map_plan.panels.values():
                plan.sparse_bins()  # Geometry only; allocate before acquisition starts.
        allocation_limit(2048)
        logging.info('Incremental map geometry prepared')
    worker, work_output = None, None
    first_frame = last_frame = None
    try:
        with zmq.Context() as context, context.socket(zmq.PULL) as socket:
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(args.endpoint)
            logging.info('Listening to %s', args.endpoint)
            jobs = prepared_messages(socket, capture, args.decode_workers, args.timeout)
            with closing(jobs):
                for parts, prepared in jobs:
                    if worker is not None and worker.poll() is not None:
                        raise RuntimeError('Analysis service exited; see analysis.log')
                    if work_output is not None and (work_output / 'result.json').exists():
                        logging.info('Provisional results (%s checksum): %s',
                                     capture.checksum_status, work_output / 'result.json')
                        work_output = None
                    if parts is None:
                        continue
                    kind = capture.accept(parts, prepared)
                    if kind == 'dheader-1.0':
                        worker = start_worker(args)
                    if kind == 'dimage-1.0':
                        last_frame = time.monotonic()
                        if first_frame is None:
                            first_frame = last_frame
                        count = len(capture.omega)
                        atomic_json(capture.root / 'status.json', dict(frames=count,
                                    expected=capture.expected, complete=False,
                                    analysis=str(work_output) if work_output else None,
                                    checksum_status=capture.checksum_status))
                        if (count % args.checkpoint == 0 and count < capture.expected
                                and work_output is None):
                            work_output = submit_analysis(worker, args, capture.snapshot())
                        if count % 100 == 0:
                            logging.info('Captured %d/%d frames', count, capture.expected)
        span = last_frame - first_frame
        atomic_json(capture.root / 'acquisition.json', dict(frames=len(capture.omega),
                    seconds=span, frames_per_second=(len(capture.omega)-1)/span if span else None))
        # Supersede the prefix while retaining static assets and fitting processes.
        work_output = submit_analysis(worker, args, capture.snapshot(), final=True)
        while worker.poll() is None:
            time.sleep(0.1)
        if worker.returncode:
            raise RuntimeError('Final analysis failed; see analysis.log')
        atomic_json(capture.root / 'status.json', dict(frames=len(capture.omega),
                    expected=capture.expected, complete=True, analysis=str(work_output),
                    checksum_status=capture.checksum_status))
        logging.info('Final results: %s', work_output)
    except BaseException as error:
        atomic_json(capture.root / 'status.json', dict(complete=False, valid=False,
                    checksum_status=capture.checksum_status, error=str(error)))
        raise
    finally:
        stop_worker(worker)
        capture.close()


def replay_messages(args):
    """Yield bounded detector messages from images or a lossless wire capture."""
    cache = None
    if args.wire_input:
        root = Path(args.wire_input)
        metadata = json.loads((root / 'header.json').read_text())
        message, header = metadata['header'], metadata['config']
        count = min(args.frames or header['nimages'] * header.get('ntrigger', 1),
                    header['nimages'] * header.get('ntrigger', 1))
        if len(list((root / 'frames').glob('*.zip'))) < count:
            raise ValueError('Wire replay source is incomplete; use --frames for a prefix')
        frame = lambda index: read_wire(root, index)
    else:
        import numpy as np
        from hexrd.hedm.experiment import HedmExperiment
        from hexrd.hedm._image_cache import FrameCache, _Readers

        experiment = HedmExperiment(args.config)
        if len(experiment.image_series_list) != 1:
            raise ValueError('The sandbox supports a single physical detector')
        series = experiment.image_series_list[0]
        cache = FrameCache(0)
        view = _Readers(cache).view(series)
        omega = next(iter(experiment.image_series.values())).omega
        count = min(args.frames or len(series), len(series))
        frame_time = 1/args.rate if args.rate else .001
        message = dict(htype='dheader-1.0', series=1, header_detail='basic')
        header = dict(x_pixels_in_detector=int(series.shape[1]),
                      y_pixels_in_detector=int(series.shape[0]),
                      omega_start=float(omega[0, 0]), omega_increment=float(np.diff(omega[0])[0]),
                      count_time=frame_time, frame_time=frame_time)
        frame = lambda index: encode_image(view.frame(cache, index), 1, index,
                                           omega[index], frame_time, args.source_scale)
    header.update(nimages=count, ntrigger=1)
    try:
        yield [json_bytes(message), json_bytes(header)]
        for index in range(count):
            yield frame(index)
        yield [json_bytes(dict(htype='dseries_end-1.0', series=message['series']))]
    finally:
        if cache is not None:
            cache.close()


def publish(args):
    import zmq

    with closing(replay_messages(args)) as source:
        prepared = None
        if args.preload_mb:
            prepared, size = [], 0
            for parts in source:
                size += sum(map(len, parts))
                if size > args.preload_mb * MIB:
                    raise MemoryError('Compressed replay exceeds --preload-mb; reduce --frames')
                prepared.append(parts)
            logging.info('Prepared %d compressed frames (%.1f MiB)', len(prepared)-2, size / MIB)
        with zmq.Context() as context, context.socket(zmq.PUSH) as socket:
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.LINGER, 1000)
            socket.bind(args.endpoint)

            def send(parts):
                deadline = time.monotonic() + args.timeout
                while True:
                    if socket.poll(100, zmq.POLLOUT):
                        try:
                            socket.send_multipart(parts, flags=zmq.NOBLOCK, copy=False)
                            return
                        except zmq.Again:
                            pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Receiver unavailable or unable to keep up')

            started, count = None, 0
            for parts in source if prepared is None else prepared:
                send(parts)
                if json.loads(parts[0])['htype'] == 'dheader-1.0':
                    started, total = time.monotonic(), json.loads(parts[1])['nimages']
                elif json.loads(parts[0])['htype'] == 'dimage-1.0':
                    count += 1
                    if args.rate:
                        time.sleep(max(0, started + count/args.rate - time.monotonic()))
                    if count % 100 == 0:
                        logging.info('Sent %d/%d frames', count, total)


def compare(left, right):
    import numpy as np

    left, right = Path(left), Path(right)
    for filename in ('scientific.npz', 'grains.npz'):
        with np.load(left / filename) as a, np.load(right / filename) as b:
            if set(a.files) != set(b.files):
                raise AssertionError(f'Different array sets: {filename}')
            for name in a.files:
                x, y = a[name], b[name]
                if x.dtype != y.dtype or x.shape != y.shape:
                    raise AssertionError(f'Scientific output differs: {name}')
                chunks = np.nditer([x, y],
                                   flags=['external_loop', 'buffered', 'zerosize_ok'],
                                   order='C', buffersize=MIB // x.dtype.itemsize)
                for a_chunk, b_chunk in chunks:
                    if a_chunk.tobytes() != b_chunk.tobytes():
                        raise AssertionError(f'Scientific output differs: {name}')
    paths = {p.relative_to(left) for p in left.rglob('*.out')}
    other = {p.relative_to(right) for p in right.rglob('*.out')}
    if paths != other:
        raise AssertionError('Different output file sets')
    for path in paths:
        if (left / path).read_bytes() != (right / path).read_bytes():
            raise AssertionError(f'Output differs: {path}')
    logging.info('BITWISE MATCH: orientations, trials, scores, fits, '
                 'and %d text files', len(paths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('publish', 'receive'):
        command = commands.add_parser(name)
        command.add_argument('config')
        command.add_argument('--endpoint', default='tcp://127.0.0.1:9999')
        command.add_argument('--timeout', type=float, default=60)
        if name == 'publish':
            command.add_argument('--rate', type=float, default=0, help='frames/s; 0 = fastest')
            command.add_argument('--frames', type=int, help='limit replay for transport tests')
            command.add_argument('--preload-mb', type=int, default=0,
                                 help='prepare compressed replay before timing; 0–512 MiB')
            command.add_argument('--source-scale', type=float, choices=(1., .5), default=1.,
                                 help='lossless emulator scale; .5 permits processed half counts')
            command.add_argument('--wire-input', help='replay an existing lossless wire capture')
        else:
            command.add_argument('output', help='new capture directory')
            command.add_argument('--checkpoint', type=int, default=360)
            command.add_argument('--decode-workers', type=int, choices=range(1, 9), default=4,
                                 help='bounded decode/map/spool threads (default: 4)')
            command.add_argument('--incremental-maps', action='store_true',
                                 help='accumulate native maps once during reception')
            command.add_argument('--wire-spool', action='store_true',
                                 help='persist original compressed messages without recompression')
            command.add_argument('--fit-support', action='store_true',
                                 help='also index exact threshold support for nearest-neighbor fitting')
            command.add_argument('--raw-archive',
                                 help='reuse a matching existing wire archive; retain full-pixel access')
            command.add_argument('--defer-checksums', action='store_true',
                                 help='explicitly pending-checksum provisional updates; validate before final results')
    command = commands.add_parser('analyze', help='offline reference or one immutable prefix')
    command.add_argument('config')
    command.add_argument('output')
    command.add_argument('--snapshot')
    command = commands.add_parser('session', help=argparse.SUPPRESS)
    command.add_argument('config')
    command.add_argument('output')
    for name in ('receive', 'analyze', 'session'):
        command = commands.choices[name]
        command.add_argument('--analysis-workers', type=int, choices=(1, 2), default=2,
                             help='persistent fitting processes within the memory budget')
        command.add_argument('--discovery-workers', type=int, choices=range(1, 9), default=4)
    command = commands.add_parser('compare')
    command.add_argument('left')
    command.add_argument('right')
    args = parser.parse_args(argv)
    frames = getattr(args, 'frames', None)
    preload = getattr(args, 'preload_mb', 0)
    if (getattr(args, 'timeout', 1) <= 0 or getattr(args, 'checkpoint', 2) < 2
            or getattr(args, 'rate', 0) < 0 or (frames is not None and frames < 1)
            or not 0 <= preload <= 512):
        parser.error('timeout/frames must be positive, rate nonnegative, checkpoint >= 2, preload 0–512')
    allowance = (768 if args.command == 'publish' and args.wire_input else
                 ANALYSIS_MIB if args.command in ('analyze', 'session') else
                 2048 if args.command == 'receive' and args.incremental_maps else 1536)
    allocation_limit(allowance)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        if args.command == 'publish':
            publish(args)
        elif args.command == 'receive':
            receive(args)
        elif args.command == 'analyze':
            analyze(args.config, args.output, args.snapshot,
                    args.analysis_workers, args.discovery_workers)
        elif args.command == 'session':
            serve_analysis(args)
        else:
            compare(args.left, args.right)
    except KeyboardInterrupt:
        logging.info('Cancelled')
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
