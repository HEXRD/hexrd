"""Exact sparse Eiger decoding, including malformed blocks and pixel extremes."""
import ctypes
from pathlib import Path
import struct

import numpy as np
import pytest

from hexrd.core.extensions import transforms as native

pytestmark = pytest.mark.skipif(not hasattr(native, '_EigerSupport'), reason='Native Eiger support unavailable')


@pytest.fixture(scope='module')
def codecs():
    hdf5plugin = pytest.importorskip('hdf5plugin')
    lz4 = pytest.importorskip('lz4.block._block')
    bitshuffle = ctypes.CDLL(str(Path(hdf5plugin.PLUGIN_PATH) / 'libh5bshuf.so'))
    bound = bitshuffle.bshuf_compress_lz4_bound
    bound.argtypes = [ctypes.c_size_t] * 3
    bound.restype = ctypes.c_size_t
    compress = bitshuffle.bshuf_compress_lz4
    compress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *([ctypes.c_size_t] * 3)]
    compress.restype = ctypes.c_int64

    def encode(values, block):
        values = np.asarray(values, dtype='<u4').ravel()
        output = np.empty(bound(values.size, 4, block), dtype='u1')
        size = compress(values.ctypes.data, output.ctypes.data, values.size, 4, block)
        assert size >= 0
        return np.frombuffer(struct.pack('>QI', values.nbytes, block * 4) + output[:size].tobytes(), dtype='u1')

    return ctypes.CDLL(lz4.__file__), encode


def test_sparse_eiger_random_bit_planes_and_thresholds(codecs):
    library, encode = codecs
    rng = np.random.default_rng(97531)
    counts = rng.integers(0, 2**32, size=8259, dtype='u4')
    payload = encode(counts, 0)
    decoder = native._EigerSupport(library, 4)
    for cut in [0, 2**32-1, *(2**bit-1 for bit in range(32)),
                *rng.integers(0, 2**32, size=32).tolist()]:
        values, pixels = decoder.support(payload, cut)
        expected = np.flatnonzero(counts > cut).astype('u4')
        assert pixels.tobytes() == expected.tobytes()
        assert values.tobytes() == counts[expected].tobytes()


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('length,block', [(1, 0), (7, 64), (8, 64), (63, 64),
                                       (64, 64), (77, 64), (4099, 0)])
def test_sparse_eiger_retains_exact_pixels_and_order(codecs, workers, length, block):
    library, encode = codecs
    counts = np.resize(np.array([0, 1, 1999, 2000, 2001, 2**24+1, 2**31+1, 2**32-1], dtype='u4'), length)
    payload = encode(counts, block)
    decoder = native._EigerSupport(library, workers)
    for threshold in (0., 1.5, 2000., 2**24, 2**32-1):
        values, pixels = decoder.support(payload, threshold)
        expected = np.flatnonzero(counts > threshold).astype('u4')
        assert pixels.tobytes() == expected.tobytes()
        assert values.tobytes() == counts[expected].tobytes()
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError, match='cancelled'):
        decoder.support(payload, 0., cancel)


def test_sparse_eiger_rejects_corruption_and_remains_usable(codecs):
    library, encode = codecs
    decoder = native._EigerSupport(library, 2)
    good = encode(np.arange(100, dtype='u4'), 64)
    for malformed in (b'', b'\0'*11, struct.pack('>QI', 2**40, 8192),
                      good[:-1].tobytes(), good.tobytes()+b'extra',
                      struct.pack('>QI', 100, 5)):
        with pytest.raises(ValueError):
            decoder.support(np.frombuffer(malformed, dtype='u1'), 0.)
    invalid_lz4 = struct.pack('>QII', 256, 256, 1) + b'\xff'
    with pytest.raises(RuntimeError, match='compressed block'):
        decoder.support(np.frombuffer(invalid_lz4, dtype='u1'), 0.)
    values, pixels = decoder.support(good, 50.)
    assert pixels.tolist() == values.tolist() == list(range(51, 100))
    for threshold in (-1., np.nan, np.inf, 2**32):
        with pytest.raises(ValueError):
            decoder.support(good, threshold)
    for workers in (0, 9):
        with pytest.raises(ValueError, match='workers'):
            native._EigerSupport(library, workers)
