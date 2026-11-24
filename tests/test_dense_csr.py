"""Native image compaction preserves SciPy's sparse pixel representation."""
import numpy as np
import pytest
from scipy.sparse import csr_array

from hexrd.core.experiment import _as_csr
from hexrd.core.extensions import transforms


@pytest.mark.parametrize('dtype', ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'f4', 'f8'])
@pytest.mark.parametrize('threshold', [0., .1, 1., 25.1, 1000., 1e100])
def test_fit_support_compaction_keeps_exact_pixels_and_order(dtype, threshold):
    values = np.random.default_rng(21).integers(0, 100, (17, 31)).astype(dtype)
    if values.dtype.kind == 'f':
        values *= .5
        edge = values.dtype.type(threshold) if threshold < 1e10 else values.dtype.type(1000.)
        values.flat[:7] = [np.nan, np.inf, -np.inf, -0., edge,
                          np.nextafter(edge, -np.inf, dtype=values.dtype),
                          np.nextafter(edge, np.inf, dtype=values.dtype)]
    if values.dtype.kind in 'if':
        values.flat[8] = -1
    before = values.tobytes()
    reference = values.copy()
    # Explicit binary64 threshold semantics, including float32's neighboring values.
    reference[(reference >= 0) & (reference.astype('f8') <= threshold)] = 0
    expected = csr_array(reference)
    actual = transforms._denseToCSR(values, threshold)
    for field, array in zip(('data', 'indices', 'indptr'), actual):
        assert array.tobytes() == getattr(expected, field).tobytes()
    assert values.tobytes() == before


def test_fit_support_rejects_unsupported_comparisons():
    for dtype in ('u8', 'i8', 'c16', '>f8'):
        assert transforms._denseToCSR(np.ones((2, 3), dtype=dtype), 1.) is None
    for threshold in (np.inf, np.nan, -np.inf):
        with pytest.raises(ValueError, match='threshold'):
            transforms._denseToCSR(np.ones((2, 3)), threshold)


@pytest.mark.parametrize('dtype', ['u4', 'f8'])
def test_fit_support_grows_only_its_occupied_buffers(dtype):
    frame = np.random.default_rng(81).integers(0, 5, size=(257, 1023)).astype(dtype)
    expected = csr_array(np.where(frame > 1., frame, 0))
    assert expected.nnz > 131072
    buffers = transforms._denseToCSR(frame, 1.)
    for name, array in zip(('data', 'indices', 'indptr'), buffers):
        assert array.tobytes() == getattr(expected, name).tobytes()


@pytest.mark.parametrize('dtype', ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8',
                                 'f4', 'f8', 'bool', '>u4', '>f8', 'c16'])
@pytest.mark.parametrize('shape', [(0, 4), (3, 0), (1, 1), (17, 31)])
def test_native_csr_matches_scipy(dtype, shape):
    frame = np.random.default_rng(7).integers(0, 6, shape).astype(dtype)
    frame[frame < 3] = 0
    if frame.dtype.kind == 'f' and frame.size > 5:
        frame.flat[:5] = [np.nan, np.inf, -np.inf, -0., .123456789]
        frame.flat[-1] = np.nextafter(frame.dtype.type(0), frame.dtype.type(1))
    if frame.dtype.kind in 'iu' and frame.size > 5:
        frame.flat[:2] = [np.iinfo(frame.dtype).min, np.iinfo(frame.dtype).max]
    before = frame.tobytes()
    try:
        expected = csr_array(frame)
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            _as_csr(frame)
        return
    actual = _as_csr(frame)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    for name in ('data', 'indices', 'indptr'):
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes()
    assert frame.tobytes() == before


@pytest.mark.parametrize('layout', ['strided', 'readonly', 'unaligned', 'sparse', 'zero', 'nan-payload'])
def test_native_csr_layout_fallbacks(layout):
    frame = np.arange(35., dtype='f8').reshape(5, 7)
    if layout == 'strided':
        frame = frame[::-1, ::2]
    elif layout == 'readonly':
        frame.flags.writeable = False
    elif layout == 'unaligned':
        frame = np.ndarray(frame.shape, dtype='f8', buffer=bytearray(frame.nbytes+1), offset=1)
        frame[:] = 7.
    elif layout == 'zero':
        frame[:] = 0
    elif layout == 'nan-payload':
        frame.view('u8').flat[:3] = [0x7ff8000000000001, 0xfff8000000000002, 0x8000000000000000]
    else:
        frame = csr_array(frame)
    expected, actual = csr_array(frame), _as_csr(frame)
    for name in ('data', 'indices', 'indptr'):
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes()


def test_native_csr_fits_when_worst_case_reservation_does_not():
    import os
    import subprocess
    import sys

    pytest.importorskip('resource')
    subprocess.run([sys.executable, '-c', """
import resource
import numpy as np
import psutil
from hexrd.core.extensions import transforms
frame = np.zeros((2048, 4096), dtype='f8')
frame[7, 9] = 1.25
limit = psutil.Process().memory_info().vms + 16 * 1024**2
assert limit < 1024**3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
data, indices, offsets = transforms._denseToCSR(frame)
assert data.tolist() == [1.25] and indices.tolist() == [9]
assert (offsets[:8] == 0).all() and (offsets[8:] == 1).all()
assert frame[7, 9] == 1.25
assert transforms._denseToCSR(frame, .5)[0].tolist() == [1.25]
"""], env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                   check=True, timeout=30)
