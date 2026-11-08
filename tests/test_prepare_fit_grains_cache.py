"""Fit-only preparation retains exact foreground pixels and metadata."""

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_fit_grains_cache.py'
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux direct I/O')


@pytest.fixture
def cache(tmp_path):
    def make(dtype, malformed=False):
        source = tmp_path / 'source.fch5'
        values = ([-0., 10., 1000., 1001., np.nan, -1., np.inf, -np.inf]
                  if dtype == 'float64' else
                  [0, 10, 1000, 1001, 2**32 - 1, 1, 2000, 25])
        values = np.tile(np.array(values, dtype=dtype), 2)
        ij = np.tile(np.column_stack(np.unravel_index(np.arange(8), (3, 4))),
                     (2, 1)).astype('uint16')
        if malformed:
            ij[1] = ij[0]
        with h5py.File(source, 'w') as f:
            f['shape'], f['nframes'], f['dtype'] = [3, 4], 3, np.bytes_(dtype)
            f['frame_ids'] = [8, 16, 0, 8, 16, 16]
            f.create_dataset('data', data=values[:, None], chunks=(5, 1))
            f.create_dataset('indices', data=ij, chunks=(5, 1))
            f['metadata/omega'] = [[0., 1.], [1., 2.], [2., 3.]]
        return source
    return make


def run(source, output, threshold='1000'):
    return subprocess.run([sys.executable, str(SCRIPT), str(source), str(output),
                           '--threshold', threshold], capture_output=True,
                          text=True, timeout=30)


@pytest.mark.parametrize('dtype', ['float64', 'uint32'])
def test_exact_foreground_and_metadata(cache, tmp_path, dtype):
    source, output = cache(dtype), tmp_path / 'fit.fch5'
    original = source.read_bytes()
    result = run(source, output)
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == original
    with h5py.File(source) as a, h5py.File(output) as b:
        assert a['data'].dtype == b['data'].dtype
        assert a['indices'].dtype == b['indices'].dtype
        assert b.attrs['FIT_ONLY_THRESHOLD'] == 1000
        for name in ('shape', 'nframes', 'dtype', 'metadata/omega'):
            assert a[name][()].tobytes() == b[name][()].tobytes()
        for old, new in zip(a['frame_ids'][()].reshape(-1, 2),
                            b['frame_ids'][()].reshape(-1, 2)):
            data = a['data'][old[0]:old[1], 0]
            keep = (data > 1000) | np.signbit(data) | ~np.isfinite(data)
            assert data[keep].tobytes() == b['data'][new[0]:new[1], 0].tobytes()
            expected = a['indices'][old[0]:old[1]][keep]
            assert expected.tobytes() == b['indices'][new[0]:new[1]].tobytes()
    assert run(source, output).returncode != 0  # Never overwrite a cache.
    assert run(source, source).returncode != 0


def test_rejects_noncanonical_input(cache, tmp_path):
    output = tmp_path / 'fit.fch5'
    result = run(cache('float64', malformed=True), output)
    assert result.returncode != 0
    assert 'Coalesce and sort' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('threshold', ['nan', 'inf', '-1'])
def test_rejects_invalid_threshold(cache, tmp_path, threshold):
    assert run(cache('float64'), tmp_path / 'fit.fch5', threshold).returncode != 0


def test_rejects_lower_threshold_on_filtered_input(cache, tmp_path):
    source = cache('float64')
    with h5py.File(source, 'r+') as f:
        f.attrs['FIT_ONLY_THRESHOLD'] = 2000
    result = run(source, tmp_path / 'fit.fch5')
    assert result.returncode != 0
    assert 'higher threshold' in result.stderr


def test_rejects_invalid_frame_bounds(cache, tmp_path):
    source = cache('float64')
    with h5py.File(source, 'r+') as f:
        f['frame_ids'][1] = 100
    result = run(source, tmp_path / 'fit.fch5')
    assert result.returncode != 0
    assert 'frame bounds' in result.stderr
