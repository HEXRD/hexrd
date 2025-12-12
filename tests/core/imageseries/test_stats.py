import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from hexrd.core.imageseries import stats

# --- Fixtures ---

@pytest.fixture
def mock_ims():
    """Create a simple mock ImageSeries."""
    ims = MagicMock()
    ims.__len__.return_value = 10
    ims.shape = (4, 4)
    ims.dtype = np.dtype(np.float32)
    ims.__getitem__.side_effect = lambda k: np.full((4, 4), float(k), dtype=np.float32)
    
    return ims

# --- Max Tests ---

def test_max(mock_ims):
    res = stats.max(mock_ims)
    assert np.all(res == 9.0)
    
    res = stats.max(mock_ims, nframes=5)
    assert np.all(res == 4.0)

def test_max_iter(mock_ims):
    gen = stats.max_iter(mock_ims, nchunk=2)
    
    res1 = next(gen)
    assert np.all(res1 == 4.0)
    
    res2 = next(gen)
    assert np.all(res2 == 9.0)
    
    with pytest.raises(StopIteration):
        next(gen)

# --- Min Tests ---

def test_min(mock_ims):
    res = stats.min(mock_ims)
    assert np.all(res == 0.0)

def test_min_iter(mock_ims):
    gen = stats.min_iter(mock_ims, nchunk=2)
    
    res1 = next(gen)
    assert np.all(res1 == 0.0)
    
    res2 = next(gen)
    assert np.all(res2 == 0.0)

# --- Average Tests ---

def test_average(mock_ims):
    res = stats.average(mock_ims)
    assert np.all(res == 4.5)
    
    res = stats.average(mock_ims, nframes=5)
    assert np.all(res == 2.0)

def test_average_iter(mock_ims):
    gen = stats.average_iter(mock_ims, nchunk=2)
    
    res1 = next(gen)
    assert np.all(res1 == 2.0)
    
    res2 = next(gen)
    assert np.all(res2 == 4.5)

def test_iterators_stop_zero_edge_case(mock_ims):
    """
    Test edge case where stop == 0.
    This occurs when nchunks == nframes.
    _chunk_stops logic produces a first stop at index 0.
    """
    nchunks = 10
    
    gen_max = stats.max_iter(mock_ims, nchunk=nchunks)
    res0 = next(gen_max)
    assert np.all(res0 == 0.0)
    
    gen_min = stats.min_iter(mock_ims, nchunk=nchunks)
    res0 = next(gen_min)
    assert np.all(res0 == 0.0)
    
    gen_avg = stats.average_iter(mock_ims, nchunk=nchunks)
    res0 = next(gen_avg)
    assert np.all(res0 == 0.0)

# --- Percentile & Median Tests ---

def test_percentile(mock_ims):
    res = stats.percentile(mock_ims, 50)
    assert np.all(res == 4.5)
    
    res = stats.percentile(mock_ims, 100)
    assert np.all(res == 9.0)

def test_median(mock_ims):
    res = stats.median(mock_ims)
    assert np.all(res == 4.5)

def test_percentile_iter(mock_ims):
    gen = stats.percentile_iter(mock_ims, 50, nchunks=2, use_buffer=False)
    
    res1 = next(gen)
    assert np.all(res1[0:2] == 4.5)
    assert np.all(res1[2:] == 0.0)
    
    res2 = next(gen)
    assert np.all(res2 == 4.5)

def test_median_iter(mock_ims):
    gen = stats.median_iter(mock_ims, nchunks=1, use_buffer=False)
    res = next(gen)
    assert np.all(res == 4.5)

# --- Utility Tests ---

def test_chunk_stops():
    stops = stats._chunk_stops(10, 2)
    np.testing.assert_array_equal(stops, [4, 9])
    
    stops = stats._chunk_stops(10, 3)
    np.testing.assert_array_equal(stops, [3, 6, 9])
    
    stops = stats._chunk_stops(10, 10)
    assert stops[0] == 0
    
    with pytest.raises(ValueError):
        stats._chunk_stops(5, 10)

def test_alloc_buffer(mock_ims):
    with patch('hexrd.core.imageseries.stats.STATS_BUFFER', 100):
        buf = stats._alloc_buffer(mock_ims, 10)
        assert buf.shape == (1, 4, 4)
        
    with patch('hexrd.core.imageseries.stats.STATS_BUFFER', 1000):
        buf = stats._alloc_buffer(mock_ims, 10)
        assert buf.shape == (10, 4, 4)

def test_toarray_buffering(mock_ims):
    arr = stats._toarray(mock_ims, 10)
    assert arr.shape == (10, 4, 4)
    assert arr[0, 0, 0] == 0.0
    
    buf = np.zeros((1, 4, 4), dtype=np.float32)
    arr = stats._toarray(mock_ims, 2, rows=(0, 2), buffer=buf)
    
    assert arr.shape == (2, 2, 4)
    assert buf[0, 0, 0] == 0.0
