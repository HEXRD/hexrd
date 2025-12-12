import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY
import numpy as np
from scipy.sparse import csr_array
import pickle
import os

from hexrd.core.imageseries.load.framecache import FrameCacheImageSeriesAdapter

# --- Fixtures ---

@pytest.fixture
def mock_npz():
    return {
        'nframes': 2, 'shape': (10, 10), 'dtype': np.array('float32'),
        '0_row': np.array([0]), '0_col': np.array([0]), '0_data': np.array([10.]),
        '1_row': np.array([1]), '1_col': np.array([1]), '1_data': np.array([20.]),
        'meta_key': 'val'
    }

@pytest.fixture
def mock_h5():
    f = MagicMock()
    f.attrs = {'HEXRD_FRAMECACHE_VERSION': 1}
    f.__enter__.return_value = f
    data = {
        'shape': np.array([10, 10]), 'nframes': np.array(2),
        'frame_ids': np.array([0, 1, 1, 2]), 'data': np.array([[10.], [20.]]),
        'indices': np.array([[0, 0], [1, 1]]), 'dtype': MagicMock(), 'metadata': MagicMock()
    }
    f.__getitem__.side_effect = lambda k: data.get(k, MagicMock())
    return f

# --- Tests ---

@patch('hexrd.core.imageseries.load.framecache.np.load')
def test_init_npz(mock_load, mock_npz):
    mock_load.return_value.__enter__.return_value = mock_npz
    a = FrameCacheImageSeriesAdapter("t.npz", style="npz")
    assert len(a) == 2 and a.shape == (10, 10) and a.metadata['meta_key'] == 'val'

@patch('hexrd.core.imageseries.load.framecache.h5py.File')
@patch('hexrd.core.imageseries.load.framecache.h5py_read_string', return_value='float32')
@patch('hexrd.core.imageseries.load.framecache.unwrap_h5_to_dict')
def test_init_fch5(mock_unwrap, mock_read, mock_file, mock_h5):
    mock_file.return_value = mock_h5
    a = FrameCacheImageSeriesAdapter("t.h5", style="fch5")
    assert len(a) == 2 and tuple(a.shape) == (10, 10)

@patch('hexrd.core.imageseries.load.framecache.yaml.load')
@patch('builtins.open', new_callable=mock_open)
def test_init_yml(mock_open, mock_yaml):
    mock_yaml.return_value = {
        'data': {'file': 'c.npz', 'nframes': 5, 'shape': [10, 10], 'dtype': 'f4'}, 'meta': {}
    }
    a = FrameCacheImageSeriesAdapter("t.yml", style="yml")
    assert len(a) == 5 and a._from_yml

def test_init_error():
    with pytest.raises(TypeError): FrameCacheImageSeriesAdapter("t", style="bad")

@patch('hexrd.core.imageseries.load.framecache._load_framecache_npz')
@patch('hexrd.core.imageseries.load.framecache.np.load')
def test_data_access(mock_load, mock_cache, mock_npz):
    mock_load.return_value.__enter__.return_value = mock_npz
    a = FrameCacheImageSeriesAdapter("t.npz", style="npz")
    
    # Mock Cache Return
    f = csr_array(np.ones((10, 10)))
    mock_cache.return_value = [f, f]
    
    # 1. Standard Access
    assert np.array_equal(a[0], f.toarray())
    # 2. Fancy Indexing (tuple)
    assert a[0, np.array([0]), np.array([0])][0] == 1.0
    # 3. Slicing
    assert a[0, 0:5].shape == (5, 10)
    # 4. Get Region
    assert a.get_region(0, ((0,5), (0,5))).shape == (5, 5)

@patch('hexrd.core.imageseries.load.framecache.np.load')
def test_metadata_parsing(mock_load, mock_npz):
    mock_load.return_value.__enter__.return_value = mock_npz
    a = FrameCacheImageSeriesAdapter("t.npz", style="npz")
    meta = {'k-array': [1], 'k': '++np.array', 'std': 1}
    res = a.load_metadata(meta)
    assert isinstance(res['k'], np.ndarray) and res['std'] == 1

@patch('hexrd.core.imageseries.load.framecache._load_framecache_npz')
@patch('hexrd.core.imageseries.load.framecache.yaml.load')
@patch('builtins.open', new_callable=mock_open)
def test_yml_paths(mock_open, mock_yaml, mock_loader):
    mock_yaml.return_value = {'data': {'file': 'c.npz', 'nframes': 1, 'shape': [1], 'dtype': 'f4'}, 'meta': {}}
    # FIX: Return a dummy frame so adapter[0] doesn't raise IndexError
    mock_loader.return_value = [MagicMock()]
    
    # Relative
    FrameCacheImageSeriesAdapter("d/t.yml", style="yml")[0]
    mock_loader.assert_called_with(filepath=os.path.join("d", "c.npz"), num_frames=1, shape=(1,), dtype=np.dtype('f4'))
    
    # Absolute
    mock_yaml.return_value['data']['file'] = '/abs/c.npz'
    FrameCacheImageSeriesAdapter("d/t.yml", style="yml")[0]
    mock_loader.assert_called_with(filepath='/abs/c.npz', num_frames=1, shape=(1,), dtype=np.dtype('f4'))

@patch('hexrd.core.imageseries.load.framecache.h5py.File')
def test_internal_loaders(mock_file, mock_h5, mock_npz):
    # Test FCH5 Loader + Threading logic via actual execution
    from hexrd.core.imageseries.load.framecache import _load_framecache_fch5, _load_framecache_npz
    _load_framecache_fch5.cache_clear()
    
    mock_file.return_value = mock_h5
    fl = _load_framecache_fch5("p.h5", 2, (10, 10), np.dtype('float32'), 2)
    assert fl[0][0,0] == 10.0
    
    # Test NPZ Loader
    _load_framecache_npz.cache_clear()
    with patch('hexrd.core.imageseries.load.framecache.np.load') as ml:
        ml.return_value.__enter__.return_value = mock_npz
        fl = _load_framecache_npz("p.npz", 2, (10, 10), np.dtype('float32'))
        assert fl[0][0,0] == 10.0

@patch('hexrd.core.imageseries.load.framecache.h5py.File')
def test_fch5_errors(mock_file):
    f = MagicMock(); f.attrs = {}; f.__enter__.return_value = f
    mock_file.return_value = f
    with pytest.raises(NotImplementedError): FrameCacheImageSeriesAdapter("t.h5", style="fch5")
    f.attrs = {'HEXRD_FRAMECACHE_VERSION': 99}
    with pytest.raises(NotImplementedError): FrameCacheImageSeriesAdapter("t.h5", style="fch5")

def test_pickle(mock_npz):
    with patch('hexrd.core.imageseries.load.framecache.np.load') as ml:
        ml.return_value.__enter__.return_value = mock_npz
        a = FrameCacheImageSeriesAdapter("t.npz", style="npz")
        assert hasattr(pickle.loads(pickle.dumps(a)), '_load_framelist_lock')