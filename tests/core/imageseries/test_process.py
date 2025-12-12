import pytest
from unittest.mock import patch
import numpy as np
from hexrd.core.imageseries.process import ProcessedImageSeries

# --- Fixtures ---

class MockImageSeries:
    """A simple mock ImageSeries for testing."""
    def __init__(self, nframes=10, shape=(4, 4), dtype=np.float32, data=None):
        self._nframes = nframes
        self.shape = shape
        self.dtype = dtype
        self.metadata = {'omega': np.arange(nframes)}
        self.get_region_called = False
        self._data_map = data if data else {}

    def __len__(self):
        return self._nframes

    def __getitem__(self, key):
        if isinstance(key, tuple):
            frame_idx = key[0]
            slices = key[1:]
            frame = self[frame_idx]
            return frame[tuple(slices)]

        if isinstance(key, int):
            if key in self._data_map:
                return self._data_map[key].astype(self.dtype)
            return np.full(self.shape, float(key), dtype=self.dtype)
        
        return np.zeros(self.shape, dtype=self.dtype)

    def get_region(self, frame_idx, region):
        self.get_region_called = True
        return np.full((2, 2), -999.0)

@pytest.fixture
def mock_series():
    return MockImageSeries()

# --- Initialization Tests ---

def test_init_basic(mock_series):
    ps = ProcessedImageSeries(mock_series, [])
    assert len(ps) == 10
    assert ps.shape == (4, 4)
    assert ps.dtype == np.float32
    assert ps.metadata is not mock_series.metadata
    np.testing.assert_array_equal(ps.metadata['omega'], mock_series.metadata['omega'])

def test_init_framelist(mock_series):
    """Test subsetting frames via frame_list."""
    ps = ProcessedImageSeries(mock_series, [], frame_list=[0, 2, 4])
    
    assert len(ps) == 3
    expected_omega = np.array([0, 2, 4])
    np.testing.assert_array_equal(ps.metadata['omega'], expected_omega)
    np.testing.assert_array_equal(ps[1], np.full((4,4), 2.0))

# --- Operation Tests ---

def test_op_dark():
    """Test dark subtraction with clipping."""
    imser = MockImageSeries()
    oplist = [('dark', np.full((4,4), 2.0))]
    ps = ProcessedImageSeries(imser, oplist)
    res = ps[5]
    assert np.all(res == 3.0)
    
    ps.oplist[0] = ('dark', np.full((4,4), 10.0))
    res = ps[5]
    assert np.all(res == 0.0)

def test_op_flip():
    """Test all flip modes."""
    data_0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    imser = MockImageSeries(shape=(2, 3), data={0: data_0})
    
    ps = ProcessedImageSeries(imser, [('flip', 'v')])
    np.testing.assert_array_equal(ps[0], [[3, 2, 1], [6, 5, 4]])
    
    ps = ProcessedImageSeries(imser, [('flip', 'h')])
    np.testing.assert_array_equal(ps[0], [[4, 5, 6], [1, 2, 3]])
    
    ps = ProcessedImageSeries(imser, [('flip', 'r180')])
    np.testing.assert_array_equal(ps[0], [[6, 5, 4], [3, 2, 1]])
    
    ps = ProcessedImageSeries(imser, [('flip', 't')])
    np.testing.assert_array_equal(ps[0], [[1, 4], [2, 5], [3, 6]])
    
    ps = ProcessedImageSeries(imser, [('flip', 'r90')])
    np.testing.assert_array_equal(ps[0], [[3, 6], [2, 5], [1, 4]])
    
    ps = ProcessedImageSeries(imser, [('flip', 'r270')])
    np.testing.assert_array_equal(ps[0], [[4, 1], [5, 2], [6, 3]])
    
    ps = ProcessedImageSeries(imser, [('flip', 'none')])
    np.testing.assert_array_equal(ps[0], [[1, 2, 3], [4, 5, 6]])

def test_op_rectangle(mock_series):
    """Test standard rectangle slicing."""
    oplist = [('rectangle', ((0, 2), (0, 2)))]
    ps = ProcessedImageSeries(mock_series, oplist)
    
    res = ps[0]
    assert res.shape == (2, 2)

def test_op_rectangle_optimized(mock_series):
    """Test optimized path when rectangle is the *first* operation."""
    oplist = [('rectangle', ((0, 2), (0, 2)))]
    ps = ProcessedImageSeries(mock_series, oplist)
    
    res = ps[0]
    
    assert mock_series.get_region_called
    assert res[0, 0] == -999.0 # Value from get_region mock

def test_op_rectangle_unoptimized(mock_series):
    """Test unoptimized rectangle path (triggered when rect is NOT first)."""
    oplist = [('flip', 'none'), ('rectangle', ((0, 2), (0, 2)))]
    ps = ProcessedImageSeries(mock_series, oplist)
    
    res = ps[0]
    
    assert not mock_series.get_region_called
    assert res.shape == (2, 2)

def test_op_rectangle_optimized_fancy_indexing(mock_series):
    """Test optimized rectangle with fancy indexing keys."""
    oplist = [('rectangle', ((0, 2), (0, 2)))]
    ps = ProcessedImageSeries(mock_series, oplist)
    
    res = ps[(0, slice(0, 1))] 
    
    assert mock_series.get_region_called
    assert res.shape == (1, 2)

def test_op_add():
    """Test add operation and type promotion."""
    imser = MockImageSeries(dtype=np.uint8, data={0: np.array([[10]])})
    
    ps = ProcessedImageSeries(imser, [('add', 5.5)])
    res = ps[0]
    
    assert res[0, 0] == 15.5
    assert res.dtype == np.float32

@patch('hexrd.core.imageseries.process.scipy.ndimage.gaussian_laplace')
def test_op_gauss_laplace(mock_gl, mock_series):
    """Test Gaussian Laplace op."""
    ps = ProcessedImageSeries(mock_series, [('gauss_laplace', 2.0)])
    _ = ps[0]
    
    mock_gl.assert_called_with(pytest.approx(np.full((4,4), 0.0)), 2.0)

# --- API & Iterator Tests ---

def test_iter_and_api(mock_series):
    ps = ProcessedImageSeries(mock_series, [])
    
    def noop(img, data): return img
    ps.addop('noop', noop)
    assert 'noop' in ps._opdict
    assert ps.oplist == []
    
    frames = list(ps)
    assert len(frames) == 10
    assert frames[0][0, 0] == 0.0
    assert frames[9][0, 0] == 9.0

def test_fancy_getitem_dispatch(mock_series):
    """Test __getitem__ with tuple keys (fancy indexing)."""
    ps = ProcessedImageSeries(mock_series, [])
    res = ps[(0, slice(0, 2))]
    assert res.shape == (2, 4)
