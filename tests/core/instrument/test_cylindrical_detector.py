import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from hexrd.core import constants as ct
from hexrd.core.instrument.cylindrical_detector import (
    CylindricalDetector, 
    _fix_branch_cut_in_gradients
)

# --- Fixtures ---

@pytest.fixture
def mock_warp():
    """Mock the _warp_to_cylinder internal function."""
    with patch('hexrd.core.instrument.cylindrical_detector._warp_to_cylinder') as m:
        m.return_value = np.array([[0.0, 0.0, 1.0]])
        yield m

@pytest.fixture
def mock_xrdutil():
    """Mock the external hexrd.hedm.xrdutil dependency."""
    with patch('hexrd.core.instrument.cylindrical_detector.xrdutil') as m:
        m.utils._dvec_to_angs.return_value = (np.array([0.1]), np.array([0.2]))
        
        m.utils._project_on_detector_cylinder.return_value = (
            np.array([[10.0, 20.0]]), 
            np.eye(3),                
            np.array([True])          
        )
        
        m.utils._unitvec_to_cylinder.return_value = np.array([0.0, 10.0, 0.0])
        m.utils._clip_to_cylindrical_detector.return_value = (
            np.array([0.0, 10.0, 0.0]), 
            None 
        )
        m.utils._dewarp_from_cylinder.return_value = np.array([5.0, 5.0])
        
        yield m

@pytest.fixture
def detector(mock_xrdutil, mock_warp):
    """Create a standard CylindricalDetector instance with mocked dependencies."""
    return CylindricalDetector(
        radius=100.0,
        rows=200,
        cols=200,
        pixel_size=(0.1, 0.1),
        tvec=[0, 0, 500],
        tilt=[0, 0, 0]
    )

# --- Initialization & Properties ---

def test_init(detector):
    assert detector.detector_type == 'cylindrical'
    assert detector.radius == 100.0
    assert detector.rows == 200
    assert detector.cols == 200

def test_properties(detector):
    assert detector.extra_config_kwargs['radius'] == 100.0
    
    detector.radius = 200.0
    assert detector.radius == 200.0
    
    assert detector.pixel_normal.shape == (detector.rows * detector.cols, 3)
    assert np.isclose(detector.angle_extent, 0.05)

# --- Coordinate Transforms ---

def test_cart_to_angles(detector, mock_warp, mock_xrdutil):
    xy_data = np.array([[10.0, 20.0]])
    
    tth_eta, _ = detector.cart_to_angles(xy_data)
    
    mock_warp.assert_called_once()
    _, kwargs = mock_warp.call_args
    np.testing.assert_array_equal(kwargs['tVec_s'], ct.zeros_3)
    
    mock_xrdutil.utils._dvec_to_angs.assert_called_once()
    
    assert tth_eta.shape == (1, 2)
    assert tth_eta[0, 0] == 0.1
    assert tth_eta[0, 1] == 0.2

def test_cart_to_angles_with_distortion(detector, mock_warp):
    mock_dist = MagicMock()
    mock_dist.apply.return_value = np.array([[11.0, 21.0]])
    
    detector._distortion = mock_dist
    
    xy_data = np.array([[10.0, 20.0]])
    detector.cart_to_angles(xy_data, apply_distortion=True)
    
    mock_dist.apply.assert_called_once_with(xy_data)
    args, _ = mock_warp.call_args
    np.testing.assert_array_equal(args[0], [[11.0, 21.0]])

def test_angles_to_cart(detector, mock_xrdutil):
    tth_eta = np.array([[0.1, 0.5]])
    
    xy = detector.angles_to_cart(tth_eta)
    
    mock_xrdutil.utils._project_on_detector_cylinder.assert_called_once()
    assert xy.shape == (1, 2)
    assert not np.isnan(xy).any()
    np.testing.assert_array_equal(xy, [[10.0, 20.0]])

def test_angles_to_cart_masked(detector, mock_xrdutil):
    mock_xrdutil.utils._project_on_detector_cylinder.return_value = (
        np.array([[10.0, 20.0]]), 
        np.eye(3), 
        np.array([False]) 
    )
    
    tth_eta = np.array([[0.1, 0.5]])
    xy = detector.angles_to_cart(tth_eta)
    
    assert np.isnan(xy).all()

def test_cart_to_dvecs(detector, mock_warp):
    xy_data = np.array([[10.0, 20.0]])
    detector.cart_to_dvecs(xy_data)
    
    mock_warp.assert_called_once()
    _, kwargs = mock_warp.call_args
    assert kwargs['normalize'] is False

# --- Beam & Pixel Physics ---

def test_beam_position(detector, mock_xrdutil):
    pos = detector.beam_position
    
    mock_xrdutil.utils._unitvec_to_cylinder.assert_called_once()
    mock_xrdutil.utils._clip_to_cylindrical_detector.assert_called_once()
    mock_xrdutil.utils._dewarp_from_cylinder.assert_called_once()
    
    np.testing.assert_array_equal(pos, [5.0, 5.0])

def test_local_normal(detector):
    normals = detector.local_normal()
    n_pix = detector.rows * detector.cols
    assert normals.shape == (n_pix, 3)
    
    norms = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

# --- Pixel Angles & Gradients (Memoized) ---

def test_pixel_angles(detector, mock_warp, mock_xrdutil):
    n_pix = detector.rows * detector.cols
    mock_warp.return_value = np.zeros((n_pix, 3))
    
    tth_mock = np.zeros(n_pix)
    eta_mock = np.zeros(n_pix)
    mock_xrdutil.utils._dvec_to_angs.return_value = (tth_mock, eta_mock)
    
    res_tth, res_eta = detector.pixel_angles()
    
    assert res_tth.shape == (detector.rows, detector.cols)
    assert res_eta.shape == (detector.rows, detector.cols)

def test_pixel_angles_with_distortion(detector, mock_warp, mock_xrdutil):
    """Test distortion branch in _pixel_angles."""
    n_pix = detector.rows * detector.cols
    
    mock_dist = MagicMock()
    mock_dist.apply.return_value = np.zeros((n_pix, 2))
    detector._distortion = mock_dist
    
    mock_warp.return_value = np.zeros((n_pix, 3))
    mock_xrdutil.utils._dvec_to_angs.return_value = (np.zeros(n_pix), np.zeros(n_pix))
    
    detector.pixel_angles()
    mock_dist.apply.assert_called_once()

def test_pixel_gradients(detector, mock_warp, mock_xrdutil):
    n_pix = detector.rows * detector.cols
    mock_warp.return_value = np.zeros((n_pix, 3))
    
    tth = np.linspace(0, 1, n_pix)
    eta = np.linspace(0, 1, n_pix)
    mock_xrdutil.utils._dvec_to_angs.return_value = (tth, eta)
    
    grad_tth = detector.pixel_tth_gradient()
    grad_eta = detector.pixel_eta_gradient()
    
    assert grad_tth.shape == (detector.rows, detector.cols)
    assert grad_eta.shape == (detector.rows, detector.cols)

# --- Calculations & Utilities ---

def test_transmission_calc(detector):
    detector.filter = MagicMock()
    detector.filter.absorption_length.return_value = 1.0
    detector.filter.thickness = 0.1
    
    detector.coating = MagicMock()
    detector.coating.absorption_length.return_value = 1.0
    detector.coating.thickness = 0.1
    
    detector.phosphor = MagicMock()
    detector.phosphor.energy_absorption_length.return_value = 1.0
    detector.phosphor.thickness = 0.1
    detector.phosphor.readout_length = 0.1
    detector.phosphor.pre_U0 = 1.0
    
    trans_coating, trans_phosphor = detector.calc_filter_coating_transmission(energy=50.0)
    
    assert trans_coating.shape == (detector.rows, detector.cols)
    assert trans_phosphor.shape == (detector.rows, detector.cols)

def test_memoization_update():
    panels = [CylindricalDetector(), CylindricalDetector()]
    CylindricalDetector.update_memoization_sizes(panels)

def test_branch_cut_fix():
    arr = np.array([0.1])
    res = _fix_branch_cut_in_gradients(arr)
    assert res == 0.1
    
    arr_cut = np.array([np.pi])
    res_cut = _fix_branch_cut_in_gradients(arr_cut)
    assert np.isclose(res_cut, 0.0, atol=1e-6)
