import pytest
import numpy as np
from hexrd.core import gridutil

# =============================================================================
# cellIndices Tests
# =============================================================================

def test_cellIndices_positive_delta():
    edges = np.array([0., 1., 2., 3., 4.])
    points = np.array([0.5, 1.5, 3.99, 4.0, -1.0])
    indices = gridutil.cellIndices(edges, points)
    
    np.testing.assert_array_equal(indices, [0, 1, 3, 3, -1])

def test_cellIndices_negative_delta():
    edges = np.array([4., 3., 2., 1., 0.])
    points = np.array([3.5, 2.5, 0.0, 5.0])
    indices = gridutil.cellIndices(edges, points)
    
    np.testing.assert_array_equal(indices, [0, 1, 3, -2])

def test_cellIndices_errors():
    with pytest.raises(AssertionError):
        gridutil.cellIndices(np.array([1.]), [1.])
        
    with pytest.raises(RuntimeError):
        gridutil.cellIndices(np.array([1., 1.]), [1.])

def test_cellIndices_nan_handling():
    edges = np.array([0, 10])
    points = np.array([np.nan, 5.0])
    
    indices = gridutil.cellIndices(edges, points)
    assert indices[0] == -1
    assert indices[1] == 0

# =============================================================================
# Connectivity Tests
# =============================================================================

def test_fill_connectivity_py_func():
    """Explicitly test the Numba kernel for coverage."""
    con = np.zeros((4, 4), dtype=np.int64)
    gridutil._fill_connectivity.py_func(con, 2, 2, 1)
    
    np.testing.assert_array_equal(con[0], [1, 0, 3, 4])

def test_cellConnectivity_2d_ul():
    con = gridutil.cellConnectivity(2, 2, p=1, origin='ul')
    assert con.shape == (4, 4)
    np.testing.assert_array_equal(con[0], [1, 0, 3, 4])
    np.testing.assert_array_equal(con[3], [5, 4, 7, 8])

def test_cellConnectivity_2d_ll():
    con_ul = gridutil.cellConnectivity(2, 2, origin='ul')
    con_ll = gridutil.cellConnectivity(2, 2, origin='ll')
    np.testing.assert_array_equal(con_ll[0], con_ul[0][::-1])

def test_cellConnectivity_3d():
    con = gridutil.cellConnectivity(1, 1, p=2)
    assert con.shape == (1, 8) 
    np.testing.assert_array_equal(con[0], [1, 0, 2, 3, 5, 4, 6, 7])

# =============================================================================
# Centroids and Areas (Numba .py_func)
# =============================================================================

def test_cellCentroids():
    crd = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    con = np.array([[0, 1, 2, 3]])
    
    cens = gridutil.cellCentroids.py_func(crd, con)
    assert cens.shape == (1, 2)
    np.testing.assert_allclose(cens[0], [0.5, 0.5])

def test_compute_areas_numba():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    con = np.array([[0, 1, 2, 3]])
    
    areas = gridutil.compute_areas.py_func(xy, con)
    assert np.isclose(areas[0], 1.0)
    
    con_cw = np.array([[0, 3, 2, 1]])
    areas_cw = gridutil.compute_areas.py_func(xy, con_cw)
    assert np.isclose(areas_cw[0], -1.0)

def test_computeArea_polygon():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(gridutil.computeArea(poly), 0.5)
    
    sq = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
    assert np.isclose(gridutil.computeArea(sq), 4.0)

# =============================================================================
# Tolerance Grid
# =============================================================================

def test_make_tolerance_grid():
    ndiv, grid = gridutil.make_tolerance_grid(0.1, 1.0, 1)
    assert ndiv == 10
    assert len(grid) == 11
    assert np.isclose(grid[0], -0.5)
    
    ndiv, grid = gridutil.make_tolerance_grid(0.1, 0.15, 1, adjust_window=True)
    assert ndiv == 2
    assert np.isclose(grid[-1] - grid[0], 0.2)
    
    ndiv, grid = gridutil.make_tolerance_grid(0.1, 1.0, 1, one_sided=True)
    assert ndiv == 20
    assert np.isclose(grid[0], -1.0)

# =============================================================================
# Geometry Clipping Tests
# =============================================================================

def test_computeIntersection():
    l1 = [[0, 0], [2, 2]]
    l2 = [[0, 2], [2, 0]]
    pt = gridutil.computeIntersection(l1, l2)
    np.testing.assert_allclose(pt, [1.0, 1.0])
    
    l3 = [[0, 1], [2, 3]]
    assert len(gridutil.computeIntersection(l1, l3)) == 0

def test_isinside():
    boundary = np.array([[0.0, 0.0], [1.0, 0.0]])

    assert gridutil.isinside(np.array([0.5, -0.5]), boundary, ccw=True)
    assert not gridutil.isinside(np.array([0.5, 0.5]), boundary, ccw=True)
    assert gridutil.isinside(np.array([0.5, 0.5]), boundary, ccw=False)
    assert gridutil.isinside(np.array([0.5, 0.0]), boundary)

def test_sutherlandHodgman():
    subject = [[0., 0.], [2., 0.], [2., 2.], [0., 2.]]
    clip = [[1., 0.], [3., 0.], [3., 2.], [1., 2.]]
    
    res = gridutil.sutherlandHodgman(subject, clip)
    area = gridutil.computeArea(res)
    assert np.isclose(area, 2.0)
    
    assert len(gridutil.sutherlandHodgman([], clip)) == 0
