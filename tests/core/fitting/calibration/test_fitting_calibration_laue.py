import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from hexrd.core.fitting.calibration.laue import (
    LaueCalibrator,
    sxcal_obj_func,
    gaussian_2d,
    gaussian_2d_int
)

# --- Fixtures ---

@pytest.fixture
def laue_calibrator():
    mock_instr = MagicMock()
    mock_instr.chi = 0.0
    mock_instr.tvec = np.zeros(3)
    mock_instr.beam_vector = np.array([0, 0, 1])

    det = MagicMock()
    det.rows, det.cols = 200, 200
    det.pixel_area = 0.01
    det.rmat, det.tvec = np.eye(3), np.zeros(3)
    det.distortion = None
    det.clip_to_panel.return_value = (None, np.array([True]))
    det.config_dict.return_value = {'mock': 'config'}
    det.angularPixelSize.return_value = 0.1
    mock_instr.detectors = {'det1': det}

    mock_mat = MagicMock()
    mock_mat.name = "TestMat"
    mock_mat.planeData.latVecOps = {'B': np.eye(3)}

    return LaueCalibrator(
        instr=mock_instr, material=mock_mat, grain_params=np.zeros(12),
        min_energy=10, max_energy=50, calibration_picks=None
    )

# --- Consolidated Basic Tests ---

def test_laue_attributes_and_setters(laue_calibrator):
    assert laue_calibrator.type == 'laue'
    np.testing.assert_array_equal(laue_calibrator.energy_cutoffs, [10, 50])
    
    mock_dist = MagicMock()
    laue_calibrator.tth_distortion = {'det1': mock_dist}
    assert laue_calibrator.tth_distortion['det1'].panel == laue_calibrator.instr.detectors['det1']

    laue_calibrator.energy_cutoffs = [20, 60]
    assert laue_calibrator.plane_data.wavelength == 60
    with pytest.raises(AssertionError):
        laue_calibrator.energy_cutoffs = [10]

# --- Autopick Logic ---

@pytest.mark.parametrize("config", [
    {"id": "success_blob"},
    {"id": "success_label", "blob": False},
    {"id": "boundary_fail", "patch_offset": -500, "expect": 0},
    {"id": "clip_fail", "clip": False, "expect": 0},
    {"id": "multi_peak", "blob_ret": [[5, 5, 1], [0, 0, 1]]},
    {"id": "fit_fail_edge", "lsq": ([10, 1, 0, 1, 0.1, 0.1, 0, 0, 0], 1), "expect": 0},
    {"id": "fit_fail_int", "lsq": ([0.1, 1, 0, 1, 5, 5, 0, 0, 0], 1), "expect": 0},
    {"id": "no_fit_mode", "fit": False},
    {"id": "no_fit_label", "fit": False, "blob": False},
    {"id": "no_peaks", "blob_ret": [], "expect": 1, "expect_nan": True},
    {"id": "distorted", "dist": True, "res_pt": [105.0, 105.0]},
])
@patch('hexrd.core.fitting.calibration.laue.nquad', return_value=(100.0, 0.1))
@patch('hexrd.core.fitting.calibration.laue.leastsq')
@patch('hexrd.core.fitting.calibration.laue.blob_log')
@patch('hexrd.core.fitting.calibration.laue.filters')
@patch('hexrd.core.fitting.calibration.laue.xrdutil')
@patch('hexrd.core.fitting.calibration.laue.xfcapi')
@patch('hexrd.core.fitting.calibration.laue.switch_xray_source')
def test_autopick_points(mock_switch, mock_xfc, mock_xrd, mock_filt, mock_blob, 
                        mock_lsq, mock_nquad, config, laue_calibrator):
    
    fit = config.get("fit", True)
    use_blob = config.get("blob", True)
    expect_count = config.get("expect", 1)
    
    mock_xfc.gvec_to_xy.return_value = np.array([[101.0, 101.0]])
    mock_filt.gaussian.return_value = np.ones((10, 10)) * 10
    if not use_blob: mock_filt.gaussian.return_value[5, 5] = 100
    
    mock_blob.return_value = np.array(config.get("blob_ret", [[5, 5, 1.0]]))
    mock_lsq.return_value = config.get("lsq", ([100, 1, 0, 1, 5.0, 5.0, 0, 0, 0], 1))

    laue_calibrator.instr.simulate_laue_pattern.return_value = {
        'det1': (np.array([[[100, 100]]]), np.array([[[1], [1], [1]]]), 
                 np.array([[[15, 45]]]), np.array([[1.0]]), np.array([[20.0]]))
    }

    row, col = np.meshgrid(np.arange(10), np.arange(10), indexing='ij')
    row += config.get("patch_offset", 0)
    mock_patch = [(np.array([[10, 11]]), np.array([[20], [21]])), 
                  (np.array([[50]]), np.array([[50]])), None, 1.0, (row, col)]
    mock_xrd.make_reflection_patches.return_value = [mock_patch]

    det = laue_calibrator.instr.detectors['det1']
    det.clip_to_panel.return_value = (None, np.array([config.get("clip", True)]))
    if config.get("dist"):
        det.distortion = MagicMock()
        det.distortion.apply_inverse.return_value = np.array(config.get("res_pt"))

    res = laue_calibrator.autopick_points({'det1': np.zeros((200, 200))}, 
                                          use_blob_detection=use_blob, fit_peaks=fit)

    mock_switch.assert_called()
    picks = res['pick_xys']['det1']
    assert len(picks) == expect_count
    
    if expect_count > 0:
        if config.get("expect_nan"):
            assert np.all(np.isnan(picks[0]))
        else:
            expected = config.get("res_pt", [101.0, 101.0])
            np.testing.assert_array_equal(picks[0], expected)

# --- Objective Function & Methods ---

@patch('hexrd.core.fitting.calibration.laue.switch_xray_source')
@patch('hexrd.core.fitting.calibration.laue.sxcal_obj_func')
def test_laue_methods_and_obj_func(mock_obj_func, mock_switch, laue_calibrator):
    laue_calibrator.data_dict = {
        'pick_xys': {'det1': [[10, 10], [np.nan, np.nan]]},
        'hkls': {'det1': [[1, 1, 1], [2, 0, 0]]}
    }

    hkls, xys = laue_calibrator._evaluate()
    assert hkls['det1'].shape == (3, 1)

    laue_calibrator.residual()
    assert mock_obj_func.call_args[0][5][0] == 5.0 # Energy scaled 0.5 * 10
    
    laue_calibrator.model()
    assert mock_obj_func.call_args[1]['sim_only'] is True

    mock_instr = laue_calibrator.instr
    mock_instr.detectors['det1'].simulate_laue_pattern.return_value = (
        [np.array([[100, 100]])], [np.zeros((1, 3))], [np.array([[10, 20]])], None, None
    )
    
    res_sim = sxcal_obj_func(np.zeros(12), mock_instr, None, 
                             {'det1': np.zeros((1, 3))}, np.eye(3), [10, 50], sim_only=True)
    assert len(res_sim['det1']) == 2 # [xy, ang]

    meas_xy = {'det1': np.array([[99.0, 99.0]])}
    res_res = sxcal_obj_func(np.zeros(12), mock_instr, meas_xy, 
                             {'det1': np.zeros((1, 3))}, np.eye(3), [10, 50], sim_only=False)
    np.testing.assert_allclose(res_res, 1.0)

# --- Math Helpers ---

def test_gaussians():
    p = [10, 0.5, 0, 0.5, 0, 0, 0, 0, 0] # Amp 10 at 0,0
    assert np.isclose(gaussian_2d(p, np.zeros((3, 3)))[0], 10.0)
    assert np.isclose(gaussian_2d_int(0, 0, *p), 10.0)