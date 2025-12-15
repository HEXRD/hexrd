from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from hexrd.core import constants as ct
from hexrd.core.instrument.detector import (
    Detector,
    _fix_indices,
    _row_edge_vec,
    _col_edge_vec,
    _interpolate_bilinear,
    _interpolate_bilinear_in_place,
)
from hexrd.core.instrument.physics_package import AbstractPhysicsPackage
from hexrd.core.material import crystallography

# --- Concrete Implementation for Testing ---


class ConcreteDetector(Detector):
    """
    Minimal concrete implementation of the abstract Detector base class.
    Used to test shared functionality.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calibration_flags = [True] * 6

    @property
    def detector_type(self):
        return "concrete"

    @property
    def pixel_normal(self):
        n = np.zeros((self.rows * self.cols, 3))
        n[:, 2] = 1.0
        return n

    def cart_to_angles(self, xy_data, **kwargs):
        xy = np.atleast_2d(xy_data)
        return xy * 0.01, np.zeros((len(xy), 3))

    def angles_to_cart(self, tth_eta, **kwargs):
        te = np.atleast_2d(tth_eta)
        return te * 100.0

    def cart_to_dvecs(self, xy_data, **kwargs):
        xy = np.atleast_2d(xy_data)
        n = len(xy)
        dvecs = np.zeros((n, 3))
        dvecs[:, 2] = 1.0
        return dvecs

    def pixel_angles(self, origin=ct.zeros_3):
        tth = np.zeros(self.shape)
        eta = np.zeros(self.shape)
        return tth, eta

    def pixel_tth_gradient(self, origin=ct.zeros_3):
        return np.zeros(self.shape)

    def pixel_eta_gradient(self, origin=ct.zeros_3):
        return np.zeros(self.shape)

    def calc_filter_coating_transmission(self, energy):
        return np.ones(self.shape), np.ones(self.shape)

    @property
    def beam_position(self):
        return np.array([0.0, 0.0])

    @property
    def physical_size(self):
        return np.array(
            [self.rows * self.pixel_size_row, self.cols * self.pixel_size_col]
        )

    @property
    def calibration_flags(self):
        return self._calibration_flags

    @calibration_flags.setter
    def calibration_flags(self, flags):
        self._calibration_flags = flags

    def clip_to_panel(self, pts, buffer_edges=True):
        return super().clip_to_panel(pts, buffer_edges=buffer_edges)


def make_physics_pkg():
    pkg = SimpleNamespace()
    pkg.sample_material = "Si"
    pkg.window_material = "Be"
    pkg.sample_density = 2.33
    pkg.window_density = 1.85
    pkg.sample_thickness = 100.0
    pkg.window_thickness = 50.0
    pkg.sample_absorption_length = lambda E: 1000.0
    pkg.window_absorption_length = lambda E: 2000.0
    pkg.pinhole_thickness = 0.0
    pkg.pinhole_diameter = 0.0
    return pkg

def _get_pyfunc(f):
    """Return the underlying Python function if numba produced .py_func"""
    return getattr(f, "py_func", f)

# --- Fixtures ---


@pytest.fixture
def mock_distortion_registry():
    class RegisteredDistortion:
        maptype = "registered"
        params = np.array([0.0])

        def apply_inverse(self, pts):
            return pts

    with patch('hexrd.core.instrument.detector.distortion_registry') as m_dr:
        m_dr.distortion_registry = {'Registered': RegisteredDistortion}
        yield RegisteredDistortion


@pytest.fixture
def base_detector():
    return ConcreteDetector()


@pytest.fixture
def physics_pkg():
    pkg = MagicMock()
    pkg.sample_material = 'Si'
    pkg.window_material = 'Be'
    pkg.sample_density = 2.33
    pkg.window_density = 1.85
    pkg.sample_thickness = 100.0
    pkg.window_thickness = 50.0
    pkg.sample_absorption_length.return_value = 1000.0
    pkg.window_absorption_length.return_value = 2000.0
    pkg.pinhole_thickness = 0.0
    pkg.pinhole_diameter = 0.0
    return pkg


@pytest.fixture
def mock_distortion_registry():
    class RegisteredDistortion:
        maptype = "registered"
        params = np.array([0.0])

        def apply_inverse(self, pts):
            return pts

        def apply(self, pts):
            return pts + np.array([10.0, 5.0])

    with patch('hexrd.core.instrument.detector.distortion_registry') as m_dr:
        m_dr.distortion_registry = {'Registered': RegisteredDistortion}
        yield RegisteredDistortion


# --- Basic Tests ---


@patch('hexrd.core.instrument.detector.calculate_incoherent_scattering')
@patch.object(ConcreteDetector, 'calc_compton_window_transmission')
@patch.object(ConcreteDetector, 'calc_compton_physics_package_transmission')
@patch.object(ConcreteDetector, 'pixel_Q')
def test_compute_compton_scattering_intensity(
    mock_pixel_Q,
    mock_trans_pkg,
    mock_trans_win,
    mock_incoherent_scat,
    base_detector,
    physics_pkg,
):
    rows, cols = base_detector.shape

    mock_q = np.ones(rows * cols) * 0.1
    mock_pixel_Q.return_value = mock_q

    mock_incoherent_scat.side_effect = [
        np.full(rows * cols, 5.0),
        np.full(rows * cols, 2.0),
    ]

    mock_trans_pkg.return_value = np.full(base_detector.shape, 0.8)
    mock_trans_win.return_value = np.full(base_detector.shape, 0.5)

    energy = 70.0
    rMat_s = np.eye(3)

    result_intensity, result_t_s, result_t_w = (
        base_detector.compute_compton_scattering_intensity(
            energy, rMat_s, physics_pkg
        )
    )

    expected_intensity = np.full(base_detector.shape, 5.0)

    np.testing.assert_allclose(result_intensity, expected_intensity)
    assert result_intensity.shape == base_detector.shape

    np.testing.assert_allclose(result_t_s, 0.8)
    np.testing.assert_allclose(result_t_w, 0.5)


def test_init_defaults(base_detector):
    d = base_detector
    assert d.rows == 2048
    assert d.cols == 2048
    assert d.pixel_size_row == 0.2
    assert d.pixel_size_col == 0.2
    np.testing.assert_array_equal(d.tvec, [0.0, 0.0, -1000.0])
    np.testing.assert_array_equal(d.tilt, [0.0, 0.0, 0.0])
    assert d.name == 'default'


def test_get_and_set_properties(base_detector, mock_distortion_registry):
    d = base_detector

    d.rows = 1024
    d.cols = 512
    assert d.shape == (1024, 512)

    d.pixel_size_row = 0.1
    d.pixel_size_col = 0.3
    np.testing.assert_array_equal(d.physical_size, [102.4, 153.6])
    d.name = 'test_detector'
    assert d.name == 'test_detector'

    d.lmfit_name
    d.pixel_area
    d.tth_distortion
    d.tth_distortion = np.zeros(d.shape)
    assert d.tth_distortion.shape == d.shape
    with pytest.raises(
        ValueError,
        match="tth_distortion must have shape equal to detector shape",
    ):
        d.tth_distortion = np.zeros((10, 10))
        d.tth_distortion = np.zeros((d.rows, d.cols, 2))
    d.tth_distortion = None
    assert d.tth_distortion is None

    d.rows = 100
    d.cols = 200

    start_r, start_c = 10, 50
    d.roi = [start_r, start_c]

    expected_roi_r = (start_r, start_r + d.rows)
    expected_roi_c = (start_c, start_c + d.cols)

    assert d.roi == (expected_roi_r, expected_roi_c)

    d.roi = None
    assert d.roi is None

    with pytest.raises(ValueError, match="roi must be a 2-element array-like"):
        d.roi = [10, 20, 30]
        d.roi = [10]

    d.row_edge_vec
    d.col_edge_vec
    d.corner_ll
    d.corner_lr
    d.corner_ul
    d.corner_ur

    d.tvec = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(d.tvec, [1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError, match="tvec must be a 3-element array-like"
    ):
        d.tvec = np.array([1.0, 2.0])
        d.tvec = np.array([1.0, 2.0, 3.0, 4.0])

    d.tilt = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(d.tilt, [0.1, 0.2, 0.3])

    with pytest.raises(
        ValueError, match="tilt must be a 3-element array-like"
    ):
        d.tilt = np.array([0.1, 0.2])
        d.tilt = np.array([0.1, 0.2, 0.3, 0.4])

    d.bvec = np.array([0.0, 1.0, 0.0])
    np.testing.assert_array_equal(d.bvec, [0.0, 1.0, 0.0])
    with pytest.raises(
        ValueError,
        match="bvec must be a 3-element array-like with unit magnitude",
    ):
        d.bvec = np.array([0.0, 1.0])
        d.bvec = np.array([0.0, 1.0, 0.0, 0.0])

    d.xrs_dist
    d.xrs_dist = 10
    assert d.xrs_dist == 10
    with pytest.raises(ValueError, match="xrs_dist must be a scalar value"):
        d.xrs_dist = np.array([10, 20])

    d.evec
    d.evec = np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_equal(d.evec, [1.0, 0.0, 0.0])
    with pytest.raises(
        ValueError,
        match="evec must be a 3-element array-like with unit magnitude",
    ):
        d.evec = np.array([1.0, 0.0])
        d.evec = np.array([1.0, 0.0, 0.0, 0.0])
        d.evec = np.array([1.0, 1.0, 0.0])  # Not unit magnitude

    d.distortion = None
    assert d.distortion is None
    d.distortion = mock_distortion_registry()
    assert isinstance(d.distortion, mock_distortion_registry)
    with pytest.raises(
        TypeError, match="Input distortion is not in registry"
    ):
        d.distortion = "invalid_distortion"

    d.normal
    d.pixel_coords
    d.pixel_solid_angles


def test_config_dict(base_detector, mock_distortion_registry):
    base_detector.saturation_level = 1000.0
    base_detector.group = 'group1'
    conf = base_detector.config_dict()
    assert conf['detector']['detector_type'] == "concrete"
    assert 'roi' not in conf['detector']['pixels']

    d_roi = ConcreteDetector(rows=2048, cols=2048, roi=[10, 20])
    d_roi.saturation_level = 1000.0
    conf_roi = d_roi.config_dict(style='yaml')

    expected_roi_flat = [10, 20]

    assert 'roi' in conf_roi['detector']['pixels']
    assert conf_roi['detector']['pixels']['roi'] == expected_roi_flat

    base_detector.distortion = mock_distortion_registry()
    conf_dist = base_detector.config_dict()
    assert 'distortion' in conf_dist['detector']

    base_detector.config_dict(panel_buffer=None)
    base_detector.config_dict(panel_buffer=np.ones(2, dtype=bool))
    base_detector.config_dict(panel_buffer=np.ones((2048, 2048), dtype=bool))

    with pytest.raises(
        ValueError,
        match="panel_buffer must be a 2-element array-like or a 2-d array",
    ):
        base_detector.config_dict(panel_buffer=np.ones(3))

    with pytest.raises(
        ValueError,
        match="panel_buffer must be a 2-element array-like or a 2-d array",
    ):
        base_detector.config_dict(panel_buffer=np.zeros((3, 3, 3)))


def test_config_dict_hdf5(base_detector):
    base_detector.saturation_level = 1000.0
    conf = base_detector.config_dict(style='hdf5')
    assert isinstance(conf, dict)


# --- Coordinate Conversions ---


def test_cartToPixel(base_detector, mock_distortion_registry):
    d = base_detector
    xy = np.array([[0.0, 0.0]])

    ij = d.cartToPixel(xy)
    assert np.allclose(ij, [[1023.5, 1023.5]])

    d.distortion = mock_distortion_registry()
    ij_dist = d.cartToPixel(xy, apply_distortion=True)

    assert np.allclose(ij_dist, [[998.5, 1073.5]])

    ij_int = d.cartToPixel(xy, pixels=True)
    assert np.allclose(ij_int, [[1024, 1024]])


def test_pixelToCart(base_detector):
    d = base_detector
    ij = np.array([[1023.5, 1023.5]])
    xy = d.pixelToCart(ij)
    assert np.allclose(xy, [[0.0, 0.0]])


# --- Interpolation ---


def test_interpolate_nearest(base_detector):
    d = base_detector
    img = np.ones((d.rows, d.cols))

    xy = np.array([[0.0, 0.0]])
    val = d.interpolate_nearest(xy, img)
    assert val[0] == 1.0

    xy_out = np.array([[10000.0, 10000.0]])
    val_out = d.interpolate_nearest(xy_out, img)
    assert np.isnan(val_out[0])

    val_out = d.interpolate_nearest(xy_out, img, pad_with_nans=False)
    assert val_out[0] == 0.0


def test_interpolate_bilinear(base_detector):
    d = base_detector
    img = np.ones((d.rows, d.cols))

    xy = np.array([[0.0, 0.0]])
    val = d.interpolate_bilinear(xy, img)
    assert val[0] == 1.0


# --- Physics Calculations ---


def test_pixel_Q(base_detector):
    q = base_detector.pixel_Q(energy=65.0)
    assert np.all(q == 0.0)


def test_pixel_compton_energy_loss(base_detector):
    e = base_detector.pixel_compton_energy_loss(energy=50.0)
    assert np.all(e == 50.0)


def test_pixel_compton_attenuation_length(base_detector):
    with patch(
        'hexrd.core.instrument.detector.calculate_linear_absorption_length'
    ) as m_calc:
        num_pixels = base_detector.rows * base_detector.cols
        m_calc.return_value = np.zeros(num_pixels)

        res = base_detector.pixel_compton_attenuation_length(
            energy=50.0, density=1.0, formula='Si'
        )
        assert res.shape == base_detector.shape


def test_calc_physics_package_transmission(base_detector, physics_pkg):
    with patch('hexrd.core.instrument.detector.angles_to_dvec') as m_a2d:
        dvecs = np.zeros((base_detector.rows * base_detector.cols, 3))
        dvecs[:, 2] = -1.0
        m_a2d.return_value = dvecs

        with np.errstate(divide='ignore', invalid='ignore'):
            trans = base_detector.calc_physics_package_transmission(
                energy=50.0, rMat_s=np.eye(3), physics_package=physics_pkg
            )
        assert trans.shape == base_detector.shape


# --- Simulation ---


def test_map_to_plane(base_detector):
    pts = np.array([[0.0, 0.0]])
    rmat_p = np.eye(3)
    tvec_p = np.array([0, 0, 0])

    res = base_detector.map_to_plane(pts, rmat_p, tvec_p)
    assert res.shape == (1, 2)


# --- Utilities ---


def test_utility_functions():
    idx = np.array([-1, 0, 5, 6])
    fixed = _fix_indices(idx, 0, 5)
    np.testing.assert_array_equal(fixed, [0, 0, 5, 5])

    edges = _row_edge_vec(10, 0.1)
    assert len(edges) == 11


def test_make_powder_rings_simple(base_detector):
    tth_list = [0.1, 0.2]

    with pytest.raises(RuntimeError, match="must supply a delta_tth"):
        base_detector.make_powder_rings(tth_list)

    mock_dist = MagicMock()
    mock_dist.apply.side_effect = lambda x, return_nominal=False: x

    tvec_nonzero = np.array([1.0, 0.0, 0.0])
    base_detector.make_powder_rings(
        tth_list, delta_tth=0.01, tth_distortion=mock_dist, tvec_s=tvec_nonzero
    )

    base_detector.make_powder_rings(
        tth_list, delta_tth=0.01, tth_distortion=mock_dist
    )
    assert mock_dist.apply.called

    class FakePlaneData:
        def __init__(self, *args):
            self.tThWidth = 0.01
            self.wavelength = 0.5

        def getTTh(self):
            return np.array([0.1, 0.2])

        def getTThRanges(self):
            return np.array([[0.095, 0.105], [0.195, 0.205]])

        def getMergedRanges(self, cullDupl=True):
            return None, np.array([[0.095, 0.105]])

    with patch('hexrd.core.instrument.detector.PlaneData', FakePlaneData):
        pd = FakePlaneData()

        base_detector.make_powder_rings(pd)
        base_detector.make_powder_rings(pd, merge_hkls=True)
        base_detector.make_powder_rings(pd, delta_tth=0.5)

    base_detector.make_powder_rings(
        tth_list, delta_tth=0.01, eta_list=[0, 90, 180, 270]
    )
    base_detector.make_powder_rings(tth_list, delta_tth=0.01, full_output=True)


def test_angularPixelSize(base_detector):
    with patch('hexrd.core.instrument.detector.xrdutil') as m_xrd:
        m_xrd.angularPixelSize.return_value = np.array([[0.1, 0.1]])
        res = base_detector.angularPixelSize([[0, 0]])
        assert res.shape == (1, 2)


def test_simulate_rotation_series(base_detector):
    pd = MagicMock()
    pd.latVecOps = {'B': np.eye(3)}
    pd.wavelength = 0.5
    pd.getTTh.return_value = np.array([0.1])

    grain_params = [np.zeros(12)]

    with patch('hexrd.core.instrument.detector.xrdutil') as m_xrd:
        m_xrd._fetch_hkls_from_planedata.return_value = np.zeros((1, 4))
        m_xrd.apply_correction_to_wavelength.return_value = 0.5
        m_xrd._filter_hkls_eta_ome.return_value = (
            np.zeros((1, 3)),
            np.zeros((1, 4)),
        )

        m_xrd._project_on_detector_plane.return_value = (
            np.array([[0.0, 0.0]]),
            np.eye(3),
            np.array([True]),
        )

        base_detector.simulate_rotation_series(pd, grain_params)

        base_detector.simulate_rotation_series(
            pd, grain_params, wavelength=0.6
        )


def test_simulate_laue_pattern(base_detector, mock_distortion_registry):
    base_detector.distortion = mock_distortion_registry()

    pd = MagicMock(spec=crystallography.PlaneData)
    pd.getSymHKLs.return_value = [np.array([[1], [0], [0]])]
    pd.latVecOps = {'B': np.eye(3)}

    with patch('hexrd.core.instrument.detector.xy_to_gvec') as m_xy2g:
        m_xy2g.return_value = ([np.array([0.1, 0.0])], np.array([[0, 0, 1]]))

        with patch(
            'hexrd.core.instrument.detector.gvec_to_xy',
            return_value=np.array([[0.0, 0.0]]),
        ):
            with np.errstate(divide='ignore', invalid='ignore'):
                base_detector.simulate_laue_pattern(
                    pd, grain_params=[np.zeros(12)]
                )

                with pytest.raises(
                    ValueError,
                    match="minEnergy must be array-like if maxEnergy is",
                ):
                    base_detector.simulate_laue_pattern(pd, maxEnergy=[1, 2])

                with pytest.raises(
                    ValueError,
                    match="maxEnergy and minEnergy must be same length",
                ):
                    base_detector.simulate_laue_pattern(
                        pd, minEnergy=[1], maxEnergy=[1, 2]
                    )

                base_detector.simulate_laue_pattern(
                    pd, minEnergy=[5.0, 10.0], maxEnergy=[15.0, 20.0]
                )


def test_simulate_laue_pattern_legacy_input(base_detector):
    hkls = np.array([[1], [0], [0]])
    bmat = np.eye(3)

    with patch(
        'hexrd.core.instrument.detector.gvec_to_xy',
        return_value=np.array([[0.0, 0.0]]),
    ):
        with patch('hexrd.core.instrument.detector.xy_to_gvec') as m_xy2g:
            m_xy2g.return_value = (
                [np.array([0.1, 0.0])],
                np.array([[0, 0, 1]]),
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                base_detector.simulate_laue_pattern((hkls, bmat))


def test_simulate_laue_pattern_error(base_detector):
    with pytest.raises(RuntimeError):
        base_detector.simulate_laue_pattern("invalid_input")


def test_polarization_factor(base_detector):
    _ = base_detector.polarization_factor(f_hor=1.0, f_vert=0.0)

    with pytest.raises(RuntimeError):
        base_detector.polarization_factor(f_hor=0.5, f_vert=0.0)

    with pytest.raises(RuntimeError):
        base_detector.polarization_factor(f_hor=-0.1, f_vert=1.1)


def test_lorentz_factor(base_detector):
    with np.errstate(divide='ignore', invalid='ignore'):
        lf = base_detector.lorentz_factor()
        assert lf.shape == base_detector.shape


def test_clip_to_panel_buffer_logic(base_detector):
    base_detector.panel_buffer = np.array([1.0, 1.0])
    pts = np.array([[0.0, 0.0], [1000.0, 1000.0]])

    cl, mk = base_detector.clip_to_panel(pts)
    assert len(cl) == 1
    assert mk[0] == True
    assert mk[1] == False

    mask = np.ones((base_detector.rows, base_detector.cols), dtype=bool)

    ij = base_detector.cartToPixel([[0.0, 0.0]], pixels=True)[0]
    mask[ij[0], ij[1]] = False

    base_detector.panel_buffer = mask
    _, mk_2d = base_detector.clip_to_panel(pts)
    assert mk_2d[0] == False
    assert mk_2d[1] == False

    pts_valid = np.array([[0.5, 0.5]])
    _, mk_valid = base_detector.clip_to_panel(pts_valid)
    assert mk_valid[0] == True


def test_clip_to_panel_invalid_buffer(base_detector):
    with pytest.raises(AssertionError):
        base_detector.panel_buffer = np.zeros((3, 3, 3))


def test_calc_compton_physics_package_transmission(physics_pkg):
    d = ConcreteDetector(rows=4, cols=3)
    d.bvec = np.array([0.0, 0.0, 1.0])
    rMat_s = np.eye(3)
    energy = 50.0

    n_pixels = d.rows * d.cols

    first = np.array([0.0, 0.0, -1.0])
    rest = np.array([0.0, 0.0, 1.0])
    dvecs = np.vstack([first] + [rest] * (n_pixels - 1))

    with patch('hexrd.core.instrument.detector.angles_to_dvec') as m_a2d:
        m_a2d.return_value = dvecs

        def fake_calc_sample(
            seca, secb, energy_in, physics_package, kind='sample'
        ):
            secb = np.asarray(secb)
            out = np.where(np.isnan(secb), np.nan, 2.0 * np.ones_like(secb))
            return out

        def fake_calc_window(secb, energy_in, physics_package):
            secb = np.asarray(secb)
            out = np.where(np.isnan(secb), np.nan, 0.5 * np.ones_like(secb))
            return out

        with patch.object(
            d, 'calc_compton_transmission', side_effect=fake_calc_sample
        ) as m_sample:
            with patch.object(
                d,
                'calc_compton_transmission_window',
                side_effect=fake_calc_window,
            ) as m_window:
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = d.calc_compton_physics_package_transmission(
                        energy, rMat_s, physics_pkg
                    )

                assert result.shape == d.shape

                flat = result.flatten()
                assert np.isnan(flat[0])
                assert np.allclose(flat[1:], 1.0, equal_nan=False)


def test_calc_compton_window_transmission_masks_and_combines():
    """Ensure masking of backscattered beams becomes NaN and that the product
    T_sample * T_window is returned using the (already-masked) secb passed."""
    d = ConcreteDetector(rows=4, cols=3)
    d.bvec = np.array([0.0, 0.0, 1.0])

    pkg = make_physics_pkg()

    n_pix = d.rows * d.cols
    first = np.array([0.0, 0.0, -1.0])
    rest = np.array([0.0, 0.0, 1.0])
    dvecs = np.vstack([first] + [rest] * (n_pix - 1))

    with patch("hexrd.core.instrument.detector.angles_to_dvec") as m_a2d:
        m_a2d.return_value = dvecs

        def fake_sample(seca, energy_in, physics_package):
            return 2.0

        def fake_window(
            seca, secb, energy_in, physics_package, *args, **kwargs
        ):
            secb_arr = np.asarray(secb)
            return np.where(
                np.isnan(secb_arr), np.nan, 0.5 * np.ones_like(secb_arr)
            )

        with patch.object(
            d, "calc_compton_transmission_sample", side_effect=fake_sample
        ):
            with patch.object(
                d, "calc_compton_transmission", side_effect=fake_window
            ):
                res = d.calc_compton_window_transmission(
                    energy=50.0, rMat_s=np.eye(3), physics_package=pkg
                )

                assert res.shape == d.shape

                flat = res.flatten()
                assert np.isnan(flat[0])

                assert np.allclose(flat[1:], 1.0, equal_nan=False)


def test_calc_compton_transmission_sample_and_window_branches():
    d = ConcreteDetector(rows=3, cols=2)
    pkg = make_physics_pkg()

    n_pix = d.rows * d.cols
    seca = np.ones(n_pix)
    secb = np.ones(n_pix)

    with patch.object(
        d,
        "pixel_compton_attenuation_length",
        return_value=np.full(n_pix, 2000.0),
    ):
        out_sample = d.calc_compton_transmission(
            seca, secb, energy=20.0, physics_package=pkg, pp_layer="sample"
        )
        assert out_sample.shape == (n_pix,)
        assert np.all(np.isfinite(out_sample))

    pkg2 = make_physics_pkg()
    pkg2.window_material = None
    out_window_none = d.calc_compton_transmission(
        seca, secb, energy=20.0, physics_package=pkg2, pp_layer="window"
    )
    assert np.all(out_window_none == 1.0)

    pkg3 = make_physics_pkg()
    pkg3.window_material = "Be"
    pkg3.window_thickness = 0.0
    out_window_zero = d.calc_compton_transmission(
        seca, secb, energy=20.0, physics_package=pkg3, pp_layer="window"
    )
    assert np.all(out_window_zero == 1.0)


def test_calc_compton_transmission_sample_function_matches_exp_formula():
    d = ConcreteDetector(rows=2, cols=2)
    pkg = make_physics_pkg()
    n_pix = d.rows * d.cols
    seca = np.full(n_pix, 2.0)
    out = d.calc_compton_transmission_sample(
        seca, energy=30.0, physics_package=pkg
    )

    mu_s = 1.0 / pkg.sample_absorption_length(30.0)
    expected_flat = np.exp(-mu_s * pkg.sample_thickness * seca)
    np.testing.assert_allclose(out, expected_flat)


def test_calc_compton_transmission_window_behavior():
    d = ConcreteDetector(rows=2, cols=2)
    pkg = make_physics_pkg()
    pkg.window_material = None
    out = d.calc_compton_transmission_window(
        np.ones(d.rows * d.cols), energy=30.0, physics_package=pkg
    )
    assert np.all(out == 1.0)

    pkg.window_material = "Be"
    pkg.window_thickness = 10.0
    pkg.window_density = 1.85
    with patch.object(
        d,
        "pixel_compton_attenuation_length",
        return_value=np.full(d.shape, 1000.0),
    ):
        secb = np.full(d.shape, 1.5)
        out2 = d.calc_compton_transmission_window(
            secb, energy=30.0, physics_package=pkg
        )
        mu_w_prime = 1.0 / 1000.0
        expected = np.exp(-mu_w_prime * pkg.window_thickness * secb)
        np.testing.assert_allclose(out2, expected)


def test_calc_effective_pinhole_area_zero_and_nonzero():
    d = ConcreteDetector(rows=2, cols=2)
    pkg = make_physics_pkg()

    pkg.pinhole_thickness = 0.0
    pkg.pinhole_diameter = 0.0
    out_ones = d.calc_effective_pinhole_area(pkg)
    assert np.all(out_ones == 1.0)

    pkg.pinhole_thickness = 1.0
    pkg.pinhole_diameter = 1000.0
    with patch.object(
        d, "pixel_angles", return_value=(np.zeros(d.shape), np.zeros(d.shape))
    ):
        out = d.calc_effective_pinhole_area(pkg)
        assert np.allclose(out, 0.5 * (np.pi / 2))


def test_calc_transmission_generic_and_phosphor():
    d = ConcreteDetector(rows=2, cols=3)
    secb = np.ones(d.rows * d.cols).reshape(d.shape)
    out1 = d.calc_transmission_generic(
        secb, thickness=0.0, absorption_length=100.0
    )
    assert np.all(out1 == 1.0)

    thickness = 2.0
    absorption_length = 100.0
    out2 = d.calc_transmission_generic(
        secb, thickness=thickness, absorption_length=absorption_length
    )
    mu = 1.0 / absorption_length
    expected = np.exp(-thickness * mu * secb)
    np.testing.assert_allclose(out2, expected)

    out3 = d.calc_transmission_phosphor(
        secb,
        thickness=0.0,
        readout_length=1.0,
        absorption_length=10.0,
        energy=10.0,
        pre_U0=0.1,
    )
    assert np.all(out3 == 1.0)

    thickness = 1.0
    readout_length = 1.0
    absorption_length = 2.0
    energy = 10.0
    pre_U0 = 0.1
    secb_arr = np.ones(d.rows * d.cols).reshape(d.shape)
    out4 = d.calc_transmission_phosphor(
        secb_arr, thickness, readout_length, absorption_length, energy, pre_U0
    )
    f1 = absorption_length * thickness
    f2 = absorption_length * readout_length
    arg = secb_arr + 1.0 / f2
    expected4 = pre_U0 * energy * ((1.0 - np.exp(-f1 * arg)) / arg)
    np.testing.assert_allclose(out4, expected4)

    physics_package=make_physics_pkg()
    physics_package.sample_thickness = 0.0
    physics_package.window_thickness = 0.0

    d.calc_transmission_sample(
        secb, secb, 20, physics_package
    )
    d.calc_transmission_window(
        secb, 20, physics_package
    )

def test_interpolate_bilinear_single_and_multiple_points():
    img = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]], dtype=float)

    cc = np.array([0.25])
    fc = np.array([0.25])
    cf = np.array([0.25])
    ff = np.array([0.25])

    i_floor = np.array([0], dtype=np.int64)
    j_floor = np.array([0], dtype=np.int64)
    i_ceil = np.array([1], dtype=np.int64)
    j_ceil = np.array([1], dtype=np.int64)

    res = _interpolate_bilinear(img, cc, fc, cf, ff, i_floor, j_floor, i_ceil, j_ceil)
    assert res.shape == (1,)
    assert pytest.approx(res[0], rel=1e-12) == 3.0  # (1+2+4+5)/4

    cc2 = np.array([0.4, 0.1])
    fc2 = np.array([0.1, 0.2])
    cf2 = np.array([0.4, 0.3])
    ff2 = np.array([0.1, 0.4])

    i_floor2 = np.array([0, 1], dtype=np.int64)
    j_floor2 = np.array([1, 1], dtype=np.int64)
    i_ceil2 = np.array([1, 2], dtype=np.int64)
    j_ceil2 = np.array([2, 2], dtype=np.int64)

    res2 = _interpolate_bilinear(img, cc2, fc2, cf2, ff2, i_floor2, j_floor2, i_ceil2, j_ceil2)
    assert res2.shape == (2,)

    expected = np.array([3.7, 7.7], dtype=float)
    np.testing.assert_allclose(res2, expected, rtol=1e-12, atol=0)


def test_interpolate_bilinear_in_place_dtype_and_inplace_behavior():
    """Call the in-place function directly and verify dtype preservation and accumulation.

    This also verifies that output_img is written to (in-place) and that if img is integer
    the output uses img.dtype (so values will be cast/truncated accordingly).
    """
    img_int = np.array([[10, 20],
                        [30, 40]], dtype=np.int64)

    cc = np.array([0.25])
    fc = np.array([0.25])
    cf = np.array([0.25])
    ff = np.array([0.25])

    i_floor = np.array([0], dtype=np.int64)
    j_floor = np.array([0], dtype=np.int64)
    i_ceil = np.array([1], dtype=np.int64)
    j_ceil = np.array([1], dtype=np.int64)

    on_panel_idx = np.array([0], dtype=np.int64)
    output = np.zeros(on_panel_idx.shape[0], dtype=img_int.dtype)

    in_place_py = _get_pyfunc(_interpolate_bilinear_in_place)
    in_place_py(img_int, cc, fc, cf, ff, i_floor, j_floor, i_ceil, j_ceil, on_panel_idx, output)

    assert output.shape == (1,)
    assert output.dtype == img_int.dtype
    assert output[0] == 25

    in_place_py(img_int, cc, fc, cf, ff, i_floor, j_floor, i_ceil, j_ceil, on_panel_idx, output)
    assert output[0] == 50