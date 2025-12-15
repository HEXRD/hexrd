import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from hexrd.core import constants as ct
from hexrd.core.instrument.detector import Detector

# --- Concrete Implementation for Testing ---


class ConcreteDetector(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calibration_flags = [True] * 6

    @property
    def detector_type(self):
        return "concrete"

    def cart_to_angles(self, xy_data, **kwargs):
        return np.zeros_like(xy_data), np.zeros((len(xy_data), 3))

    def angles_to_cart(self, tth_eta, **kwargs):
        return np.zeros_like(tth_eta)

    def cart_to_dvecs(self, xy_data, **kwargs):
        return np.zeros((len(xy_data), 3))

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

    def clip_to_panel(self, pts, buffer=None):
        return super().clip_to_panel(pts, buffer)


# --- Fixtures ---


@pytest.fixture
def mock_distortion_registry():
    class RegisteredDistortion:
        maptype = "registered"
        params = np.array([0.0])

    with patch('hexrd.core.instrument.detector.distortion_registry') as m_dr:
        m_dr.distortion_registry = {'Registered': RegisteredDistortion}
        yield RegisteredDistortion


@pytest.fixture
def base_detector():
    return ConcreteDetector()


# --- Initialization Tests ---


def test_init_defaults(base_detector):
    d = base_detector
    assert d.rows == 2048
    assert d.cols == 2048
    assert d.pixel_size_row == 0.2
    assert d.pixel_size_col == 0.2
    np.testing.assert_array_equal(d.tvec, [0.0, 0.0, -1000.0])
    np.testing.assert_array_equal(d.tilt, [0.0, 0.0, 0.0])
    assert d.name == 'default'


def test_init_custom():
    d = ConcreteDetector(
        rows=100,
        cols=200,
        pixel_size=(0.5, 0.1),
        tvec=[10, 20, 30],
        tilt=[1, 2, 3],
        name="custom_det",
    )
    assert d.rows == 100
    assert d.cols == 200
    assert d.pixel_size_row == 0.5
    assert d.pixel_size_col == 0.1
    np.testing.assert_array_equal(d.tvec, [10, 20, 30])
    np.testing.assert_array_equal(d.tilt, [1, 2, 3])
    assert d.name == "custom_det"


# --- Property Tests ---


def test_physical_properties(base_detector):
    d = base_detector
    assert np.isclose(d.physical_size[0], 409.6)
    assert np.isclose(d.physical_size[1], 409.6)


def test_calibration_flags(base_detector):
    flags = base_detector.calibration_flags
    assert len(flags) >= 6
    base_detector.calibration_flags = [False] * len(flags)
    assert not any(base_detector.calibration_flags)


# --- Pixel Coordinate Tests ---


def test_pixel_coords_generation():
    rows, cols = 10, 20
    p_row, p_col = 1.0, 2.0
    d = ConcreteDetector(rows=rows, cols=cols, pixel_size=(p_row, p_col))

    coords = d.pixel_coords
    assert len(coords) == 2
    py, px = coords

    assert py.shape == (rows, cols)
    assert px.shape == (rows, cols)

    assert np.isclose(px[0, 1] - px[0, 0], p_col)
    assert np.isclose(abs(py[1, 0] - py[0, 0]), p_row)


# --- Distortion Handling ---


def test_distortion_property(mock_distortion_registry):
    d = ConcreteDetector()
    ValidDistortion = mock_distortion_registry

    assert d.distortion is None

    valid_dist = ValidDistortion()
    d.distortion = valid_dist
    assert d.distortion is valid_dist

    class InvalidDistortion:
        pass

    with pytest.raises(TypeError, match='Input distortion is not in registry'):
        d.distortion = InvalidDistortion()

    d.distortion = None
    assert d.distortion is None


# --- Configuration Serialization ---


def test_config_dict():
    d = ConcreteDetector(
        rows=100, cols=200, pixel_size=(0.1, 0.2), name="test_conf"
    )
    d.saturation_level = 1000.0

    conf = d.config_dict()

    assert isinstance(conf, dict)
    assert conf['detector']['detector_type'] == "concrete"
    assert conf['detector']['pixels']['rows'] == 100
    assert conf['detector']['pixels']['columns'] == 200


def test_config_dict_yaml_style():
    d = ConcreteDetector()
    d.saturation_level = 1000.0

    conf = d.config_dict(style='yaml')
    assert 'detector_type' in conf['detector'].keys()
    assert isinstance(conf['detector']['transform']['tilt'], list)


# --- Geometry / Clipping ---


def test_clip_to_panel():
    d = ConcreteDetector(rows=100, cols=100, pixel_size=(0.1, 0.1))

    pts_in = np.array([[0.0, 0.0], [1.0, 1.0]])
    pts_out = np.array([[100.0, 100.0]])
    pts = np.vstack([pts_in, pts_out])

    _, mask = d.clip_to_panel(pts)

    assert len(mask) == 3
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == False


# --- Memoization Utilities ---


def test_update_memoization_sizes():
    panels = [ConcreteDetector(), ConcreteDetector()]
    Detector.update_memoization_sizes(panels)
