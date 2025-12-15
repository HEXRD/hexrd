import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from hexrd.core.imageseries.omega import (
    OmegaImageSeries,
    OmegaWedges,
    OmegaSeriesError,
)

# --- OmegaWedges Tests ---


def test_omegawedges_basic():
    ow = OmegaWedges(20)

    ow.addwedge(0.0, 10.0, 10)
    assert ow.nwedges == 1
    assert ow.wframes == 10

    ow.addwedge(20.0, 30.0, 10)
    assert ow.nwedges == 2
    assert ow.wframes == 20

    omegas = ow.omegas
    assert omegas.shape == (20, 2)
    assert omegas[0, 0] == 0.0
    assert omegas[0, 1] == 1.0
    assert omegas[10, 0] == 20.0

    ow.delwedge(1)
    assert ow.nwedges == 1
    assert ow.wframes == 10


def test_omegawedges_error():
    ow = OmegaWedges(20)
    ow.addwedge(0, 10, 5)
    with pytest.raises(OmegaSeriesError, match="does not match"):
        _ = ow.omegas


@patch('numpy.save')
def test_omegawedges_save(mock_save):
    ow = OmegaWedges(10)
    ow.addwedge(0, 10, 10)
    ow.save_omegas("test.npy")
    mock_save.assert_called()


# --- OmegaImageSeries Tests ---


@pytest.fixture
def mock_ims():
    ims = MagicMock()
    ims.__len__.return_value = 10
    ims.metadata = {}
    return ims


def test_omegaseries_init_error_no_meta(mock_ims):
    with pytest.raises(OmegaSeriesError, match="no omega metadata"):
        OmegaImageSeries(mock_ims)


def test_omegaseries_init_error_mismatch(mock_ims):
    mock_ims.metadata['omega'] = np.zeros((5, 2))
    with pytest.raises(OmegaSeriesError, match="omega array mismatch"):
        OmegaImageSeries(mock_ims)


def test_omegaseries_init_valid(mock_ims):
    om = np.zeros((10, 2))
    om[:, 0] = np.arange(10)
    om[:, 1] = np.arange(1, 11)
    mock_ims.metadata['omega'] = om

    ois = OmegaImageSeries(mock_ims)

    assert ois.nwedges == 1
    w = ois.wedge(0)
    assert w['ostart'] == 0.0
    assert w['ostop'] == 10.0
    assert w['delta'] == 1.0


def test_make_wedges_non_contiguous(mock_ims):
    om = np.zeros((10, 2))
    om[0:5, 0] = np.arange(5)
    om[0:5, 1] = np.arange(1, 6)
    om[5:10, 0] = np.arange(20, 25)
    om[5:10, 1] = np.arange(21, 26)

    mock_ims.metadata['omega'] = om
    ois = OmegaImageSeries(mock_ims)

    assert ois.nwedges == 2
    assert ois.wedge(0)['ostop'] == 5.0
    assert ois.wedge(1)['ostart'] == 20.0


def test_make_wedges_delta_change(mock_ims):
    om = np.zeros((10, 2))
    om[0:5, 0] = np.arange(5)
    om[0:5, 1] = np.arange(1, 6)
    om[5:10, 0] = np.arange(5, 7.5, 0.5)
    om[5:10, 1] = np.arange(5.5, 8.0, 0.5)

    mock_ims.metadata['omega'] = om
    ois = OmegaImageSeries(mock_ims)

    assert ois.nwedges == 2
    assert ois.wedge(0)['delta'] == 1.0
    assert ois.wedge(1)['delta'] == 0.5


def test_make_wedges_decreasing_error(mock_ims):
    om = np.zeros((10, 2))
    om[0, 0] = 1.0
    om[0, 1] = 0.0
    mock_ims.metadata['omega'] = om

    with pytest.raises(OmegaSeriesError, match="must be increasing"):
        OmegaImageSeries(mock_ims)


# --- Mapping Tests ---


def test_omega_to_frame(mock_ims):
    om = np.zeros((20, 2))
    om[0:10, 0] = np.arange(10)
    om[0:10, 1] = np.arange(1, 11)
    om[10:20, 0] = np.arange(20, 30)
    om[10:20, 1] = np.arange(21, 31)

    mock_ims.__len__.return_value = 20
    mock_ims.metadata['omega'] = om
    ois = OmegaImageSeries(mock_ims)

    f, w = ois.omega_to_frame(5.5)
    assert f == 5
    assert w == 0

    f, w = ois.omega_to_frame(25.5)
    assert f == 15
    assert w == 1

    f, w = ois.omega_to_frame(15.0)
    assert f == -1

    f, w = ois.omega_to_frame(365.5)
    assert f == 5


def test_omegarange_to_frames(mock_ims):
    om = np.zeros((20, 2))
    om[0:10, 0] = np.arange(350, 360)
    om[0:10, 1] = np.arange(351, 361)
    om[10:20, 0] = np.arange(0, 10)
    om[10:20, 1] = np.arange(1, 11)

    mock_ims.__len__.return_value = 20
    mock_ims.metadata['omega'] = om
    ois = OmegaImageSeries(mock_ims)

    assert ois.omegarange_to_frames(100, 101) == ()
    assert ois.omegarange_to_frames(355, 101) == ()

    res = ois.omegarange_to_frames(352.1, 354.9)
    assert res == [2, 3, 4]

    res = ois.omegarange_to_frames(358.1, 1.9)
    assert res == [8, 9, 10, 11]


def test_omegarange_gap(mock_ims):
    om = np.zeros((20, 2))
    om[0:10, 0] = np.arange(10)
    om[0:10, 1] = np.arange(1, 11)
    om[10:20, 0] = np.arange(20, 30)
    om[10:20, 1] = np.arange(21, 31)

    mock_ims.__len__.return_value = 20
    mock_ims.metadata['omega'] = om
    ois = OmegaImageSeries(mock_ims)

    res = ois.omegarange_to_frames(5.0, 25.0)
    assert res == ()
