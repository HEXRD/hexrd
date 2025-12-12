import pickle

import numpy as np
import h5py
import pytest

from hexrd.core.imageseries.load.hdf5 import HDF5ImageSeriesAdapter


# -----------------------
# Helpers
# -----------------------
def make_h5_file_2d(path, group='entry', dset='images', shape=(10, 8)):
    """Create an HDF5 file with a group and a 2-D dataset and return filename."""
    fname = str(path)
    with h5py.File(fname, 'w') as f:
        g = f.create_group(group)
        data = np.arange(shape[0] * shape[1], dtype=np.int32).reshape(shape)
        g.create_dataset(dset, data=data)
        g.attrs['a'] = 1
        g.attrs['b'] = "text"
    return fname, np.arange(shape[0] * shape[1], dtype=np.int32).reshape(shape)


def make_h5_file_3d(path, group='entry', dset='images', shape=(5, 7, 6)):
    fname = str(path)
    with h5py.File(fname, 'w') as f:
        g = f.create_group(group)
        data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        g.create_dataset(dset, data=data)
        g.attrs['meta'] = 123
    return fname, data


# -----------------------
# 2-D dataset tests
# -----------------------
def test_2d_adapter_basic(tmp_path):
    fname, arr = make_h5_file_2d(str(tmp_path / "h2d.h5"))
    adapter = HDF5ImageSeriesAdapter(fname, path='entry')

    assert adapter.metadata['a'] == 1
    assert adapter.shape == arr.shape
    assert len(adapter) == 1

    np.testing.assert_array_equal(adapter[0], arr)
    with pytest.raises(IndexError): _ = adapter[1]
    with pytest.raises(IndexError): adapter.get_region(0, ((1, 3), (2, 5)))

    assert len(list(adapter)) == 1

    reloaded = pickle.loads(pickle.dumps(adapter))
    assert reloaded._ndim == 2
    assert reloaded.shape == arr.shape
    reloaded.close()
    
    adapter.close()

    with h5py.File(fname, 'r') as f:
        a2 = HDF5ImageSeriesAdapter(f, path='entry')
        np.testing.assert_array_equal(a2[0], arr)

def test_2d_adapter_del_warns_on_close_error(tmp_path, monkeypatch):
    p = tmp_path / "h2d_warn.h5"
    fname, _ = make_h5_file_2d(str(p))

    adapter = HDF5ImageSeriesAdapter(fname, path='entry')
    
    def bad_close():
        raise RuntimeError("boom")
    adapter.close = bad_close

    with pytest.warns(UserWarning, match="could not close"):
        adapter.__del__()


# -----------------------
# 3-D dataset tests
# -----------------------
def test_3d_adapter_indexing_and_shape(tmp_path):
    fname, data = make_h5_file_3d(str(tmp_path / "h3d.h5"), shape=(5, 4, 3))
    adapter = HDF5ImageSeriesAdapter(fname, path='entry')

    assert len(adapter) == 5
    assert adapter.shape == (4, 3)
    assert adapter.dtype == data.dtype

    np.testing.assert_array_equal(adapter[2], data[2])
    assert adapter[(1, 2, 1)] == data[1, 2, 1]

    region = ((0, 2), (0, 2))
    np.testing.assert_array_equal(adapter.get_region(1, region), data[1, :2, :2])
    
    adapter.close()


def test_invalid_ndim_raises(tmp_path):
    fname = str(tmp_path / "bad.h5")
    with h5py.File(fname, 'w') as f:
        g = f.create_group('entry')
        arr = np.zeros((2, 2, 2, 2))
        g.create_dataset('images', data=arr)

    with pytest.raises(RuntimeError):
        HDF5ImageSeriesAdapter(fname, path='entry')


# -----------------------
# Pickle round-trip (uses __getstate__/__setstate__)
# -----------------------
def test_pickle_roundtrip(tmp_path):
    fname, _ = make_h5_file_2d(str(tmp_path / "hpick.h5"))
    adapter = HDF5ImageSeriesAdapter(fname, path='entry')

    reloaded = pickle.loads(pickle.dumps(adapter))

    assert reloaded.dtype == adapter.dtype
    assert reloaded.shape == adapter.shape
    
    reloaded.close()
    adapter.close()