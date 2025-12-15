import pytest
import numpy as np
from hexrd.core.imageseries.load.array import ArrayImageSeriesAdapter


@pytest.fixture
def data_3d():
    return np.arange(100).reshape(2, 5, 10)


def test_array_adapter(data_3d):
    adapter = ArrayImageSeriesAdapter(None, data=data_3d, meta={'key': 'val'})
    assert len(adapter) == 2
    assert adapter.shape == (5, 10)
    assert adapter.dtype == data_3d.dtype
    assert adapter.metadata['key'] == 'val'

    frame = adapter[0]
    np.testing.assert_array_equal(frame, data_3d[0])
    frame[0, 0] = 999
    assert adapter[0][0, 0] != 999
    assert len(list(adapter)) == 2

    adapter_2d = ArrayImageSeriesAdapter(None, data=np.ones((5, 5)))
    assert len(adapter_2d) == 1
    assert adapter_2d._data.ndim == 3

    with pytest.raises(RuntimeError):
        ArrayImageSeriesAdapter(None, data=np.zeros((2, 2, 2, 2)))
