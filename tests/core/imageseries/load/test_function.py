import pytest
import numpy as np
from hexrd.core.imageseries.load.function import FunctionImageSeriesAdapter

# --- Fixtures ---


@pytest.fixture
def frame_generator():
    """A simple function that generates frames based on index."""

    def _gen(i):
        return np.full((10, 10), i, dtype=np.float32)

    return _gen


# --- Tests ---


def test_function_adapter_init(frame_generator):
    """Test initialization and property extraction."""
    meta = {'pixel_size': 0.1}
    adapter = FunctionImageSeriesAdapter(
        None, func=frame_generator, num_frames=5, meta=meta
    )

    assert len(adapter) == 5
    assert adapter.shape == (10, 10)
    assert adapter.dtype == np.float32
    assert adapter.metadata == meta


def test_function_adapter_access(frame_generator):
    """Test __getitem__ behavior."""
    adapter = FunctionImageSeriesAdapter(
        None, func=frame_generator, num_frames=5
    )

    frame_2 = adapter[2]
    np.testing.assert_array_equal(
        frame_2, np.full((10, 10), 2.0, dtype=np.float32)
    )

    with pytest.raises(Exception, match="Only int keys are supported"):
        _ = adapter[0:2]

    with pytest.raises(Exception, match="Only int keys are supported"):
        _ = adapter[(0, 0)]


def test_function_adapter_iteration(frame_generator):
    """Test iterator protocol."""
    adapter = FunctionImageSeriesAdapter(
        None, func=frame_generator, num_frames=3
    )

    # Consume iterator
    frames = list(adapter)

    assert len(frames) == 3
    assert frames[0][0, 0] == 0.0
    assert frames[2][0, 0] == 2.0


def test_function_adapter_missing_args():
    """Ensure it raises KeyError if required kwargs are missing."""

    def dummy(i):
        return np.zeros((1, 1))

    with pytest.raises(KeyError):
        FunctionImageSeriesAdapter(None, num_frames=10)

    with pytest.raises(KeyError):
        FunctionImageSeriesAdapter(None, func=dummy)
