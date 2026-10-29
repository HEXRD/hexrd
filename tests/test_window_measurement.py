"""Shared window measurement keeps reads bounded and preserves omega slots."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.core.extensions import transforms as native
from hexrd.hedm._image_cache import BoundedSeries, FrameCache, StreamingView, View
from hexrd.hedm._staged_spots import Window, measure_windows


@pytest.mark.parametrize('dtype', ['O', '>f8', 'f2'])
def test_unsupported_dtype_uses_reference_extraction_without_reading(dtype):
    reader = SimpleNamespace(shape=(1, 1), dtype=np.dtype(dtype), nframes=1)
    assert BoundedSeries(View(reader), FrameCache(0), {}, 1).native_view() is None


@pytest.mark.parametrize('points', [40000, 80000])
def test_large_frames_use_bounded_tiles_and_preserve_omega_slots(points):
    calls = []

    class Reader:
        shape, dtype, nframes = (1, points), np.dtype('f4'), 2

        def read_positions(self, frames, cache):
            return -frames

        def sample_pixels(self, frame, rows, cols, cache):
            assert 0 < len(rows) <= 65536
            calls.append((frame, len(rows)))
            return (1 + cols % 17 + 20 * frame).astype('f4')

    cache = FrameCache(0)
    shapes = np.array([[2, 1, points], [2, 1, points]])
    windows, stats, _ = Window.capture(
        StreamingView(View(Reader(), dense=True), cache),
        np.array([[0, 1], [0, 1]]),
        np.array([0, points, 2 * points]),
        np.zeros(2 * points, dtype='i8'),
        np.tile(np.arange(points), 2),
        shapes,
        0,
    )
    assert measure_windows(windows, cache) == 2
    assert calls[0][0] == 1
    assert sum(n for _, n in calls) == 4 * points
    cols = np.arange(points)
    pixels = np.r_[(1 + cols % 17).astype('f4'), (21 + cols % 17).astype('f4')]
    expected = native._spotStats(pixels, np.array([0, len(pixels)]), shapes[:1], 0)
    for actual, wanted in zip(stats, expected):
        np.testing.assert_array_equal(actual, np.repeat(wanted, 2, axis=0), strict=True)


@pytest.mark.parametrize(
    'shapes', [np.array([[0, 1, 1]]), np.array([[1, 0, 1]]), np.array([[1, 1, 2]])]
)
def test_invalid_geometry_is_rejected_before_sampling(shapes):
    reader = SimpleNamespace(shape=(1, 1), dtype=np.dtype('f4'), nframes=1)
    with pytest.raises(ValueError, match='window dimensions'):
        StreamingView(View(reader), FrameCache(0)).extract(
            np.array([[0]]), np.array([0, 1]), np.array([0]), np.array([0]), shapes, 0
        )


@pytest.mark.parametrize('bad', [np.ones(1, dtype='f8'), np.ones((1, 1), dtype='f4')])
def test_sampler_must_return_the_declared_pixel_type_and_shape(bad):
    class Reader:
        shape, dtype, nframes = (1, 1), np.dtype('f4'), 1

        def sample_pixels(self, *args):
            return bad

    with pytest.raises(ValueError, match='image shape/dtype'):
        StreamingView(View(Reader()), FrameCache(0)).extract(
            np.array([[0]]),
            np.array([0, 1]),
            np.array([0]),
            np.array([0]),
            np.array([[1, 1, 1]]),
            0,
        )
