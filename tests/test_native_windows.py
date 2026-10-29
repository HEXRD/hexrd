"""Native window execution must preflight unsupported data before sampling."""

import numpy as np
import pytest

from hexrd.core.extensions import transforms as native


def no_callback(*args):
    pytest.fail('native preflight performed image work')


@pytest.mark.parametrize('dtype', ['O', '>f8', 'f2'])
def test_unsupported_window_dtype_is_declined_before_image_work(dtype):
    assert (
        native._measureWindows(
            [],
            np.empty((0, 3), dtype='i8'),
            np.dtype(dtype),
            no_callback,
            no_callback,
            no_callback,
        )
        is None
    )


def window_request(points):
    return (
        np.zeros(points, dtype='i4'),
        np.arange(points, dtype='i4'),
        np.array([[1, 1, points]], dtype='i8'),
        0,
        (
            np.zeros((1, 3)),
            np.zeros(1, dtype='f4'),
            np.zeros(1, dtype='f4'),
            np.zeros(1, dtype=bool),
        ),
        True,
    )


@pytest.mark.parametrize('points', [40000, 80000])
def test_large_frames_use_bounded_tiles_and_preserve_omega_slots(points):
    requests = [window_request(points), window_request(points)]
    requests = [
        (*r[:2], np.array([[2, 1, points]], dtype='i8'), *r[3:]) for r in requests
    ]
    events = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0]], dtype='i8')
    calls, ticks = [], []

    def sample(frame, rows, cols):
        assert 0 < len(rows) <= 65536
        calls.append((frame, len(rows)))
        return (1 + cols % 17 + 20 * frame).astype('f4')

    assert (
        native._measureWindows(
            requests,
            events,
            np.dtype('f4'),
            sample,
            lambda: None,
            ticks.append,
        )
        == 2
    )
    assert sum(n for _, n in calls) == 4 * points
    assert ticks == [0, 2]
    cols = np.arange(points)
    pixels = np.r_[(1 + cols % 17).astype('f4'), (21 + cols % 17).astype('f4')]
    expected = native._spotStats(
        pixels, np.array([0, len(pixels)], dtype='i8'), requests[0][2], 0
    )
    for request in requests:
        for actual, wanted in zip(request[4], expected):
            np.testing.assert_array_equal(actual, wanted, strict=True)


def test_duplicate_window_slot_is_rejected_before_image_work():
    with pytest.raises(ValueError, match='frame event'):
        native._measureWindows(
            [window_request(1)],
            np.array([[0, 0, 0], [0, 0, 0]], dtype='i8'),
            np.dtype('f4'),
            no_callback,
            no_callback,
            no_callback,
        )
