from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING
import importlib.resources

import numpy as np

import hexrd.core.resources.panel_buffers

if TYPE_CHECKING:
    from hexrd.core.instrument.detector import Detector


def panel_buffer_as_2d_array(panel: Detector) -> np.ndarray:
    # Take whatever the panel buffer is and convert it to a 2D array
    if panel.panel_buffer is None:
        # Just make a panel buffer with all True values
        return np.ones(panel.shape, dtype=bool)
    elif isinstance(panel.panel_buffer, str):
        # Load the known panel buffer and return
        return panel_buffer_from_str(panel.panel_buffer, panel)
    elif panel.panel_buffer.shape == (2,):
        # The two floats are specifying the borders in mm for x and y.
        # Convert to pixel borders. Swap x and y so we have i, j in pixels.
        borders = np.round(
            [
                panel.panel_buffer[1] / panel.pixel_size_row,
                panel.panel_buffer[0] / panel.pixel_size_col,
            ]
        ).astype(int)

        # Convert to array
        panel_buffer = np.zeros(panel.shape, dtype=bool)

        # We can't do `-borders[i]` since that doesn't work for 0,
        # so we must do `panel.shape[i] - borders[i]` instead.
        panel_buffer[
            borders[0] : panel.shape[0] - borders[0],
            borders[1] : panel.shape[1] - borders[1],
        ] = True
        return panel_buffer
    elif panel.panel_buffer.ndim == 2:
        return panel.panel_buffer

    raise NotImplementedError(panel.panel_buffer.ndim)


def panel_buffer_from_str(name: str, panel: Detector) -> np.ndarray:
    buffer = _load_panel_buffer_from_file(name)

    # If the buffer shape matches the panel shape, we are done
    if buffer.shape == panel.shape:
        return buffer

    # If the size of the buffer is larger than the detector size,
    # try to see if we can use ROIs to cut out the region we need.
    roi = panel.roi
    if roi is None:
        msg = (
            f'Buffer shape {buffer.shape} does not match '
            f'panel shape {panel.shape} for buffer name: {name}'
        )
        raise RuntimeError(msg)

    roi_buffer = buffer[roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]
    if roi_buffer.shape != panel.shape:
        msg = (
            f'Buffer shape {buffer.shape} does not match '
            f'panel shape {panel.shape} for buffer name: {name}. '
            f'Attempted to utilize the panel ROI {panel.roi} to '
            'extract the region needed, but the final shape did not match'
        )
        raise RuntimeError(msg)

    return roi_buffer


def valid_panel_buffer_names() -> list[str]:
    dir_path = importlib.resources.files(hexrd.core.resources.panel_buffers)
    return [file_.name.removesuffix(".npz") for file_ in dir_path.iterdir() if file_.name.endswith('.npz')]


# Cache this so we only read from disk once
@cache
def _load_panel_buffer_from_file(name: str) -> np.ndarray:
    path = importlib.resources.files(hexrd.core.resources.panel_buffers).joinpath(
        f'{name}.npz'
    )
    if not path.is_file():
        raise NotImplementedError(f'Unknown panel buffer name: {name}')

    npz = np.load(str(path))
    buffer = npz['panel_buffer']

    # Since the output here is memoized, make sure this is never modified
    buffer.flags.writeable = False
    return buffer
