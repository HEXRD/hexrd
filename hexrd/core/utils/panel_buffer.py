import numpy as np

from hexrd.core.instrument.detector import Detector


def panel_buffer_as_2d_array(panel: Detector) -> np.ndarray:
    # Take whatever the panel buffer is and convert it to a 2D array
    if panel.panel_buffer is None:
        # Just make a panel buffer with all True values
        return np.ones(panel.shape, dtype=bool)
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
