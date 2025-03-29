import numpy as np

from hexrd.instrument import HEDMInstrument


def in_range(x: np.ndarray, xrange: tuple[float, float]) -> np.ndarray:
    # Generic function to determine which `x` values are in the range `xrange`
    # Returns an array of booleans indicating which ones were in range
    return (xrange[0] <= x) & (x < xrange[1])


def chunk_spots_into_subpanels(
    spot_arrays: dict[str, np.ndarray],
    instr: HEDMInstrument,
) -> dict[str, np.ndarray]:
    # Break up the spots into subpanels, and remap the coordinates
    # to be within the subpanel's coordinates.
    all_groups = instr.detector_groups
    if not all_groups:
        # There are no subpanels...
        return spot_arrays

    subpanel_spot_arrays = {}
    for group, array in spot_arrays.items():
        for det_key, panel in instr.detectors_in_group(group).items():
            # Extract all spots that belong to this subpanel
            on_panel_rows = in_range(array[:, 0], panel.roi[0]) & in_range(
                array[:, 1], panel.roi[1]
            )
            if np.any(on_panel_rows):
                # Extract the spots on the subpanel
                subpanel_array = array[on_panel_rows]

                # Adjust the i, j coordinates for this subpanel
                subpanel_array[:, 0] -= panel.roi[0][0]
                subpanel_array[:, 1] -= panel.roi[1][0]
                subpanel_spot_arrays[det_key] = subpanel_array
            else:
                subpanel_spot_arrays[det_key] = np.empty((0,))

    return subpanel_spot_arrays
