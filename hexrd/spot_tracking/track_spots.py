import copy
from pathlib import Path

import h5py
import numpy as np

from hexrd.instrument import HEDMInstrument
from hexrd.utils.decorators import memoize

from .chunk_spots import chunk_spots_into_subpanels
from .spot import Spot
from .spot_tracker import SpotTracker, TrackedSpot


# First key is detector key, second key is spot ID, and the
# final list is the list of spots (one for each frame a tracked
# spot is on) for that spot ID.
TrackSpotsOutputType = dict[str, dict[int, list[TrackedSpot]]]


# Track the spots in the raw images
def track_spots(
    spots_filename: Path,
    num_images: int,
    det_keys: list[str],
) -> TrackSpotsOutputType:
    spot_trackers = {k: SpotTracker() for k in det_keys}
    tracked_spots = {k: {} for k in det_keys}
    with h5py.File(spots_filename, 'r') as rf:
        for frame_index in range(num_images):
            print('Tracking spots for frame:', frame_index)
            for det_key in det_keys:
                # Pull the spots from the file
                path = f'{det_key}/{frame_index}'
                spots_array = rf[path][()]

                # Convert to list of Spot
                spots = [Spot.from_array(x) for x in spots_array]

                # Next, track spots
                tracker = spot_trackers[det_key]
                tracker.track_spots(spots, frame_index)

                # Now update our tracked list
                spot_dict = tracked_spots[det_key]
                for spot_id, spot in tracker.current_spots.items():
                    spot_list = spot_dict.setdefault(spot_id, [])
                    spot_list.append(copy.deepcopy(spot))

    return tracked_spots


# Compute x, y, w, omega, and omega width for every spot
def compute_mean_spot(
    spot_list: list[TrackedSpot],
    omegas: np.ndarray,
) -> np.ndarray:
    omega_ranges = np.radians([omegas[s.frame_index] for s in spot_list])
    omega_values = [np.mean(x) for x in omega_ranges]
    sums = np.array([s.sum for s in spot_list])
    widths = np.array([s.w for s in spot_list])
    max_width = widths.max()
    coords = np.array([(s.i, s.j) for s in spot_list])
    n_frames = len(spot_list)
    max_int = max(s.max for s in spot_list)
    sum_int = sum(s.sum for s in spot_list)

    # We are using a width-weighted omega as the average omega, currently
    sum_weighted_omega = (omega_values * sums).sum() / (sums.sum())
    sum_weighted_coords = (coords * sums[:, np.newaxis]).sum(
        axis=0
    ) / sums.sum()

    # We are using the full range of omegas
    omega_width = (omega_ranges[-1][1] - omega_ranges[0][0]) / 2
    return np.asarray((
        *sum_weighted_coords,
        max_width,
        sum_weighted_omega,
        omega_width,
        n_frames,
        max_int,
        sum_int,
    ))


def combine_spots(
    tracked_spots: TrackSpotsOutputType,
    omegas: np.ndarray,
) -> dict[str, np.ndarray]:
    # Combine spots on different frames that appear to belong to the
    # same HKL, and perform weighted averages for computing their x, y
    # and omega values.
    spot_arrays = {}
    for det_key, spots_dict in tracked_spots.items():
        array = np.empty((len(spots_dict), 8), dtype=float)
        for i, spot_list in enumerate(spots_dict.values()):
            array[i] = compute_mean_spot(spot_list, omegas)

        spot_arrays[det_key] = array

    return spot_arrays


def track_combine_and_chunk_spots(
    spots_filename: Path,
    instr: HEDMInstrument,
    num_images: int,
    omegas: np.ndarray,
) -> dict[str, np.ndarray]:
    # First, check if the spots have been pre-tracked. If they have
    # been, we don't need to track them again.
    # FIXME: this is all going to change with the new spot finding/tracking
    with h5py.File(spots_filename, 'r') as rf:
        k = '_pretracked_spots'
        if k in rf:
            # Shortcut! The spots were pre-tracked...
            print(
                'Pre-tracked spots found in the spots data file. '
                'Skipping spot tracking and using those...'
            )
            return {
                det_key: rf[f'{k}/{det_key}'][()]
                for det_key in instr.detectors
            }

    # Use detector groups if they are available, because the spots
    # will have been identified from the raw images
    det_keys = instr.detector_groups
    was_chunked = True
    if not det_keys:
        # Must not be chunked into groups
        det_keys = list(instr.detectors)
        was_chunked = False

    spot_arrays = _track_and_combine_spots(
        spots_filename,
        num_images,
        det_keys,
        omegas,
    )
    if was_chunked:
        spot_arrays = chunk_spots_into_subpanels(spot_arrays, instr)

    return spot_arrays


# Memoize this. It involves extracting the spots from the file,
# tracking them, and combining them. This might be repeated once for
# each grain, and memoizing it makes it so we don't have to repeat those
# steps.
@memoize(maxsize=2)
def _track_and_combine_spots(
    spots_filename: Path,
    num_images: int,
    det_keys: list[str],
    omegas: np.ndarray,
) -> dict[str, np.ndarray]:
    tracked_spots = track_spots(spots_filename, num_images, det_keys)
    return combine_spots(tracked_spots, omegas)
