import copy
from pathlib import Path

import h5py
import numpy as np

from hexrd.instrument import HEDMInstrument
from hexrd.material.crystallography import PlaneData
from hexrd.rotations import mapAngle

from spot_finder import Spot
from spot_tracker import SpotTracker, TrackedSpot

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
        array = np.empty((len(spots_dict), 6), dtype=float)
        for i, spot_list in enumerate(spots_dict.values()):
            array[i] = compute_mean_spot(spot_list, omegas)

        spot_arrays[det_key] = array

    return spot_arrays


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


# First key is detector key. Second key is grain ID. Third key is
# array name. Choices are 'hkls', 'sim_xys', 'sim_angs', 'meas_xys',
# 'assigned_spots', 'meas_angs', and 'num_frames'
# All arrays are the same length, and a value at index `i` always
# corresponds to the HKL at index `i`.
AssignSpotsOutputType = dict[str, dict[int, dict[str, np.ndarray]]]


def assign_spots_to_hkls(
    spot_arrays: dict[str, np.ndarray],
    instr: HEDMInstrument,
    tolerances: np.ndarray,  # tth, eta, omega in radians
    eta_period: tuple[int],
    plane_data: PlaneData,
    grain_params: np.ndarray,
    eta_ranges: list[tuple[float, float]],
    omega_ranges: list[tuple[float, float]],
    omega_period: tuple[float, float],
) -> AssignSpotsOutputType:

    # Simulate the spots
    simulated_results = instr.simulate_rotation_series(
        plane_data,
        grain_params,
        eta_ranges,
        omega_ranges,
        omega_period,
    )

    # Loop over detectors and grain IDs and try to locate their matching spots
    ret = {}
    for det_key, sim_results in simulated_results.items():
        panel = instr.detectors[det_key]
        spot_array = spot_arrays[det_key][:, [0, 1, 3]]

        # Compute tth, eta for this spot array
        # We will use these to compare to the simulated spot
        # results and match spots with HKLs.
        # First convert to cartesian
        xys = panel.pixelToCart(spot_array[:, :2])

        # Next convert to angles. Apply the distortion.
        ang_crds, _ = panel.cart_to_angles(
            xys,
            tvec_s=instr.tvec,
            apply_distortion=True,
        )

        # Map the angles to our eta period
        ang_crds[:, 1] = mapAngle(ang_crds[:, 1], eta_period)

        # Stack the omegas on the end
        ang_spot_coords = np.hstack((ang_crds, spot_array[:, [2]]))
        num_frames = spot_arrays[det_key][:, 5]

        # Grab some simulated HKLs
        sim_all_hkls = sim_results[1]
        sim_all_xys = sim_results[3]

        # results[2] is angles, but these don't take into account things like
        # grain centroid shifts. It's more accurate to compute the angles from
        # the xys.

        detector_assigned_spots = []
        grain_hkl_assignments = ret.setdefault(det_key, {})
        for grain_id, sim_xys in enumerate(sim_all_xys):
            sim_omegas = sim_results[2][grain_id][:, 2]
            sim_hkls = sim_all_hkls[grain_id]

            tvec_c = grain_params[grain_id][3:6]
            angles, _ = panel.cart_to_angles(sim_xys, tvec_c=tvec_c)

            # Fix eta period
            angles[:, 1] = mapAngle(angles[:, 1], eta_period)

            # Add the omegas
            angles = np.hstack((angles, sim_omegas[:, np.newaxis]))

            # Create the hkl assignments array
            hkl_assignments = np.full(len(sim_hkls), -1, dtype=int)
            skipped_hkls = []
            spots_assigned = []
            for i, ang_crd in enumerate(angles):
                # Find the closest spot
                differences = abs(ang_crd - ang_spot_coords)
                distances = np.sqrt((differences**2).sum(axis=1))
                min_idx = distances.argmin()

                # Verify that the differences are within the tolerances
                if not np.all(differences[min_idx] < tolerances):
                    # Not within the tolerance...
                    skipped_hkls.append(sim_hkls[i])
                    continue

                hkl_assignments[i] = min_idx
                spots_assigned.append(min_idx)

            if skipped_hkls:
                # FIXME: better handling here
                # This just means we identified some spots that were
                # not paired with HKLs. That might be okay, because
                # we might have not simulated all of the HKLs.
                print(
                    f'For grain {grain_id} on detector {det_key}, did not '
                    'find spots to match the following simulated HKLs '
                    '(perhaps they are low structure factor):',
                    [x.tolist() for x in skipped_hkls],
                )

            spots_assigned = np.asarray(spots_assigned)
            assigned_indices_sorted, counts = np.unique(
                spots_assigned,
                return_counts=True,
            )
            if np.any(counts > 1):
                # FIXME: better handling here
                # This means two different HKLs were assigned to the same spot.
                # We'll definitely have to figure out what to do about this...
                print(
                    f'WARNING!!! {grain_id} on detector {det_key}, '
                    'some spots were assigned twice!',
                    counts[counts > 1],
                )

            # Keep track of all spots assigned on this detector, so
            # we can figure out if any spots were assigned to multiple
            # grains.
            detector_assigned_spots.append(assigned_indices_sorted)

            cart_spot_coords = np.empty((len(spots_assigned), 3))
            meas_angs = np.empty((len(spots_assigned), 3))
            if spots_assigned.size != 0:
                cart_spot_coords[:, :2] = panel.pixelToCart(
                    spot_array[spots_assigned][:, [0, 1]]
                )
                if panel.distortion is not None:
                    # Apply the distortion
                    cart_spot_coords[:, :2] = panel.distortion.apply_inverse(
                        cart_spot_coords[:, :2]
                    )

                cart_spot_coords[:, 2] = spot_array[spots_assigned, 2]
                meas_angs = ang_spot_coords[spots_assigned]

            keep_hkls = hkl_assignments != -1

            # Store our assignments. `hkls[i]` is the HKL that corresponds
            # to both `sim_xys[i]`, `meas_xys[i]`, and `meas_angs[i]`
            grain_hkl_assignments[grain_id] = {
                'hkls': sim_hkls[keep_hkls],
                'sim_xys': sim_xys[keep_hkls],
                'sim_angs': angles[keep_hkls],
                'meas_xys': cart_spot_coords,
                'spots_assigned': spots_assigned,
                'meas_angs': meas_angs,
                'num_frames': num_frames[spots_assigned],
            }

        # Check if any spots assigned to HKLs from one grain were also assigned
        # to HKLs on another grain
        detector_assigned_indices_sorted, counts = np.unique(
            np.hstack(detector_assigned_spots),
            return_counts=True,
        )
        if np.any(counts > 1):
            # FIXME: better handling here
            print(
                f'WARNING!!! On detector {det_key}, '
                'some spots were assigned to multiple grains!',
                counts[counts > 1],
            )

    return ret
