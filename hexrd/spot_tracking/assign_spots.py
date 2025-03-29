import numpy as np
from scipy.spatial import KDTree

from hexrd.instrument import HEDMInstrument
from hexrd.material.crystallography import PlaneData
from hexrd.rotations import angularDifference, mapAngle


# First key is detector key. Second key is grain ID. Third key is
# array name. Choices are 'hkls', 'sim_xys', 'sim_angs', 'meas_xys',
# 'assigned_spots', 'meas_angs', and 'spot_num_frames'
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
    n_frames: int,
) -> AssignSpotsOutputType:

    # Simulate the spots
    simulated_results = instr.simulate_rotation_series(
        plane_data,
        grain_params,
        eta_ranges,
        omega_ranges,
        omega_period,
    )

    if len(omega_ranges) > 1:
        # We convert the omegas to frame coordinates. This is a
        # lot simpler if we assume a single omega range.
        msg = 'We can only use a single omega range right now'
        raise NotImplementedError(msg)

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
        spot_num_frames = spot_arrays[det_key][:, 5]
        max_int = spot_arrays[det_key][:, 6]
        sum_int = spot_arrays[det_key][:, 7]

        # We verified earlier that there should only be one omega range.
        # We could do more than one, but it's easier to just assume one
        # for now...
        min_ome, max_ome = omega_ranges[0]

        def omegas_to_frame_pixels(omegas: np.ndarray) -> np.ndarray:
            return (omegas - min_ome) / (max_ome - min_ome) * n_frames

        # Compute measured pixels
        meas_pixels = spot_array.copy()
        # Convert the omegas to frame pixels
        meas_pixels[:, 2] = omegas_to_frame_pixels(meas_pixels[:, 2])

        kd_tree = KDTree(meas_pixels)

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
            sim_angles, _ = panel.cart_to_angles(sim_xys, tvec_c=tvec_c)

            # Fix eta period
            sim_angles[:, 1] = mapAngle(sim_angles[:, 1], eta_period)

            # Add the omegas
            sim_angles = np.hstack((sim_angles, sim_omegas[:, np.newaxis]))

            # Compute simulated spots in pixel coordinates as well.
            # We will assign spots to HKLs according to minimum pixel
            # distance (like `pull_spots()` does), rather than angular
            # distances. However, we will still compare angles to verify
            # they are within the specified tolerances.
            sim_pixels = panel.cartToPixel(
                sim_xys,
                apply_distortion=True,
            )

            frame_pixels = omegas_to_frame_pixels(sim_omegas)
            sim_pixels = np.hstack((sim_pixels, frame_pixels[:, np.newaxis]))

            # Create the hkl assignments array
            hkl_assignments = np.full(len(sim_hkls), -1, dtype=int)
            skipped_hkls = []
            spots_assigned = []
            for i, ang_crd in enumerate(sim_angles):
                # Find the closest spot. Include wrapping around to other side.
                d1, min_idx1 = kd_tree.query(sim_pixels[i])
                d2, min_idx2 = kd_tree.query(sim_pixels[i] - [0, 0, n_frames])
                min_idx = min_idx1 if d1 < d2 else min_idx2

                # Use special function to take into account angular wrapping
                ang_differences = angularDifference(
                    ang_crd,
                    ang_spot_coords[min_idx],
                )

                # Verify that the differences are within the tolerances
                if not np.all(ang_differences < tolerances):
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
                'sim_angs': sim_angles[keep_hkls],
                'meas_xys': cart_spot_coords,
                'spots_assigned': spots_assigned,
                'meas_angs': meas_angs,
                'max_int': max_int[spots_assigned],
                'sum_int': sum_int[spots_assigned],
                'spot_num_frames': spot_num_frames[spots_assigned],
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
