import h5py
import numpy as np
import yaml

import pytest

from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument.hedm_instrument import HEDMInstrument
from hexrd.material.material import load_materials_hdf5
from hexrd.spot_tracking.assign_spots import assign_spots_to_hkls
from hexrd.spot_tracking.track_spots import track_combine_and_chunk_spots


@pytest.fixture
def ruby_hedm_dir(example_repo_path):
    return example_repo_path / 'dexelas' / 'ruby'


@pytest.fixture
def ruby_dexelas_images_dir(example_repo_path):
    return example_repo_path / 'NIST_ruby/multiruby_dexelas/imageseries'


@pytest.fixture
def dexelas_instrument(ruby_hedm_dir):
    # Load the instrument
    with open(ruby_hedm_dir / 'dual_dexelas_composite.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    return HEDMInstrument(conf)


@pytest.fixture
def ruby_material(ruby_hedm_dir):
    # Load the material
    with h5py.File(ruby_hedm_dir / 'ruby.h5', 'r') as rf:
        return load_materials_hdf5(rf)['ruby']


@pytest.fixture
def grain_params(ruby_hedm_dir):
    return np.load(ruby_hedm_dir / 'grain_params.npy')


@pytest.fixture
def dexelas_images(ruby_dexelas_images_dir):
    d = ruby_dexelas_images_dir
    return {
        'ff1': d / 'mruby-0129_000004_ff1_000012-cachefile.npz',
        'ff2': d / 'mruby-0129_000004_ff2_000012-cachefile.npz',
    }


@pytest.fixture
def spots_filename(example_repo_path):
    return example_repo_path / 'spot_finding/spots_file.h5'


def test_spot_tracking(dexelas_instrument, ruby_material, grain_params,
                       dexelas_images, spots_filename):

    instr = dexelas_instrument
    material = ruby_material
    image_files = dexelas_images

    # Load in the raw imageseries for each Dexela detector
    raw_ims_dict = {}
    for k, filename in image_files.items():
        raw_ims_dict[k] = imageseries.open(filename, 'frame-cache')

    # Grab the number of images and the omega period
    num_images = len(raw_ims_dict[k])
    omegas = raw_ims_dict[k].metadata['omega']
    omega_period = np.radians((omegas[0][0], omegas[-1][1]))
    eta_period = (-np.pi, np.pi)

    # Just assume the whole eta and omega ranges are used
    eta_ranges = [eta_period]
    omega_ranges = [omega_period]

    # Break up the imageseries into their subpanels
    ims_dict = {}
    for det_key, panel in instr.detectors.items():
        ops = [
            ('rectangle', panel.roi),
        ]

        raw_key = det_key[:3]
        ims = raw_ims_dict[raw_key]
        ims_dict[det_key] = OmegaImageSeries(ProcessedImageSeries(ims, ops))

    plane_data = material.planeData

    # Track, combine, and chunk spots into subpanels
    spot_arrays = track_combine_and_chunk_spots(
        spots_filename,
        instr,
        num_images,
        omegas,
    )

    # Set tolerances for tth, eta, and omega
    tth_tol = np.radians(0.25)
    eta_tol = np.radians(1.0)
    ome_tol = np.radians(1.5)
    tolerances = np.array([tth_tol, eta_tol, ome_tol])

    # Now assign spots to HKLs
    assigned_spots = assign_spots_to_hkls(
        spot_arrays,
        instr,
        tolerances,
        eta_period,
        plane_data,
        grain_params,
        eta_ranges,
        omega_ranges,
        omega_period,
        num_images,
    )

    # Now run `pull_spots()` and verify that our assigned spots match
    ref_spots_dict = {}
    for i, params in enumerate(grain_params):
        kwargs = {
            'plane_data': plane_data,
            'grain_params': params,
            'tth_tol': np.degrees(tth_tol),
            'eta_tol': np.degrees(eta_tol),
            'ome_tol': np.degrees(ome_tol),
            'imgser_dict': ims_dict,
            'npdiv': 4,
            'threshold': 25.0,
            'eta_ranges': eta_ranges,
            'ome_period': omega_period,
            'dirname': None,
            'filename': None,
            'return_spot_list': False,
            'quiet': True,
            'check_only': False,
            'interp': 'nearest',
        }
        ref_spots_dict[i] = instr.pull_spots(**kwargs)

    # Keep track of which HKLs from `pull_spots()` were matched or
    # unmatched.
    num_matched = 0
    num_unmatched = 0

    # Keep track of the distances too
    distances = []
    ome_diffs = []

    matched_spots = {}
    for grain_id, ref_spots_output in ref_spots_dict.items():
        ref_completeness, ref_grain_spots = ref_spots_output
        for det_key, ref_det_spots in ref_grain_spots.items():
            # Grab the measured spots
            meas_spots = assigned_spots[det_key][grain_id]

            # Keep track of which spots were assigned and which were not
            matched_spots.setdefault(det_key, {})
            these_matched_spots = np.zeros(len(meas_spots['hkls']), dtype=bool)
            matched_spots[det_key][grain_id] = these_matched_spots

            for ref_spot in ref_det_spots:
                hkl = ref_spot[2]
                ref_meas_angs = ref_spot[6]
                ref_omega = ref_meas_angs[2]
                ref_meas_xy = ref_spot[7]

                if np.any(np.isnan(ref_meas_xy)):
                    # This is not a real spot...
                    # I don't know why `pull_spots()` sometimes does this...
                    continue

                matching_idx = -1
                for row in range(len(meas_spots['hkls'])):
                    if np.array_equal(meas_spots['hkls'][row], hkl):
                        # Found a matching HKL!
                        matching_idx = row
                        break

                if matching_idx == -1:
                    print(
                        f'Warning! Did not find a match on {det_key} for HKL: '
                        f'{hkl}'
                    )
                    num_unmatched += 1
                    continue

                # Compute the distance between the reference measured xy and
                # our own measured xy
                distance = np.sqrt((
                    (ref_meas_xy - meas_spots['meas_xys'][matching_idx][:2])
                    ** 2
                ).sum())
                distances.append(distance)

                # Compute difference in omega between the reference and ours
                ome_diff = abs(
                    ref_omega - meas_spots['meas_xys'][matching_idx][2]
                )
                ome_diffs.append(ome_diff)

                # Indicate this spot was assigned
                these_matched_spots[matching_idx] = True
                num_matched += 1

    max_omega_diff = max(ome_diffs)
    max_distance = max(distances)
    percent_found = num_matched / (num_matched + num_unmatched) * 100

    # There should be at least 1156 spots matched. This should be all
    # of the spots from `pull_spots()`
    assert num_matched >= 1156

    # There should not be any that we missed
    assert num_unmatched == 0

    # The mean of the distances should be very small
    mean_distance = np.mean(distances)
    assert mean_distance < 0.003
    assert max_distance < 0.5

    mean_ome_diff = np.mean(ome_diffs)
    assert np.degrees(mean_ome_diff) < 0.07
    assert np.degrees(max_omega_diff) < 0.51

    # Verify that we matched all of the
    print(
        'Percentage of spots that `pull_spots()` found that we also found:',
        f'{percent_found:.2f}%',
    )

    print(f'Mean distance (xy): {np.mean(distances):.4f}')
    print(f'Max distance (xy): {max_distance:.4f}')
    print(f'Mean omega diff (degrees): {np.degrees(np.mean(ome_diffs)):.4f}')
    print(f'Max omega diff (degrees): {np.degrees(max_omega_diff):.4f}')
    num_extra_hkls = 0
    for det_key, det_assignments in matched_spots.items():
        for grain_id, grain_assignments in det_assignments.items():
            if np.any(~grain_assignments):
                extra_hkls = assigned_spots[det_key][grain_id]['hkls'][
                    ~grain_assignments
                ]
                num_extra_hkls += len(extra_hkls)
                # print('Extra HKLs:', extra_hkls)

    print(
        'Number of HKLs we found that `pull_spots()` did not find:',
        num_extra_hkls
    )

    # This is the total number of spots matched
    expected_total_spots_found = 1243
    assert num_matched + num_extra_hkls >= expected_total_spots_found
