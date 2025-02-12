from typing import Optional

import lmfit
import numpy as np

from hexrd.core.instrument import (
    calc_angles_from_beam_vec,
    calc_beam_vec,
    Detector,
    HEDMInstrument,
)
from hexrd.core.rotations import (
    angleAxisOfRotMat,
    expMapOfQuat,
    make_rmat_euler,
    quatOfRotMat,
    RotMatEuler,
    rotMatOfExpMap,
)
from hexrd.core.material.unitcell import _lpname
from .relative_constraints import RelativeConstraints, RelativeConstraintsType
from hexrd.core.fitting.calibration.relative_constraints import (
    RelativeConstraints,
    RelativeConstraintsType,
)


# First is the axes_order, second is extrinsic
DEFAULT_EULER_CONVENTION = ('zxz', False)

EULER_CONVENTION_TYPES = dict | tuple | None


def create_instr_params(
    instr, euler_convention=DEFAULT_EULER_CONVENTION, relative_constraints=None
):
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    parms_list = []

    # This supports either single beam or multi-beam
    beam_param_names = create_beam_param_names(instr)
    for beam_name, beam in instr.beam_dict.items():
        azim, pol = calc_angles_from_beam_vec(beam['vector'])
        energy = beam['energy']

        names = beam_param_names[beam_name]
        parms_list.append((names['beam_polar'], pol, False, pol - 1, pol + 1))
        parms_list.append(
            (names['beam_azimuth'], azim, False, azim - 1, azim + 1)
        )
        parms_list.append(
            (names['beam_energy'], energy, False, energy - 0.2, energy + 0.2)
        )

    parms_list.append(
        (
            'instr_chi',
            np.degrees(instr.chi),
            False,
            np.degrees(instr.chi) - 1,
            np.degrees(instr.chi) + 1,
        )
    )
    parms_list.append(('instr_tvec_x', instr.tvec[0], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_y', instr.tvec[1], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_z', instr.tvec[2], False, -np.inf, np.inf))

    if (
        relative_constraints is None
        or relative_constraints.type == RelativeConstraintsType.none
    ):
        add_unconstrained_detector_parameters(
            instr,
            euler_convention,
            parms_list,
        )
    elif relative_constraints.type == RelativeConstraintsType.group:
        add_group_constrained_detector_parameters(
            instr,
            euler_convention,
            parms_list,
            relative_constraints,
        )
    elif relative_constraints.type == RelativeConstraintsType.system:
        add_system_constrained_detector_parameters(
            instr,
            euler_convention,
            parms_list,
            relative_constraints,
        )
    else:
        raise NotImplementedError(relative_constraints.type)

    return parms_list


def add_unconstrained_detector_parameters(instr, euler_convention, parms_list):
    for det_name, panel in instr.detectors.items():
        det = det_name.replace('-', '_')

        angles = detector_angles_euler(panel, euler_convention)
        angle_names = param_names_euler_convention(det, euler_convention)

        for name, angle in zip(angle_names, angles):
            parms_list.append((name, angle, False, angle - 2, angle + 2))

        parms_list.append(
            (
                f'{det}_tvec_x',
                panel.tvec[0],
                True,
                panel.tvec[0] - 1,
                panel.tvec[0] + 1,
            )
        )
        parms_list.append(
            (
                f'{det}_tvec_y',
                panel.tvec[1],
                True,
                panel.tvec[1] - 0.5,
                panel.tvec[1] + 0.5,
            )
        )
        parms_list.append(
            (
                f'{det}_tvec_z',
                panel.tvec[2],
                True,
                panel.tvec[2] - 1,
                panel.tvec[2] + 1,
            )
        )
        if panel.distortion is not None:
            p = panel.distortion.params
            for ii, pp in enumerate(p):
                parms_list.append(
                    (
                        f'{det}_distortion_param_{ii}',
                        pp,
                        False,
                        -np.inf,
                        np.inf,
                    )
                )
        if panel.detector_type.lower() == 'cylindrical':
            parms_list.append(
                (f'{det}_radius', panel.radius, False, -np.inf, np.inf)
            )


def _add_constrained_detector_parameters(
    euler_convention: EULER_CONVENTION_TYPES,
    parms_list: list[tuple],
    prefix: str,
    constraint_params: dict,
):
    tvec = constraint_params['translation']
    tilt = constraint_params['tilt']

    if euler_convention is not None:
        # Convert the tilt to the specified Euler convention
        normalized = normalize_euler_convention(euler_convention)
        rme = RotMatEuler(
            np.zeros(
                3,
            ),
            axes_order=normalized[0],
            extrinsic=normalized[1],
        )

        rme.rmat = _tilt_to_rmat(tilt, None)
        tilt = np.degrees(rme.angles)

    tvec_names = [
        f'{prefix}_tvec_x',
        f'{prefix}_tvec_y',
        f'{prefix}_tvec_z',
    ]
    tvec_deltas = [1, 1, 1]

    tilt_names = param_names_euler_convention(prefix, euler_convention)
    tilt_deltas = [2, 2, 2]

    for i, name in enumerate(tvec_names):
        value = tvec[i]
        delta = tvec_deltas[i]
        parms_list.append((name, value, True, value - delta, value + delta))

    for i, name in enumerate(tilt_names):
        value = tilt[i]
        delta = tilt_deltas[i]
        parms_list.append((name, value, True, value - delta, value + delta))


def add_system_constrained_detector_parameters(
    instr: HEDMInstrument,
    euler_convention: EULER_CONVENTION_TYPES,
    parms_list: list[tuple],
    relative_constraints: RelativeConstraints,
):
    prefix = 'system'
    constraint_params = relative_constraints.params
    _add_constrained_detector_parameters(
        euler_convention,
        parms_list,
        prefix,
        constraint_params,
    )


def add_group_constrained_detector_parameters(
    instr: HEDMInstrument,
    euler_convention: EULER_CONVENTION_TYPES,
    parms_list: list[tuple],
    relative_constraints: RelativeConstraints,
):
    for group in instr.detector_groups:
        prefix = group.replace('-', '_')
        constraint_params = relative_constraints.params[group]
        _add_constrained_detector_parameters(
            euler_convention,
            parms_list,
            prefix,
            constraint_params,
        )


def create_beam_param_names(instr: HEDMInstrument) -> dict[str, str]:
    param_names = {}
    for k, v in instr.beam_dict.items():
        prefix = f'{k}_' if instr.has_multi_beam else ''
        param_names[k] = {
            'beam_polar': f'{prefix}beam_polar',
            'beam_azimuth': f'{prefix}beam_azimuth',
            'beam_energy': f'{prefix}beam_energy',
        }
    return param_names


def fix_detector_y(
    instr: HEDMInstrument,
    params: lmfit.Parameters,
    relative_constraints: Optional[RelativeConstraints] = None,
):
    """Fix the y translation for all detectors"""
    if relative_constraints is None:
        rtype = RelativeConstraintsType.none
    else:
        rtype = relative_constraints.type

    if rtype == RelativeConstraintsType.none:
        prefixes = list(instr.detectors)
    elif rtype == RelativeConstraintsType.group:
        prefixes = instr.detector_groups
    elif rtype == RelativeConstraintsType.system:
        prefixes = ['system']

    prefixes = [x.replace('-', '_') for x in prefixes]
    names = [f'{x}_tvec_y' for x in prefixes]
    for name in names:
        params[name].vary = False


def update_instrument_from_params(
    instr,
    params,
    euler_convention=DEFAULT_EULER_CONVENTION,
    relative_constraints: Optional[RelativeConstraints] = None,
):
    """
    this function updates the instrument from the
    lmfit parameter list. we don't have to keep track
    of the position numbers as the variables are named
    variables. this will become the standard in the
    future since bound constraints can be very easily
    implemented.
    """
    if not isinstance(params, lmfit.Parameters):
        msg = (
            'Only lmfit.Parameters is acceptable input. ' f'Received: {params}'
        )
        raise NotImplementedError(msg)

    # This supports single XRS or multi XRS
    beam_param_names = create_beam_param_names(instr)
    for xrs_name, param_names in beam_param_names.items():
        energy = params[param_names['beam_energy']].value
        azim = params[param_names['beam_azimuth']].value
        pola = params[param_names['beam_polar']].value

        instr.beam_dict[xrs_name]['energy'] = energy
        instr.beam_dict[xrs_name]['vector'] = calc_beam_vec(azim, pola)

    # Trigger any needed updates from beam modifications
    instr.beam_dict_modified()

    chi = np.radians(params['instr_chi'].value)
    instr.chi = chi

    instr_tvec = [
        params['instr_tvec_x'].value,
        params['instr_tvec_y'].value,
        params['instr_tvec_z'].value,
    ]
    instr.tvec = np.r_[instr_tvec]

    if (
        relative_constraints is None
        or relative_constraints.type == RelativeConstraintsType.none
    ):
        update_unconstrained_detector_parameters(
            instr,
            params,
            euler_convention,
        )
    elif relative_constraints.type == RelativeConstraintsType.group:
        update_group_constrained_detector_parameters(
            instr,
            params,
            euler_convention,
            relative_constraints,
        )
    elif relative_constraints.type == RelativeConstraintsType.system:
        update_system_constrained_detector_parameters(
            instr,
            params,
            euler_convention,
            relative_constraints,
        )
    else:
        raise NotImplementedError(relative_constraints.type)


def update_unconstrained_detector_parameters(instr, params, euler_convention):
    for det_name, detector in instr.detectors.items():
        det = det_name.replace('-', '_')
        set_detector_angles_euler(detector, det, params, euler_convention)

        tvec = np.r_[
            params[f'{det}_tvec_x'].value,
            params[f'{det}_tvec_y'].value,
            params[f'{det}_tvec_z'].value,
        ]
        detector.tvec = tvec
        if detector.detector_type.lower() == 'cylindrical':
            rad = params[f'{det}_radius'].value
            detector.radius = rad

        distortion_str = f'{det}_distortion_param'
        if any(distortion_str in p for p in params):
            if detector.distortion is None:
                raise RuntimeError(f"distortion discrepancy for '{det}'!")
            else:
                names = np.sort([p for p in params if distortion_str in p])
                distortion = np.r_[[params[n].value for n in names]]
                try:
                    detector.distortion.params = distortion
                except AssertionError:
                    raise RuntimeError(
                        f"distortion for '{det}' "
                        f"expects {len(detector.distortion.params)} "
                        f"params but got {len(distortion)}"
                    )


def _update_constrained_detector_parameters(
    detectors: list[Detector],
    params: dict,
    rotation_center: np.ndarray,
    euler_convention: EULER_CONVENTION_TYPES,
    prefix: str,
    constraint_params: dict,
):
    tvec = constraint_params['translation']
    tilt = constraint_params['tilt']

    tvec_names = [
        f'{prefix}_tvec_x',
        f'{prefix}_tvec_y',
        f'{prefix}_tvec_z',
    ]
    tilt_names = param_names_euler_convention(prefix, euler_convention)

    # Just like the detectors, we will apply tilt first and then translation
    # Only apply these transforms if they were marked "Vary".

    if any(params[x].vary for x in tilt_names):
        # Find the change in tilt, create an rmat, then apply to detector tilts
        # and translations.
        new_tilt = np.array([params[x].value for x in tilt_names])

        # The old tilt was in the None convention
        old_rmat = _tilt_to_rmat(tilt, None)
        new_rmat = _tilt_to_rmat(new_tilt, euler_convention)

        # Compute the rmat used to convert from old to new
        rmat_diff = new_rmat @ old_rmat.T

        # Rotate each detector using the rmat_diff
        for panel in detectors:
            panel.tilt = _rmat_to_tilt(rmat_diff @ panel.rmat)

            # Also rotate the detectors about the rotation center
            panel.tvec = (
                rmat_diff @ (panel.tvec - rotation_center) + rotation_center
            )

        # Update the tilt
        tilt[:] = _rmat_to_tilt(new_rmat)

    if any(params[x].vary for x in tvec_names):
        # Find the change in center and shift all tvecs
        new_tvec = np.array([params[x].value for x in tvec_names])

        diff = new_tvec - tvec
        for panel in detectors:
            panel.tvec += diff

        # Update the tvec
        tvec[:] = new_tvec


def update_system_constrained_detector_parameters(
    instr: HEDMInstrument,
    params: dict,
    euler_convention: EULER_CONVENTION_TYPES,
    relative_constraints: RelativeConstraints,
):
    detectors = list(instr.detectors.values())

    # Get the center of rotation (depending on the settings)
    rotation_center = relative_constraints.center_of_rotation(instr)
    prefix = 'system'
    constraint_params = relative_constraints.params

    _update_constrained_detector_parameters(
        detectors,
        params,
        rotation_center,
        euler_convention,
        prefix,
        constraint_params,
    )


def update_group_constrained_detector_parameters(
    instr: HEDMInstrument,
    params: dict,
    euler_convention: EULER_CONVENTION_TYPES,
    relative_constraints: RelativeConstraints,
):
    for group in instr.detector_groups:
        detectors = list(instr.detectors_in_group(group).values())

        # Get the center of rotation (depending on the settings)
        rotation_center = relative_constraints.center_of_rotation(instr, group)
        prefix = group.replace('-', '_')
        constraint_params = relative_constraints.params[group]

        _update_constrained_detector_parameters(
            detectors,
            params,
            rotation_center,
            euler_convention,
            prefix,
            constraint_params,
        )


def _tilt_to_rmat(
    tilt: np.ndarray, euler_convention: dict | tuple
) -> np.ndarray:
    # Convert the tilt to exponential map parameters, and then
    # to the rotation matrix, and return.
    if euler_convention is None:
        return rotMatOfExpMap(tilt)

    normalized = normalize_euler_convention(euler_convention)
    return make_rmat_euler(
        np.radians(tilt),
        axes_order=normalized[0],
        extrinsic=normalized[1],
    )


def _rmat_to_tilt(rmat: np.ndarray) -> np.ndarray:
    phi, n = angleAxisOfRotMat(rmat)
    return phi * n.flatten()


def create_tth_parameters(
    instr: HEDMInstrument,
    meas_angles: dict[str, np.ndarray],
) -> list[lmfit.Parameter]:

    prefixes = tth_parameter_prefixes(instr)
    parms_list = []
    for xray_source, angles in meas_angles.items():
        prefix = prefixes[xray_source]
        for ii, tth in enumerate(angles):
            ds_ang = []
            for k, v in tth.items():
                if v is not None:
                    ds_ang.append(v[:, 0])

            if not ds_ang:
                continue

            val = np.degrees(np.mean(np.hstack(ds_ang)))

            parms_list.append(
                (f'{prefix}{ii}', val, True, val - 5.0, val + 5.0)
            )

    return parms_list


def tth_parameter_prefixes(instr: HEDMInstrument) -> dict[str, str]:
    """Generate tth parameter prefixes according to beam names"""
    prefix = 'DS_ring_'
    beam_names = instr.beam_names
    if len(beam_names) == 1:
        return {beam_names[0]: prefix}

    return {name: f'{name}_{prefix}' for name in beam_names}


def create_material_params(material, refinements=None):
    # The refinements should be in reduced format
    refine_idx = 0

    parms_list = []
    for i, lp_name in enumerate(_lpname):
        if not material.unitcell.is_editable(lp_name):
            continue

        if i < 3:
            # Lattice length
            # Convert to angstroms
            multiplier = 10
            diff = 0.1
        else:
            # Lattice angle
            multiplier = 1
            diff = 0.5

        refine = True if refinements is None else refinements[refine_idx]

        val = material.lparms[i] * multiplier
        parms_list.append(
            (
                f'{material.name}_{lp_name}',
                val,
                refine,
                val - diff,
                val + diff,
            )
        )

        refine_idx += 1

    return parms_list


def update_material_from_params(params, material):
    new_lparms = material.lparms
    for i, lp_name in enumerate(_lpname):
        param_name = f'{material.name}_{lp_name}'
        if param_name in params:
            if i < 3:
                # Lattice length
                # Convert to nanometers
                multiplier = 0.1
            else:
                # Lattice angle
                multiplier = 1

            new_lparms[i] = params[param_name].value * multiplier

    material.lparms = new_lparms

    if 'beam_energy' in params:
        # Make sure the beam energy is up-to-date from the instrument
        material.planeData.wavelength = params['beam_energy'].value


def grain_param_names(base_name):
    return [f'{base_name}_grain_param_{i}' for i in range(12)]


def create_grain_params(base_name, grain, refinements=None):
    param_names = grain_param_names(base_name)
    if refinements is None:
        refinements = [True] * len(param_names)

    parms_list = []
    for i, name in enumerate(param_names):
        parms_list.append(
            (
                name,
                grain[i],
                refinements[i],
                grain[i] - 2,
                grain[i] + 2,
            )
        )
    return parms_list


def rename_to_avoid_collision(params, all_params):
    # Rename any params to avoid name collisions
    current_names = [x[0] for x in all_params]
    new_params = []
    old_to_new_mapping = {}
    for param in params:
        new_param_names = [x[0] for x in new_params]
        all_param_names = current_names + new_param_names
        old = param[0]
        if param[0] in all_param_names:
            i = 1
            while f'{i}_{param[0]}' in all_param_names:
                i += 1
            param = (f'{i}_{param[0]}', *param[1:])
        old_to_new_mapping[old] = param[0]
        new_params.append(param)

    return new_params, old_to_new_mapping


def add_engineering_constraints(params, engineering_constraints):
    if engineering_constraints == 'TARDIS':
        # Since these plates always have opposite signs in y, we can add
        # their absolute values to get the difference.
        dist_plates = np.abs(params['IMAGE_PLATE_2_tvec_y']) + np.abs(
            params['IMAGE_PLATE_4_tvec_y']
        )

        min_dist = 22.83
        max_dist = 23.43
        # if distance between plates exceeds a certain value, then cap it
        # at the max/min value and adjust the value of tvec_ys
        if dist_plates > max_dist:
            delta = np.abs(dist_plates - max_dist)
            dist_plates = max_dist
            params['IMAGE_PLATE_2_tvec_y'].value = (
                params['IMAGE_PLATE_2_tvec_y'].value + 0.5 * delta
            )
            params['IMAGE_PLATE_4_tvec_y'].value = (
                params['IMAGE_PLATE_4_tvec_y'].value - 0.5 * delta
            )
        elif dist_plates < min_dist:
            delta = np.abs(dist_plates - min_dist)
            dist_plates = min_dist
            params['IMAGE_PLATE_2_tvec_y'].value = (
                params['IMAGE_PLATE_2_tvec_y'].value - 0.5 * delta
            )
            params['IMAGE_PLATE_4_tvec_y'].value = (
                params['IMAGE_PLATE_4_tvec_y'].value + 0.5 * delta
            )

        params.add(
            'tardis_distance_between_plates',
            value=dist_plates,
            min=min_dist,
            max=max_dist,
            vary=True,
        )
        expr = 'tardis_distance_between_plates - abs(IMAGE_PLATE_2_tvec_y)'
        params['IMAGE_PLATE_4_tvec_y'].expr = expr


class LmfitValidationException(Exception):
    pass


def validate_params_list(params_list):
    # Make sure there are no duplicate names
    duplicate_names = []
    for i, x in enumerate(params_list):
        for y in params_list[i + 1 :]:
            if x[0] == y[0]:
                duplicate_names.append(x[0])

    if duplicate_names:
        msg = f'Duplicate names found in params list: {duplicate_names}'
        raise LmfitValidationException(msg)


EULER_PARAM_NAMES_MAPPING = {
    None: ('expmap_x', 'expmap_y', 'expmap_z'),
    ('xyz', True): ('euler_x', 'euler_y', 'euler_z'),
    ('zxz', False): ('euler_z', 'euler_xp', 'euler_zpp'),
}


def normalize_euler_convention(euler_convention):
    if isinstance(euler_convention, dict):
        return (
            euler_convention['axes_order'],
            euler_convention['extrinsic'],
        )

    return euler_convention


def param_names_euler_convention(base, euler_convention):
    normalized = normalize_euler_convention(euler_convention)
    return [f'{base}_{x}' for x in EULER_PARAM_NAMES_MAPPING[normalized]]


def detector_angles_euler(panel, euler_convention):
    if euler_convention is None:
        # Return exponential map parameters
        return panel.tilt

    normalized = normalize_euler_convention(euler_convention)
    rmat = panel.rmat
    rme = RotMatEuler(
        np.zeros(3),
        axes_order=normalized[0],
        extrinsic=normalized[1],
    )

    rme.rmat = rmat
    return np.degrees(rme.angles)


def set_detector_angles_euler(panel, base_name, params, euler_convention):
    normalized = normalize_euler_convention(euler_convention)
    names = param_names_euler_convention(base_name, euler_convention)

    angles = []
    for name in names:
        angles.append(params[name].value)

    angles = np.asarray(angles)

    if euler_convention is None:
        # No conversion needed
        panel.tilt = angles
        return

    rmat = make_rmat_euler(
        np.radians(angles),
        axes_order=normalized[0],
        extrinsic=normalized[1],
    )

    panel.tilt = expMapOfQuat(quatOfRotMat(rmat))
