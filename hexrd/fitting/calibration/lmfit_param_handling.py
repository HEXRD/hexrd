import lmfit
import numpy as np

from hexrd.instrument import (
    calc_angles_from_beam_vec,
    calc_beam_vec,
    HEDMInstrument,
)
from hexrd.rotations import (
    expMapOfQuat,
    make_rmat_euler,
    quatOfRotMat,
    RotMatEuler,
    rotMatOfExpMap
)
from hexrd.material.unitcell import _lpname
import copy

# First is the axes_order, second is extrinsic
DEFAULT_EULER_CONVENTION = ('zxz', False)


def create_instr_params(instr, euler_convention=DEFAULT_EULER_CONVENTION):
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    parms_list = []

    # This supports either single beam or multi-beam
    beam_param_names = create_beam_param_names(instr)
    for beam_name, beam in instr.beam_dict.items():
        azim, pol = calc_angles_from_beam_vec(beam['vector'])
        energy = beam['energy']

        names = beam_param_names[beam_name]
        parms_list.append((
            names['beam_polar'], pol, False, pol - 1, pol + 1
        ))
        parms_list.append((
            names['beam_azimuth'], azim, False, azim - 1, azim + 1
        ))
        parms_list.append((
            names['beam_energy'], energy, False, energy - 0.2, energy + 0.2
        ))

    parms_list.append(('instr_chi', np.degrees(instr.chi),
                       False, np.degrees(instr.chi)-1,
                       np.degrees(instr.chi)+1))
    parms_list.append(('instr_tvec_x', instr.tvec[0], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_y', instr.tvec[1], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_z', instr.tvec[2], False, -np.inf, np.inf))
    for det_name, panel in instr.detectors.items():
        det = det_name.replace('-', '_')

        angles = detector_angles_euler(panel, euler_convention)
        angle_names = param_names_euler_convention(det, euler_convention)

        for name, angle in zip(angle_names, angles):
            parms_list.append((name,
                               angle,
                               False,
                               angle - 2,
                               angle + 2))

        parms_list.append((f'{det}_tvec_x',
                           panel.tvec[0],
                           True,
                           panel.tvec[0]-1,
                           panel.tvec[0]+1))
        parms_list.append((f'{det}_tvec_y',
                           panel.tvec[1],
                           True,
                           panel.tvec[1]-0.5,
                           panel.tvec[1]+0.5))
        parms_list.append((f'{det}_tvec_z',
                           panel.tvec[2],
                           True,
                           panel.tvec[2]-1,
                           panel.tvec[2]+1))
        if panel.distortion is not None:
            p = panel.distortion.params
            for ii, pp in enumerate(p):
                parms_list.append((f'{det}_distortion_param_{ii}', pp,
                                   False, -np.inf, np.inf))
        if panel.detector_type.lower() == 'cylindrical':
            parms_list.append((f'{det}_radius', panel.radius, False,
                               -np.inf, np.inf))

    return parms_list

def create_instr_params_fiddle(instr, euler_convention=DEFAULT_EULER_CONVENTION):
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    parms_list = []
    if instr.has_multi_beam:
        for k, v in instr.multi_beam_dict.items():
            azim, pol = calc_angles_from_beam_vec(v['beam_vector'])
            pname = f'{k}_beam_polar'
            aname = f'{k}_beam_azimuth'
            parms_list.append((pname, pol, False, pol-1, pol+1))
            parms_list.append((aname, azim, False, azim-1, azim+1))

            bname = f'{k}_beam_energy'
            beam_energy = v['beam_energy']
            parms_list.append((bname,
                               beam_energy,
                               False,
                               beam_energy-0.2,
                               beam_energy+0.2))
    else:
        azim, pol = calc_angles_from_beam_vec(instr.beam_vector)
        parms_list.append(('beam_polar', pol, False, pol-1, pol+1))
        parms_list.append(('beam_azimuth', azim, False, azim-1, azim+1))

        parms_list.append(('beam_energy',
                           instr.beam_energy,
                           False,
                           instr.beam_energy-0.2,
                           instr.beam_energy+0.2))

    parms_list.append(('instr_chi', np.degrees(instr.chi),
                       False, np.degrees(instr.chi)-1,
                       np.degrees(instr.chi)+1))
    parms_list.append(('instr_tvec_x', instr.tvec[0], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_y', instr.tvec[1], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_z', instr.tvec[2], False, -np.inf, np.inf))

    parms_list.append(('instr_tilt_x', instr.tilt[0], False, -np.inf, np.inf))
    parms_list.append(('instr_tilt_y', instr.tilt[1], False, -np.inf, np.inf))
    parms_list.append(('instr_tilt_z', instr.tilt[2], False, -np.inf, np.inf))

    for det_name, panel in instr.detectors.items():
        if panel.distortion is not None:
                p = panel.distortion.params
                for ii, pp in enumerate(p):
                    parms_list.append((f'{det}_distortion_param_{ii}', pp,
                                       False, -np.inf, np.inf))

    return parms_list

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


def update_instrument_from_params(instr, params, euler_convention):
    """
    this function updates the instrument from the
    lmfit parameter list. we don't have to keep track
    of the position numbers as the variables are named
    variables. this will become the standard in the
    future since bound constraints can be very easily
    implemented.
    """
    if not isinstance(params, lmfit.Parameters):
        msg = ('Only lmfit.Parameters is acceptable input. '
               f'Received: {params}')
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

    instr_tvec = [params['instr_tvec_x'].value,
                  params['instr_tvec_y'].value,
                  params['instr_tvec_z'].value]
    instr.tvec = np.r_[instr_tvec]

    for det_name, detector in instr.detectors.items():
        det = det_name.replace('-', '_')
        set_detector_angles_euler(detector, det, params, euler_convention)

        tvec = np.r_[params[f'{det}_tvec_x'].value,
                     params[f'{det}_tvec_y'].value,
                     params[f'{det}_tvec_z'].value]
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

def update_instrument_from_params_fiddle(instr, params, euler_convention):
    """
    this function is specifically meant for the fiddle
    instrument. Only a global translation and rotation
    parameter controls the calibration. the relative
    locations/orientations of the icarus detectors are 
    fixed.
    """
    if not isinstance(params, lmfit.Parameters):
        msg = ('Only lmfit.Parameters is acceptable input. '
               f'Received: {params}')
        raise NotImplementedError(msg)

    instr.beam_energy = params['beam_energy'].value

    azim = params['beam_azimuth'].value
    pola = params['beam_polar'].value
    instr.beam_vector = calc_beam_vec(azim, pola)

    instr_tvec = [params['instr_tvec_x'].value,
                  params['instr_tvec_y'].value,
                  params['instr_tvec_z'].value]
    instr.tvec = np.r_[instr_tvec]

    chi = np.radians(params['instr_chi'].value)
    instr.chi = chi

    instr_tilt = [params['instr_tilt_x'].value,
                  params['instr_tilt_y'].value,
                  params['instr_tilt_z'].value]
    instr.tilt = np.r_[instr_tilt]

    update_detector_pretilt(instr)

    for det_name, detector in instr.detectors.items():
        det = det_name.replace('-', '_')
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

            parms_list.append((f'{prefix}{ii}',
                               val,
                               True,
                               val-5.,
                               val+5.))

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
        parms_list.append((
            f'{material.name}_{lp_name}',
            val,
            refine,
            val - diff,
            val + diff,
        ))

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


def grain_param_names(mat_name):
    return [f'{mat_name}_grain_param_{i}' for i in range(12)]


def create_grain_params(mat_name, grain, refinements=None):
    param_names = grain_param_names(mat_name)
    if refinements is None:
        refinements = [True] * len(param_names)

    parms_list = []
    for i, name in enumerate(param_names):
        parms_list.append((
            name,
            grain[i],
            refinements[i],
            grain[i] - 2,
            grain[i] + 2,
        ))
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
        dist_plates = (np.abs(params['IMAGE_PLATE_2_tvec_y']) +
                       np.abs(params['IMAGE_PLATE_4_tvec_y']))

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

        params.add('tardis_distance_between_plates',
                   value=dist_plates,
                   min=min_dist,
                   max=max_dist,
                   vary=True)
        expr = 'tardis_distance_between_plates - abs(IMAGE_PLATE_2_tvec_y)'
        params['IMAGE_PLATE_4_tvec_y'].expr = expr


class LmfitValidationException(Exception):
    pass


def validate_params_list(params_list):
    # Make sure there are no duplicate names
    duplicate_names = []
    for i, x in enumerate(params_list):
        for y in params_list[i + 1:]:
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
        np.zeros(3,),
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

def update_detector_pretilt(instr):
    for det_name, panel in instr.detectors.items():
        panel.pretilt = instr.tilt
