import lmfit
import numpy as np
from hexrd.instrument import calc_angles_from_beam_vec
from hexrd.rotations import RotMatEuler

def make_lmfit_params(instr,
                      meas_angles=None,
                      engineering_constraints=None,
                      calibration_type='structureless',
                      plane_data=None):
    """helper function to form a lmfit parameter class
    to be used by a generic calibrator class.

    Parameters
    ----------
    instr                   : hexrd.instrument.HEDMInstrument
                              instrument to be refined
    meas_angles             : list
                              intial guess for the line positions
                              in structureless calibration
    engineering_constraints : str
                              if 'TARDIS' then some extra constraints
                              are added 
    calibration_type        : str
                              'structureless', 'fast_powder',
                              'laue' or 'composite'. this keyword 
                              decides what parameters are added

    planeData               : hexrd.material.crystallography.PlaneData

    Returns
    -------
    params : lmfit.Parameters
        lmfit Parameter class object with all refinable variables
    """
    params = lmfit.Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    all_params = []
    add_instr_params(all_params,
                     instr)
    if calibration_type == 'structureless':
        if not isinstance(meas_angles, list):
            msg = f'incorrect input type for meas_angles'
            raise TypeError('msg')

        if len(meas_angles) > 0:
            add_tth_parameters(all_params,
                               meas_angles)
        else:
            msg = f'empty meas_angles list'
            raise ValueError(msg)

    params.add_many(*all_params)
    if engineering_constraints == 'TARDIS':
        # Since these plates always have opposite signs in y, we can add
        # their absolute values to get the difference.
        dist_plates = (np.abs(params['IMAGE_PLATE_2_tvec_y'])+
                       np.abs(params['IMAGE_PLATE_4_tvec_y']))

        min_dist = 22.83
        max_dist = 23.43
        # if distance between plates exceeds a certain value, then cap it
        # at the max/min value and adjust the value of tvec_ys
        if dist_plates > max_dist:
            delta = np.abs(dist_plates - max_dist)
            dist_plates = max_dist
            params['IMAGE_PLATE_2_tvec_y'].value = (
            params['IMAGE_PLATE_2_tvec_y'].value +
                0.5*delta)
            params['IMAGE_PLATE_4_tvec_y'].value = (
            params['IMAGE_PLATE_4_tvec_y'].value -
                0.5*delta)
        elif dist_plates < min_dist:
            delta = np.abs(dist_plates - min_dist)
            dist_plates = min_dist
            params['IMAGE_PLATE_2_tvec_y'].value = (
            params['IMAGE_PLATE_2_tvec_y'].value -
                0.5*delta)
            params['IMAGE_PLATE_4_tvec_y'].value = (
            params['IMAGE_PLATE_4_tvec_y'].value +
                0.5*delta)
        params.add('tardis_distance_between_plates',
                         value=dist_plates,
                         min=min_dist,
                         max=max_dist,
                         vary=True)
        expr = 'tardis_distance_between_plates - abs(IMAGE_PLATE_2_tvec_y)'
        params['IMAGE_PLATE_4_tvec_y'].expr = expr

    return params

def add_instr_params(parms_list,
                     instr):
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    if isinstance(instr.beam_vector, np.ndarray):
        azim, pol = calc_angles_from_beam_vec(instr.beam_vector)
        parms_list.append(('beam_polar', pol, False, pol-1, pol+1))
        parms_list.append(('beam_azimuth', azim, False, azim-1, azim+1))
        
    elif isinstance(instr.beam_vector, dict):
        for k, v in instr.beam_vectors.items():
            azim, pol = calc_angles_from_beam_vec(v)
            pname = f'{k}_beam_polar'
            aname = f'{k}_beam_azimuth'
            parms_list.append((pname, pol, False, pol-1, pol+1))
            parms_list.append((aname, azim, False, azim-1, azim+1))
    parms_list.append(('instr_chi', np.degrees(instr.chi),
                       False, np.degrees(instr.chi)-1,
                       np.degrees(instr.chi)+1))
    parms_list.append(('instr_tvec_x', instr.tvec[0], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_y', instr.tvec[1], False, -np.inf, np.inf))
    parms_list.append(('instr_tvec_z', instr.tvec[2], False, -np.inf, np.inf))
    euler_convention = {'axes_order': 'zxz',
                        'extrinsic': False}
    for det_name, panel in instr.detectors.items():
        det = det_name.replace('-', '_')
        rmat = panel.rmat
        rme = RotMatEuler(np.zeros(3,),
                          **euler_convention)
        rme.rmat = rmat
        euler = np.degrees(rme.angles)

        parms_list.append((f'{det}_euler_z',
                           euler[0],
                           False,
                           euler[0]-2,
                           euler[0]+2))
        parms_list.append((f'{det}_euler_xp',
                           euler[1],
                           False,
                           euler[1]-2,
                           euler[1]+2))
        parms_list.append((f'{det}_euler_zpp',
                           euler[2],
                           False,
                           euler[2]-2,
                           euler[2]+2))

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
            for ii,pp in enumerate(p):
                parms_list.append((f'{det}_distortion_param_{ii}',pp,
                                   False, -np.inf, np.inf))
        if panel.detector_type.lower() == 'cylindrical':
            parms_list.append((f'{det}_radius', panel.radius, False, -np.inf, np.inf))

def add_tth_parameters(parms_list,
                       meas_angles):
    for ii,tth in enumerate(meas_angles):
        ds_ang = np.empty([0,])
        for k,v in tth.items():
            if v is not None:
                ds_ang = np.concatenate((ds_ang, v[:,0]))
        if ds_ang.size != 0:
            val = np.degrees(np.mean(ds_ang))
            parms_list.append((f'DS_ring_{ii}',
                               val,
                               True,
                               val-5.,
                               val+5.))