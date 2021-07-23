#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a cmd-line callable utility for converting older-style
instrument YAML configuration files to hexrd >=0.6

Examples
--------
    $ python convert_instrument_config.py --help
    $ python convert_instrument_config.py <old_param.yml> -o new_param.yml

Notes
-----

    Note that the format changes include the following:
    * All translation vector components now have the key `translation` rather
      than `t_vec_*` for clarity.
    * The `tilt_angles` for each detector are now under `tilt`, and have been
      changed from extrinsic XYZ Euler angles (in radians) to the associated
      rotation matrix invariants; specifically angle*axis (dimensionless).
      This was done to more easily facilitate the use of different user-
      specified Euler angle conventions in external interfaces.

Todo:
    * Add to hexrd/scripts
"""

import argparse
import yaml
from hexrd.rotations import make_rmat_euler, angleAxisOfRotMat

# =============================================================================
# PARAMETERS
# =============================================================================

# top level keys
ROOT_KEYS = ['beam', 'oscillation_stage', 'detectors']
OLD_OSCILL_KEYS = ['t_vec_s', 'chi']
NEW_OSCILL_KEYS = ['translation', 'chi']
OLD_TRANSFORM_KEYS = ['t_vec_d', 'tilt_angles']
NEW_TRANSFORM_KEYS = ['translation', 'tilt']


# =============================================================================
# FUNCTIONS
# =============================================================================

def convert_instrument_config(old_cfg, output=None):
    """
    Convert v0.5.x style instrument YAML config file to v0.6.x style.

    Parameters
    ----------
    old_cfg : str
        The filename of the v0.5.x style instrument YAML config.
    output : str, optional
        The filename to write the updated v0.6.x instrument YAML config to.
        The default is None, in which case the new YAML format is printed to
        stdout.

    Raises
    ------
    RuntimeError
        Raises exception if input config has any unrecognized keys.

    Returns
    -------
    None.

    """
    icfg = yaml.safe_load(open(old_cfg, 'r'))

    new_cfg = dict.fromkeys(icfg)

    # %% first beam
    new_cfg['beam'] = icfg['beam']

    # %% next, calibration crystal if applicable
    calib_key = 'calibration_crystal'
    if calib_key in icfg.keys():
        new_cfg[calib_key] = icfg[calib_key]

    # %% sample stage
    old_dict = icfg['oscillation_stage']
    tmp_dict = dict.fromkeys(NEW_OSCILL_KEYS)
    for tk in zip(OLD_OSCILL_KEYS, NEW_OSCILL_KEYS):
        tmp_dict[tk[1]] = old_dict[tk[0]]
    new_cfg['oscillation_stage'] = tmp_dict

    # %% detectors
    new_cfg['detectors'] = dict.fromkeys(icfg['detectors'])
    det_block_keys = ['pixels', 'saturation_level', 'transform', 'distortion']
    for det_id, source_params in icfg['detectors'].items():
        new_dict = {}
        for key in det_block_keys:
            if key != 'transform':
                try:
                    new_dict[key] = source_params[key]
                except(KeyError):
                    if key == 'distortion':
                        continue
                    elif key == 'saturation_level':
                        new_dict[key] = 2**16
                    else:
                        raise RuntimeError(
                            "unrecognized parameter key '%s'" % key
                        )
            else:
                old_dict = source_params[key]
                tmp_dict = dict.fromkeys(NEW_TRANSFORM_KEYS)
                for tk in zip(OLD_TRANSFORM_KEYS, NEW_TRANSFORM_KEYS):
                    if tk[0] == 't_vec_d':
                        tmp_dict[tk[1]] = old_dict[tk[0]]
                    elif tk[0] == 'tilt_angles':
                        xyz_angles = old_dict[tk[0]]
                        rmat = make_rmat_euler(xyz_angles,
                                               'xyz',
                                               extrinsic=True)
                        phi, n = angleAxisOfRotMat(rmat)
                        tmp_dict[tk[1]] = (phi*n.flatten()).tolist()
                new_dict[key] = tmp_dict
        new_cfg['detectors'][det_id] = new_dict

    # %% dump new file
    if output is None:
        print(new_cfg)
    else:
        yaml.dump(new_cfg, open(output, 'w'))
    return


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="convert a v0.5.x instrument config to v0.6.x format"
    )

    parser.add_argument(
        'instrument_cfg',
        help="v0.5 style instrument config YAML file"
    )

    parser.add_argument(
        '-o', '--output-file',
        help="output file name",
        type=str,
        default=""
    )

    args = parser.parse_args()

    old_cfg = args.instrument_cfg
    output_file = args.output_file

    if len(output_file) == 0:
        output_file = None

    convert_instrument_config(old_cfg, output=output_file)
