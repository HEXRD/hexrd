import copy

import numpy as np

from .laue import LaueCalibrator
from .powder import PowderCalibrator


class CompositeCalibration:
    def __init__(self, instr, processed_picks, img_dict):
        self.instr = instr
        self.original_instr = copy.deepcopy(instr)
        self.npi = len(self.instr.calibration_parameters)
        self.data = processed_picks
        calibrator_list = []
        params = []
        param_flags = []
        for pick_data in processed_picks:
            if pick_data['type'] == 'powder':
                # flags for calibrator
                lpflags = [i[1] for i in pick_data['refinements']]
                flags = np.hstack(
                    [self.instr.calibration_flags, lpflags]
                )
                param_flags.append(lpflags)

                kwargs = {
                    'instr': self.instr,
                    'plane_data': pick_data['plane_data'],
                    'img_dict': img_dict,
                    'flags': flags,
                    'tth_distortion': pick_data['tth_distortion'],
                }
                calib = PowderCalibrator(**kwargs)

                params.append(calib.full_params[-calib.npe:])
                calibrator_list.append(calib)

            elif pick_data['type'] == 'laue':
                # flags for calibrator
                gparams = pick_data['options']['crystal_params']
                min_energy = pick_data['options']['min_energy']
                max_energy = pick_data['options']['max_energy']

                gpflags = [i[1] for i in pick_data['refinements']]
                flags = np.hstack(
                    [self.instr.calibration_flags, gpflags]
                )
                param_flags.append(gpflags)
                calib = LaueCalibrator(
                    self.instr, pick_data['plane_data'],
                    gparams, flags,
                    min_energy=min_energy, max_energy=max_energy
                )
                params.append(calib.full_params[-calib.npe:])
                calibrator_list.append(calib)

        self.calibrators = calibrator_list
        self.params = np.hstack(params)
        self.param_flags = np.hstack(param_flags)
        self.full_params = np.hstack(
            [self.instr.calibration_parameters, self.params]
        )
        self.flags = np.hstack(
            [self.instr.calibration_flags, self.param_flags]
        )

    def reduced_params(self):
        return self.full_params[self.flags]

    def residual(self, reduced_params, pick_data_list):
        # first update a copy of the full parameter list
        full_params = np.array(self.full_params)
        full_params[self.flags] = reduced_params
        instr_params = full_params[:self.npi]
        addtl_params = full_params[self.npi:]

        def powder_residual(calib, these_reduced_params, data_dict):
            # Convert our data_dict into the input data format that
            # the powder calibrator is expecting.
            calibration_data = {}
            for det_key in data_dict['hkls']:
                # MUST use original instrument to convert these coordinates
                # so that we are consistent. This is because the instrument
                # gets modified with each call to `residual()`.
                panel = self.original_instr.detectors[det_key]

                picks_list = []
                for i, hkl in enumerate(data_dict['hkls'][det_key]):
                    picks = data_dict['picks'][det_key][i]
                    if not picks:
                        continue

                    # Each row consists of 8 columns:
                    # First two are measured x, y
                    # Third is unknown (appears unused in the PowderCalibrator)
                    # Four - six are the hkl indices
                    # Seven and Eight are unknown (appears unused here)

                    # We are going to convert our data into rows of 6 columns.
                    # We will fill in zero for the third column, and ignore
                    # the last two columns.

                    # Convert the angles to Cartesian
                    cartesian_picks = panel.angles_to_cart(
                        np.radians(picks),
                        apply_distortion=True,
                    )
                    repeated_zero = np.repeat([[0]], len(cartesian_picks),
                                              axis=0)
                    repeated_hkl = np.repeat([hkl], len(cartesian_picks),
                                             axis=0)
                    formatted = np.hstack((cartesian_picks, repeated_zero,
                                           repeated_hkl))
                    picks_list.append(formatted)

                calibration_data[det_key] = picks_list

            return calib.residual(these_reduced_params, calibration_data)

        def laue_residual(calib, these_reduced_params, data_dict):
            return calib.residual(these_reduced_params, data_dict)

        residual_funcs = {
            PowderCalibrator: powder_residual,
            LaueCalibrator: laue_residual,
        }

        # loop calibrators and collect residuals
        ii = 0
        residual = []
        for ical, calib in enumerate(self.calibrators):
            # make copy offull params for this calibrator
            these_full_params = np.hstack(
                [instr_params, addtl_params[ii:ii + calib.npe]]
            )

            # pull out reduced list
            these_reduced_params = these_full_params[calib.flags]

            # call to calibrator residual api with proper index into pick data
            f = residual_funcs[type(calib)]
            residual.append(
                f(calib, these_reduced_params, pick_data_list[ical])
            )

            # advance alibrator extra parameter offset
            ii += calib.npe

        # return single hstacked residual
        return np.hstack(residual)
