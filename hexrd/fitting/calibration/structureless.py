import lmfit
import numpy as np

from hexrd.instrument import calc_angles_from_beam_vec
from hexrd.rotations import RotMatEuler


class StructurelessCalibrator:
    """
    this class implements the equivalent of the
    powder calibrator but without constraining
    the optimization to a structure. in this
    implementation, the location of the constant
    two theta line that a set of points lie on
    is also an optimization parameter.

    unlike the previous implementations, this routine
    is based on the lmfit module to implement the
    more complicated constraints for the TARDIS box

    if TARDIS_constraints are set to True, then the following
    additional linear constraint is added to the calibration

    22.83 mm <= |IMAGE-PLATE-2 tvec[1]| + |IMAGE-PLATE-2 tvec[1]| <= 23.43 mm

    """
    def __init__(self,
                 instr,
                 data,
                 tth_distortion=None,
                 engineering_constraints=None):

        self._instr = instr
        self._data = data
        self._tth_distortion = tth_distortion
        self._engineering_constraints = engineering_constraints
        self.make_lmfit_params()
        self.set_minimizer()

    def make_lmfit_params(self):
        self.params = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        all_params = []
        self.add_instr_params(all_params)
        self.add_tth_parameters(all_params)
        self.params.add_many(*all_params)
        if self.engineering_constraints == 'TARDIS':
            # Since these plates always have opposite signs in y, we can add
            # their absolute values to get the difference.
            dist_plates = (np.abs(self.params['IMAGE_PLATE_2_tvec_y'])+
                           np.abs(self.params['IMAGE_PLATE_4_tvec_y']))

            min_dist = 22.83
            max_dist = 23.43
            # if distance between plates exceeds a certain value, then cap it
            # at the max/min value and adjust the value of tvec_ys
            if dist_plates > max_dist:
                delta = np.abs(dist_plates - max_dist)
                dist_plates = max_dist
                self.params['IMAGE_PLATE_2_tvec_y'].value = (
                    self.params['IMAGE_PLATE_2_tvec_y'].value +
                    0.5*delta)
                self.params['IMAGE_PLATE_4_tvec_y'].value = (
                    self.params['IMAGE_PLATE_4_tvec_y'].value -
                    0.5*delta)
            elif dist_plates < min_dist:
                delta = np.abs(dist_plates - min_dist)
                dist_plates = min_dist
                self.params['IMAGE_PLATE_2_tvec_y'].value = (
                    self.params['IMAGE_PLATE_2_tvec_y'].value -
                    0.5*delta)
                self.params['IMAGE_PLATE_4_tvec_y'].value = (
                    self.params['IMAGE_PLATE_4_tvec_y'].value +
                    0.5*delta)
            self.params.add('tardis_distance_between_plates',
                             value=dist_plates,
                             min=min_dist,
                             max=max_dist,
                             vary=True)
            expr = 'tardis_distance_between_plates - abs(IMAGE_PLATE_2_tvec_y)'
            self.params['IMAGE_PLATE_4_tvec_y'].expr = expr

    def add_instr_params(self, parms_list):
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        instr = self.instr
        parms_list.append(('beam_energy',instr.beam_energy,
                            False, instr.beam_energy-0.2,
                            instr.beam_energy+0.2))
        azim, pol = calc_angles_from_beam_vec(instr.beam_vector)
        parms_list.append(('beam_polar', pol, False, pol-2, pol+2))
        parms_list.append(('beam_azimuth', azim, False, azim-2, azim+2))
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

    def add_tth_parameters(self, parms_list):
        angs = self.meas_angles
        for ii,tth in enumerate(angs):
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

    @property
    def engineering_params(self):
        ret = []
        if self.engineering_constraints == 'TARDIS':
            ret.append('tardis_distance_between_plates')
        return ret

    def calc_residual(self, params):
        self.instr.update_from_lmfit_parameter_list(params)
        residual = np.empty([0,])
        for ii, (rng, corr_rng) in enumerate(zip(self.meas_angles, self.tth_correction)):
            for det_name, panel in self.instr.detectors.items():
                if rng[det_name] is not None:
                    if rng[det_name].size != 0:
                        tth_rng = params[f'DS_ring_{ii}'].value
                        tth_updated = np.degrees(rng[det_name][:,0])
                        delta_tth = tth_updated - tth_rng
                        if corr_rng[det_name] is not None:
                            delta_tth -= np.degrees(corr_rng[det_name])
                        residual = np.concatenate((residual, delta_tth))

        return residual

    def set_minimizer(self):
        self.fitter = lmfit.Minimizer(self.calc_residual,
                                      self.params,
                                      nan_policy='omit')

    def run_calibration(self,
                        method='least_squares',
                        odict=None):
        """
        odict is the options dictionary
        """
        if odict is None:
            odict = {}

        if method == 'least_squares':
            fdict = {
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "verbose": 2,
                "max_nfev": 1000,
                "x_scale": "jac",
                "method": "trf",
                "jac": "3-point",
            }
            fdict.update(odict)

            self.res = self.fitter.least_squares(self.params,
                                                 **fdict)
        else:
            fdict = odict
            self.res = self.fitter.scalar_minimize(method=method,
                                                   params=self.params,
                                                   max_nfev=50000,
                                                   **fdict)

        self.params = self.res.params
        # res = self.fitter.least_squares(**fdict)
        return self.res

    @property
    def nrings(self):
        """
        return dictionary over panels with number
        of DS rings on each panel
        """
        return len(data)

    @property
    def tth_distortion(self):
        return self._tth_distortion

    @tth_distortion.setter
    def tth_distortion(self, v):
        self._tth_distortion = v
        # No need to update lmfit parameters

    @property
    def engineering_constraints(self):
        return self._engineering_constraints

    @engineering_constraints.setter
    def engineering_constraints(self, v):
        if v == self._engineering_constraints:
            return

        valid_settings = [
            None,
            'None',
            'TARDIS',
        ]
        if v not in valid_settings:
            valid_str = ', '.join(map(valid_settings, str))
            msg = (
                f'Invalid engineering constraint "{v}". Valid constraints '
                f'are: "{valid_str}"'
            )
            raise Exception(msg)

        self._engineering_constraints = v
        self.make_lmfit_params()

    @property
    def instr(self):
        return self._instr

    @instr.setter
    def instr(self, ins):
        self._instr = ins
        self.make_lmfit_params()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dat):
        self._data = dat
        self.make_lmfit_params()

    @property
    def residual(self):
        return self.calc_residual(self.params)

    @property
    def meas_angles(self):
        """
        this property will return a dictionary
        of angles based on current instrument
        parameters.
        """
        ang_list = []
        for rng in self.data:
            ang_dict = dict.fromkeys(self.instr.detectors)
            for det_name, meas_xy in rng.items():

                panel = self.instr.detectors[det_name]
                angles, _ = panel.cart_to_angles(
                                            meas_xy,
                                            tvec_s=self.instr.tvec,
                                            apply_distortion=True)
                ang_dict[det_name] = angles
            ang_list.append(ang_dict)

        return ang_list

    @property
    def tth_correction(self):
        corr_list = []
        for rng in self.data:
            corr_dict = dict.fromkeys(self.instr.detectors)
            if self.tth_distortion is not None:
                for det_name, meas_xy in rng.items():
                    # !!! sd has ref to detector so is updated
                    sd = self.tth_distortion[det_name]
                    tth_corr = sd.apply(meas_xy, return_nominal=False)[:,0]
                    corr_dict[det_name] = tth_corr
            corr_list.append(corr_dict)
        return corr_list
