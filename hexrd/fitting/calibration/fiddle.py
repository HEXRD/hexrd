import lmfit
import numpy as np

from hexrd.utils.hkl import hkl_to_str, str_to_hkl
from .lmfit_param_handling import (
    create_instr_params_fiddle,
    create_tth_parameters,
    DEFAULT_EULER_CONVENTION,
    update_instrument_from_params_fiddle,
)

class Fiddle_structureless:
    """this class is a special case of the calibration.
    The instrument is a multi-detector instrument. However,
    the relative locations of each panel is very tightly
    constrained. Therefore, we constrain the relative 
    translation and tilt of each panel w.r.t Icarus sensor 2
    (chosen at random). There is a "global" tvec and tilt
    angle which can translate and rotate the entire instrument.
    There is no overall tvec in the instrument class which will
    be used for translation. However, there is no global tilt
    angles with all degrees of freedom. So that will have to be
    manually dealt with.

    @Author: Saransh Singh, Lawrence Livermore National Lab
             saransh1@llnl.gov
    """
    def __init__(self,
                 instr,
                 data,
                 tth_distortion=None,
                 euler_convention=DEFAULT_EULER_CONVENTION):

        self._instr = instr
        self._data = data
        self._tth_distortion = tth_distortion
        self.euler_convention = euler_convention
        self.make_lmfit_params_fiddle()
        self.set_minimizer()

    def make_lmfit_params_fiddle(self):
        params = []
        params += create_instr_params_fiddle(self.instr, self.euler_convention)
        params += create_tth_parameters(self.meas_angles)

        params_dict = lmfit.Parameters()
        params_dict.add_many(*params)

        self.params = params_dict
        return params_dict

    def calc_residual(self, params):
        update_instrument_from_params_fiddle(
            self.instr,
            params,
            self.euler_convention,
        )

        residual = np.empty([0,])
        for ii, (rng, corr_rng) in enumerate(zip(self.meas_angles,
                                                 self.tth_correction)):
            for det_name, panel in self.instr.detectors.items():
                if rng[det_name] is not None:
                    if rng[det_name].size != 0:
                        tth_rng = params[f'DS_ring_{ii}'].value
                        tth_updated = np.degrees(rng[det_name][:, 0])
                        delta_tth = tth_updated - tth_rng
                        if corr_rng[det_name] is not None:
                            delta_tth -= np.degrees(corr_rng[det_name])
                        residual = np.concatenate((residual, delta_tth))

        return residual/len(residual)

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
                **odict
            }

            self.res = self.fitter.least_squares(self.params,
                                                 **odict)
        else:
            self.res = self.fitter.scalar_minimize(method=method,
                                                   params=self.params,
                                                   max_nfev=50000,
                                                   **odict)

        self.params = self.res.params
        # res = self.fitter.least_squares(**fdict)
        return self.res

    @property
    def nrings(self):
        """
        return dictionary over panels with number
        of DS rings on each panel
        """
        return len(self.data)

    @property
    def tth_distortion(self):
        return self._tth_distortion

    @tth_distortion.setter
    def tth_distortion(self, v):
        self._tth_distortion = v
        # No need to update lmfit parameters

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
                    tth_corr = sd.apply(meas_xy, return_nominal=False)[:, 0]
                    corr_dict[det_name] = tth_corr
            corr_list.append(corr_dict)
        return corr_list

    @property
    def two_XRS(self):
        return self.instr.has_multi_beam

class Fiddle_composite:
    """this class is a special case of the powder calibration.
    The instrument is a multi-detector instrument. However,
    the relative locations of each panel is very tightly
    constrained. Therefore, we constrain the relative 
    translation and tilt of each panel w.r.t Icarus sensor 2
    (chosen at random). There is a "global" tvec and tilt
    angle which can translate and rotate the entire instrument.
    There is no overall tvec in the instrument class which will
    be used for translation. However, there is no global tilt
    angles with all degrees of freedom. So that will have to be
    manually dealt with.

    @Author: Saransh Singh, Lawrence Livermore National Lab
             saransh1@llnl.gov
    """
    def __init__(self,
                 instr,
                 materials,
                 picks,
                 tth_distortion=None,
                 euler_convention=DEFAULT_EULER_CONVENTION):

        pass

    @property
    def calibration_picks(self):
        # Convert this from our internal data dict format
        picks = {}
        for det_key, data in self.data_dict.items():
            picks[det_key] = {}
            for ringset in data:
                for row in ringset:
                    # Rows 3, 4, and 5 are the hkl
                    hkl_str = hkl_to_str(row[3:6].astype(int))
                    picks[det_key].setdefault(hkl_str, [])
                    # Rows 0 and 1 are the xy coordinates
                    picks[det_key][hkl_str].append(row[:2].tolist())

        return picks

    @calibration_picks.setter
    def calibration_picks(self, v):
        # Convert this to our internal data dict format
        data_dict = {}
        for det_key, hkl_picks in v.items():
            data_dict[det_key] = []
            for hkl_str, picks in hkl_picks.items():
                if len(picks) == 0:
                    # Just skip over it
                    continue

                data = np.zeros((len(picks), 8), dtype=np.float64)
                # Rows 0 and 1 are the xy coordinates
                data[:, :2] = np.asarray(picks)
                # Rows 3, 4, and 5 are the hkl
                data[:, 3:6] = str_to_hkl(hkl_str)
                data_dict[det_key].append(data)

        self.data_dict = data_dict

