import lmfit
import numpy as np

from .lmfit_param_handling import (
    create_instr_params,
    create_tth_parameters,
    DEFAULT_EULER_CONVENTION,
    update_instrument_from_params,
)

class Fiddle:
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
                 engineering_constraints=None,
                 euler_convention=DEFAULT_EULER_CONVENTION):

        self._instr = instr
        self._data = data
        self._tth_distortion = tth_distortion
        self._engineering_constraints = engineering_constraints
        self.euler_convention = euler_convention
        self.make_lmfit_params()
        self.set_minimizer()

    def set_minimizer(self):
        self.fitter = lmfit.Minimizer(self.calc_residual,
                                      self.params,
                                      nan_policy='omit')