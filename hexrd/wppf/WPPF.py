# standard imports
# ---------
from os import path
import time
import warnings


# 3rd party imports
# -----------------
import h5py

import lmfit

import numpy as np

from scipy.interpolate import CubicSpline

# hexrd imports
# -------------
from hexrd import constants
from hexrd.imageutil import snip1d_quad
from hexrd.material import Material
from hexrd.utils.multiprocess_generic import GenericMultiprocessing
from hexrd.valunits import valWUnit
from hexrd.wppf.peakfunctions import (
    calc_rwp,
    computespectrum_pvfcj,
    computespectrum_pvtch,
    computespectrum_pvpink,
    calc_Iobs_pvfcj,
    calc_Iobs_pvtch,
    calc_Iobs_pvpink,
)
from hexrd.wppf import wppfsupport
from hexrd.wppf.spectrum import Spectrum
from hexrd.wppf.parameters import Parameters
from hexrd.wppf.phase import (
    Phases_LeBail,
    Phases_Rietveld,
    Material_LeBail,
    Material_Rietveld,
)


class LeBail:
    """
    ===========================================================================

    >> @AUTHOR: Saransh Singh, Lawrence Livermore National Lab,
                saransh1@llnl.gov
    >> @DATE: 05/19/2020 SS 1.0 original
              09/11/2020 SS 1.1 expt_spectrum, params and phases now have
              multiple input option for easy integration with hexrdgui
              09/14/2020 SS 1.2 bkgmethod is now a dictionary. if method is
              'chebyshev', the the value specifies the degree of the polynomial
              to use for background estimation
              01/22/2021 SS 1.3 added intensity_init option to initialize
              intensity with structure factors if the user so chooses
              01/22/2021 SS 1.4 added option to specify background via a
              filename or numpy array
              03/12/2021 SS 1.5 added _generate_default_parameter function

    >> @DETAILS: this is the main LeBail class and contains all the refinable
                 parameters for the analysis. Since the LeBail method has no
                 structural information during refinement, the refinable
                 parameters for this model will be:

                 1. a, b, c, alpha, beta, gamma : unit cell parameters
                 2. U, V, W : cagliotti paramaters
                 3. 2theta_0 : Instrumental zero shift error
                 4. eta1, eta2, eta3 : weight factor for gaussian vs lorentzian

                 @NOTE: All angles are always going to be in degrees

    >> @PARAMETERS:
        expt_spectrum: name of file or numpy array or Spectrum
                       class of experimental intensity
        params: yaml file or dictionary or Parameter class
        phases: yaml file or dictionary or Phases_Lebail class
        wavelength: dictionary of wavelengths
        bkgmethod: method to estimate background. either spline or chebyshev
                   fit or filename or numpy array
                   (last two options added 01/22/2021 SS)
        Intensity_init: if set to none, then some power of 10 is used.
                        User has option to pass in dictionary of structure
                        factors. must ensure that the size of structure factor
                        matches the possible reflections (added 01/22/2021 SS)
    ============================================================================
    """

    def _nm(x):
        return valWUnit("lp", "length", x, "nm")

    def __init__(
        self,
        expt_spectrum=None,
        params=None,
        phases=None,
        wavelength={
            "kalpha1": [_nm(0.15406), 1.0],
            "kalpha2": [_nm(0.154443), 1.0]
        },
        bkgmethod={"spline": None},
        intensity_init=None,
        peakshape="pvfcj",
    ):

        self.peakshape = peakshape
        self.bkgmethod = bkgmethod
        self.intensity_init = intensity_init

        # self.initialize_expt_spectrum(expt_spectrum)
        self.spectrum_expt = expt_spectrum

        if wavelength is not None:
            self.wavelength = wavelength

        self._tstart = time.time()

        self.phases = phases

        self.params = params

        self.initialize_Icalc()

        self.computespectrum()

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart
        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

    def __str__(self):
        resstr = "<LeBail Fit class>\nParameters of \
        the model are as follows:\n"
        resstr += self.params.__str__()
        return resstr

    def checkangle(ang, name):

        if np.abs(ang) > 180.0:
            warnings.warn(
                name
                + " : the absolute value of angles \
                                seems to be large > 180 degrees"
            )

    def params_vary_off(self):
        """
        no params are varied
        """
        for p in self.params:
            self.params[p].vary = False

    def params_vary_on(self):
        """
        all params are varied
        """
        for p in self.params:
            self.params[p].vary = True

    def dump_hdf5(self, file):
        """
        >> @AUTHOR: Saransh Singh, Lawrence Livermore National Lab,
                    saransh1@llnl.gov
        >> @DATE:   01/19/2021 SS 1.0 original
        >> @DETAILS: write out the hdf5 file with all the spectrum, parameters
                     and phases pecified by filename or h5py.File object
        """
        if isinstance(file, str):
            fexist = path.isfile(file)
            if fexist:
                fid = h5py.File(file, "r+")
            else:
                fid = h5py.File(file, "x")

        elif isinstance(file, h5py.File):
            fid = file

        else:
            raise RuntimeError(
                "Parameters: dump_hdf5 Pass in a filename \
                string or h5py.File object"
            )

        self.phases.dump_hdf5(fid)
        self.params.dump_hdf5(fid)
        self.spectrum_expt.dump_hdf5(fid, "experimental")
        self.spectrum_sim.dump_hdf5(fid, "simulated")
        self.background.dump_hdf5(fid, "background")

    def initialize_bkg(self):
        """
        the cubic spline seems to be the ideal route in terms
        of determining the background intensity. this involves
        selecting a small (~5) number of points from the spectrum,
        usually called the anchor points. a cubic spline interpolation
        is performed on this subset to estimate the overall background.
        scipy provides some useful routines for this

        the other option implemented is the chebyshev polynomials. this
        basically automates the background determination and removes the
        user from the loop which is required for the spline type background.
        """
        if self.bkgmethod is None:
            self._background = []
            for tth in self.tth_list:
                self._background.append(Spectrum(x=tth, y=np.zeros(tth.shape)))

        elif "spline" in self.bkgmethod.keys():
            self._background = []
            self.selectpoints()
            for i, pts in enumerate(self.points):
                tth = self._spectrum_expt[i]._x
                x = pts[:, 0]
                y = pts[:, 1]
                self._background.append(self.splinefit(x, y, tth))

        elif "chebyshev" in self.bkgmethod.keys():
            self.chebyshevfit()

        elif "file" in self.bkgmethod.keys():
            if len(self._spectrum_expt) > 1:
                raise RuntimeError(
                    "initialize_bkg: \
                    file input not allowed for \
                    masked spectra."
                )
            else:
                bkg = Spectrum.from_file(self.bkgmethod["file"])
                x = bkg.x
                y = bkg.y
                cs = CubicSpline(x, y)

                yy = cs(self.tth_list)

                self._background = [Spectrum(x=self.tth_list[0], y=yy)]

        elif "array" in self.bkgmethod.keys():
            if len(self._spectrum_expt) > 1:
                raise RuntimeError(
                    "initialize_bkg: \
                    file input not allowed for \
                    masked spectra."
                )
            else:
                x = self.bkgmethod["array"][:, 0]
                y = self.bkgmethod["array"][:, 1]
                cs = CubicSpline(x, y)

                yy = cs(self._tth_list)

                self._background = [Spectrum(x=self.tth_list, y=yy)]

        elif "snip1d" in self.bkgmethod.keys():
            self._background = []
            for i, s in enumerate(self._spectrum_expt):
                if not self.tth_step:
                    ww = 3
                else:
                    if self.tth_step[i] > 0.0:
                        ww = np.rint(
                            self.bkgmethod["snip1d"][0] / self.tth_step[i]
                        ).astype(np.int32)
                    else:
                        ww = 3

                numiter = self.bkgmethod["snip1d"][1]

                yy = np.squeeze(
                    snip1d_quad(np.atleast_2d(s.y), w=ww, numiter=numiter)
                )
                self._background.append(Spectrum(x=self._tth_list[i], y=yy))

    def chebyshevfit(self):
        """
        03/08/2021 SS spectrum_expt is a list now. accounting
        for that change
        """
        self._background = []
        degree = self.bkgmethod["chebyshev"]

        for i, s in enumerate(self._spectrum_expt):
            tth = self._tth_list[i]
            if s.y.shape[0] <= degree:
                self._background.append(Spectrum(x=tth, y=np.zeros(tth.shape)))
            else:
                p = np.polynomial.Chebyshev.fit(
                    tth, s.y, degree, w=self._weights[i] ** 4
                )
                self._background.append(Spectrum(x=tth, y=p(tth)))

    def selectpoints(self):
        """
        03/08/2021 SS spectrum_expt is a list now. accounting
        for that change
        """
        # Keep matplotlib as an optional dependency
        # FIXME: move this function to where it is needed
        from pylab import plot, ginput, close, title, xlabel, ylabel

        self.points = []
        for i, s in enumerate(self._spectrum_expt):
            txt = (
                f"Select points for background estimation;\n"
                f"click middle mouse button when done. segment # {i}"
            )
            title(txt)

            plot(s.x, s.y, "-k")
            xlabel("2$\theta$")
            ylabel("intensity (a.u.)")

            self.points.append(np.asarray(ginput(0, timeout=-1)))

            close()

    # cubic spline fit of background using custom points chosen from plot
    def splinefit(self, x, y, tth):
        """
        03/08/2021 SS adding tth as input. this is the
        list of points for which background is estimated
        """
        cs = CubicSpline(x, y)
        bkg = cs(tth)
        return Spectrum(x=tth, y=bkg)

    def calctth(self):
        self.tth = {}
        self.hkls = {}
        self.dsp = {}
        for p in self.phases:
            self.tth[p] = {}
            self.hkls[p] = {}
            self.dsp[p] = {}
            for k, l in self.phases.wavelength.items():
                t = self.phases[p].getTTh(l[0].getVal('nm'))
                allowed = self.phases[p].wavelength_allowed_hkls
                t = t[allowed]
                hkl = self.phases[p].hkls[allowed, :]
                dsp = self.phases[p].dsp[allowed]
                tth_min = min(self.tth_min)
                tth_max = max(self.tth_max)
                limit = np.logical_and(t >= tth_min, t <= tth_max)
                self.tth[p][k] = t[limit]
                self.hkls[p][k] = hkl[limit, :]
                self.dsp[p][k] = dsp[limit]

    def initialize_Icalc(self):
        """
        @DATE 01/22/2021 SS modified the function so Icalc can be initialized
        with a dictionary of structure factors
        """

        self.Icalc = {}

        if self.intensity_init is None:
            if self.spectrum_expt._y.max() > 0:
                n10 = np.floor(np.log10(self.spectrum_expt._y.max())) - 2
            else:
                n10 = 0

            for p in self.phases:
                self.Icalc[p] = {}
                for k in self.phases.wavelength.keys():
                    self.Icalc[p][k] = \
                        (10 ** n10) * np.ones(self.tth[p][k].shape)

        elif isinstance(self.intensity_init, dict):
            """
            first check if intensities for all phases are present in the
            passed dictionary
            """
            for p in self.phases:
                if p not in self.intensity_init:
                    raise RuntimeError(
                        "LeBail: Intensity was initialized\
                     using custom values. However, initial values for one \
                     or more phases seem to be missing from the dictionary."
                    )
                self.Icalc[p] = {}

                """
                now check that the size of the initial intensities provided is
                consistent with the number of reflections
                    (size of initial intensity > size of hkl is allowed.
                     the unused values are ignored.)
                for this we need to step through the different wavelengths in
                the spectrum and check each of them
                """
                for k in self.phases.wavelength:
                    if k not in self.intensity_init[p]:
                        raise RuntimeError(
                            "LeBail: Intensity was initialized\
                         using custom values. However, initial values for one \
                         or more wavelengths in spectrum seem to be missing \
                         from the dictionary."
                        )

                    if (self.tth[p][k].shape[0]
                            <= self.intensity_init[p][k].shape[0]):
                        self.Icalc[p][k] = self.intensity_init[p][k][
                            0:self.tth[p][k].shape[0]
                        ]
        else:
            raise RuntimeError(
                "LeBail: Intensity_init must be either\
                 None or a dictionary"
            )

    def computespectrum(self):
        """
        >> @AUTHOR: Saransh Singh, Lawrence Livermore National Lab,
                    saransh1@llnl.gov
        >> @DATE:   06/08/2020 SS 1.0 original
        >> @DETAILS: compute the simulated spectrum
        """
        x = self.tth_list
        y = np.zeros(x.shape)
        tth_list = np.ascontiguousarray(self.tth_list)

        for iph, p in enumerate(self.phases):

            for k, l in self.phases.wavelength.items():

                Ic = self.Icalc[p][k]

                shft_c = np.cos(0.5 * np.radians(self.tth[p][k])) * self.shft
                trns_c = np.sin(np.radians(self.tth[p][k])) * self.trns
                tth = self.tth[p][k] + self.zero_error + shft_c + trns_c

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                # n = np.min((tth.shape[0], Ic.shape[0]))
                shkl = self.phases[p].shkl
                name = self.phases[p].name
                eta_n = f"self.{name}_eta_fwhm"
                eta_fwhm = eval(eta_n)
                strain_direction_dot_product = 0.0
                is_in_sublattice = False

                if self.peakshape == 0:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        self.HL,
                        self.SL,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                        self.xn,
                        self.wn,
                    )

                elif self.peakshape == 1:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                    )

                elif self.peakshape == 2:
                    args = (
                        np.array([self.alpha0, self.alpha1]),
                        np.array([self.beta0, self.beta1]),
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                    )

                y += self.computespectrum_fcn(*args)

        self._spectrum_sim = Spectrum(x=x, y=y)

        P = calc_num_variables(self.params)

        errvec, self.Rwp, self.gofF = calc_rwp(
            self.spectrum_sim.data_array,
            self.spectrum_expt.data_array,
            self.weights.data_array,
            P,
        )
        return errvec

    def CalcIobs(self):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    06/08/2020 SS 1.0 original
        >> @DETAILS: this is one of the main functions to partition the expt
                     intensities to overlapping peaks in the calculated pattern
        """

        self.Iobs = {}
        spec_expt = self.spectrum_expt.data_array
        spec_sim = self.spectrum_sim.data_array
        tth_list = np.ascontiguousarray(self.tth_list)
        for iph, p in enumerate(self.phases):

            self.Iobs[p] = {}
            for k, l in self.phases.wavelength.items():

                Ic = self.Icalc[p][k]

                shft_c = np.cos(0.5 * np.radians(self.tth[p][k])) * self.shft
                trns_c = np.sin(np.radians(self.tth[p][k])) * self.trns
                tth = self.tth[p][k] + self.zero_error + shft_c + trns_c

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                # n = np.min((tth.shape[0], Ic.shape[0]))  # !!! not used
                shkl = self.phases[p].shkl
                name = self.phases[p].name
                eta_n = f"self.{name}_eta_fwhm"
                eta_fwhm = eval(eta_n)
                strain_direction_dot_product = 0.0
                is_in_sublattice = False

                if self.peakshape == 0:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        self.HL,
                        self.SL,
                        self.xn,
                        self.wn,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                elif self.peakshape == 1:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                elif self.peakshape == 2:
                    args = (
                        np.array([self.alpha0, self.alpha1]),
                        np.array([self.beta0, self.beta1]),
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                self.Iobs[p][k] = self.calc_Iobs_fcn(*args)

    def calcRwp(self, params):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >> @DETAILS: this routine computes the rwp for a set of parameters.
                     the parameters are used to set the values in the LeBail
                     class too
        """

        self._set_params_vals_to_class(params, init=False, skip_phases=False)
        self._update_shkl(params)

        errvec = self.computespectrum()

        return errvec

    def initialize_lmfit_parameters(self):

        params = lmfit.Parameters()

        for p in self.params:
            par = self.params[p]
            if par.vary:
                params.add(p, value=par.value, min=par.lb, max=par.ub)

        return params

    def update_parameters(self):

        for p in self.res.params:
            par = self.res.params[p]
            self.params[p].value = par.value

    def RefineCycle(self, print_to_screen=True):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    06/08/2020 SS 1.0 original
                     01/28/2021 SS 1.1 added optional print_to_screen argument
        >> @DETAILS: this is one refinement cycle for the least squares,
                     typically few  10s to 100s of cycles may be required for
                     convergence
        """
        self.CalcIobs()
        self.Icalc = self.Iobs

        self.res = self.Refine()
        if self.res is not None:
            self.update_parameters()
        self.niter += 1
        self.Rwplist = np.append(self.Rwplist, self.Rwp)
        self.gofFlist = np.append(self.gofFlist, self.gofF)

        if print_to_screen:
            print(
                "Finished iteration. Rwp: \
                {:.3f} % goodness of fit: {:.3f}".format(
                    self.Rwp * 100.0, self.gofF
                )
            )

    def Refine(self):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >> @DETAILS: this routine performs the least squares refinement for all
                     variables which are allowed to be varied.
        """

        params = self.initialize_lmfit_parameters()

        if len(params) > 0:
            fdict = {
                "ftol": 1e-6,
                "xtol": 1e-6,
                "gtol": 1e-6,
                "verbose": 0,
                "max_nfev": 1000,
                "method": "trf",
                "jac": "2-point",
            }
            fitter = lmfit.Minimizer(self.calcRwp, params)

            res = fitter.least_squares(**fdict)
            return res
        else:
            msg = "nothing to refine. updating intensities"
            print(msg)
            errvec = self.computespectrum()
            return getattr(self, 'res', None)

    def updatespectrum(self):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    11/23/2020 SS 1.0 original
                     03/05/2021 SS 1.1 added computation and update of Rwp
        >> @DETAILS: this routine computes the spectrum for an updated list of
                     parameters intended to be used for sensitivity and
                     identifiability analysis
        """

        """
        the err variable is the difference between
        simulated and experimental spectra
        """
        # ???: is this supposed to return something, or is it incomplete?
        params = self.initialize_lmfit_parameters()
        errvec = self.calcRwp(params)

    def _update_shkl(self, params):
        """
        if certain shkls are refined, then update
        them using the params arg. else use values from
        the parameter class
        """
        shkl_dict = {}
        for p in self.phases:
            shkl_name = self.phases[p].valid_shkl
            eq_const = self.phases[p].eq_constraints
            mname = self.phases[p].name
            key = [f"{mname}_{s}" for s in shkl_name]
            for s, k in zip(shkl_name, key):
                if k in params:
                    shkl_dict[s] = params[k].value
                else:
                    shkl_dict[s] = self.params[k].value

            self.phases[p].shkl = wppfsupport._fill_shkl(shkl_dict, eq_const)

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, Uinp):
        self._U = Uinp
        return

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, Vinp):
        self._V = Vinp
        return

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, Winp):
        self._W = Winp
        return

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, Xinp):
        self._X = Xinp
        return

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Yinp):
        self._Y = Yinp
        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

    @property
    def Hcag(self):
        return self._Hcag

    @Hcag.setter
    def Hcag(self, val):
        self._Hcag = val

    @property
    def HL(self):
        return self._HL

    @HL.setter
    def HL(self, val):
        self._HL = val

    @property
    def SL(self):
        return self._SL

    @SL.setter
    def SL(self, val):
        self._SL = val

    @property
    def alpha0(self):
        return self._alpha0

    @alpha0.setter
    def alpha0(self, val):
        self._alpha0 = val

    @property
    def alpha1(self):
        return self._alpha1

    @alpha1.setter
    def alpha1(self, val):
        self._alpha1 = val

    @property
    def beta0(self):
        return self._beta0

    @beta0.setter
    def beta0(self, val):
        self._beta0 = val

    @property
    def beta1(self):
        return self._beta1

    @beta1.setter
    def beta1(self, val):
        self._beta1 = val

    @property
    def tth_list(self):
        if isinstance(self.spectrum_expt._x, np.ma.MaskedArray):
            return self.spectrum_expt._x.filled()
        else:
            return self.spectrum_expt._x

    @property
    def zero_error(self):
        return self._zero_error

    @zero_error.setter
    def zero_error(self, value):
        self._zero_error = value
        return

    @property
    def eta_fwhm(self):
        return self._eta_fwhm

    @eta_fwhm.setter
    def eta_fwhm(self, val):
        self._eta_fwhm = val

    @property
    def shft(self):
        return self._shft

    @shft.setter
    def shft(self, val):
        self._shft = val

    @property
    def trns(self):
        return self._trns

    @trns.setter
    def trns(self, val):
        self._trns = val

    @property
    def peakshape(self):
        return self._peakshape

    @peakshape.setter
    def peakshape(self, val):
        """
        @TODO make sure the parameter list
        is updated when the peakshape changes
        """
        if isinstance(val, str):
            if val == "pvfcj":
                self._peakshape = 0
            elif val == "pvtch":
                self._peakshape = 1
            elif val == "pvpink":
                self._peakshape = 2
            else:
                msg = (
                    "invalid peak shape string. "
                    "must be: \n"
                    "1. pvfcj: pseudo voight (Finger, Cox, Jephcoat)\n"
                    "2. pvtch: pseudo voight (Thompson, Cox, Hastings)\n"
                    "3. pvpink: Pink beam (Von Dreele)"
                )
                raise ValueError(msg)
        elif isinstance(val, int):
            if val >= 0 and val <= 2:
                self._peakshape = val
            else:
                msg = (
                    "invalid peak shape int. "
                    "must be: \n"
                    "1. 0: pseudo voight (Finger, Cox, Jephcoat)\n"
                    "2. 1: pseudo voight (Thompson, Cox, Hastings)\n"
                    "3. 2: Pink beam (Von Dreele)"
                )
                raise ValueError(msg)

        """
        update parameters
        """
        if hasattr(self, "params"):
            params = wppfsupport._generate_default_parameters_LeBail(
                self.phases, self.peakshape
            )
            for p in params:
                if p in self.params:
                    params[p] = self.params[p]
            self._params = params
            self._set_params_vals_to_class(params, init=True, skip_phases=True)
            self.computespectrum()

    @property
    def computespectrum_fcn(self):
        if self.peakshape == 0:
            return computespectrum_pvfcj
        elif self.peakshape == 1:
            return computespectrum_pvtch
        elif self.peakshape == 2:
            return computespectrum_pvpink

    @property
    def calc_Iobs_fcn(self):
        if self.peakshape == 0:
            return calc_Iobs_pvfcj
        elif self.peakshape == 1:
            return calc_Iobs_pvtch
        elif self.peakshape == 2:
            return calc_Iobs_pvpink

    @property
    def spectrum_expt(self):
        vector_list = [s.y for s in self._spectrum_expt]

        spec_masked = join_regions(
            vector_list, self.global_index, self.global_shape
        )
        return Spectrum(x=self._tth_list_global, y=spec_masked)

    @spectrum_expt.setter
    def spectrum_expt(self, expt_spectrum):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
                     09/11/2020 SS 1.1 multiple data types accepted as input
                     09/14/2020 SS 1.2 background method chebyshev now has user
                       specified polynomial degree
                     03/03/2021 SS 1.3 moved the initialization to the property
                       definition of self.spectrum_expt
                     05/03/2021 SS 2.0 moved weight calculation and background
                       initialization to the property definition
                     03/05/2021 SS 2.1 adding support for masked array
                       np.ma.MaskedArray
                     03/08/2021 SS 3.0 spectrum_expt is now a list to deal with
                        the masked arrays
        >> @DETAILS: load the experimental spectum of 2theta-intensity
        """
        if expt_spectrum is not None:
            if isinstance(expt_spectrum, Spectrum):
                """
                directly passing the spectrum class
                """
                self._spectrum_expt = [expt_spectrum]
                # self._spectrum_expt.nan_to_zero()
                self.global_index = [(0, expt_spectrum.shape[0])]
                self.global_mask = np.zeros(
                    [
                        expt_spectrum.shape[0],
                    ],
                    dtype=np.bool,
                )
                self._tth_list = [s._x for s in self._spectrum_expt]
                self._tth_list_global = expt_spectrum._x
                self.offset = False

            elif isinstance(expt_spectrum, np.ndarray):
                """
                initialize class using a nx2 array
                """
                if np.ma.is_masked(expt_spectrum):
                    """
                    @date 03/05/2021 SS 1.0 original
                    this is an instance of masked array where there are
                    nans in the spectrum. this will have to be handled with
                    a lot of care. steps are as follows:
                    1. if array is masked array, then check if any values are
                       masked or not.
                    2. if they are then the spectrum_expt is a list of
                       individual islands of the spectrum, each with its own
                       background
                    3. Every place where spectrum_expt is used, we will do a
                       type test to figure out the logic of the operations
                    """
                    expt_spec_list, gidx = separate_regions(expt_spectrum)
                    self.global_index = gidx
                    self.global_shape = expt_spectrum.shape[0]
                    self.global_mask = expt_spectrum.mask[:, 1]
                    self._spectrum_expt = []
                    for s in expt_spec_list:
                        self._spectrum_expt.append(
                            Spectrum(
                                x=s[:, 0],
                                y=s[:, 1],
                                name="expt_spectrum"
                            )
                        )

                else:
                    max_ang = expt_spectrum[-1, 0]
                    if max_ang < np.pi:
                        warnings.warn(
                            "angles are small and appear to \
                            be in radians. please check"
                        )

                    self._spectrum_expt = [
                        Spectrum(
                            x=expt_spectrum[:, 0],
                            y=expt_spectrum[:, 1],
                            name="expt_spectrum",
                        )
                    ]

                    self.global_index = [
                        (0, self._spectrum_expt[0].x.shape[0])
                    ]
                    self.global_shape = expt_spectrum.shape[0]
                    self.global_mask = np.zeros(
                        [
                            expt_spectrum.shape[0],
                        ],
                        dtype=np.bool,
                    )

                self._tth_list = [s._x for s in self._spectrum_expt]
                self._tth_list_global = expt_spectrum[:, 0]
                self.offset = False

            elif isinstance(expt_spectrum, str):
                """
                load from a text file
                undefined behavior if text file has nans
                """
                if path.exists(expt_spectrum):
                    self._spectrum_expt = [
                        Spectrum.from_file(expt_spectrum, skip_rows=0)
                    ]
                    # self._spectrum_expt.nan_to_zero()
                    self.global_index = [
                        (0, self._spectrum_expt[0].x.shape[0])
                    ]
                    self.global_shape = self._spectrum_expt[0].x.shape[0]
                    self.global_mask = np.zeros(
                        [
                            self.global_shape,
                        ],
                        dtype=np.bool,
                    )
                else:
                    raise FileNotFoundError(
                        "input spectrum file doesn't exist."
                    )

                self._tth_list = [self._spectrum_expt[0]._x]
                self._tth_list_global = self._spectrum_expt[0]._x
                self.offset = False

            """
            03/08/2021 SS tth_min and max are now lists
            """
            self.tth_max = []
            self.tth_min = []
            self.ntth = []
            for s in self._spectrum_expt:
                self.tth_max.append(s.x.max())
                self.tth_min.append(s.x.min())
                self.ntth.append(s.x.shape[0])

            """
            03/02/2021 SS added tth_step for some computations
            related to snip background estimation
            @TODO this will not work for masked spectrum
            03/08/2021 tth_step is a list now
            """
            self.tth_step = []
            for tmi, tma, nth in zip(self.tth_min, self.tth_max, self.ntth):
                if nth > 1:
                    self.tth_step.append((tma - tmi) / nth)
                else:
                    self.tth_step.append(0.0)

            """
            @date 03/03/2021 SS
            there are cases when the intensity in the spectrum is
            negative. our approach will be to offset the spectrum to make all
            the values positive for the computation and then finally offset it
            when the computation has finished.
            03/08/2021 all quantities are lists now
            """
            for s in self._spectrum_expt:
                self.offset = []
                self.offset_val = []
                if np.any(s.y < 0.0):
                    self.offset.append(True)
                    self.offset_val.append(s.y.min())
                    s.y = s.y - s.y.min()

            """
            @date 09/24/2020 SS
            catching the cases when intensity is zero.
            for these points the weights will become
            infinite. therefore, these points will be
            masked out and assigned a weight of zero.
            In addition, if any points have negative
            intensity, they will also be assigned a zero
            weight
            03/08/2021 SS everything is a list now
            """
            self._weights = []
            for s in self._spectrum_expt:
                mask = s.y <= 0.0
                ww = np.zeros(s.y.shape)
                """also initialize statistical weights
                for the error calculation"""
                ww[~mask] = 1.0 / np.sqrt(s.y[~mask])
                self._weights.append(ww)

            self.initialize_bkg()
        else:
            raise RuntimeError("expt_spectrum setter: spectrum is None")

    @property
    def spectrum_sim(self):
        tth, inten = self._spectrum_sim.data
        inten[self.global_mask] = np.nan
        inten += self.background.y

        return Spectrum(x=tth, y=inten)

    @property
    def background(self):
        vector_list = [s.y for s in self._background]

        bkg_masked = join_regions(
            vector_list, self.global_index, self.global_shape
        )
        return Spectrum(x=self.tth_list, y=bkg_masked)

    @property
    def weights(self):
        weights_masked = join_regions(
            self._weights, self.global_index, self.global_shape
        )
        return Spectrum(x=self.tth_list, y=weights_masked)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param_info):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
                     09/11/2020 SS 1.1 modified to accept multiple input types
                     03/05/2021 SS 2.0 moved everything to the property setter
        >> @DETAILS: initialize parameter list from file. if no file given,
                     then initialize to some default values
                     (lattice constants are for CeO2)
        """
        from scipy.special import roots_legendre

        xn, wn = roots_legendre(16)
        self.xn = xn[8:]
        self.wn = wn[8:]

        if param_info is not None:
            if isinstance(param_info, Parameters):
                """
                directly passing the parameter class
                """
                self._params = param_info
                params = param_info

            else:
                params = Parameters()

                if isinstance(param_info, dict):
                    """
                    initialize class using dictionary read from the yaml file
                    """
                    for k, v in param_info.items():
                        params.add(
                            k,
                            value=np.float(v[0]),
                            lb=np.float(v[1]),
                            ub=np.float(v[2]),
                            vary=np.bool(v[3]),
                        )

                elif isinstance(param_info, str):
                    """
                    load from a yaml file
                    """
                    if path.exists(param_info):
                        params.load(param_info)
                    else:
                        raise FileNotFoundError(
                            "input spectrum file doesn't exist."
                        )

                """
                this part initializes the lattice parameters in the
                """
                for p in self.phases:
                    wppfsupport._add_lp_to_params(params, self.phases[p])

                self._params = params
        else:
            """
            first three are cagliotti parameters
            next two are the lorentz paramters
            final is the zero instrumental peak position error
            mixing factor calculated by Thomax, Cox, Hastings formula
            """
            params = wppfsupport._generate_default_parameters_LeBail(
                self.phases, self.peakshape
            )
            self._params = params

        self._set_params_vals_to_class(params, init=True, skip_phases=True)

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, phase_info):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    06/08/2020 SS 1.0 original
                     09/11/2020 SS 1.1 multiple different ways to initialize
                       phases
                     09/14/2020 SS 1.2 added phase initialization from
                       material.Material class
                     03/05/2021 SS 2.0 moved everything to property setter
        >> @DETAILS: load the phases for the LeBail fits
        """

        if phase_info is not None:
            if isinstance(phase_info, Phases_LeBail):
                """
                directly passing the phase class
                """
                self._phases = phase_info

            else:

                if hasattr(self, "wavelength"):
                    if self.wavelength is not None:
                        p = Phases_LeBail(wavelength=self.wavelength)
                else:
                    p = Phases_LeBail()

                if isinstance(phase_info, dict):
                    """
                    initialize class using a dictionary with key as
                    material file and values as the name of each phase
                    """
                    for material_file in phase_info:
                        material_names = phase_info[material_file]
                        if not isinstance(material_names, list):
                            material_names = [material_names]
                        p.add_many(material_file, material_names)

                elif isinstance(phase_info, str):
                    """
                    load from a yaml file
                    """
                    if path.exists(phase_info):
                        p.load(phase_info)
                    else:
                        raise FileNotFoundError("phase file doesn't exist.")

                elif isinstance(phase_info, Material):
                    p[phase_info.name] = Material_LeBail(
                        fhdf=None, xtal=None,
                        dmin=None, material_obj=phase_info
                    )

                elif isinstance(phase_info, list):
                    for mat in phase_info:
                        p[mat.name] = Material_LeBail(
                            fhdf=None, xtal=None,
                            dmin=None, material_obj=mat
                        )

                        p.num_phases += 1

                    for mat in p:
                        p[mat].pf = 1.0 / p.num_phases

                self._phases = p

        self.calctth()

        for p in self.phases:
            (
                self.phases[p].valid_shkl,
                self.phases[p].eq_constraints,
                self.phases[p].rqd_index,
                self.phases[p].trig_ptype,
            ) = wppfsupport._required_shkl_names(self.phases[p])

    def _set_params_vals_to_class(self, params, init=False, skip_phases=False):
        """
        @date 03/12/2021 SS 1.0 original
        take values in parameters and set the
        corresponding class values with the same
        name
        """
        for p in params:
            if init:
                setattr(self, p, params[p].value)
            else:
                if hasattr(self, p):
                    setattr(self, p, params[p].value)

        if not skip_phases:

            updated_lp = False

            for p in self.phases:
                mat = self.phases[p]
                """
                PART 1: update the lattice parameters
                """
                lp = []
                lpvary = False
                pre = p + "_"

                name = [f"{pre}{x}" for x in wppfsupport._lpname]

                for nn in name:
                    if nn in params:
                        if params[nn].vary:
                            lpvary = True
                        lp.append(params[nn].value)
                    elif nn in self.params:
                        lp.append(self.params[nn].value)

                if not lpvary:
                    pass
                else:
                    lp = self.phases[p].Required_lp(lp)
                    mat.lparms = np.array(lp)
                    mat._calcrmt()
                    updated_lp = True

            if updated_lp:
                self.calctth()


def _nm(x):
    return valWUnit("lp", "length", x, "nm")


def extract_intensities(
    polar_view,
    tth_array,
    params=None,
    phases=None,
    wavelength={"kalpha1": _nm(0.15406), "kalpha2": _nm(0.154443)},
    bkgmethod={"chebyshev": 10},
    intensity_init=None,
    termination_condition={"rwp_perct_change": 0.05, "max_iter": 100},
    peakshape="pvtch",
):
    """
    ===========================================================================

    >> @AUTHOR:  Saransh Singh, Lanwrence Livermore National Lab,
                 saransh1@llnl.gov
    >> @DATE:    01/28/2021 SS 1.0 original
                 03/03/2021 SS 1.1 removed detector_mask since polar_view is
                 now a masked array
    >> @DETAILS: this function is used for extracting the experimental pole
                 figure intensities from the polar 2theta-eta map. The workflow
                 is to simply run the LeBail class, in parallel, over the
                 different azimuthal profiles and return the Icalc values for
                 the different wavelengths in the calculation. For now, the
                 multiprocessing is done using the multiprocessing module which
                 comes natively with python. Extension to MPI will be done
                 later if necessary.
    >> @PARAMS   polar_view: mxn array with the polar view. the parallelization
                             is done !!! this is now a masked numpy array !!!
                             over "m" i.e. the eta dimension
                    tth_array: nx1 array with two theta values at each sampling
                               point
                    params: parameter values for the LeBail class. Could be in
                            the form of yaml file, dictionary or Parameter
                            class
                    phases: materials to use in intensity extraction. could be
                            a list of material objects, or file or dictionary
                    wavelength: dictionary of wavelengths to be used in the
                                computation
                    bkgmethod: "spline" or "chebyshev" or "snip"
                                default is chebyshev
                    intensity_init: initial intensities for each reflection.
                                    If none, then it is specified to some power
                                    of 10 depending on maximum intensity in
                                    spectrum (only used for powder simulator)
    ============================================================================
    """

    # prepare the data file to distribute suing multiprocessing
    data_inp_list = []

    # check if the dimensions all match
    if polar_view.shape[1] != tth_array.shape[0]:
        raise RuntimeError(
            "WPPF : extract_intensities : \
                            inconsistent dimensions \
                            of polar_view and tth_array variables."
        )

    non_zeros_index = []
    for i in range(polar_view.shape[0]):
        d = polar_view[i, :]
        # make sure that there is atleast one nonzero pixel

        if np.sum(~d.mask) > 1:
            data = np.ma.stack((tth_array, d)).T
            data_inp_list.append(data)
            non_zeros_index.append(i)

    kwargs = {
        "params": params,
        "phases": phases,
        "wavelength": wavelength,
        "bkgmethod": bkgmethod,
        "termination_condition": termination_condition,
        "peakshape": peakshape,
    }

    P = GenericMultiprocessing()
    results = P.parallelise_function(
        data_inp_list, single_azimuthal_extraction, **kwargs
    )

    """
    process the outputs from the multiprocessing to make the
    simulated polar views, tables of hkl <--> intensity etc. at
    each azimuthal location
    in this section, all the rows which had no pixels
    falling on detector will be handles
    separately
    """
    pv_simulated = np.zeros(polar_view.shape)
    extracted_intensities = []
    hkls = []
    tths = []
    for i in range(len(non_zeros_index)):
        idx = non_zeros_index[i]
        xp, yp, rwp, Icalc, hkl, tth = results[i]

        intp_int = np.interp(tth_array, xp, yp, left=0.0, right=0.0)

        pv_simulated[idx, :] = intp_int

        extracted_intensities.append(Icalc)
        hkls.append(hkl)
        tths.append(tth)

    """
    make the values outside detector NaNs and convert to masked array
    """
    pv_simulated[polar_view.mask] = np.nan
    pv_simulated = np.ma.masked_array(
        pv_simulated, mask=np.isnan(pv_simulated)
    )

    return extracted_intensities, hkls, tths, non_zeros_index, pv_simulated


def single_azimuthal_extraction(
    expt_spectrum,
    params=None,
    phases=None,
    wavelength={"kalpha1": _nm(0.15406), "kalpha2": _nm(0.154443)},
    bkgmethod={"chebyshev": 10},
    intensity_init=None,
    termination_condition=None,
    peakshape="pvtch",
):

    kwargs = {
        "expt_spectrum": expt_spectrum,
        "params": params,
        "phases": phases,
        "wavelength": wavelength,
        "bkgmethod": bkgmethod,
        "peakshape": peakshape,
    }

    # get termination conditions for the LeBail refinement
    del_rwp = termination_condition["rwp_perct_change"]
    max_iter = termination_condition["max_iter"]

    L = LeBail(**kwargs)

    rel_error = 1.0
    init_error = 1.0
    niter = 0

    # when change in Rwp < 0.05% or reached maximum iteration
    while rel_error > del_rwp and niter < max_iter:
        L.RefineCycle(print_to_screen=False)
        rel_error = 100.0 * np.abs((L.Rwp - init_error))
        init_error = L.Rwp
        niter += 1

    res = (L.spectrum_sim._x, L.spectrum_sim._y, L.Rwp, L.Iobs, L.hkls, L.tth)
    return res


class Rietveld:
    """
    ===========================================================================

    >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                 saransh1@llnl.gov
    >> @DATE:    01/08/2020 SS 1.0 original
                 07/13/2020 SS 2.0 complete rewrite to include new
                   parameter/material/pattern class
                 02/01/2021 SS 2.1 peak shapes from Thompson,Cox and Hastings
                   formula using X anf Y parameters for the lorentzian peak
                   widths
    >> @DETAILS: this is the main rietveld class and contains all the refinable
                 parameters for the analysis. the member classes are as follows
                 (in order of initialization):
                    1. Spectrum     contains the experimental spectrum
                    2. Background   contains the background extracted from
                                    spectrum
                    3. Refine       contains all the machinery for refinement
    ============================================================================
    """

    def __init__(
        self,
        expt_spectrum=None,
        params=None,
        phases=None,
        wavelength={"kalpha1": [_nm(0.15406), 1.0],
                    "kalpha2": [_nm(0.154443), 0.52]},
        bkgmethod={"spline": None},
        peakshape="pvfcj",
        shape_factor=1.0,
        particle_size=1.0,
        phi=0.0,
    ):

        self.bkgmethod = bkgmethod
        self.shape_factor = shape_factor
        self.particle_size = particle_size
        self.phi = phi
        self.peakshape = peakshape
        self.spectrum_expt = expt_spectrum

        self._tstart = time.time()

        if wavelength is not None:
            self.wavelength = wavelength
            for k, v in self.wavelength.items():
                v[0] = valWUnit("lp", "length", v[0].getVal("nm"), "nm")

        self.phases = phases

        self.params = params

        self.PolarizationFactor()
        self.computespectrum()

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart

        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

    def __str__(self):
        resstr = "<Rietveld Fit class>\nParameters of \
        the model are as follows:\n"
        resstr += self.params.__str__()
        return resstr

    def initialize_bkg(self):
        """
        the cubic spline seems to be the ideal route in terms
        of determining the background intensity. this involves
        selecting a small (~5) number of points from the spectrum,
        usually called the anchor points. a cubic spline interpolation
        is performed on this subset to estimate the overall background.
        scipy provides some useful routines for this

        the other option implemented is the chebyshev polynomials. this
        basically automates the background determination and removes the
        user from the loop which is required for the spline type background.
        """
        if self.bkgmethod is None:
            self._background = []
            for tth in self.tth_list:
                self._background.append(Spectrum(x=tth, y=np.zeros(tth.shape)))

        elif "spline" in self.bkgmethod.keys():
            self._background = []
            self.selectpoints()
            for i, pts in enumerate(self.points):
                tth = self._spectrum_expt[i]._x
                x = pts[:, 0]
                y = pts[:, 1]
                self._background.append(self.splinefit(x, y, tth))

        elif "chebyshev" in self.bkgmethod.keys():
            self.chebyshevfit()

        elif "file" in self.bkgmethod.keys():
            if len(self._spectrum_expt) > 1:
                raise RuntimeError(
                    "initialize_bkg: \
                    file input not allowed for \
                    masked spectra."
                )
            else:
                bkg = Spectrum.from_file(self.bkgmethod["file"])
                x = bkg.x
                y = bkg.y
                cs = CubicSpline(x, y)

                yy = cs(self.tth_list)

                self._background = [Spectrum(x=self.tth_list[0], y=yy)]

        elif "array" in self.bkgmethod.keys():
            if len(self._spectrum_expt) > 1:
                raise RuntimeError(
                    "initialize_bkg: \
                    file input not allowed for \
                    masked spectra."
                )
            else:
                x = self.bkgmethod["array"][:, 0]
                y = self.bkgmethod["array"][:, 1]
                cs = CubicSpline(x, y)

                yy = cs(self._tth_list)

                self._background = [Spectrum(x=self.tth_list, y=yy)]

        elif "snip1d" in self.bkgmethod.keys():
            self._background = []
            for i, s in enumerate(self._spectrum_expt):
                if not self.tth_step:
                    ww = 3
                else:
                    if self.tth_step[i] > 0.0:
                        ww = np.rint(
                            self.bkgmethod["snip1d"][0] / self.tth_step[i]
                        ).astype(np.int32)
                    else:
                        ww = 3

                numiter = self.bkgmethod["snip1d"][1]

                yy = np.squeeze(
                    snip1d_quad(np.atleast_2d(s.y), w=ww, numiter=numiter)
                )
                self._background.append(Spectrum(x=self._tth_list[i], y=yy))

    def chebyshevfit(self):
        """
        03/08/2021 SS spectrum_expt is a list now. accounting
        for that change
        """
        self._background = []
        degree = self.bkgmethod["chebyshev"]
        for i, s in enumerate(self._spectrum_expt):
            tth = self._tth_list[i]
            p = np.polynomial.Chebyshev.fit(
                tth, s.y, degree, w=self._weights[i] ** 2
            )
            self._background.append(Spectrum(x=tth, y=p(tth)))

    def selectpoints(self):
        """
        03/08/2021 SS spectrum_expt is a list now. accounting
        for that change
        """

        # Keep matplotlib as an optional dependency
        from pylab import plot, ginput, close, title, xlabel, ylabel

        self.points = []
        for i, s in enumerate(self._spectrum_expt):
            txt = (
                f"Select points for background estimation;\n"
                f"click middle mouse button when done. segment # {i}"
            )
            title(txt)

            plot(s.x, s.y, "-k")
            xlabel("2$\theta$")
            ylabel("intensity (a.u.)")

            self.points.append(np.asarray(ginput(0, timeout=-1)))

            close()

    # cubic spline fit of background using custom points chosen from plot
    def splinefit(self, x, y, tth):
        """
        03/08/2021 SS adding tth as input. this is the
        list of points for which background is estimated
        """
        cs = CubicSpline(x, y)
        bkg = cs(tth)
        return Spectrum(x=tth, y=bkg)

    def calctth(self):
        self.tth = {}
        self.hkls = {}
        self.dsp = {}
        self.limit = {}
        for p in self.phases:
            self.tth[p] = {}
            self.hkls[p] = {}
            self.dsp[p] = {}
            self.limit[p] = {}

            for k, l in self.phases.wavelength.items():
                t = self.phases[p][k].getTTh(l[0].getVal('nm'))
                allowed = self.phases[p][k].wavelength_allowed_hkls
                t = t[allowed]
                hkl = self.phases[p][k].hkls[allowed, :]
                dsp = self.phases[p][k].dsp[allowed]
                tth_min = min(self.tth_min)
                tth_max = max(self.tth_max)
                limit = np.logical_and(t >= tth_min, t <= tth_max)
                self.limit[p][k] = limit
                self.tth[p][k] = t[limit]
                self.hkls[p][k] = hkl[limit, :]
                self.dsp[p][k] = dsp[limit]

    def calcsf(self):
        self.sf = {}
        self.sf_raw = {}
        self.extinction = {}
        self.absorption = {}
        for p in self.phases:
            self.sf[p] = {}
            self.sf_raw[p] = {}
            self.extinction[p] = {}
            self.absorption[p] = {}
            for k, l in self.phases.wavelength.items():
                w = l[0].getVal("nm")
                w_int = l[1]
                tth = self.tth[p][k]
                # allowed = self.phases[p][k].wavelength_allowed_hkls
                # limit = self.limit[p][k]
                self.sf[p][k], self.sf_raw[p][k] = \
                    self.phases[p][k].CalcXRSF(w, w_int)

                self.extinction[p][k] = self.phases[p][k].calc_extinction(
                    10.0 * w,
                    tth,
                    self.sf_raw[p][k],
                    self.shape_factor,
                    self.particle_size,
                )
                self.absorption[p][k] = self.phases[p][k].calc_absorption(
                    tth, self.phi, 10.0 * w
                )

    def PolarizationFactor(self):

        # tth = self.tth
        self.LP = {}
        Ph = self.Ph
        for p in self.phases:
            self.LP[p] = {}
            for k, l in self.phases.wavelength.items():
                t = np.radians(self.tth[p][k])
                self.LP[p][k] = (
                    (1 + Ph * np.cos(t) ** 2)
                    / np.cos(0.5 * t)
                    / np.sin(0.5 * t) ** 2
                    #/ (2.0 * (1 + Ph))
                )

    def computespectrum(self):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab
        >> @EMAIL:      saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    compute the simulated spectrum
        """
        x = self.tth_list
        y = np.zeros(x.shape)
        tth_list = np.ascontiguousarray(self.tth_list)

        for iph, p in enumerate(self.phases):

            for k, l in self.phases.wavelength.items():

                shft_c = np.cos(0.5 * np.radians(self.tth[p][k])) * self.shft
                trns_c = np.sin(np.radians(self.tth[p][k])) * self.trns
                tth = self.tth[p][k] + self.zero_error + shft_c + trns_c

                pf = self.phases[p][k].pf / self.phases[p][k].vol ** 2
                sf = self.sf[p][k]
                lp = self.LP[p][k]
                extinction = self.extinction[p][k]
                absorption = self.absorption[p][k]

                n = np.min((tth.shape[0], sf.shape[0], lp.shape[0]))

                tth = tth[:n]
                sf = sf[:n]
                lp = lp[:n]
                extinction = extinction[:n]
                absorption = absorption[:n]

                Ic = self.scale * pf * sf * lp  # *extinction*absorption

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]

                shkl = self.phases[p][k].shkl
                name = self.phases[p][k].name
                eta_n = f"self.{name}_eta_fwhm"
                eta_fwhm = eval(eta_n)
                strain_direction_dot_product = 0.0
                is_in_sublattice = False

                if self.peakshape == 0:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        self.HL,
                        self.SL,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                        self.xn,
                        self.wn,
                    )

                elif self.peakshape == 1:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                    )

                elif self.peakshape == 2:
                    args = (
                        np.array([self.alpha0, self.alpha1]),
                        np.array([self.beta0, self.beta1]),
                        np.array([self.U, self.V, self.W]),
                        self.P,
                        np.array([self.X, self.Y]),
                        np.array([self.Xe, self.Ye, self.Xs]),
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        strain_direction_dot_product,
                        is_in_sublattice,
                        tth_list,
                        Ic,
                    )

                y += self.computespectrum_fcn(*args)

        self._spectrum_sim = Spectrum(x=x, y=y)

        P = calc_num_variables(self.params)

        errvec, self.Rwp, self.gofF = calc_rwp(
            self.spectrum_sim.data_array,
            self.spectrum_expt.data_array,
            self.weights.data_array,
            P,
        )
        return errvec

    def calcRwp(self, params):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >> @DETAILS: this routine computes the weighted error between
                     calculated and experimental spectra. goodness of fit is
                     also calculated. the weights are the inverse squareroot of
                     the experimental intensities
        """

        """
        the err variable is the difference between
        simulated and experimental spectra
        """
        self._set_params_vals_to_class(params, init=False, skip_phases=False)
        self._update_shkl(params)
        errvec = self.computespectrum()

        return errvec

    def initialize_lmfit_parameters(self):

        params = lmfit.Parameters()

        for p in self.params:
            par = self.params[p]
            if par.vary:
                params.add(p, value=par.value, min=par.lb, max=par.ub)

        return params

    def update_parameters(self):

        for p in self.res.params:
            par = self.res.params[p]
            self.params[p].value = par.value

    def Refine(self):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >> @DETAILS: this routine performs the least squares refinement for all
                     variables that are allowed to be varied.
        """

        params = self.initialize_lmfit_parameters()

        if len(params) > 0:
            fdict = {
                "ftol": 1e-6,
                "xtol": 1e-6,
                "gtol": 1e-6,
                "verbose": 0,
                "max_nfev": 1000,
                "method": "trf",
                "jac": "2-point",
            }

            fitter = lmfit.Minimizer(self.calcRwp, params)

            self.res = fitter.least_squares(**fdict)

            self.update_parameters()

            self.niter += 1
            self.Rwplist = np.append(self.Rwplist, self.Rwp)
            self.gofFlist = np.append(self.gofFlist, self.gofF)

            print(
                "Finished iteration. Rwp: {:.3f} % goodness of \
                  fit: {:.3f}".format(
                    self.Rwp * 100.0, self.gofF
                )
            )
        else:
            print("Nothing to refine...")

    def _set_params_vals_to_class(self, params, init=False, skip_phases=False):
        """
        @date: 03/12/2021 SS 1.0 original
        @details: set the values from parameters to the Rietveld class
        """
        for p in params:
            if init:
                setattr(self, p, params[p].value)
            else:
                if hasattr(self, p):
                    setattr(self, p, params[p].value)
        if not skip_phases:
            updated_lp = False
            updated_atominfo = False
            pf = np.zeros([self.phases.num_phases,])
            pf_cur = self.phases.phase_fraction.copy()
            for ii, p in enumerate(self.phases):
                name = f"{p}_phase_fraction"
                if name in params:
                    pf[ii] = params[name].value
                else:
                    pf[ii] = pf_cur[ii]

                for lpi in self.phases[p]:
                    mat = self.phases[p][lpi]

                    """
                    PART 1: update the lattice parameters
                    """
                    lp = []
                    lpvary = False
                    pre = f"{p}_"
                    name = [f"{pre}{x}" for x in wppfsupport._lpname]
                    for nn in name:
                        if nn in params:
                            if params[nn].vary:
                                lpvary = True
                            lp.append(params[nn].value)
                        elif nn in self.params:
                            lp.append(self.params[nn].value)

                    if not lpvary:
                        pass
                    else:
                        lp = self.phases[p][lpi].Required_lp(lp)
                        self.phases[p][lpi].lparms = np.array(lp)
                        updated_lp = True
                    """
                    PART 2: update the atom info
                    """
                    atom_type = mat.atom_type
                    atom_label = wppfsupport._getnumber(atom_type)

                    for i in range(atom_type.shape[0]):
                        Z = atom_type[i]
                        elem = constants.ptableinverse[Z]
                        nx = f"{p}_{elem}{atom_label[i]}_x"
                        ny = f"{p}_{elem}{atom_label[i]}_y"
                        nz = f"{p}_{elem}{atom_label[i]}_z"
                        oc = f"{p}_{elem}{atom_label[i]}_occ"

                        if mat.aniU:
                            Un = []
                            for j in range(6):
                                Un.append(
                                    (
                                        f"{p}_{elem}"
                                        f"{atom_label[i]}"
                                        f"_{wppfsupport._nameU[j]}"
                                    )
                                )
                        else:
                            dw = f"{p}_{elem}{atom_label[i]}_dw"

                        if nx in params:
                            x = params[nx].value
                            updated_atominfo = True
                        else:
                            x = self.params[nx].value

                        if ny in params:
                            y = params[ny].value
                            updated_atominfo = True
                        else:
                            y = self.params[ny].value

                        if nz in params:
                            z = params[nz].value
                            updated_atominfo = True
                        else:
                            z = self.params[nz].value

                        if oc in params:
                            oc = params[oc].value
                            updated_atominfo = True
                        else:
                            oc = self.params[oc].value

                        if mat.aniU:
                            U = []
                            for j in range(6):
                                if Un[j] in params:
                                    updated_atominfo = True
                                    U.append(params[Un[j]].value)
                                else:
                                    U.append(self.params[Un[j]].value)
                            U = np.array(U)
                            mat.U[i, :] = U
                        else:
                            if dw in params:
                                dw = params[dw].value
                                updated_atominfo = True
                            else:
                                dw = self.params[dw].value
                            mat.U[i] = dw

                        mat.atom_pos[i, :] = np.array([x, y, z, oc])

                    if mat.aniU:
                        mat.calcBetaij()
                    if updated_lp:
                        mat._calcrmt()

                if updated_lp:
                    self.calctth()

                if updated_lp or updated_atominfo:
                    self.calcsf()

            self.phases.phase_fraction = pf

    def _update_shkl(self, params):
        """
        if certain shkls are refined, then update
        them using the params arg. else use values from
        the parameter class
        """
        shkl_dict = {}
        for p in self.phases:
            for k in self.phases[p]:
                shkl_name = self.phases[p][k].valid_shkl
                eq_const = self.phases[p][k].eq_constraints
                mname = self.phases[p][k].name
                key = [f"{mname}_{s}" for s in shkl_name]
                for s, kk in zip(shkl_name, key):
                    if kk in params:
                        shkl_dict[s] = params[kk].value
                    else:
                        shkl_dict[s] = self.params[kk].value

                self.phases[p][k].shkl = wppfsupport._fill_shkl(
                    shkl_dict, eq_const
                )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param_info):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                    bsaransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >>           07/15/2020 SS 1.1 modified to add lattice parameters,
                       atom positions and isotropic DW factors
                     02/01/2021 SS 2.0 modified to follow same input style as
                       LeBail class with inputs of Parameter class, dict or
                       filename valid
        >> @DETAILS: initialize parameter list from file. if no file given,
                     then initialize to some default values
                     (lattice constants are for CeO2)
        """
        from scipy.special import roots_legendre

        xn, wn = roots_legendre(16)
        self.xn = xn[8:]
        self.wn = wn[8:]
        if param_info is not None:
            if isinstance(param_info, Parameters):
                """
                directly passing the parameter class
                """
                self._params = param_info
                params = param_info
            else:
                params = Parameters()

                if isinstance(param_info, dict):
                    """
                    initialize class using dictionary read from the yaml file
                    """
                    for k, v in param_info.items():
                        params.add(
                            k,
                            value=np.float(v[0]),
                            lb=np.float(v[1]),
                            ub=np.float(v[2]),
                            vary=np.bool(v[3]),
                        )

                elif isinstance(param_info, str):
                    """
                    load from a yaml file
                    """
                    if path.exists(param_info):
                        params.load(param_info)
                    else:
                        raise FileNotFoundError(
                            "input spectrum file doesn't exist."
                        )

                    """
                    this part initializes the lattice parameters, atom
                    positions in asymmetric unit, occupation and the isotropic
                    debye waller factor. the anisotropic DW factors will be
                    added in the future
                    """
                    for p in self.phases:
                        for lpi in self.phases[p]:
                            wppfsupport._add_atominfo_to_params(
                                params, self.phases[p][lpi]
                            )

                self._params = params

        else:
            """
            first three are cagliotti parameters
            next are the lorentz paramters
            a scale parameter and
            final is the zero instrumental peak position error
            """
            params = wppfsupport._generate_default_parameters_Rietveld(
                self.phases, self.peakshape
            )
            self._params = params

        self._set_params_vals_to_class(params, init=True, skip_phases=True)

    @property
    def spectrum_expt(self):
        vector_list = [s.y for s in self._spectrum_expt]

        spec_masked = join_regions(
            vector_list, self.global_index, self.global_shape
        )
        return Spectrum(x=self._tth_list_global, y=spec_masked)

    @spectrum_expt.setter
    def spectrum_expt(self, expt_spectrum):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
                     09/11/2020 SS 1.1 multiple data types accepted as input
                     09/14/2020 SS 1.2 background method chebyshev now has user
                       specified polynomial degree
                     03/03/2021 SS 1.3 moved the initialization to the property
                       definition of self.spectrum_expt
                     05/03/2021 SS 2.0 moved weight calculation and background
                       initialization to the property definition
                     03/05/2021 SS 2.1 adding support for masked array
                       np.ma.MaskedArray
                     03/08/2021 SS 3.0 spectrum_expt is now a list to deal with
                       the masked arrays
        >> @DETAILS: load the experimental spectum of 2theta-intensity
        """
        if expt_spectrum is not None:
            if isinstance(expt_spectrum, Spectrum):
                """
                directly passing the spectrum class
                """
                self._spectrum_expt = [expt_spectrum]
                # self._spectrum_expt.nan_to_zero()
                self.global_index = [(0, expt_spectrum.shape[0])]
                self.global_mask = np.zeros(
                    [
                        expt_spectrum.shape[0],
                    ],
                    dtype=np.bool,
                )
                self._tth_list = [s._x for s in self._spectrum_expt]
                self._tth_list_global = expt_spectrum._x
                self.offset = False

            elif isinstance(expt_spectrum, np.ndarray):
                """
                initialize class using a nx2 array
                """
                if np.ma.is_masked(expt_spectrum):
                    """
                    @date 03/05/2021 SS 1.0 original
                    this is an instance of masked array where there are
                    nans in the spectrum. this will have to be handled with
                    a lot of care. steps are as follows:
                    1. if array is masked array, then check if any values are
                    masked or not.
                    2. if they are then the spectrum_expt is a list of
                    individual islands of the spectrum, each with its own
                    background
                    3. Every place where spectrum_expt is used, we will do a
                    type test to figure out the logic of the operations
                    """
                    expt_spec_list, gidx = separate_regions(expt_spectrum)
                    self.global_index = gidx
                    self.global_shape = expt_spectrum.shape[0]
                    self.global_mask = expt_spectrum.mask[:, 1]
                    self._spectrum_expt = []
                    for s in expt_spec_list:
                        self._spectrum_expt.append(
                            Spectrum(x=s[:, 0],
                                     y=s[:, 1],
                                     name="expt_spectrum")
                        )

                else:
                    max_ang = expt_spectrum[-1, 0]
                    if max_ang < np.pi:
                        warnings.warn(
                            "angles are small and appear to \
                            be in radians. please check"
                        )

                    self._spectrum_expt = [
                        Spectrum(
                            x=expt_spectrum[:, 0],
                            y=expt_spectrum[:, 1],
                            name="expt_spectrum",
                        )
                    ]

                    self.global_index = [
                        (0, self._spectrum_expt[0].x.shape[0])
                    ]
                    self.global_shape = expt_spectrum.shape[0]
                    self.global_mask = np.zeros(
                        [
                            expt_spectrum.shape[0],
                        ],
                        dtype=np.bool,
                    )

                self._tth_list = [s._x for s in self._spectrum_expt]
                self._tth_list_global = expt_spectrum[:, 0]
                self.offset = False

            elif isinstance(expt_spectrum, str):
                """
                load from a text file
                undefined behavior if text file has nans
                """
                if path.exists(expt_spectrum):
                    self._spectrum_expt = [
                        Spectrum.from_file(expt_spectrum, skip_rows=0)
                    ]
                    # self._spectrum_expt.nan_to_zero()
                    self.global_index = [
                        (0, self._spectrum_expt[0].x.shape[0])
                    ]
                    self.global_shape = self._spectrum_expt[0].x.shape[0]
                    self.global_mask = np.zeros(
                        [
                            self.global_shape,
                        ],
                        dtype=np.bool,
                    )
                else:
                    raise FileNotFoundError(
                        "input spectrum file doesn't exist."
                    )

                self._tth_list = [self._spectrum_expt[0]._x]
                self._tth_list_global = self._spectrum_expt[0]._x
                self.offset = False

            """
            03/08/2021 SS tth_min and max are now lists
            """
            self.tth_max = []
            self.tth_min = []
            self.ntth = []
            for s in self._spectrum_expt:
                self.tth_max.append(s.x.max())
                self.tth_min.append(s.x.min())
                self.ntth.append(s.x.shape[0])

            """
            03/02/2021 SS added tth_step for some computations
            related to snip background estimation
            @TODO this will not work for masked spectrum
            03/08/2021 tth_step is a list now
            """
            self.tth_step = []
            for tmi, tma, nth in zip(self.tth_min, self.tth_max, self.ntth):
                if nth > 1:
                    self.tth_step.append((tma - tmi) / nth)
                else:
                    self.tth_step.append(0.0)

            """
            @date 03/03/2021 SS
            there are cases when the intensity in the spectrum is
            negative. our approach will be to offset the spectrum to make all
            the values positive for the computation and then finally offset it
            when the computation has finished.
            03/08/2021 all quantities are lists now
            """
            for s in self._spectrum_expt:
                self.offset = []
                self.offset_val = []
                if np.any(s.y < 0.0):
                    self.offset.append(True)
                    self.offset_val.append(s.y.min())
                    s.y = s.y - s.y.min()

            """
            @date 09/24/2020 SS
            catching the cases when intensity is zero.
            for these points the weights will become
            infinite. therefore, these points will be
            masked out and assigned a weight of zero.
            In addition, if any points have negative
            intensity, they will also be assigned a zero
            weight
            03/08/2021 SS everything is a list now
            """
            self._weights = []
            for s in self._spectrum_expt:
                mask = s.y <= 0.0
                ww = np.zeros(s.y.shape)
                """also initialize statistical weights
                for the error calculation"""
                ww[~mask] = 1.0 / np.sqrt(s.y[~mask])
                self._weights.append(ww)

            self.initialize_bkg()
        else:
            raise RuntimeError("expt_spectrum setter: spectrum is None")

    @property
    def spectrum_sim(self):
        tth, inten = self._spectrum_sim.data
        inten[self.global_mask] = np.nan
        inten += self.background.y

        return Spectrum(x=tth, y=inten)

    @property
    def background(self):
        vector_list = [s.y for s in self._background]

        bkg_masked = join_regions(
            vector_list, self.global_index, self.global_shape
        )
        return Spectrum(x=self.tth_list, y=bkg_masked)

    @property
    def weights(self):
        weights_masked = join_regions(
            self._weights, self.global_index, self.global_shape
        )
        return Spectrum(x=self.tth_list, y=weights_masked)

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, phase_info):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    06/08/2020 SS 1.0 original
                     02/01/2021 SS 2.0 modified to follow same input style as
                       LeBail class with inputs of Material class, dict, list
                       or filename valid
        >> @DETAILS: load the phases for the LeBail fits
        """

        if phase_info is not None:
            if isinstance(phase_info, Phases_Rietveld):
                """
                directly passing the phase class
                """
                self._phases = phase_info
            else:

                if hasattr(self, "wavelength"):
                    if self.wavelength is not None:
                        p = Phases_Rietveld(wavelength=self.wavelength)
                else:
                    p = Phases_Rietveld()

                if isinstance(phase_info, dict):
                    """
                    initialize class using a dictionary with key as
                    material file and values as the name of each phase
                    """
                    for material_file in phase_info:
                        material_names = phase_info[material_file]
                        if not isinstance(material_names, list):
                            material_names = [material_names]
                        p.add_many(material_file, material_names)

                elif isinstance(phase_info, str):
                    """
                    load from a yaml file
                    """
                    if path.exists(phase_info):
                        p.load(phase_info)
                    else:
                        raise FileNotFoundError("phase file doesn't exist.")

                elif isinstance(phase_info, Material):
                    if not p.phase_dict:
                        p[phase_info.name] = {}

                    for k, v in self.wavelength.items():
                        E = (
                            1.0e6
                            * constants.cPlanck
                            * constants.cLight
                            / constants.cCharge
                            / v[0].getVal('nm')
                        )
                        phase_info.beamEnergy = valWUnit(
                            "kev", "ENERGY", E, "keV"
                        )
                        p[phase_info.name][k] = Material_Rietveld(
                            fhdf=None, xtal=None,
                            dmin=None, material_obj=phase_info
                        )
                        p[phase_info.name][k].pf = 1.0
                    p.num_phases = 1

                elif isinstance(phase_info, list):
                    for mat in phase_info:
                        p[mat.name] = {}
                        for k, v in self.wavelength.items():
                            E = (
                                1.0e6
                                * constants.cPlanck
                                * constants.cLight
                                / constants.cCharge
                                / v[0].getVal('nm')
                            )
                            mat.beamEnergy = valWUnit(
                                "kev", "ENERGY", E, "keV"
                            )
                            p[mat.name][k] = Material_Rietveld(
                                fhdf=None, xtal=None,
                                dmin=None, material_obj=mat
                            )
                        p.num_phases += 1

                    for mat in p:
                        for k, v in self.wavelength.items():
                            p[mat][k].pf = 1.0 / p.num_phases
                self._phases = p

        self.calctth()
        self.calcsf()

        for p in self.phases:
            for k in self.phases[p]:
                (
                    self._phases[p][k].valid_shkl,
                    self._phases[p][k].eq_constraints,
                    self._phases[p][k].rqd_index,
                    self._phases[p][k].trig_ptype,
                ) = wppfsupport._required_shkl_names(self._phases[p][k])

    @property
    def peakshape(self):
        return self._peakshape

    @peakshape.setter
    def peakshape(self, val):
        """
        @TODO make sure the parameter list
        is updated when the peakshape changes
        """
        if isinstance(val, str):
            if val == "pvfcj":
                self._peakshape = 0
            elif val == "pvtch":
                self._peakshape = 1
            elif val == "pvpink":
                self._peakshape = 2
            else:
                msg = (
                    "invalid peak shape string. "
                    "must be: \n"
                    "1. pvfcj: pseudo voight (Finger, Cox, Jephcoat)\n"
                    "2. pvtch: pseudo voight (Thompson, Cox, Hastings)\n"
                    "3. pvpink: Pink beam (Von Dreele)"
                )
                raise ValueError(msg)
        elif isinstance(val, int):
            if val >= 0 and val <= 2:
                self._peakshape = val
            else:
                msg = (
                    "invalid peak shape int. "
                    "must be: \n"
                    "1. 0: pseudo voight (Finger, Cox, Jephcoat)\n"
                    "2. 1: pseudo voight (Thompson, Cox, Hastings)\n"
                    "3. 2: Pink beam (Von Dreele)"
                )
                raise ValueError(msg)

        """
        update parameters
        """
        if hasattr(self, "params"):
            params = wppfsupport._generate_default_parameters_Rietveld(
                self.phases, self.peakshape
            )
            for p in params:
                if p in self.params:
                    params[p] = self.params[p]
            self._params = params
            self._set_params_vals_to_class(params, init=True, skip_phases=True)
            self.computespectrum()

    @property
    def computespectrum_fcn(self):
        if self.peakshape == 0:
            return computespectrum_pvfcj
        elif self.peakshape == 1:
            return computespectrum_pvtch
        elif self.peakshape == 2:
            return computespectrum_pvpink

    @property
    def calc_Iobs_fcn(self):
        if self.peakshape == 0:
            return calc_Iobs_pvfcj
        elif self.peakshape == 1:
            return calc_Iobs_pvtch
        elif self.peakshape == 2:
            return calc_Iobs_pvpink

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, Uinp):
        self._U = Uinp
        return

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, Vinp):
        self._V = Vinp
        return

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, Winp):
        self._W = Winp
        return

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, Pinp):
        self._P = Pinp
        return

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, Xinp):
        self._X = Xinp
        return

    @property
    def Xe(self):
        return self._Xe

    @Xe.setter
    def Xe(self, Xeinp):
        self._Xe = Xeinp
        return

    @property
    def Xs(self):
        return self._Xs

    @Xs.setter
    def Xs(self, Xsinp):
        self._Xs = Xsinp
        return

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Yinp):
        self._Y = Yinp
        return

    @property
    def Ye(self):
        return self._Ye

    @Ye.setter
    def Ye(self, Yeinp):
        self._Ye = Yeinp
        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

    @property
    def Hcag(self):
        return self._Hcag

    @Hcag.setter
    def Hcag(self, val):
        self._Hcag = val

    @property
    def tth_list(self):
        if isinstance(self.spectrum_expt._x, np.ma.MaskedArray):
            return self.spectrum_expt._x.filled()
        else:
            return self.spectrum_expt._x

    @property
    def zero_error(self):
        return self._zero_error

    @zero_error.setter
    def zero_error(self, value):
        self._zero_error = value
        return

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        return

    @property
    def Ph(self):
        return self._Ph

    @Ph.setter
    def Ph(self, val):
        self._Ph = val

    @property
    def eta_fwhm(self):
        return self._eta_fwhm

    @eta_fwhm.setter
    def eta_fwhm(self, val):
        self._eta_fwhm = val


def calc_num_variables(params):
    P = 0
    for pp in params:
        if params[pp].vary:
            P += 1
    return P


def separate_regions(masked_spec_array):
    """
    utility function for separating array into separate
    islands as dictated by mask. this function was taken from
    stackoverflow
    https://stackoverflow.com/questions/43385877/
    efficient-numpy-subarrays-extraction-from-a-mask
    """
    array = masked_spec_array.data
    mask = ~masked_spec_array.mask[:, 1]
    m0 = np.concatenate(([False], mask, [False]))
    idx = np.flatnonzero(m0[1:] != m0[:-1])
    gidx = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    return [array[idx[i]: idx[i + 1], :] for i in range(0, len(idx), 2)], gidx


def join_regions(vector_list, global_index, global_shape):
    """
    @author Saransh Singh Lawrence Livermore National Lab
    @date 03/08/2021 SS 1.0 original
    @details utility function for joining different pieces of masked array
    into one masked array
    """
    out_vector = np.empty(
        [
            global_shape,
        ]
    )
    out_vector[:] = np.nan
    for s, ids in zip(vector_list, global_index):
        out_vector[ids[0]: ids[1]] = s

    # out_array = np.ma.masked_array(out_array, mask = np.isnan(out_array))
    return out_vector

peakshape_dict = {
    "pvfcj": "pseudo-voight (finger, cox, jephcoat)",
    "pvtch": "pseudo-voight (thompson, cox, hastings)",
    "pvpink": "pseudo-voight (von dreele)",
}
