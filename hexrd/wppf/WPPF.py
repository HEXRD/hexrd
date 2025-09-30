# standard imports
# ---------
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import copy
from functools import partial
from os import path
import time
import warnings

# 3rd party imports
# -----------------
import h5py
import lmfit
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre

# hexrd imports
# -------------
from hexrd import constants
from hexrd.imageutil import snip1d_quad
from hexrd.material import Material
from hexrd.transforms.xfcapi import angles_to_gvec
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
from hexrd.wppf.phase import (
    Phases_LeBail,
    Phases_Rietveld,
    Material_LeBail,
    Material_Rietveld,
)


class AbstractWPPF(ABC):
    """Methods used by both LeBail and Rietveld"""

    # Abstract methods which must be defined for each WPPF type
    @abstractmethod
    def generate_default_parameters(self) -> lmfit.Parameters:
        pass

    @abstractmethod
    def _add_phase_params(self, params: lmfit.Parameters):
        pass

    @abstractmethod
    def _set_phase_params_vals_to_class(self, params: lmfit.Parameters):
        pass

    @abstractmethod
    def _get_phase(self, name: str, wavelength_type: str):
        pass

    # Shared methods which each WPPF type uses
    def __str__(self):
        cls_name = self.__class__.__name__
        resstr = (
            f"<{cls_name} Fit class>\n"
            "resParameters of the model are as follows:\n"
        )
        resstr += self.params.__str__()
        return resstr

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

    @property
    def num_vary(self) -> int:
        return sum(x.vary for x in self.params.values())

    @property
    def bkgmethod(self):
        return self._bkgmethod

    @bkgmethod.setter
    def bkgmethod(self, v):
        self._bkgmethod = v
        if "chebyshev" in v and hasattr(self, 'bkg_coef'):
            degree = v["chebyshev"]
            # In case the degree has changed, slice off any extra at the end,
            # and in case it is less, pad with zeros.
            if len(self.bkg_coef) > degree + 1:
                self.bkg_coef = self.bkg_coef[:degree + 1]
            elif len(self.bkg_coef) < degree + 1:
                pad_width = (0, degree + 1 - len(self.bkg_coef))
                self.bkg_coef = np.pad(self.bkg_coef, pad_width)

    def chebyshevfit(self):
        """
        03/08/2021 SS spectrum_expt is a list now. accounting
        for that change
        """
        self._background = []
        self.bkg_coef = self.cheb_init_coef
        for i, s in enumerate(self._spectrum_expt):
            tth = self._tth_list[i]
            self._background.append(Spectrum(x=tth, y=self.init_bkg(tth)))

    # cubic spline fit of background using custom points chosen from plot
    def splinefit(self, points):
        """
        03/08/2021 SS adding tth as input. this is the
        list of points for which background is estimated
        """
        self._background = []
        x = points[:, 0]
        y = points[:, 1]
        cs = CubicSpline(x, y)
        for i, s in enumerate(self._spectrum_expt):
            tth = self._tth_list[i]
            bkg = cs(tth)
            self._background.append(Spectrum(x=tth, y=bkg))

    @property
    def cheb_coef(self):
        if "chebyshev" in self.bkgmethod:
            return self.bkg_coef
        else:
            return None

    @property
    def cheb_polynomial(self):
        return np.polynomial.Chebyshev(
            self.cheb_coef,
            domain=[self.tth_list[0], self.tth_list[-1]]
        )

    @property
    def cheb_init_coef(self):
        if "chebyshev" in self.bkgmethod:
            return self.init_bkg.coef
        else:
            return None

    def reset_background_params(self):
        # Reset background parameters to their initial values
        if "chebyshev" not in self.bkgmethod:
            return

        params = self.params
        for i in range(len(self.bkg_coef)):
            name = f'bkg_{i}'
            if name in params:
                params[name].value = self.bkg_coef[i]

    def _update_bkg(self, params):
        """
        update the background coefficients for
        the chebyshev polynomials
        """
        if "chebyshev" in self.bkgmethod:
            coef = self.bkg_coef.copy()
            for p in params:
                if "bkg" in p:
                    indx = int(p.split("_")[1])
                    coef[indx] = params[p].value
            self.bkg_coef = coef
        else:
            return

    def _update_shkl(self, params):
        """
        if certain shkls are refined, then update
        them using the params arg. else use values from
        the parameter class
        """
        shkl_dict = {}
        for p in self.phases:
            for k in self.phases.wavelength:
                phase = self._get_phase(p, k)
                shkl_name = phase.valid_shkl
                eq_const = phase.eq_constraints
                mname = phase.name
                key = [f"{mname}_{s}" for s in shkl_name]
                for s, k in zip(shkl_name, key):
                    if k in params:
                        shkl_dict[s] = params[k].value
                    else:
                        shkl_dict[s] = self.params[k].value

                phase.shkl = wppfsupport._fill_shkl(shkl_dict, eq_const)

    def update_parameters(self):
        for p in self.res.params:
            par = self.res.params[p]
            self.params[p].value = par.value

    def calctth(self):
        self.tth = {}
        self.hkls = {}
        self.dsp = {}
        self.limit = {}
        self.sf_hkl_factors = {}
        self.sf_lfactor = {}
        for p in self.phases:
            self.tth[p] = {}
            self.hkls[p] = {}
            self.dsp[p] = {}
            self.limit[p] = {}
            self.sf_hkl_factors[p] = {}
            self.sf_lfactor[p] = {}
            for k, l in self.phases.wavelength.items():
                phase = self._get_phase(p, k)
                t = phase.getTTh(l[0].getVal('nm'))
                sf_f, lfact_sf = phase.get_sf_hkl_factors()
                allowed = phase.wavelength_allowed_hkls
                t = t[allowed]
                hkl = phase.hkls[allowed, :]
                dsp = phase.dsp[allowed]
                tth_min = min(self.tth_min)
                tth_max = max(self.tth_max)
                limit = np.logical_and(t >= tth_min, t <= tth_max)
                self.limit[p][k] = limit
                self.tth[p][k] = t[limit]
                self.hkls[p][k] = hkl[limit, :]
                self.dsp[p][k] = dsp[limit]
                if sf_f is not None and lfact_sf is not None:
                    sf_f = sf_f[allowed]
                    lfact_sf = lfact_sf[allowed]
                    self.sf_hkl_factors[p][k] = sf_f[limit]
                    self.sf_lfactor[p][k] = lfact_sf[limit]

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
        self._update_bkg(params)
        errvec = self.computespectrum()

        return errvec

    @property
    def spectrum_sim(self):
        tth, inten = self._spectrum_sim.data
        inten[self.global_mask] = np.nan
        inten += self.background.y

        return Spectrum(x=tth, y=inten)

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
                    dtype=bool,
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
                        dtype=bool,
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
                        dtype=bool,
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
                ww[~mask] = 1.0 / s.y[~mask]
                self._weights.append(ww)

            self.initialize_bkg()
        else:
            raise RuntimeError("expt_spectrum setter: spectrum is None")

    @property
    def background(self):
        if "chebyshev" in self.bkgmethod:
            vector_list = [self.cheb_polynomial(t) for t in self._tth_list]
        else:
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
        xn, wn = roots_legendre(16)
        self.xn = xn[8:]
        self.wn = wn[8:]

        if param_info is not None:
            if isinstance(param_info, lmfit.Parameters):
                """
                directly passing the parameter class
                """
                self._params = param_info
                params = param_info

            else:
                params = lmfit.Parameters()

                if isinstance(param_info, dict):
                    """
                    initialize class using dictionary read from the yaml file
                    """
                    for k, v in param_info.items():
                        params.add(
                            k,
                            value=float(v[0]),
                            min=float(v[1]),
                            max=float(v[2]),
                            vary=bool(v[3]),
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
                self._add_phase_params(params)
                self._params = params
        else:
            """
            first three are cagliotti parameters
            next two are the lorentz paramters
            final is the zero instrumental peak position error
            mixing factor calculated by Thomax, Cox, Hastings formula
            """
            params = self.generate_default_parameters()
            self._params = params

        self._set_params_vals_to_class(params, init=True, skip_phases=True)

    def _set_params_vals_to_class(self, params, init=False, skip_phases=False):
        """
        @date: 03/12/2021 SS 1.0 original
        @details: set the values from parameters to the WPPF class
        """
        for p in params:
            if init or hasattr(self, p):
                setattr(self, p, params[p].value)

        if not skip_phases:
            self._set_phase_params_vals_to_class(params)

        if self.amorphous_model is not None:
            self._set_amorphous_params_vals_to_class(params)

    def _set_amorphous_params_vals_to_class(self, params: lmfit.Parameters):
        if self.amorphous_model is None:
            # Can't do anything
            return

        scale = {}
        shift = {}
        center = {}
        fwhm = {}
        for key in self.amorphous_model.scale:
            nn = f'{key}_amorphous_scale'
            if nn in params:
                scale[key] = params[nn].value
            else:
                scale[key] = self.amorphous_model.scale[key]

            if self.amorphous_model.model_type == "experimental":
                nn = f'{key}_amorphous_shift'
                if nn in params:
                    shift[key] = params[nn].value
                else:
                    shift[key] = self.amorphous_model.shift[key]

            if self.amorphous_model.model_type in ("split_gaussian",
                                                   "split_pv"):
                nn = f'{key}_amorphous_center'
                if nn in params:
                    center[key] = params[nn].value
                else:
                    center[key] = self.amorphous_model.center[key]

            if self.amorphous_model.model_type == "split_gaussian":
                nnl = f'{key}_amorphous_fwhm_l'
                if nnl in params:
                    fwhm_l = params[nnl].value
                else:
                    fwhm_l = self.amorphous_model.fwhm[key][0]

                nnr = f'{key}_amorphous_fwhm_r'
                if nnr in params:
                    fwhm_r = params[nnr].value
                else:
                    fwhm_r = self.amorphous_model.fwhm[key][1]

                fwhm[key] = np.array([fwhm_l,
                                      fwhm_r
                                      ])
            elif self.amorphous_model.model_type == "split_pv":
                nnl = f'{key}_amorphous_fwhm_g_l'
                if nnl in params:
                    fwhm_g_l = params[nnl].value
                else:
                    fwhm_g_l = self.amorphous_model.fwhm[key][0]

                nnl = f'{key}_amorphous_fwhm_l_l'
                if nnl in params:
                    fwhm_l_l = params[nnl].value
                else:
                    fwhm_l_l = self.amorphous_model.fwhm[key][1]

                nnr = f'{key}_amorphous_fwhm_g_r'
                if nnr in params:
                    fwhm_g_r = params[nnr].value
                else:
                    fwhm_g_r = self.amorphous_model.fwhm[key][2]

                nnr = f'{key}_amorphous_fwhm_l_r'
                if nnr in params:
                    fwhm_l_r = params[nnr].value
                else:
                    fwhm_l_r = self.amorphous_model.fwhm[key][3]

                fwhm[key] = np.array([fwhm_g_l,
                                      fwhm_l_l,
                                      fwhm_g_r,
                                      fwhm_l_r
                                      ])

        self.amorphous_model.scale = scale

        if self.amorphous_model.model_type == "experimental":
            self.amorphous_model.shift = shift

        elif self.amorphous_model.model_type in ["split_gaussian",
                                                 "split_pv"]:
            self.amorphous_model.center = center
            self.amorphous_model.fwhm = fwhm

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

            points = self.bkgmethod["spline"]
            if not points:
                raise RuntimeError(
                    "Background points must be set to use spline"
                )

            self.splinefit(np.squeeze(np.asarray(points)))

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

    @property
    def init_bkg(self):
        degree = self.bkgmethod["chebyshev"]
        x = np.empty([0, ])
        y = np.empty([0, ])
        wts = np.empty([0, ])
        for i, s in enumerate(self._spectrum_expt):
            tth = self._tth_list[i]
            wt = self._weights[i]
            x = np.append(x, tth)
            y = np.append(y, s._y)
            wts = np.append(wts, wt)
        p = np.polynomial.Chebyshev.fit(x, y, degree, w=wts)
        return p

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
            params = self.generate_default_parameters()
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
    def tth_list(self):
        if isinstance(self.spectrum_expt._x, np.ma.MaskedArray):
            return self.spectrum_expt._x.filled()
        else:
            return self.spectrum_expt._x

    @property
    def total_area(self):
        tth, intensity = self.spectrum_sim.data
        _, background = self.background.data
        total_intensity = intensity-background
        '''put some guard rails around the total intensity
        to protect against nans in the values
        '''
        mask = np.isnan(total_intensity)
        sum_area = np.trapz(total_intensity[~mask], tth[~mask])
        return sum_area

    @property
    def DOC(self):
        if self.amorphous_model is None:
            return 1.

        amorphous_area = self.amorphous_model.integrated_area
        return 1. - amorphous_area / self.total_area


def _nm(x):
    return valWUnit("lp", "length", x, "nm")


class LeBail(AbstractWPPF):
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
        amorphous_model=None,
        reset_background_params=True,
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

        self.amorphous_model = amorphous_model

        self.params = params

        self.initialize_Icalc()

        self.computespectrum()

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart
        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

        if reset_background_params:
            # Reset background parameters to their initial values
            self.reset_background_params()
        else:
            self._update_bkg(self.params)

    def generate_default_parameters(self) -> lmfit.Parameters:
        return wppfsupport._generate_default_parameters_LeBail(
            self.phases,
            self.peakshape,
            self.bkgmethod,
            amorphous_model=self.amorphous_model,
        )

    def _add_phase_params(self, params: lmfit.Parameters):
        for p in self.phases:
            wppfsupport._add_lp_to_params(params, self.phases[p])

    def _set_phase_params_vals_to_class(self, params: lmfit.Parameters):
        updated_lp = False
        for p in self.phases:
            mat = self.phases[p]

            """
            PART 1: update stacking fault and twin beta parameters
            """
            if mat.sgnum == 225:
                sf_alpha_name = f"{p}_sf_alpha"
                twin_beta_name = f"{p}_twin_beta"
                if params[sf_alpha_name].vary:
                    mat.sf_alpha = params[sf_alpha_name].value
                if params[twin_beta_name].vary:
                    mat.twin_beta = params[twin_beta_name].value

            """
            PART 2: update the lattice parameters
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

            if lpvary:
                lp = self.phases[p].Required_lp(lp)
                mat.lparms = np.array(lp)
                mat._calcrmt()
                updated_lp = True

        if updated_lp:
            self.calctth()

    def _get_phase(self, name: str, wavelength_type: str):
        # LeBail just ignores the wavelength type for phases
        return self.phases[name]

    def initialize_Icalc(self):
        """
        @DATE 01/22/2021 SS modified the function so Icalc can be initialized
        with a dictionary of structure factors
        """

        self.Icalc = {}

        if self.intensity_init is None:
            if self.spectrum_expt._y.max() > 0:
                n10 = np.floor(np.log10(self.spectrum_expt._y.max())) - 1
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

                name = self.phases[p].name
                lam = l[0].getVal("nm")
                Ic = self.Icalc[p][k]

                shft_c = np.cos(0.5 * np.radians(self.tth[p][k])) * self.shft
                trns_c = np.sin(np.radians(self.tth[p][k])) * self.trns

                if self.phases[p].sf_alpha is None:
                    sf_shift = 0.0
                    Xs = np.zeros(Ic.shape)
                else:
                    alpha = getattr(self, f"{p}_sf_alpha")
                    beta = getattr(self, f"{p}_twin_beta")
                    sf_shift = alpha*np.tan(np.radians(0.5*self.tth[p][k])) *\
                        self.sf_hkl_factors[p][k]
                    Xs = np.degrees(0.9 * (1.5 * alpha + beta) * (
                        self.sf_lfactor[p][k] * lam / self.phases[p].lparms[0])
                    )

                tth = self.tth[p][k] + self.zero_error + \
                    shft_c + trns_c + sf_shift

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                # n = np.min((tth.shape[0], Ic.shape[0]))
                shkl = self.phases[p].shkl
                eta_fwhm = getattr(self, f"{name}_eta_fwhm")

                X = getattr(self, f"{name}_X")
                Y = getattr(self, f"{name}_Y")
                P = getattr(self, f"{name}_P")
                XY = np.array([X, Y])

                if self.peakshape == 0:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        P,
                        XY,
                        Xs,
                        shkl,
                        eta_fwhm,
                        self.HL,
                        self.SL,
                        tth,
                        dsp,
                        hkls,
                        tth_list,
                        Ic,
                        self.xn,
                        self.wn,
                    )

                elif self.peakshape == 1:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        P,
                        XY,
                        Xs,
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        tth_list,
                        Ic,
                    )

                elif self.peakshape == 2:
                    args = (
                        np.array([self.alpha0, self.alpha1]),
                        np.array([self.beta0, self.beta1]),
                        np.array([self.U, self.V, self.W]),
                        P,
                        XY,
                        Xs,
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        tth_list,
                        Ic,
                    )

                y += self.computespectrum_fcn(*args)

        if self.amorphous_model is not None:
            y += self.amorphous_model.amorphous_lineout

        self._spectrum_sim = Spectrum(x=x, y=y)

        errvec, self.Rwp, self.gofF = calc_rwp(
            self.spectrum_sim.data_array,
            self.spectrum_expt.data_array,
            self.weights.data_array,
            self.num_vary,
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

                name = self.phases[p].name
                Ic = self.Icalc[p][k]
                lam = l[0].getVal("nm")

                shft_c = np.cos(0.5 * np.radians(self.tth[p][k])) * self.shft
                trns_c = np.sin(np.radians(self.tth[p][k])) * self.trns

                sf_shift = 0.0
                Xs = np.zeros(Ic.shape)
                if self.phases[p].sf_alpha is not None:
                    alpha = getattr(self, f"{p}_sf_alpha")
                    beta = getattr(self, f"{p}_twin_beta")
                    sf_shift = alpha*np.tan(np.radians(0.5*self.tth[p][k])) *\
                        self.sf_hkl_factors[p][k]
                    Xs = np.degrees(0.9 * (1.5 * alpha + beta) * (
                        self.sf_lfactor[p][k] * lam / self.phases[p].lparms[0]
                    ))

                tth = self.tth[p][k] + self.zero_error + \
                    shft_c + trns_c + sf_shift

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                # n = np.min((tth.shape[0], Ic.shape[0]))  # !!! not used
                shkl = self.phases[p].shkl
                eta_fwhm = getattr(self, f"{name}_eta_fwhm")

                X = getattr(self, f"{name}_X")
                Y = getattr(self, f"{name}_Y")
                P = getattr(self, f"{name}_P")
                XY = np.array([X, Y])

                if self.peakshape == 0:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        P,
                        XY,
                        Xs,
                        shkl,
                        eta_fwhm,
                        self.HL,
                        self.SL,
                        self.xn,
                        self.wn,
                        tth,
                        dsp,
                        hkls,
                        tth_list,
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                elif self.peakshape == 1:
                    args = (
                        np.array([self.U, self.V, self.W]),
                        P,
                        XY,
                        Xs,
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
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
                        P,
                        XY,
                        Xs,
                        shkl,
                        eta_fwhm,
                        tth,
                        dsp,
                        hkls,
                        tth_list,
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                self.Iobs[p][k] = self.calc_Iobs_fcn(*args)

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
            msg = (f"Finished iteration. Rwp: "
                   f"{self.Rwp*100.0:.2f} % and chi^2: {self.gofF:.2f}")
            print(msg)

    def Refine(self):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >> @DETAILS: this routine performs the least squares refinement for all
                     variables which are allowed to be varied.
        """
        if self.num_vary > 0:
            fdict = {
                "ftol": 1e-6,
                "xtol": 1e-6,
                "gtol": 1e-6,
                "verbose": 0,
                "max_nfev": 1000,
                "method": "trf",
                "jac": "2-point",
            }
            fitter = lmfit.Minimizer(self.calcRwp, self.params)

            res = fitter.least_squares(**fdict)
            return res
        else:
            msg = "nothing to refine. updating intensities"
            print(msg)
            self.computespectrum()
            return getattr(self, 'res', None)

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
        if phase_info is None:
            self._on_phases_changed()
            return

        if isinstance(phase_info, Phases_LeBail):
            """
            directly passing the phase class
            """
            self._phases = phase_info
            self._on_phases_changed()
            return

        p = Phases_LeBail(wavelength=getattr(self, "wavelength", None))

        if isinstance(phase_info, dict):
            """
            initialize class using a dictionary with key as
            material file and values as the name of each phase
            """
            for material_file, material_names in phase_info.items():
                if not isinstance(material_names, list):
                    material_names = [material_names]
                p.add_many(material_file, material_names)
        elif isinstance(phase_info, str):
            """
            load from a yaml file
            """
            p.load(phase_info)
        else:
            mat_list = []
            if isinstance(phase_info, Material):
                mat_list = [phase_info]
            elif isinstance(phase_info, list):
                mat_list = phase_info

            for mat in mat_list:
                p[mat.name] = Material_LeBail(
                    fhdf=None, xtal=None,
                    dmin=None, material_obj=mat
                )

            p.reset_phase_fractions()

        self._phases = p
        self._on_phases_changed()

    def _on_phases_changed(self):
        self.calctth()
        for p in self.phases:
            (
                self.phases[p].valid_shkl,
                self.phases[p].eq_constraints,
                self.phases[p].rqd_index,
                self.phases[p].trig_ptype,
            ) = wppfsupport._required_shkl_names(self.phases[p])


class Rietveld(AbstractWPPF):
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
        amorphous_model=None,
        reset_background_params=True,
        texture_model=None,
        eta_min=-180,
        eta_max=180,
        eta_step=5.,
    ):

        self.bkgmethod = bkgmethod
        self.shape_factor = shape_factor
        self.particle_size = particle_size
        self.phi = phi
        self.peakshape = peakshape
        self.spectrum_expt = expt_spectrum
        self.amorphous_model = amorphous_model

        # extent of 2D data in the azimuth
        # important for texture analysis
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_step = eta_step

        self._tstart = time.time()

        if wavelength is not None:
            self.wavelength = wavelength
            for k, v in self.wavelength.items():
                v[0] = valWUnit("lp", "length", v[0].getVal("nm"), "nm")

        self.phases = phases
        self.texture_model = texture_model

        self.params = params

        self.PolarizationFactor()
        self.computespectrum()

        self._tstop = time.time()
        self.tinit = self._tstop - self._tstart

        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

        if reset_background_params:
            # Reset background parameters to their initial values
            self.reset_background_params()
        else:
            self._update_bkg(self.params)

    def generate_default_parameters(self) -> lmfit.Parameters:
        return wppfsupport._generate_default_parameters_Rietveld(
            self.phases,
            self.peakshape,
            self.bkgmethod,
            init_val=self.cheb_init_coef,
            amorphous_model=self.amorphous_model,
            texture_model=self.texture_model,
        )

    def _add_phase_params(self, params: lmfit.Parameters):
        for p in self.phases:
            for lpi in self.phases[p]:
                wppfsupport._add_atominfo_to_params(
                    params, self.phases[p][lpi]
                )

    def _set_phase_params_vals_to_class(self, params: lmfit.Parameters):
        # These indicate that *any* materials updated their lattice
        # parameters or atom info.
        updated_lp = False
        updated_atominfo = False

        pf = np.zeros([self.phases.num_phases, ])
        for ii, p in enumerate(self.phases):
            name = f"{p}_phase_fraction"
            pf[ii] = params[name].value

            for lpi in self.phases[p]:
                mat = self.phases[p][lpi]

                # This indicate *this* material updated its lattice parameter
                # or atom info.
                lpvary = False
                atominfo_vary = False

                """
                PART 1: update stacking fault and twin beta parameters
                """
                if mat.sgnum == 225:
                    sf_alpha_name = f"{p}_sf_alpha"
                    twin_beta_name = f"{p}_twin_beta"
                    if params[sf_alpha_name].vary:
                        mat.sf_alpha = params[sf_alpha_name].value
                    if params[twin_beta_name].vary:
                        mat.twin_beta = params[twin_beta_name].value

                """
                PART 2: update the lattice parameters
                """
                lp = []
                pre = f"{p}_"
                name = [f"{pre}{x}" for x in wppfsupport._lpname]
                for nn in name:
                    if nn in params:
                        if params[nn].vary:
                            lpvary = True
                        lp.append(params[nn].value)

                if lpvary:
                    lp = mat.Required_lp(lp)
                    mat.lparms = np.array(lp)
                    mat._calcrmt()

                """
                PART 3: update the atom info
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

                    atominfo_vary = any(
                        params[name].vary for name in (nx, ny, nz, oc)
                    )
                    x = params[nx].value
                    y = params[ny].value
                    z = params[nz].value
                    oc = params[oc].value

                    if mat.aniU:
                        U = []
                        for j in range(6):
                            param = params[Un[j]]
                            if param.vary:
                                atominfo_vary = True
                            U.append(param.value)
                        mat.U[i, :] = np.array(U)
                    else:
                        if params[dw].vary:
                            atominfo_vary = True

                        mat.U[i] = params[dw].value

                    mat.atom_pos[i, :] = np.array([x, y, z, oc])

                if mat.aniU:
                    mat.calcBetaij()

                if lpvary:
                    updated_lp = True

                if atominfo_vary:
                    updated_atominfo = True

        if updated_lp:
            self.calctth()

        if updated_lp or updated_atominfo:
            self.calcsf()

        self.phases.phase_fraction = pf / np.sum(pf)

    def _get_phase(self, name: str, wavelength_type: str):
        # Rietveld uses the wavelength type
        return self.phases[name][wavelength_type]

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
                    # / (2.0 * (1 + Ph))
                )

    def compute_intensities(self):
        '''this function computes the intensities of the
        xray diffraction excluding any texture. this function
        will replace part of the code in the computespectrum
        function
        '''
        Ic = {}
        for iph, p in enumerate(self.phases):
            Ic[p] = {}
            for k, l in self.phases.wavelength.items():
                tth = self.tth[p][k]
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

                Ic[p][k] = self.scale * pf * sf * lp  # *extinction*absorption
        return Ic

    def compute_tth_after_shifts(self,
                                 p,
                                 k):
        '''another helper function to be used by both 
        Rietveld.computspectrum and Rietveld.computespectrum_2d

        Parameters
        ----------

        p: str
            name of the phase
        k: str
            wavelength key
        '''
        name = self.phases[p][k].name
        lam = self.phases.wavelength[k][0].getVal("nm")
        shft_c = np.cos(0.5 * np.radians(self.tth[p][k])) * self.shft
        trns_c = np.sin(np.radians(self.tth[p][k])) * self.trns
        if self.phases[p][k].sf_alpha is None:
            sf_shift = 0.0
            Xs = np.zeros(self.tth[p][k].shape)
        else:
            alpha = getattr(self, f"{p}_sf_alpha")
            beta = getattr(self, f"{p}_twin_beta")
            sf_shift = alpha*np.tan(np.radians(0.5*self.tth[p][k])) *\
                self.sf_hkl_factors[p][k]
            Xs = np.degrees(0.9*(1.5*alpha+beta)*(
                self.sf_lfactor[p][k] *
                lam/self.phases[p][k].lparms[0]))
        tth = self.tth[p][k] + self.zero_error + \
            shft_c + trns_c + sf_shift

        return tth, Xs


    def computespectrum_phase(self,
                              p,
                              k,
                              Ic,
                              texture_factor=None):
        '''this is a helper function so which is use by both the 
        Rietveld.computspectrum and Rietveld.computespectrum_2d
        function to avoid code repetition.

        Parameters
        ----------

        p: str
            name of the phase
        k: str
            wavelength key
        texture_factor: numpy.ndarray
            azimuthally averaged texture factor
        '''
        tth_list = np.ascontiguousarray(self.tth_list)
        name = self.phases[p][k].name

        tth, Xs = self.compute_tth_after_shifts(p, k)

        Icmod = Ic.copy()
        if not texture_factor is None:
            n = np.min((tth.shape[0], Ic.shape[0], texture_factor.shape[0]))
            tth = tth[:n]
            Icmod = Icmod[:n] * texture_factor[:n]  # *extinction*absorption

        dsp = self.dsp[p][k]
        hkls = self.hkls[p][k]
        shkl = self.phases[p][k].shkl
        eta_fwhm = getattr(self, f"{name}_eta_fwhm")
        X = getattr(self, f"{name}_X")
        Y = getattr(self, f"{name}_Y")
        P = getattr(self, f"{name}_P")
        XY = np.array([X, Y])

        if self.peakshape == 0:
            args = (
                np.array([self.U, self.V, self.W]),
                P,
                XY,
                Xs,
                shkl,
                eta_fwhm,
                self.HL,
                self.SL,
                tth,
                dsp,
                hkls,
                tth_list,
                Icmod,
                self.xn,
                self.wn,
            )
        elif self.peakshape == 1:
            args = (
                np.array([self.U, self.V, self.W]),
                P,
                XY,
                Xs,
                shkl,
                eta_fwhm,
                tth,
                dsp,
                hkls,
                tth_list,
                Icmod,
            )
        elif self.peakshape == 2:
            args = (
                np.array([self.alpha0, self.alpha1]),
                np.array([self.beta0, self.beta1]),
                np.array([self.U, self.V, self.W]),
                P,
                XY,
                Xs,
                shkl,
                eta_fwhm,
                tth,
                dsp,
                hkls,
                tth_list,
                Icmod,
            )
        return self.computespectrum_fcn(*args)


    def computespectrum(self):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab
        >> @EMAIL:      saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    compute the simulated spectrum
        """

        x = self.tth_list
        y = np.zeros(x.shape)
        Icomputed = self.compute_intensities()
        for iph, p in enumerate(self.phases):

            for k, l in self.phases.wavelength.items():

                if self.texture_model[p] is None:
                    texture_factor = None
                else:
                    texture_factor = self.texture_model[p].calc_texture_factor(
                                            self.params,
                                            eta_min=self.eta_min,
                                            eta_max=self.eta_max)
                y += self.computespectrum_phase(p,
                                    k,
                                    Icomputed[p][k],
                                    texture_factor=texture_factor)

        if self.amorphous_model is not None:
            y += self.amorphous_model.amorphous_lineout

        self._spectrum_sim = Spectrum(x=x, y=y)

        errvec, self.Rwp, self.gofF = calc_rwp(
            self.spectrum_sim.data_array,
            self.spectrum_expt.data_array,
            self.weights.data_array,
            self.num_vary,
        )
        return errvec

    def computespectrum_2D(self):
        '''this function computes the 2D pattern for the 
        Rietevld model. if there is no texture, the pattern
        is  uniform in the azimuthal direction. if there is
        a texture model present, then the azimuthal intensities
        are modulated based on the texture model.

        Parameters
        -----------

        eta_min: float
            minimum azimuthal angle
        eta_max: float
            maximum azimuthal angle
        eta_step: float
            step size in azimuth. this determines the
            dimensions in the azimuthal direction

        Returns
        --------
        simulated_2d: np.ndarray
            simulated 2D diffraction pattern
        '''
        x = self.tth_list
        y = np.zeros(x.shape)

        azimuth = np.arange(self.eta_min,
                            self.eta_max,
                            self.eta_step)
        nazimuth = azimuth.shape[0]

        Icomputed = self.compute_intensities()

        if self.texture_model is None:

            for iph, p in enumerate(self.phases):
                for k, l in self.phases.wavelength.items():
                   y += self.computespectrum_phase(p,
                                                   k,
                                                   Icomputed[p][k],
                                                   texture_factor=None)
            y[self.global_mask] = np.nan
            y += self.background.y
            if self.amorphous_model is not None:
                y += self.amorphous_model.amorphous_lineout

            self.simulated_2d = np.tile(y, (nazimuth-1, 1))

            return

        else:
            '''get pole figure intensities around the azimuth
            '''
            self.simulated_2d = np.empty([nazimuth-1, x.shape[0]])
            azimuth_texture_factor = {}
            for iph, p in enumerate(self.phases):
                if p in self.texture_model:
                    self.texture_model[p].calc_pf_rings(
                                        self.params,
                                        eta_min=self.eta_min,
                                        eta_max=self.eta_max,
                                        eta_step=self.eta_step,
                                        calc_type='spectrum_2d')
                    azimuth_texture_factor[p] = self.texture_model[p].intensities_rings_2d
                else:
                    azimuth_texture_factor[p] = None

            for irow in range(1, nazimuth):
                y = np.zeros(x.shape)
                for iph, p in enumerate(self.phases):

                    for k, l in self.phases.wavelength.items():
                        hkls = self.hkls[p][k]
                        texture_factor = None

                        if not azimuth_texture_factor[p] is None:
                            texture_factor = np.empty([hkls.shape[0], ])
                            for ii,h in enumerate(hkls):
                                hkey = tuple(h)
                                texture_factor[ii] = azimuth_texture_factor[p][hkey][irow]

                        y += self.computespectrum_phase(p,
                                                        k,
                                                        Icomputed[p][k],
                                                        texture_factor=texture_factor)

                y[self.global_mask] = np.nan
                y += self.background.y
                if self.amorphous_model is not None:
                    y += self.amorphous_model.amorphous_lineout

                self.simulated_2d[irow-1, :] = y

    def Refine(self):
        """
        >> @AUTHOR:  Saransh Singh, Lawrence Livermore National Lab,
                     saransh1@llnl.gov
        >> @DATE:    05/19/2020 SS 1.0 original
        >> @DETAILS: this routine performs the least squares refinement for all
                     variables that are allowed to be varied.
        """
        if self.num_vary > 0:
            fdict = {
                "ftol": 1e-6,
                "xtol": 1e-6,
                "gtol": 1e-6,
                "verbose": 0,
                "max_nfev": 1000,
                "method": "trf",
                "jac": "2-point",
            }

            fitter = lmfit.Minimizer(self.calcRwp, self.params)

            self.res = fitter.least_squares(**fdict)

            self.update_parameters()

            self.niter += 1
            self.Rwplist = np.append(self.Rwplist, self.Rwp)
            self.gofFlist = np.append(self.gofFlist, self.gofF)

            msg = (f"Finished iteration. Rwp: "
                   f"{self.Rwp*100.0:.2f} % and chi^2: {self.gofF:.2f}")
            print(msg)
        else:
            print("Nothing to refine...")

    def RefineTexture(self):
        for name, model in self.texture_model.items():
            if model is None:
                continue

            print(f'Refining texture parameters for "{name}"')
            results = model.calculate_harmonic_coefficients(self.params)

        # Set the results to the final one
        self.res = results

        self.computespectrum()
        self.niter += 1
        self.Rwplist = np.append(self.Rwplist, self.Rwp)
        self.gofFlist = np.append(self.gofFlist, self.gofF)

        msg = (f"Finished iteration. Rwp: "
               f"{self.Rwp*100.0:.2f} % and chi^2: {self.gofF:.2f}")
        print(msg)

    def texture_parameters_vary(self,
                                vary=False):
        '''helper function to turn texture related
        parameters on or off
        '''
        for phase_name in self.phases:
            prefix = f'{phase_name}_c_'
            for p in self.params:
                if p.startswith(prefix):
                    self.params[p].vary = vary

    @property
    def any_texture_params_varied(self):
        for phase_name in self.phases:
            prefix = f'{phase_name}_c_'
            for param in self.params.values():
                if param.name.startswith(prefix) and param.vary:
                    return True

        return False

    @property
    def texture_models_have_pfdata(self):
        for model in self.texture_model.values():
            if not model.pfdata:
                return False

        return True

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

        if phase_info is None:
            self._on_phases_changed()
            return

        if isinstance(phase_info, Phases_Rietveld):
            """
            directly passing the phase class
            """
            self._phases = phase_info
            self._on_phases_changed()
            return

        p = Phases_Rietveld(wavelength=getattr(self, "wavelength", None))
        if isinstance(phase_info, dict):
            """
            initialize class using a dictionary with key as
            material file and values as the name of each phase
            """
            for material_file, material_names in phase_info.items():
                if not isinstance(material_names, list):
                    material_names = [material_names]
                p.add_many(material_file, material_names)
        elif isinstance(phase_info, str):
            """
            load from a yaml file
            """
            p.load(phase_info)
        else:
            mat_list = []
            if isinstance(phase_info, Material):
                mat_list = [phase_info]
            elif isinstance(phase_info, list):
                mat_list = phase_info

            for mat in mat_list:
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

            p.reset_phase_fractions()

        self._phases = p
        self._on_phases_changed()

    def _on_phases_changed(self):
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
    def texture_model(self):
        return self._texture_model

    @texture_model.setter
    def texture_model(self, valdict):
        '''only dictionary key value pairs are acceptable.
        key should match name of the phase. if a certain
        phase is not present, texture model for that phase
        is set to None
        '''
        if valdict is None:
            valdict = {}

        if not isinstance(valdict, dict):
            msg = (
                'only dictionary input allowed '
                'where key are name of phase and '
                'value is the harmonic_model instance'
            )
            raise ValueError(msg)

        self._texture_model = valdict
        for phase_name in self.phases:
            if phase_name not in valdict:
                self._texture_model[phase_name] = None

    @property
    def texture_index(self):
        res = {}
        for p in self.phases:
            if self.texture_model[p] is not None:
                res[p] = self.texture_model[p].J(self.params)
            else:
                res[p] = 1.
        return res

    @property
    def eta_min(self):
        return self._eta_min

    @eta_min.setter
    def eta_min(self, val):
        self._eta_min = np.radians(val)

    @property
    def eta_max(self):
        return self._eta_max

    @eta_max.setter
    def eta_max(self, val):
        self._eta_max = np.radians(val)

    @property
    def eta_step(self):
        return self._eta_step

    @eta_step.setter
    def eta_step(self, val):
        self._eta_step = np.radians(val)

    def compute_texture_data(self,
                             pv_binned: np.ndarray,
                             bvec: np.ndarray | None = None,
                             evec: np.ndarray | None = None,
                             azimuthal_interval: float = 5):
        """Compute texture data to use in texture refinement

        Using the current parameters on the Rietveld object, fit the peaks
        to LeBail models in order to determine their intensities, and then
        compute a dictionary of texture contributions. These texture
        contributions are automatically set on the `pfdata` of the texture
        models.

        A simulated spectrum is returned.
        """
        # Use only the first lambda key, since it doesn't matter for LeBail
        lambda_key = next(iter(self.wavelength))
        mats = [v[lambda_key] for v in self.phases.phase_dict.values()]

        # Create LeBail params to use for intensity-finding. When possible,
        # we'll use the same params currently on the Rietveld object.
        bkg_method = {'chebyshev': 1}
        params = wppfsupport._generate_default_parameters_LeBail(
            mats, 1, bkg_method)

        for p in params:
            params[p].value = self.params[p].value

        # Ensure these are marked as `Vary`
        params['U'].vary = True
        params['V'].vary = True
        params['W'].vary = True

        # Allow lattice constants to vary as well
        for mat in mats:
            lp_names = [f"{mat.name}_{x}" for x in wppfsupport._lpname]
            for name in lp_names:
                if name in params:
                    params[name].vary = True

        # Compute the current intensities from Rietveld
        ints_computed = self.compute_intensities()

        # Extract the intensities from the data
        mask = np.isnan(pv_binned)
        results = extract_intensities(**{
            'polar_view': np.ma.masked_array(pv_binned, mask=mask),
            'tth_array': self.tth_list,
            'params': params,
            'phases': mats,
            'wavelength': self.wavelength,
            'bkgmethod': bkg_method,
            'intensity_init': ints_computed,
            'termination_condition': {
                "rwp_perct_change": 0.05,
                "max_iter": 10,
            },
            'peakshape': "pvtch",
        })

        # we have to divide by the computed instensites
        # to get the texture contribution
        for mat_key, model in self.texture_model.items():
            if model is None:
                continue

            pfdata = {}
            for ii in range(pv_binned.shape[0]):
                eta = np.radians(-100 + (ii + 1) * azimuthal_interval)
                t = results[2][ii][mat_key][lambda_key]
                hkl = results[1][ii][mat_key][lambda_key]
                ints = results[0][ii][mat_key][lambda_key]
                ints_comp = ints_computed[mat_key][lambda_key]

                nn = np.min((t.shape[0], ints_comp.shape[0]))
                for jj in range(nn):
                    angs = np.atleast_2d([np.radians(t[jj]), eta, 0])
                    v = angles_to_gvec(angs, beam_vec=bvec, eta_vec=evec)
                    data = np.hstack((
                        v,
                        np.atleast_2d(ints[jj] / ints_comp[jj]),
                    ))

                    hkey = tuple(hkl[jj, :])
                    if hkey in pfdata:
                        # Stack with previous data
                        data = np.vstack((pfdata[hkey], data))

                    pfdata[hkey] = data

            # Now set the texture data on the texture model
            model.pfdata = pfdata

        return results[4]


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
    func = partial(single_azimuthal_extraction, **kwargs)

    # We found that 3 max workers in a thread pool performed
    # the best on our example dataset. This is likely because
    # there is already parallelism going on in each thread.
    # ProcessPoolExecutor was always slower.
    max_workers = 3
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, data_inp_list))
    else:
        results = [func(x) for x in data_inp_list]

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
        # Make a deepcopy of params we pass to LeBail since it will modify them
        "params": copy.deepcopy(params),
        "phases": phases,
        "wavelength": wavelength,
        "bkgmethod": bkgmethod,
        "peakshape": peakshape,
        "intensity_init": intensity_init
    }
    L = LeBail(**kwargs)

    # get termination conditions for the LeBail refinement
    del_rwp = termination_condition["rwp_perct_change"]
    max_iter = termination_condition["max_iter"]

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


peakshape_dict = {
    "pvfcj": "pseudo-voight (finger, cox, jephcoat)",
    "pvtch": "pseudo-voight (thompson, cox, hastings)",
    "pvpink": "pseudo-voight (von dreele)",
}
