import importlib.resources
import numpy as np
import warnings
from hexrd.wppf.peakfunctions import \
calc_rwp, computespectrum_pvfcj, \
computespectrum_pvtch,\
computespectrum_pvpink,\
calc_Iobs_pvfcj,\
calc_Iobs_pvtch,\
calc_Iobs_pvpink
from hexrd.wppf import wppfsupport
from hexrd.imageutil import snip1d, snip1d_quad
from hexrd.material import Material
from hexrd.valunits import valWUnit

from hexrd import instrument
from hexrd import imageseries
from hexrd.imageseries import omega
from hexrd.xrdutil import PolarView
import time

class LeBailCalibrator:
    """
    ======================================================================
    ======================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, 
                    saransh1@llnl.gov
    >> @DATE:       08/24/2021 SS 1.0 original

    >> @DETAILS:    new lebail class which is specifically designed for 
    instrument calibration using the LeBail method. in this partcular
    implementation, instead of using the iterative method for computing the
    peak intensities, we will use them as parameters in the optimization
    problem. the inputs include the instrument, the imageseries, material, 
    peakshapes etc.

    >> @PARAMETERS  instrument the instrument class in hexrd
                    img_dict dictionary of images with same keys as 
                    detectors in the instrument class

    ======================================================================
    ======================================================================
    """
    def __init__(self,
                 instrument,
                 img_dict,
                 extent=(0.,90.,0.,360.),
                 pixel_size=(0.1, 1.0),
                 params=None,
                 phases=None,
                 azimuthal_step=5.0,
                 bkgmethod={'spline': None},
                 peakshape="pvtch",
                 intensity_init=None,
                 apply_solid_angle_correction=False,
                 apply_lp_correction=False,
                 polarization=None):

        self.bkgmethod = bkgmethod
        self.peakshape = peakshape
        self.extent = extent
        self.pixel_size = pixel_size
        self.azimuthal_step = azimuthal_step

        self.img_dict = img_dict

        self._tstart = time.time()

        self.sacorrection = apply_solid_angle_correction
        self.lpcorrection = apply_lp_correction
        self.polarization = polarization

        self.instrument = instrument

        self.phases = phases
        self.params = params

        self.intensity_init = intensity_init

        # self.initialize_Icalc()

        # self.computespectrum()

        self._tstop = time.time()
        
        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

    def __str__(self):
        resstr = '<LeBail Fit class>\nParameters of \
        the model are as follows:\n'
        resstr += self.params.__str__()
        return resstr

    def calctth(self):
        self.tth = {}
        self.hkls = {}
        self.dsp = {}
        for p in self.phases:
            self.tth[p] = {}
            self.hkls[p] = {}
            self.dsp[p] = {}
            for k, l in self.phases.wavelength.items():
                t = self.phases[p].getTTh(l[0].value)
                allowed = self.phases[p].wavelength_allowed_hkls
                t = t[allowed]
                hkl = self.phases[p].hkls[allowed, :]
                dsp = self.phases[p].dsp[allowed]
                tth_min = min(self.tth_min)
                tth_max = max(self.tth_max)
                limit = np.logical_and(t >= tth_min,
                                       t <= tth_max)
                self.tth[p][k] = t[limit]
                self.hkls[p][k] = hkl[limit, :]
                self.dsp[p][k] = dsp[limit]

    def initialize_Icalc(self):
        """
        @DATE 01/22/2021 SS modified the function so Icalc can be initialized with
        a dictionary of structure factors
        """

        self.Icalc = {}

        if(self.intensity_init is None):
            if self.spectrum_expt._y.max() > 0:
                n10 = np.floor(np.log10(self.spectrum_expt._y.max())) - 1
            else:
                n10 = 0

            for p in self.phases:
                self.Icalc[p] = {}
                for k, l in self.phases.wavelength.items():

                    self.Icalc[p][k] = (10**n10) * \
                        np.ones(self.tth[p][k].shape)

        elif(isinstance(self.intensity_init, dict)):
            """
                first check if intensities for all phases are present in the
                passed dictionary
            """
            for p in self.phases:
                if p not in self.intensity_init:
                    raise RuntimeError("LeBail: Intensity was initialized\
                     using custom values. However, initial values for one \
                     or more phases seem to be missing from the dictionary.")
                self.Icalc[p] = {}

                """
                now check that the size of the initial intensities provided is consistent
                with the number of reflections (size of initial intensity > size of hkl is allowed.
                the unused values are ignored.)

                for this we need to step through the different wavelengths in the spectrum and check
                each of them
                """
                for l in self.phases.wavelength:
                    if l not in self.intensity_init[p]:
                        raise RuntimeError("LeBail: Intensity was initialized\
                         using custom values. However, initial values for one \
                         or more wavelengths in spectrum seem to be missing \
                         from the dictionary.")

                    if(self.tth[p][l].shape[0] <=
                       self.intensity_init[p][l].shape[0]):
                        self.Icalc[p][l] = \
                            self.intensity_init[p][l][0:self.tth[p]
                                                      [l].shape[0]]
        else:
            raise RuntimeError(
                "LeBail: Intensity_init must be either\
                 None or a dictionary")

    def computespectrum(self):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    compute the simulated spectrum
        """
        x = self.tth_list
        y = np.zeros(x.shape)
        tth_list = np.ascontiguousarray(self.tth_list)

        for iph, p in enumerate(self.phases):

            for k, l in self.phases.wavelength.items():

                Ic = self.Icalc[p][k]

                shft_c = np.cos(0.5*np.radians(self.tth[p][k]))*self.shft
                trns_c = np.sin(np.radians(self.tth[p][k]))*self.trns
                tth = self.tth[p][k] + \
                      self.zero_error + \
                      shft_c + \
                      trns_c

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                n = np.min((tth.shape[0], Ic.shape[0]))
                shkl = self.phases[p].shkl
                name = self.phases[p].name
                eta_n = f"self.{name}_eta_fwhm"
                eta_fwhm = eval(eta_n)
                strain_direction_dot_product = 0.
                is_in_sublattice = False

                if self.peakshape == 0:
                    args = (np.array([self.U, self.V, self.W]),
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
                            Ic, self.xn, self.wn)

                elif self.peakshape == 1:
                    args = (np.array([self.U, self.V, self.W]),
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
                            Ic)

                elif self.peakshape == 2:
                    args = (np.array([self.alpha0, self.alpha1]),
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
                            Ic)

                y += self.computespectrum_fcn(*args)

        self._spectrum_sim = Spectrum(x=x, y=y)

        P = calc_num_variables(self.params)

        errvec, self.Rwp, self.gofF = calc_rwp(
            self.spectrum_sim.data_array,
            self.spectrum_expt.data_array,
            self.weights.data_array,
            P)
        return errvec

    def prepare_polarview(self):
        self.masked = self.pv.warp_image(self.img_dict, \
                                        pad_with_nans=True, \
                                        do_interpolation=True)

        self.fulllineout = self.masked.sum(axis=0) / np.sum(~self.masked.mask, axis=0)
        self.prepare_lineouts()

    def prepare_lineouts(self):
        self.lineouts = []
        if hasattr(self, 'masked'):
           azch = self.azimuthal_chunks
           tth = self.tth_list
           for ii in range(azch.shape[0]-1):
                istr = azch[ii]
                istp = azch[ii+1]
                lo = self.masked[istr:istp,:].sum(axis=0) / \
                np.sum(~self.masked[istr:istp,:].mask, axis=0)
                data = np.ma.vstack((tth,lo)).T
                self.lineouts.append(data)


    def calcrwp(self, params):
        """
        this is one of the main functions which differs fundamentally
        to the regular Lebail class. this function computes the residual
        as a colletion of residuals at different eta, omega values.
        for the most traditional HED case, we will have only a single value 
        of omega. However, there will be support for the more complicated
        HEDM case in the future.
        """
        self.set_params_vals_to_class(params)

        errvec = self.computespectrum()
        return errvec

    @property
    def instrument(self):
        return self._instrument
    

    @instrument.setter
    def instrument(self, ins):
        if isinstance(ins, instrument.HEDMInstrument):
            self._instrument = ins
            self.pv = PolarView(self.extent[0:2],
                                ins, 
                                eta_min=self.extent[2], 
                                eta_max=self.extent[3], 
                                pixel_size=self.pixel_size)

            self.prepare_polarview()
            # self.computespectrum()
        else:
            msg = "input is not an instrument class."
            raise RuntimeError(msg)

    # @property
    # def omegaimageseries(self):
    #     return self._omegaims
    
    # @omegaimageseries.setter
    # def omegaimageseries(self, oims):
    #     if isinstance(oims, omega.OmegaImageSeries):
    #         self._omegaims
    #     else:
    #         msg = "input is not omega image series."
    #         raise RuntimeError(msg)

    @property
    def init_time(self):
        return self._tstop - self._tstart

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, ext):
        self._extent = ext

        if hasattr(self, "instrument"):
            if hasattr(self, "pixel_size"):
                self.pv = PolarView(ext[0:2],
                                self.instrument, 
                                eta_min=ext[2], 
                                eta_max=ext[3], 
                                pixel_size=self.pixel_size)
                self.prepare_polarview()

    """
    this property returns a azimuthal range over which
    the summation is performed to get the lineouts
    """
    @property
    def azimuthal_chunks(self):
        extent = self.extent
        step = self.azimuthal_step
        azlim = extent[2:]
        pxsz = self.pixel_size[1]
        shp = self.masked.shape[0]
        npix = int(np.round(step/pxsz))
        return np.r_[np.arange(0,shp,npix),shp]

    @property
    def tth_list(self):
        extent = self.extent
        shp = self.masked.shape[1]
        tthlim = extent[0:2]
        return np.linspace(tthlim[0],tthlim[1],shp)
    

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, px_sz):
        self._pixel_size = px_sz

        if hasattr(self, "instrument"):
            if hasattr(self, "extent"):
                self.pv = PolarView(self.extent[0:2],
                            ins, 
                            eta_min=self.extent[2], 
                            eta_max=self.extent[3], 
                            pixel_size=px_sz)
                self.prepare_polarview()

    @property
    def sacorrection(self):
        return self._sacorrection

    @sacorrection.setter
    def sacorrection(self, val):
        if isinstance(val, bool):
            self._sacorrection = val
            if hasattr(self, 'instrument'):
                self.prepare_polarview()
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)

    @property
    def lpcorrection(self):
        return self._lpcorrection

    @lpcorrection.setter
    def lpcorrection(self, val):
        if isinstance(val, bool):
            self._lpcorrection = val
            if hasattr(self, 'instrument'):
                self.prepare_polarview()
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)


    @property
    def img_dict(self):

        imd = self._img_dict.copy()

        if self.sacorrection:
            for dname, det in self.instrument.detectors.items():
                solid_angs = det.pixel_solid_angles
                imd[dname] = imd[dname] / solid_angs

        if self.lpcorrection:
            hpol, vpol = self.polarization
            for dname, det in self.instrument.detectors.items():
                lp = det.lorentz_polarization_factor(hpol, vpol)
                imd[dname] = imd[dname] / lp

        return imd
    
    @img_dict.setter
    def img_dict(self, imd):
        self._img_dict = imd
        if hasattr(self, 'instrument'):
            self.prepare_polarview()

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, val):
        if val is None:
            self._polarization = (0.5, 0.5)
        else:
            self._polarization = val

    @property
    def azimuthal_step(self):
        return self._azimuthal_step
    
    @azimuthal_step.setter
    def azimuthal_step(self, val):
        self._azimuthal_step = val
        self.prepare_lineouts()