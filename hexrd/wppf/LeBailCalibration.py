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
from hexrd.wppf import wppfsupport, LeBail, parameters
from hexrd.wppf.phase import Phases_LeBail, Material_LeBail
from hexrd.imageutil import snip1d, snip1d_quad
from hexrd.material import Material
from hexrd.valunits import valWUnit
from hexrd.constants import keVToAngstrom

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

        self.initialize_Icalc()

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
                tth_min = self.tth_min
                tth_max = self.tth_max
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

        for ii in range(len(self.lineouts)):

            Icalc = {}
            g = {}
            for p in self.phases:
                Icalc[p] = {}
                for k, l in self.phases.wavelength.items():
                    Icalc[p][k] = np.zeros(self.tth[p][k].shape)

            prefix = f"azpos{ii}"
            self.Icalc[prefix] = Icalc
            wppfsupport._add_intensity_parameters(self.params,self.hkls,Icalc,prefix)

        self.refine_intensities = False


    def prepare_polarview(self):
        self.masked = self.pv.warp_image(self.img_dict, \
                                        pad_with_nans=True, \
                                        do_interpolation=True)

        self.fulllineout = self.masked.sum(axis=0) / np.sum(~self.masked.mask, axis=0)
        self.prepare_lineouts()

    def prepare_lineouts(self):
        self.lineouts = {}
        if hasattr(self, 'masked'):
           azch = self.azimuthal_chunks
           tth = self.tth_list
           for ii in range(azch.shape[0]-1):
                istr = azch[ii]
                istp = azch[ii+1]
                lo = self.masked[istr:istp,:].sum(axis=0) / \
                np.sum(~self.masked[istr:istp,:].mask, axis=0)
                data = np.ma.vstack((tth,lo)).T
                key = f"azpos_{ii}"
                self.lineouts[key] = data


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

        # return errvec

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
    def wavelength(self):
        lam = keVToAngstrom(self.instrument.beam_energy)
        return {"lam1": 
                [valWUnit('lp', 'length', lam, 'angstrom'),1.0]}
    
    @property
    def refine_intensities(self):
        return self._refine_intensities

    @refine_intensities.setter
    def refine_intensities(self, val):
        if isinstance(val, bool):
            self._refine_intensities = val
            prefix = "azpos"
            for ii,(azpos,Icalc) in enumerate(self.Icalc.items()):
                for p in Icalc:
                    for k in Icalc[p]:
                        shape = Icalc[p][k].shape[0]
                        pname = [f"{prefix}{ii}_{p}_{k}_I{g}" 
                        for jj,g in zip(range(shape),self.hkls[p][k])]
                        for jj in range(shape):
                            self.params[pname[jj]].vary = val
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)
    
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

    @property
    def tth_min(self):
        return self.extent[0]
    
    @property
    def tth_max(self):
        return self.extent[1]
    
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
                msg = (f"invalid peak shape string. "
                    f"must be: \n"
                    f"1. pvfcj: pseudo voight (Finger, Cox, Jephcoat)\n"
                    f"2. pvtch: pseudo voight (Thompson, Cox, Hastings)\n"
                    f"3. pvpink: Pink beam (Von Dreele)")
                raise ValueError(msg)
        elif isinstance(val, int):
            if val >=0 and val <=2:
                self._peakshape = val
            else:
                msg = (f"invalid peak shape int. "
                    f"must be: \n"
                    f"1. 0: pseudo voight (Finger, Cox, Jephcoat)\n"
                    f"2. 1: pseudo voight (Thompson, Cox, Hastings)\n"
                    f"3. 2: Pink beam (Von Dreele)")
                raise ValueError(msg)

        """
        update parameters
        """
        if hasattr(self, 'params'):
            params = wppfsupport._generate_default_parameters_Rietveld(
                    self.phases, self.peakshape)
            for p in params:
                if p in self.params:
                    params[p] = self.params[p]
            self._params = params


    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, phase_info):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
                        09/11/2020 SS 1.1 multiple different ways to initialize phases
                        09/14/2020 SS 1.2 added phase initialization from material.Material class
                        03/05/2021 SS 2.0 moved everything to property setter
        >> @DETAILS:    load the phases for the LeBail fits
        """

        if(phase_info is not None):
            if(isinstance(phase_info, Phases_LeBail)):
                """
                directly passing the phase class
                """
                self._phases = phase_info

            else:

                if(hasattr(self, 'wavelength')):
                    if(self.wavelength is not None):
                        p = Phases_LeBail(wavelength=self.wavelength)
                else:
                    p = Phases_LeBail()

                if(isinstance(phase_info, dict)):
                    """
                    initialize class using a dictionary with key as
                    material file and values as the name of each phase
                    """
                    for material_file in phase_info:
                        material_names = phase_info[material_file]
                        if(not isinstance(material_names, list)):
                            material_names = [material_names]
                        p.add_many(material_file, material_names)

                elif(isinstance(phase_info, str)):
                    """
                    load from a yaml file
                    """
                    if(path.exists(phase_info)):
                        p.load(phase_info)
                    else:
                        raise FileError('phase file doesn\'t exist.')

                elif(isinstance(phase_info, Material)):
                    p[phase_info.name] = Material_LeBail(
                        fhdf=None,
                        xtal=None,
                        dmin=None,
                        material_obj=phase_info)

                elif(isinstance(phase_info, list)):
                    for mat in phase_info:
                        p[mat.name] = Material_LeBail(
                            fhdf=None,
                            xtal=None,
                            dmin=None,
                            material_obj=mat)

                        p.num_phases += 1

                    for mat in p:
                        p[mat].pf = 1.0/p.num_phases

                self._phases = p

        self.calctth()

        for p in self.phases:
            self.phases[p].valid_shkl, \
            self.phases[p].eq_constraints, \
            self.phases[p].rqd_index, \
            self.phases[p].trig_ptype = \
            wppfsupport._required_shkl_names(self.phases[p])

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param_info):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
                        09/11/2020 SS 1.1 modified to accept multiple input types
                        03/05/2021 SS 2.0 moved everything to the property setter
        >> @DETAILS:    initialize parameter list from file. if no file given, then initialize
                        to some default values (lattice constants are for CeO2)
        """
        from scipy.special import roots_legendre
        xn, wn = roots_legendre(16)
        self.xn = xn[8:]
        self.wn = wn[8:]

        if(param_info is not None):
            if(isinstance(param_info, Parameters)):
                """
                directly passing the parameter class
                """
                self._params = param_info
                params = param_info

            else:
                params = Parameters()

                if(isinstance(param_info, dict)):
                    """
                    initialize class using dictionary read from the yaml file
                    """
                    for k, v in param_info.items():
                        params.add(k, value=np.float(v[0]),
                                   lb=np.float(v[1]), ub=np.float(v[2]),
                                   vary=np.bool(v[3]))

                elif(isinstance(param_info, str)):
                    """
                    load from a yaml file
                    """
                    if(path.exists(param_info)):
                        params.load(param_info)
                    else:
                        raise FileError('input spectrum file doesn\'t exist.')

                """
                this part initializes the lattice parameters in the
                """
                for p in self.phases:
                    wppfsupport._add_lp_to_params(
                        params, self.phases[p])

                self._params = params
        else:

            params = wppfsupport._generate_default_parameters_LeBail(
                self.phases, self.peakshape)
            self.lebail_param_list = [p for p in params]
            wppfsupport._add_detector_geometry(params, self.instrument)
            self._params = params

class LeBaillight:
    """
    just a lightweight LeBail class which does only the
    simple computation of diffraction spectrum given the 
    parameters and intensity values
    """
    def __init__(self,
                name,
                lineout,
                lebail_param_list,
                params):

        self.name = name
        self.lebail_param_list = lebail_param_list
        self.params = params
        self.lineout = lineout

    def computespectrum(self):
        pass
        # x = self.tth_list
        # y = np.zeros(x.shape)
        # tth_list = np.ascontiguousarray(self.tth_list)

        # for iph, p in enumerate(self.phases):

        #     for k, l in self.phases.wavelength.items():

        #         Ic = self.Icalc[p][k]

        #         shft_c = np.cos(0.5*np.radians(self.tth[p][k]))*self.shft
        #         trns_c = np.sin(np.radians(self.tth[p][k]))*self.trns
        #         tth = self.tth[p][k] + \
        #               self.zero_error + \
        #               shft_c + \
        #               trns_c

        #         dsp = self.dsp[p][k]
        #         hkls = self.hkls[p][k]
        #         n = np.min((tth.shape[0], Ic.shape[0]))
        #         shkl = self.phases[p].shkl
        #         name = self.phases[p].name
        #         eta_n = f"self.{name}_eta_fwhm"
        #         eta_fwhm = eval(eta_n)
        #         strain_direction_dot_product = 0.
        #         is_in_sublattice = False

        #         if self.peakshape == 0:
        #             args = (np.array([self.U, self.V, self.W]),
        #                     self.P,
        #                     np.array([self.X, self.Y]),
        #                     np.array([self.Xe, self.Ye, self.Xs]),
        #                     shkl,
        #                     eta_fwhm,
        #                     self.HL,
        #                     self.SL,
        #                     tth,
        #                     dsp,
        #                     hkls,
        #                     strain_direction_dot_product,
        #                     is_in_sublattice,
        #                     tth_list,
        #                     Ic, self.xn, self.wn)

        #         elif self.peakshape == 1:
        #             args = (np.array([self.U, self.V, self.W]),
        #                     self.P,
        #                     np.array([self.X, self.Y]),
        #                     np.array([self.Xe, self.Ye, self.Xs]),
        #                     shkl,
        #                     eta_fwhm,
        #                     tth,
        #                     dsp,
        #                     hkls,
        #                     strain_direction_dot_product,
        #                     is_in_sublattice,
        #                     tth_list,
        #                     Ic)

        #         elif self.peakshape == 2:
        #             args = (np.array([self.alpha0, self.alpha1]),
        #                     np.array([self.beta0, self.beta1]),
        #                     np.array([self.U, self.V, self.W]),
        #                     self.P,
        #                     np.array([self.X, self.Y]),
        #                     np.array([self.Xe, self.Ye, self.Xs]),
        #                     shkl,
        #                     eta_fwhm,
        #                     tth,
        #                     dsp,
        #                     hkls,
        #                     strain_direction_dot_product,
        #                     is_in_sublattice,
        #                     tth_list,
        #                     Ic)

        #         y += self.computespectrum_fcn(*args)

        # self._spectrum_sim = Spectrum(x=x, y=y)


    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        """
        set the local value of the parameters to the 
        global values from the calibrator class
        """
        if isinstance(params, parameters.Parameters):
            if hasattr(self, 'params'):
                for p in params:
                    if (p in self.lebail_param_list) or (self.name in p):
                        self._params[p].value = params[p].value
                        self._params[p].ub = params[p].ub
                        self._params[p].lb = params[p].lb
                        self._params[p].vary = params[p].vary
            else:
                self._params = parameters.Parameters()
                for p in params:
                    if (p in self.lebail_param_list) or (self.name in p):
                        self._params.add(name=p,
                                            value=params[p].value,
                                            ub=params[p].ub,
                                            lb=params[p].lb,
                                            vary=params[p].vary)

            self.computespectrum()
        else:
            msg = "only Parameters class permitted"
            raise ValueError(msg)

    @property
    def lineout(self):
        return self._lineout
    
    @lineout.setter
    def lineout(self,lo):
        if isinstance(lo,np.ma.MaskedArray):
            self._lineout = lo
        else:
            msg = f"only masked arrays input is allowed."
            raise ValueError(msg)

    @property
    def Icalc(self):
        Ic = []
        for p in self.params:
            if self.name in p:
                Ic.append(self.params[p].value)
        return np.array(Ic)
    
