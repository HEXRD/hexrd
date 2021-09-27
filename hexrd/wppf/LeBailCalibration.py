import importlib.resources
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import lmfit
import warnings
from hexrd.wppf.peakfunctions import \
calc_rwp, computespectrum_pvfcj, \
computespectrum_pvtch,\
computespectrum_pvpink,\
calc_Iobs_pvfcj,\
calc_Iobs_pvtch,\
calc_Iobs_pvpink
from hexrd.wppf.spectrum import Spectrum
from hexrd.wppf import wppfsupport, LeBail
from hexrd.wppf.parameters import Parameters
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
                 bkgmethod={'chebyshev': 3},
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

        self.calc_simulated()

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
                    Icalc[p][k] = 100.0*np.ones(self.tth[p][k].shape)

            prefix = f"azpos{ii}"
            self.Icalc[prefix] = Icalc
            wppfsupport._add_intensity_parameters(self.params,self.hkls,Icalc,prefix)

        self.refine_intensities = False
        self.refine_background = False
        self.refine_instrument = False


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
                key = f"azpos{ii}"
                self.lineouts[key] = data


    def computespectrum(self,
                        instr_updated,
                        lp_updated):
        """
        this function calls the computespectrum function in the 
        lebaillight class for all the azimuthal positions and 
        accumulates the error vector from each of those lineouts.
        this is more or less a book keeping function rather
        """
        errvec = np.empty([0,])
        wss = 0.0
        den = 0.0
        for k,v in self.lineouts_sim.items():
            if instr_updated:
                v.lineout = self.lineouts[k]
            if lp_updated:
                v.tth = self.tth
                v.hkls = self.hkls
                v.dsp = self.dsp
            v.shkl = self.shkl
            v.params = self.params

            v.computespectrum()
            ww = v.weights
            mask = v.mask
            evec = ww*(v.spectrum_expt.data_array[:,1] -
                   v.spectrum_sim.data_array[:,1])**2
            evec = np.sqrt(evec)
            errvec = np.concatenate((errvec,evec))

            weighted_expt = ww*v.spectrum_expt.data_array[:,1]**2

            wss += np.trapz(evec, v.tth_list[~mask[:,1]])
            den += np.trapz(weighted_expt, v.tth_list[~mask[:,1]])

        rwp = np.sqrt(wss/den)

        return errvec, rwp

    def calcrwp(self, params):
        """
        this is one of the main functions which differs fundamentally
        to the regular Lebail class. this function computes the residual
        as a colletion of residuals at different eta, omega values.
        for the most traditional HED case, we will have only a single value 
        of omega. However, there will be support for the more complicated
        HEDM case in the future.
        """
        lp_updated = self.update_param_vals(params)
        self.update_shkl(params)
        instr_updated = self.update_instrument(params)
        errvec, rwp = self.computespectrum(instr_updated,
                                      lp_updated)

        print(rwp*100.)

        return errvec


    def initialize_lmfit_parameters(self):

        params = lmfit.Parameters()

        for p in self.params:
            par = self.params[p]
            if(par.vary):
                params.add(p, value=par.value, min=par.lb, max=par.ub)

        return params

    def Refine(self):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >> @DETAILS:    this routine performs the least squares refinement for all variables
                        which are allowed to be varied.
        """

        params = self.initialize_lmfit_parameters()

        fdict = {'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6,
         'verbose': 0, 'max_nfev': 1000, 'method':'trf',
         'jac':'2-point'}
        fitter = lmfit.Minimizer(self.calcrwp, params)

        res = fitter.least_squares(**fdict)
        return res

    def update_param_vals(self,
                          params):
        """
        @date 03/12/2021 SS 1.0 original
        take values in parameters and set the
        corresponding class values with the same
        name
        """
        for p in params:
            if params[p].vary:
                if p in self.params:
                    self.params[p].value = params[p].value
                    self.params[p].lb = params[p].min
                    self.params[p].ub = params[p].max


        updated_lp = False

        for p in self.phases:
            mat = self.phases[p]
            """
            PART 1: update the lattice parameters
            """
            lp = []
            lpvary = False
            pre = p + '_'

            name = [f"{pre}{x}" for x in wppfsupport._lpname]

            for nn in name:
                if nn in params:
                    if params[nn].vary:
                        lpvary = True
                    lp.append(params[nn].value)
                elif nn in self.params:
                    lp.append(self.params[nn].value)

            if(not lpvary):
                pass
            else:
                lp = self.phases[p].Required_lp(lp)
                mat.lparms = np.array(lp)
                mat._calcrmt()
                updated_lp = True

        if updated_lp:
            self.calctth()

        return updated_lp

    def update_shkl(self, params):
        """
        if certain shkls are refined, then update
        them using the params arg. else use values from
        the parameter class
        """
        updated_shkl = False
        shkl_dict = {}
        for p in self.phases:
            shkl_name = self.phases[p].valid_shkl
            eq_const = self.phases[p].eq_constraints
            mname = self.phases[p].name
            key = [f"{mname}_{s}" for s in shkl_name]
            for s,k in zip(shkl_name,key):
                if k in params:
                    shkl_dict[s] = params[k].value
                else:
                    shkl_dict[s] = self.params[k].value

            self.phases[p].shkl = wppfsupport._fill_shkl(\
                shkl_dict, eq_const)

    def update_instrument(self, params):
        instr_updated = False
        for key,det in self._instrument.detectors.items():
            for ii in range(3):
                pname = f"{key}_tvec{ii}"
                if pname in params:
                    if params[pname].vary:
                        det.tvec[ii] = params[pname].value
                        instr_updated = True
                pname = f"{key}_tilt{ii}"
                if pname in params:
                    if params[pname].vary:
                        det.tilt[ii] = params[pname].value
                        instr_updated = True

        if instr_updated:
            self.prepare_polarview()
        return instr_updated

    @property
    def bkgdegree(self):
        if "chebyshev" in self.bkgmethod.keys():
            return self.bkgmethod["chebyshev"]
    

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

    def striphkl(self, g):
        return str(g)[1:-1].replace(" ","") 
    
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
                        pname = [f"{prefix}{ii}_{p}_{k}_I{self.striphkl(g)}" 
                        for jj,g in zip(range(shape),self.hkls[p][k])]
                        for jj in range(shape):
                            self.params[pname[jj]].vary = val
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)

    @property
    def refine_background(self):
        return self._refine_background
    
    @refine_background.setter
    def refine_background(self, val):
        if isinstance(val, bool):
            self._refine_background = val
            prefix = "azpos"
            for ii in range(len(self.lineouts)):
                pname = [f"{prefix}{ii}_bkg_C{jj}" for jj in range(self.bkgdegree)]
                for p in pname:
                    self.params[p].vary = val
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)

    @property
    def refine_instrument(self):
        return self._refine_instrument
    
    @refine_instrument.setter
    def refine_instrument(self, val):
        if isinstance(val, bool):
            self._refine_instrument = val
            for key in self.instrument.detectors:
                pnametvec = [f"{key}_tvec{i}" for i in range(3)]
                pnametilt = [f"{key}_tilt{i}" for i in range(3)]
                for ptv,pti in zip(pnametvec,pnametilt):
                    self.params[ptv].vary = val
                    self.params[pti].vary = val
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
            wppfsupport._add_background(params, self.lineouts, self.bkgdegree)
            self._params = params

    @property
    def shkl(self):
        shkl = {}
        for p in self.phases:
            shkl[p] = self.phases[p].shkl
        return shkl
    

    def calc_simulated(self):
        self.lineouts_sim = {}
        for key, lo in self.lineouts.items():
            self.lineouts_sim[key] = LeBaillight(key,
                                                 lo,
                                                 self.tth,
                                                 self.hkls,
                                                 self.dsp,
                                                 self.shkl,
                                                 self.lebail_param_list,
                                                 self.params,
                                                 self.peakshape,
                                                 self.bkgmethod)

class LeBaillight:
    """
    just a lightweight LeBail class which does only the
    simple computation of diffraction spectrum given the 
    parameters and intensity values
    """
    def __init__(self,
                name,
                lineout,
                tth,
                hkls,
                dsp,
                shkl,
                lebail_param_list,
                params,
                peakshape,
                bkgmethod):

        self.name = name
        self.lebail_param_list = lebail_param_list
        self.peakshape = peakshape
        self.params = params
        self.lineout = lineout
        self.shkl = shkl
        self.tth = tth
        self.hkls = hkls
        self.dsp = dsp
        self.bkgmethod = bkgmethod
        self.computespectrum()

    def computespectrum(self):
        
        x = self.tth_list
        y = np.zeros(x.shape)
        tth_list = np.ascontiguousarray(self.tth_list)

        for p in self.tth:
            for k in self.tth[p]:

                Ic = self.Icalc[p][k]

                shft_c = np.cos(0.5*np.radians(self.tth[p][k]))*self.params["shft"].value
                trns_c = np.sin(np.radians(self.tth[p][k]))*self.params["trns"].value
                tth = self.tth[p][k] + \
                      self.params["zero_error"].value + \
                      shft_c + \
                      trns_c

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                n = np.min((tth.shape[0], Ic.shape[0]))
                shkl = self.shkl[p]
                name = p
                eta_n = f"{name}_eta_fwhm"
                eta_fwhm = self.params[eta_n].value
                strain_direction_dot_product = 0.
                is_in_sublattice = False

                cag = np.array([self.params["U"].value,
                                self.params["V"].value,
                                self.params["W"].value])
                gaussschrerr = self.params["P"].value
                lorbroad = np.array([self.params["X"].value,
                                     self.params["Y"].value])
                anisbroad = np.array([self.params["Xe"].value,
                                      self.params["Ye"].value,
                                      self.params["Xs"].value])
                if self.peakshape == 0:
                    HL = self.params["HL"].value
                    SL = self.params["SL"].value
                    args = (cag,
                            gaussschrerr,
                            lorbroad,
                            anisbroad,
                            shkl,
                            eta_fwhm,
                            HL,
                            SL,
                            tth,
                            dsp,
                            hkls,
                            strain_direction_dot_product,
                            is_in_sublattice,
                            tth_list,
                            Ic, 
                            self.xn, 
                            self.wn)

                elif self.peakshape == 1:
                    args = (cag,
                            gaussschrerr,
                            lorbroad,
                            anisbroad,
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
                    alpha = np.array([self.params["alpha0"].value,
                                      self.params["alpha1"].value])
                    beta = np.array([self.params["beta0"].value,
                                     self.params["beta1"].value])
                    args = (alpha,
                            beta,
                            cag,
                            gaussschrerr,
                            lorbroad,
                            anisbroad,
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

    @property
    def weights(self):
        lo = self.lineout
        mask = self.mask[:,1]
        weights = 1./np.sqrt(lo.data[:,1])
        weights = weights[~mask]

        return weights
    

    @property
    def bkgdegree(self):
        if "chebyshev" in self.bkgmethod.keys():
            return self.bkgmethod["chebyshev"]

    @property
    def background(self):
        tth, I = self._spectrum_sim.data
        mask = self.mask[:,1]
    
        pname = [f"{self.name}_bkg_C{ii}" 
        for ii in range(self.bkgdegree)]

        coef = [self.params[p].value for p in pname]
        c = Chebyshev(coef,domain=[tth[0],tth[-1]])
        bkg = c(tth)
        bkg[mask] = np.nan
        return bkg

    @property
    def spectrum_sim(self):
        tth, I = self._spectrum_sim.data
        mask = self.mask[:,1]
        I[mask] = np.nan
        I += self.background

        return Spectrum(x=tth, y=I)
    
    @property
    def spectrum_expt(self):
        d = self.lineout.data
        mask = self.mask[:,1]
        d[mask,1] = np.nan
        return Spectrum(x=d[:,0], y=d[:,1])    

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        """
        set the local value of the parameters to the 
        global values from the calibrator class
        """
        from scipy.special import roots_legendre
        xn, wn = roots_legendre(16)
        self.xn = xn[8:]
        self.wn = wn[8:]
        if isinstance(params, Parameters):
            if hasattr(self, 'params'):
                for p in params:
                    if (p in self.lebail_param_list) or (self.name in p):
                        self._params[p].value = params[p].value
                        self._params[p].ub = params[p].ub
                        self._params[p].lb = params[p].lb
                        self._params[p].vary = params[p].vary
            else:
                self._params = Parameters()
                for p in params:
                    if (p in self.lebail_param_list) or (self.name in p):
                        self._params.add(name=p,
                                            value=params[p].value,
                                            ub=params[p].ub,
                                            lb=params[p].lb,
                                            vary=params[p].vary)

            # if hasattr(self, "tth") and \
            # hasattr(self, "dsp") and \
            # hasattr(self, "lineout"):
            #     self.computespectrum()
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

        # if hasattr(self, "tth"):
        #     self.computespectrum()

    @property
    def mask(self):
        return self.lineout.mask

    @property
    def tth_list(self):
        return self.lineout[:,0].data


    @property
    def tth(self):
        return self._tth
    

    @tth.setter
    def tth(self, val):
        if isinstance(val, dict):
            self._tth = val
            # if hasattr(self,"dsp"):
            #     self.computespectrum()
        else:
            msg = (f"two theta vallues need "
                   f"to be in a dictionary")
            raise ValueError(msg)

    @property
    def hkls(self):
        return self._hkls
    @hkls.setter
    def hkls(self, val):
        if isinstance(val, dict):
            self._hkls = val
            # if hasattr(self,"tth") and\
            # hasattr(self,"dsp") and\
            # hasattr(self,"lineout"):
            #     self.computespectrum()
        else:
            msg = (f"two theta vallues need "
                   f"to be in a dictionary")
            raise ValueError(msg)

    @property
    def dsp(self):
        return self._dsp
    
    @dsp.setter
    def dsp(self, val):
        if isinstance(val, dict):
            self._dsp = val
            # self.computespectrum()
        else:
            msg = (f"two theta vallues need "
                   f"to be in a dictionary")
            raise ValueError(msg)
            
    @property
    def Icalc(self):
        Ic = {}
        for p in self.tth:
            Ic[p] = {}
            for k in self.tth[p]:
                vname = f"{self.name}_{p}_{k}_I"
                I = []
                for param in self.params:
                    if vname in param:
                        I.append(self.params[param].value)
                Ic[p][k] = np.array(I)
    
        return Ic

    @property
    def computespectrum_fcn(self):
        if self.peakshape == 0:
            return computespectrum_pvfcj
        elif self.peakshape == 1:
            return computespectrum_pvtch
        elif self.peakshape == 2:
            return computespectrum_pvpink