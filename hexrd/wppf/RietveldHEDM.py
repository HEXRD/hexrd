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
from hexrd.wppf.texture import harmonic_model,\
pole_figures, inverse_pole_figures
from hexrd import instrument
from hexrd import imageseries
from hexrd.imageseries import omega

class RietveldHEDM:
    """
    ======================================================================
    ======================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       08/24/2021 SS 1.0 original

    >> @DETAILS:    new rietveld class which is specifically designed to
    handle the HEDM style datasets. the input to this class will be
    the imageseries and the instrument class, along with the usual material,
    peakshape, background method etc. this class will also provide the
    flexibility to add the instrument parameters as refinable parameters
    and allow full 2-d fitting using the rietveld forward model. however,
    the most important test case is going to the texture inversion using the
    generalized axis distribution function for powder like HEDM datasets.
    ======================================================================
    ======================================================================
    """
    def __init__(self,
                 instrument,
                 omegaimageseries,
                 params=None,
                 phases=None,
                 bkgmethod={'chebyshev': 3},
                 peakshape='pvfcj',
                 shape_factor=1.,
                 particle_size=1.,
                 phi=0.,
                 texture_model=None):

        self.bkgmethod = bkgmethod
        self.shape_factor = shape_factor
        self.particle_size = particle_size
        self.phi = phi
        self.peakshape = peakshape
        self.texture_model = texture_model

        self._tstart = time.time()

        self.instrument = instrument
        self.omegaimageseries = omegaimageseries

        self.phases = phases
        self.params = params

        # self.PolarizationFactor()
        # self.computespectrum()

        self._tstop = time.time()

        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

    def remove_texture_parameters(self):
        if hasattr(self.texture_model):
            if self.texture_model_name == "harmonic":
                names = list(self.texture_model.coeff_loc.keys())
                if "phon" in self.params:
                    del self.params["phon"]
                for n in names:
                    if n in self.params:
                        del self.params[n]
        else:
            pass

    def add_texture_parameters(self):
        if self.texture_model_name == "harmonic":
            names = list(self.texture_model.coeff_loc.keys())
            if not "phon" in self.params:
                self.params.add("phon", value=0.0, min=0.0)
            else:
                self.params[n].value = 0.0
                self.params[n].vary = True
                self.params[n].min = 0.0
                self.params[n].max = np.inf
            for n in names:
                if not n in self.params:
                    self.params.add(n, value=0.0)
                else:
                    self.params[n].value = 0.0
                    self.params[n].vary = True
                    self.params[n].min = -np.inf
                    self.params[n].max = np.inf

    def init_texturemodel(self,
                     eta_data=(-180.,180.,10.),
                     omega_data=(-180.,180.,10.),
                     sample_symmetry="cylindrical",
                     max_degree=10
                     ):
    """
    this function initializes the texture model for the Rietveld
    class. the user needs to specify the start and stop ranges
    as well as the step sizes for eta and omega angles for the
    HEDM datasets. The range for tth is already a part of the class.
    this is likely going to be a fairly complicated function with
    a lot of moving parts. one texture model will be initialized for
    each member of the phase class. the coefficients of the texture
    model will not be a part of the main parameter class, since there
    are a lot of these coefficients. instead we will use the parameter
    class initialized inside the texture model.
    """

    pass
    # self.texturemodel = {}
    # eta = np.arange(eta_data[0],
    #                 eta_data[1]+1.,
    #                 eta_data[2])
    # omega = np.arange(omega_data[0],
    #                 omega_data[1]+1.,
    #                 omega_data[2])

    # eta = np.radians(eta)
    # omega = np.radians(omega)
    # ones = np.ones(eta.shape)

    # for p in self.phases:
    #     mat = self.phases[p]
    #     for k in self.phases.wavelength.keys()[0]:
    #         tth = self.tth[p][k]
    #         hkls = self.hkls[p][k]
    #         pfdata = {}
    #         for t,g in zip(tth,hkls):
    #             key = str(g)[1:-1].replace(" ","")
    #             tthtile = np.tile(t,[eta.shape[0],])
    #             data = None
    #             for o in omega:
    #                 omgtile = np.tile(o,[eta.shape[0],])
    #                 tmp = np.vstack((tthtile,eta,omgtile,ones)).T
    #                 if data is None:
    #                     data = tmp
    #                 else:
    #                     data = np.vstack((data,tmp))
    #             pfdata[key] = data

    #         pf = pole_figures(mat,
    #                           hkls,
    #                           pfdata)
    #         self.texturemodel[p] = harmonic_model(pf,
    #                                 sample_symmetry,
    #                                 max_degree)

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, ins):
        if isinstance(ins, instrument):
            self._instrument = ins
            self.PolarizationFactor()
            self.computespectrum()
        else:
            msg = "input is not an instrument class."
            raise RuntimeError(msg)

    @property
    def omegaimageseries(self):
        return self._omegaims

    @omegaimageseries.setter
    def omegaimageseries(self, oims):
        if isinstance(oims, omega.OmegaImageSeries):
            self._omegaims
        else:
            msg = "input is not omega image series."
            raise RuntimeError(msg)

    @property
    def init_time(self):
        return self._tstop - self._tstart

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
        >> @DETAILS:    load the phases for the Rietveld fits
        """

        if(phase_info is not None):
            if(isinstance(phase_info, Phases_Rietveld)):
                """
                directly passing the phase class
                """
                self._phases = phase_info

            else:

                if(hasattr(self, 'wavelength')):
                    if(self.wavelength is not None):
                        p = Phases_Rietveld(wavelength=self.wavelength)
                else:
                    p = Phases_Rietveld()

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
                    p[phase_info.name] = Material_Rietveld(
                        fhdf=None,
                        xtal=None,
                        dmin=None,
                        material_obj=phase_info)

                elif(isinstance(phase_info, list)):
                    for mat in phase_info:
                        p[mat.name] = Material_Rietveld(
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
            pl = wppfsupport._generate_default_parameters_Rietveld(
                self.phases, self.peakshape, ptype="lmfit")
            self.rietveld_param_list = [p for p in pl]
            if(isinstance(param_info, Parameters_lmfit)):
                """
                directly passing the parameter class
                """
                self._params = param_info
                params = param_info
            else:
                params = Parameters_lmfit()

                if(isinstance(param_info, dict)):
                    """
                    initialize class using dictionary read from the yaml file
                    """
                    for k in param_info:
                        v = param_info[k]
                        params.add(k, value=np.float(v[0]),
                                   min=np.float(v[1]), max=np.float(v[2]),
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

            params = wppfsupport._generate_default_parameters_Rietveld(
                self.phases, self.peakshape, ptype="lmfit")
            self.rietveld_param_list = [p for p in params]
            wppfsupport._add_detector_geometry(params, self.instrument)
            wppfsupport._add_background(params, self.lineouts, self.bkgdegree)
            self._params = params

    @property
    def wavelength(self):
        lam = keVToAngstrom(self.instrument.beam_energy)
        return {"lam1":
                [valWUnit('lp', 'length', lam, 'angstrom'),1.0]}

    @property
    def tth_list(self):
        extent = self.extent
        shp = self.masked.shape[1]
        tthlim = extent[0:2]
        return np.linspace(self.tth_min,
                           self.tth_max,
                           shp)

    @property
    def shkl(self):
        shkl = {}
        for p in self.phases:
            shkl[p] = self.phases[p].shkl
        return shkl

    @property
    def texture_model(self):
        return self._texture_model

    @texture_model.setter
    def texture_model(self, model):
        if isinstance(model, str):

            if model == "harmonic":
                self.init_texture_model_harmonic()
            else:
                msg = f"unknown texture model"
        elif model is None:
            """
            set model to None and remove extra
            parameters from the Rietveld parameter
            list
            """
            self._texture_model = None
            self.remove_texture_parameters()

    @property
    def bkgdegree(self):
        if "chebyshev" in self.bkgmethod.keys():
            return self.bkgmethod["chebyshev"]

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