import importlib.resources
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import lmfit
import warnings
from hexrd.powder.wppf.peakfunctions import (
    calc_rwp,
    computespectrum_pvfcj,
    computespectrum_pvtch,
    computespectrum_pvpink,
    calc_Iobs_pvfcj,
    calc_Iobs_pvtch,
    calc_Iobs_pvpink,
)
from hexrd.powder.wppf.spectrum import Spectrum
from hexrd.powder.wppf import wppfsupport, LeBail
from hexrd.powder.wppf.parameters import Parameters
from lmfit import Parameters as Parameters_lmfit
from hexrd.powder.wppf.phase import Phases_LeBail, Material_LeBail
from hexrd.core.imageutil import snip1d, snip1d_quad
from hexrd.core.material import Material
from hexrd.core.valunits import valWUnit
from hexrd.core.constants import keVToAngstrom

from hexrd.core import instrument
from hexrd.core import imageseries
from hexrd.core.imageseries import omega
from hexrd.core.projections.polar import PolarView
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

    def __init__(
        self,
        instrument,
        img_dict,
        extent=(0.0, 90.0, 0.0, 360.0),
        pixel_size=(0.1, 1.0),
        params=None,
        phases=None,
        azimuthal_step=5.0,
        bkgmethod={'chebyshev': 3},
        peakshape="pvtch",
        intensity_init=None,
        apply_solid_angle_correction=False,
        apply_lp_correction=False,
        polarization=None,
    ):

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

        self.nfev = 0
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
                limit = np.logical_and(t >= tth_min, t <= tth_max)
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
            prefix = f"azpos{ii}"
            lo = self.lineouts[prefix].data[:, 1]
            if self.intensity_init is None:
                if np.nanmax(lo) > 0:
                    n10 = np.floor(np.log10(np.nanmax(lo))) - 2
                else:
                    n10 = 0

            for p in self.phases:
                Icalc[p] = {}
                for k, l in self.phases.wavelength.items():
                    Icalc[p][k] = (10**n10) * np.ones(self.tth[p][k].shape)

            self.Icalc[prefix] = Icalc

        self.refine_background = False
        self.refine_instrument = False

    def prepare_polarview(self):
        self.masked = self.pv.warp_image(
            self.img_dict, pad_with_nans=True, do_interpolation=True
        )
        lo = self.masked.sum(axis=0) / np.sum(~self.masked.mask, axis=0)
        self.fulllineout = np.vstack((self.tth_list, lo)).T
        self.prepare_lineouts()

    def prepare_lineouts(self):
        self.lineouts = {}
        if hasattr(self, 'masked'):
            azch = self.azimuthal_chunks
            tth = self.tth_list
            for ii in range(azch.shape[0] - 1):
                istr = azch[ii]
                istp = azch[ii + 1]
                lo = self.masked[istr:istp, :].sum(axis=0) / np.sum(
                    ~self.masked[istr:istp, :].mask, axis=0
                )
                data = np.ma.vstack((tth, lo)).T
                key = f"azpos{ii}"
                self.lineouts[key] = data

    def computespectrum(self, instr_updated, lp_updated):
        """
        this function calls the computespectrum function in the
        lebaillight class for all the azimuthal positions and
        accumulates the error vector from each of those lineouts.
        this is more or less a book keeping function rather
        """
        errvec = np.empty([0])
        rwp = []

        for k, v in self.lineouts_sim.items():
            v.params = self.params
            if instr_updated:
                v.lineout = self.lineouts[k]
            if lp_updated:
                v.tth = self.tth
                v.hkls = self.hkls
                v.dsp = self.dsp
            v.shkl = self.shkl
            v.CalcIobs()
            v.computespectrum()

            ww = v.weights
            evec = ww * (v.spectrum_expt._y - v.spectrum_sim._y) ** 2
            evec = np.sqrt(evec)
            evec = np.nan_to_num(evec)
            errvec = np.concatenate((errvec, evec))

            weighted_expt = np.nan_to_num(ww * v.spectrum_expt._y**2)

            wss = np.trapz(evec, v.tth_list)
            den = np.trapz(weighted_expt, v.tth_list)
            r = np.sqrt(wss / den) * 100.0
            if ~np.isnan(r):
                rwp.append(r)

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
        errvec, rwp = self.computespectrum(instr_updated, lp_updated)
        self.Rwp = np.mean(rwp)
        self.nfev += 1
        self.Rwplist = np.append(self.Rwplist, self.Rwp)

        if np.mod(self.nfev, 10) == 0:
            msg = (
                f"refinement ongoing... \n weighted residual at "
                f"iteration # {self.nfev} = {self.Rwp}\n"
            )
            print(msg)

        return errvec

    def initialize_lmfit_parameters(self):

        params = self.params
        params = lmfit.Parameters()

        for p in self.params:
            par = self.params[p]
            if par.vary:
                params.add(
                    p, value=par.value, min=par.min, max=par.max, vary=True
                )
        return params

    def Refine(self):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       05/19/2020 SS 1.0 original
        >> @DETAILS:    this routine performs the least squares refinement for all variables
                        which are allowed to be varied.
        """

        params = self.initialize_lmfit_parameters()

        # fdict = {'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6, 'factor':1000}
        fitter = lmfit.Minimizer(self.calcrwp, params)

        fdict = {'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6}
        res = fitter.least_squares(**fdict)
        # res = fitter.leastsq(**fdict)
        self.res = res

        if self.res.success:
            msg = (
                f"\n \n optimization successful: {self.res.message}. \n"
                f"weighted residual error = {self.Rwp}"
            )
        else:
            msg = (
                f"\n \n optimization unsuccessful: {self.res.message}. \n"
                f"weighted residual error = {self.Rwp}"
            )

        print(msg)

    def update_param_vals(self, params):
        """
        @date 03/12/2021 SS 1.0 original
        take values in parameters and set the
        corresponding class values with the same
        name
        """
        for p in params:
            self._params[p].value = params[p].value
            self._params[p].min = params[p].min
            self._params[p].max = params[p].max

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

            if not lpvary:
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
            for s, k in zip(shkl_name, key):
                if k in params:
                    shkl_dict[s] = params[k].value
                else:
                    shkl_dict[s] = self.params[k].value

            self.phases[p].shkl = wppfsupport._fill_shkl(shkl_dict, eq_const)

    def update_instrument(self, params):
        instr_updated = False
        for key, det in self._instrument.detectors.items():
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
            self.pv = PolarView(
                self.extent[0:2],
                ins,
                eta_min=self.extent[2],
                eta_max=self.extent[3],
                pixel_size=self.pixel_size,
            )

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
                self.pv = PolarView(
                    ext[0:2],
                    self.instrument,
                    eta_min=ext[2],
                    eta_max=ext[3],
                    pixel_size=self.pixel_size,
                )
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
        npix = int(np.round(step / pxsz))
        return np.r_[np.arange(0, shp, npix), shp]

    @property
    def tth_list(self):
        return np.squeeze(np.degrees(self.pv.angular_grid[1][0, :]))

    @property
    def wavelength(self):
        lam = keVToAngstrom(self.instrument.beam_energy)
        return {"lam1": [valWUnit('lp', 'length', lam, 'angstrom'), 1.0]}

    def striphkl(self, g):
        return str(g)[1:-1].replace(" ", "")

    @property
    def refine_background(self):
        return self._refine_background

    @refine_background.setter
    def refine_background(self, val):
        if "chebyshev" in self.bkgmethod.keys():
            if isinstance(val, bool):
                self._refine_background = val
                prefix = "azpos"
                for ii in range(len(self.lineouts)):
                    pname = [
                        f"{prefix}{ii}_bkg_C{jj}"
                        for jj in range(self.bkgdegree)
                    ]
                    for p in pname:
                        self.params[p].vary = val
            else:
                msg = "only boolean values accepted"
                raise ValueError(msg)
        else:
            msg = f"background method doesn't support refinement."
            print(msg)

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
                for ptv, pti in zip(pnametvec, pnametilt):
                    self.params[ptv].vary = val
                    self.params[pti].vary = val
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)

    @property
    def refine_translation(self):
        return self._refine_tilt

    @refine_translation.setter
    def refine_translation(self, val):
        if isinstance(val, bool):
            self._refine_translation = val
            for key in self.instrument.detectors:
                pnametvec = [f"{key}_tvec{i}" for i in range(3)]
                for ptv in pnametvec:
                    self.params[ptv].vary = val
        else:
            msg = "only boolean values accepted"
            raise ValueError(msg)

    @property
    def refine_tilt(self):
        return self._refine_tilt

    @refine_tilt.setter
    def refine_tilt(self, val):
        if isinstance(val, bool):
            self._refine_tilt = val
            for key in self.instrument.detectors:
                pnametilt = [f"{key}_tilt{i}" for i in range(3)]
                for pti in pnametilt:
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
                self.pv = PolarView(
                    self.extent[0:2],
                    ins,
                    eta_min=self.extent[2],
                    eta_max=self.extent[3],
                    pixel_size=px_sz,
                )
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
                lp = det.polarization_factor(hpol, vpol) * det.lorentz_factor()
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
        return self.extent[0] + self.pixel_size[0] * 0.5

    @property
    def tth_max(self):
        return self.extent[1] - +self.pixel_size[0] * 0.5

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
                    f"invalid peak shape string. "
                    f"must be: \n"
                    f"1. pvfcj: pseudo voight (Finger, Cox, Jephcoat)\n"
                    f"2. pvtch: pseudo voight (Thompson, Cox, Hastings)\n"
                    f"3. pvpink: Pink beam (Von Dreele)"
                )
                raise ValueError(msg)
        elif isinstance(val, int):
            if val >= 0 and val <= 2:
                self._peakshape = val
            else:
                msg = (
                    f"invalid peak shape int. "
                    f"must be: \n"
                    f"1. 0: pseudo voight (Finger, Cox, Jephcoat)\n"
                    f"2. 1: pseudo voight (Thompson, Cox, Hastings)\n"
                    f"3. 2: Pink beam (Von Dreele)"
                )
                raise ValueError(msg)

        """
        update parameters
        """
        if hasattr(self, 'params'):
            params = wppfsupport._generate_default_parameters_Rietveld(
                self.phases, self.peakshape
            )
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

        if phase_info is not None:
            if isinstance(phase_info, Phases_LeBail):
                """
                directly passing the phase class
                """
                self._phases = phase_info

            else:

                if hasattr(self, 'wavelength'):
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
                        raise FileError('phase file doesn\'t exist.')

                elif isinstance(phase_info, Material):
                    p[phase_info.name] = Material_LeBail(
                        fhdf=None,
                        xtal=None,
                        dmin=None,
                        material_obj=phase_info,
                    )

                elif isinstance(phase_info, list):
                    for mat in phase_info:
                        p[mat.name] = Material_LeBail(
                            fhdf=None, xtal=None, dmin=None, material_obj=mat
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

        if param_info is not None:
            pl = wppfsupport._generate_default_parameters_LeBail(
                self.phases, self.peakshape, ptype="lmfit"
            )
            self.lebail_param_list = [p for p in pl]
            if isinstance(param_info, Parameters_lmfit):
                """
                directly passing the parameter class
                """
                self._params = param_info
                params = param_info
            else:
                params = Parameters_lmfit()

                if isinstance(param_info, dict):
                    """
                    initialize class using dictionary read from the yaml file
                    """
                    for k in param_info:
                        v = param_info[k]
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
                        raise FileError('input spectrum file doesn\'t exist.')

                """
                this part initializes the lattice parameters in the
                """
                for p in self.phases:
                    wppfsupport._add_lp_to_params(params, self.phases[p])

                self._params = params
        else:

            params = wppfsupport._generate_default_parameters_LeBail(
                self.phases, self.peakshape, ptype="lmfit"
            )
            self.lebail_param_list = [p for p in params]
            wppfsupport._add_detector_geometry(params, self.instrument)
            if "chebyshev" in self.bkgmethod.keys():
                wppfsupport._add_background(
                    params, self.lineouts, self.bkgdegree
                )
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
            self.lineouts_sim[key] = LeBaillight(
                key,
                lo,
                self.Icalc[key],
                self.tth,
                self.hkls,
                self.dsp,
                self.shkl,
                self.lebail_param_list,
                self.params,
                self.peakshape,
                self.bkgmethod,
            )


class LeBaillight:
    """
    just a lightweight LeBail class which does only the
    simple computation of diffraction spectrum given the
    parameters and intensity values
    """

    def __init__(
        self,
        name,
        lineout,
        Icalc,
        tth,
        hkls,
        dsp,
        shkl,
        lebail_param_list,
        params,
        peakshape,
        bkgmethod,
    ):

        self.name = name
        self.lebail_param_list = lebail_param_list
        self.peakshape = peakshape
        self.params = params
        self.lineout = lineout
        self.Icalc = Icalc
        self.shkl = shkl
        self.tth = tth
        self.hkls = hkls
        self.dsp = dsp
        self.bkgmethod = bkgmethod
        self.computespectrum()
        # self.CalcIobs()
        # self.computespectrum()

    def computespectrum(self):

        x = self.tth_list
        y = np.zeros(x.shape)
        tth_list = np.ascontiguousarray(self.tth_list)

        for p in self.tth:
            for k in self.tth[p]:

                Ic = self.Icalc[p][k]

                shft_c = (
                    np.cos(0.5 * np.radians(self.tth[p][k]))
                    * self.params["shft"].value
                )
                trns_c = (
                    np.sin(np.radians(self.tth[p][k]))
                    * self.params["trns"].value
                )
                tth = (
                    self.tth[p][k]
                    + self.params["zero_error"].value
                    + shft_c
                    + trns_c
                )

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                n = np.min((tth.shape[0], Ic.shape[0]))
                shkl = self.shkl[p]
                name = p
                eta_n = f"{name}_eta_fwhm"
                eta_fwhm = self.params[eta_n].value
                strain_direction_dot_product = 0.0
                is_in_sublattice = False

                cag = np.array(
                    [
                        self.params["U"].value,
                        self.params["V"].value,
                        self.params["W"].value,
                    ]
                )
                gaussschrerr = self.params["P"].value
                lorbroad = np.array(
                    [self.params["X"].value, self.params["Y"].value]
                )
                anisbroad = np.array(
                    [
                        self.params["Xe"].value,
                        self.params["Ye"].value,
                        self.params["Xs"].value,
                    ]
                )
                if self.peakshape == 0:
                    HL = self.params["HL"].value
                    SL = self.params["SL"].value
                    args = (
                        cag,
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
                        self.wn,
                    )

                elif self.peakshape == 1:
                    args = (
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
                        Ic,
                    )

                elif self.peakshape == 2:
                    alpha = np.array(
                        [
                            self.params["alpha0"].value,
                            self.params["alpha1"].value,
                        ]
                    )
                    beta = np.array(
                        [
                            self.params["beta0"].value,
                            self.params["beta1"].value,
                        ]
                    )
                    args = (
                        alpha,
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
                        Ic,
                    )

                y += self.computespectrum_fcn(*args)

        self._spectrum_sim = Spectrum(x=x, y=y)

    def CalcIobs(self):
        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/08/2020 SS 1.0 original
        >> @DETAILS:    this is one of the main functions to partition the expt intensities
                        to overlapping peaks in the calculated pattern
        """

        self.Iobs = {}
        spec_expt = self.spectrum_expt.data_array
        spec_sim = self.spectrum_sim.data_array
        tth_list = np.ascontiguousarray(self.tth_list)
        for p in self.tth:

            self.Iobs[p] = {}
            for k in self.tth[p]:

                Ic = self.Icalc[p][k]

                shft_c = (
                    np.cos(0.5 * np.radians(self.tth[p][k]))
                    * self.params["shft"].value
                )
                trns_c = (
                    np.sin(np.radians(self.tth[p][k]))
                    * self.params["trns"].value
                )
                tth = (
                    self.tth[p][k]
                    + self.params["zero_error"].value
                    + shft_c
                    + trns_c
                )

                dsp = self.dsp[p][k]
                hkls = self.hkls[p][k]
                n = np.min((tth.shape[0], Ic.shape[0]))
                shkl = self.shkl[p]
                name = p
                eta_n = f"{name}_eta_fwhm"
                eta_fwhm = self.params[eta_n].value
                strain_direction_dot_product = 0.0
                is_in_sublattice = False

                cag = np.array(
                    [
                        self.params["U"].value,
                        self.params["V"].value,
                        self.params["W"].value,
                    ]
                )
                gaussschrerr = self.params["P"].value
                lorbroad = np.array(
                    [self.params["X"].value, self.params["Y"].value]
                )
                anisbroad = np.array(
                    [
                        self.params["Xe"].value,
                        self.params["Ye"].value,
                        self.params["Xs"].value,
                    ]
                )

                if self.peakshape == 0:
                    HL = self.params["HL"].value
                    SL = self.params["SL"].value
                    args = (
                        cag,
                        gaussschrerr,
                        lorbroad,
                        anisbroad,
                        shkl,
                        eta_fwhm,
                        HL,
                        SL,
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
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                elif self.peakshape == 2:
                    alpha = np.array(
                        [
                            self.params["alpha0"].value,
                            self.params["alpha1"].value,
                        ]
                    )
                    beta = np.array(
                        [
                            self.params["beta0"].value,
                            self.params["beta1"].value,
                        ]
                    )
                    args = (
                        alpha,
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
                        Ic,
                        spec_expt,
                        spec_sim,
                    )

                self.Iobs[p][k] = self.calc_Iobs_fcn(*args)
        self.Icalc = self.Iobs

    @property
    def weights(self):
        lo = self.lineout
        weights = np.divide(1.0, np.sqrt(lo.data[:, 1]))
        weights[np.isinf(weights)] = 0.0

        return weights

    @property
    def bkgdegree(self):
        if "chebyshev" in self.bkgmethod.keys():
            return self.bkgmethod["chebyshev"]

    @property
    def tth_step(self):
        return self.lineout.data[1, 0] - self.lineout.data[0, 0]

    @property
    def background(self):
        tth, I = self.spectrum_expt.data
        mask = self.mask[:, 1]

        if "chebyshev" in self.bkgmethod.keys():
            pname = [f"{self.name}_bkg_C{ii}" for ii in range(self.bkgdegree)]

            coef = [self.params[p].value for p in pname]
            c = Chebyshev(coef, domain=[tth[0], tth[-1]])
            bkg = c(tth)
            bkg[mask] = np.nan

        elif 'snip1d' in self.bkgmethod.keys():
            ww = np.rint(self.bkgmethod["snip1d"][0] / self.tth_step).astype(
                np.int32
            )

            numiter = self.bkgmethod["snip1d"][1]

            bkg = np.squeeze(
                snip1d_quad(np.atleast_2d(I), w=ww, numiter=numiter)
            )
            bkg[mask] = np.nan
        return bkg

    @property
    def spectrum_sim(self):
        tth, I = self._spectrum_sim.data
        mask = self.mask[:, 1]
        # I[mask] = np.nan
        I += self.background

        return Spectrum(x=tth, y=I)

    @property
    def spectrum_expt(self):
        d = self.lineout.data
        mask = self.mask[:, 1]
        # d[mask,1] = np.nan
        return Spectrum(x=d[:, 0], y=d[:, 1])

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        """
        set the local value of the parameters to the
        global values from the calibrator class
        """
        if isinstance(params, Parameters_lmfit):
            if hasattr(self, '_params'):
                for p in params:
                    if (p in self.lebail_param_list) or (self.name in p):
                        self._params[p].value = params[p].value
                        self._params[p].max = params[p].max
                        self._params[p].min = params[p].min
                        self._params[p].vary = params[p].vary
            else:
                from scipy.special import roots_legendre

                xn, wn = roots_legendre(16)
                self.xn = xn[8:]
                self.wn = wn[8:]
                self._params = Parameters_lmfit()
                for p in params:
                    if (p in self.lebail_param_list) or (self.name in p):
                        self._params.add(
                            name=p,
                            value=params[p].value,
                            max=params[p].max,
                            min=params[p].min,
                            vary=params[p].vary,
                        )

            # if hasattr(self, "tth") and \
            # hasattr(self, "dsp") and \
            # hasattr(self, "lineout"):
            #     self.computespectrum()
        else:
            msg = "only lmfit Parameters class permitted"
            raise ValueError(msg)

    @property
    def lineout(self):
        return self._lineout

    @lineout.setter
    def lineout(self, lo):
        if isinstance(lo, np.ma.MaskedArray):
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
        return self.lineout[:, 0].data

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
            msg = f"two theta vallues need " f"to be in a dictionary"
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
            msg = f"two theta vallues need " f"to be in a dictionary"
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
            msg = f"two theta vallues need " f"to be in a dictionary"
            raise ValueError(msg)

    @property
    def mask(self):
        return self.lineout.mask

    @property
    def tth_list(self):
        return self.lineout[:, 0].data

    @property
    def Icalc(self):
        return self._Icalc

    @Icalc.setter
    def Icalc(self, I):
        self._Icalc = I

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
