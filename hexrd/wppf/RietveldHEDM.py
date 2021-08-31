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
                 bkgmethod={'spline': None},
                 peakshape='pvfcj',
                 shape_factor=1.,
                 particle_size=1.,
                 phi=0.):

        self.bkgmethod = bkgmethod
        self.shape_factor = shape_factor
        self.particle_size = particle_size
        self.phi = phi
        self.peakshape = peakshape

        self._tstart = time.time()

        self.instrument = instrument
        self.omegaimageseries = omegaimageseries

        self.phases = phases
        self.params = params

        self.PolarizationFactor()
        self.computespectrum()

        self._tstop = time.time()
        
        self.niter = 0
        self.Rwplist = np.empty([0])
        self.gofFlist = np.empty([0])

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

# def init_texturemodel(self,
#                      eta_data=(-180.,180.,10.),
#                      omega_data=(-180.,180.,10.),
#                      sample_symmetry="-1",
#                      max_degree=10
#                      ):
#     """
#     this function initializes the texture model for the Rietveld
#     class. the user needs to specify the start and stop ranges 
#     as well as the step sizes for eta and omega angles for the 
#     HEDM datasets. The range for tth is already a part of the class.
#     this is likely going to be a fairly complicated function with
#     a lot of moving parts. one texture model will be initialized for 
#     each member of the phase class. the coefficients of the texture 
#     model will not be a part of the main parameter class, since there 
#     are a lot of these coefficients. instead we will use the parameter
#     class initialized inside the texture model.
#     """

#     self.texturemodel = {}
#     eta = np.arange(eta_data[0], 
#                     eta_data[1]+1.,
#                     eta_data[2])
#     omega = np.arange(omega_data[0], 
#                     omega_data[1]+1.,
#                     omega_data[2])

#     eta = np.radians(eta)
#     omega = np.radians(omega)
#     ones = np.ones(eta.shape)

#     for p in self.phases:
#         mat = self.phases[p]
#         for k in self.phases.wavelength.keys()[0]:
#             tth = self.tth[p][k]
#             hkls = self.hkls[p][k]
#             pfdata = {}
#             for t,g in zip(tth,hkls):
#                 key = str(g)[1:-1].replace(" ","")
#                 tthtile = np.tile(t,[eta.shape[0],])
#                 data = None
#                 for o in omega:
#                     omgtile = np.tile(o,[eta.shape[0],])
#                     tmp = np.vstack((tthtile,eta,omgtile,ones)).T
#                     if data is None:
#                         data = tmp
#                     else:
#                         data = np.vstack((data,tmp))
#                 pfdata[key] = data

#             pf = pole_figures(mat, 
#                               hkls, 
#                               pfdata)
#             self.texturemodel[p] = harmonic_model(pf,
#                                     sample_symmetry,
#                                     max_degree)