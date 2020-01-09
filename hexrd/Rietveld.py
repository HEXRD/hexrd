import numpy as np
from hexrd import material
import warnings
from hexrd.imageutil import snip1d

class Rietveld:

	''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	01/08/2020 SS 1.0 original
    >> @DETAILS:  	this is the main rietveld class and contains all the refinable parameters
    				for the analysis. the member classes are as follows (in order of initialization):

    				1. Spectrum  		contains the experimental spectrum
    				2. Background  		contains the background extracted from spectrum
    				3. Peak 			contains everything to characterize a peak
    				4. Refine 			contains all the machinery for refinement
    	======================================================================================================== 
		======================================================================================================== 
	'''

	def __init__(self, pdata, inp_spectrum, spec_tth_min, spec_tth_max):

		'''
			these contain the main refinement parameters for the rietveld code
		'''

		# Cagliotti parameters for the half width of the peak
		self._U = 1.0e-3
		self._V = 1.0e-3
		self._W = 1.0e-3

		# Pseudo-Voight mixing parameters
		self._eta1 = 4.0e-1
		self._eta2 = 1.0e-2
		self._eta3 = 1.0e-2

		# peak intensity
		sf 			= pdata.get_structFact()
		sf 			= sf / np.amax(sf)
		self._Imax 	= 1.0

		# initialize the spectrum class
		self.S  = self.Spectrum(inp_spectrum, spec_tth_min, spec_tth_max)

		# initialize the background class
		self.B = self.Background(self.S.spec_arr, method='snip', snipw=8, numiter=2, deg=4)

		self.generate_tthlist()

		# initialize the refine class
		#self.Refine()

		# initialize simulated spectrum
		self.initialize_spectrum(pdata)

	def checkangle(ang, name):

		if(np.abs(ang) > 2.0 * np.pi):
			warnings.warn(name + " : the angles seems to be too large. \
							Please check if you've converted to radians.")

	def CagliottiH(self, tth):

		tanth 	= np.tan(0.5 * tth)
		self.H 	= np.sqrt(self.U * tanth**2 + self.V * tanth + self.W)

	def MixingFact(self, tth):
		self.eta = self.eta1 + self.eta2 * tth + self.eta3 * tth**2

		if(self.eta > 1.0):
			self.eta = 1.0

		elif(self.eta< 0.0):
			self.eta = 0.0

	def Gaussian(self, tth):
		beta = 0.5 * self.H * np.sqrt(np.pi / np.log(2.0))
		self.GaussianI = self.Imax * np.exp(-np.pi * (self.tth_list - tth)**2 / beta**2)

	def Lorentzian(self, tth):
		w = 0.5 * self.H
		self.LorentzI = self.Imax * (w**2 / (w**2 + (self.tth_list - tth)**2 ) )

	def PseudoVoight(self, tth):
		self.Gaussian(tth)
		self.Lorentzian(tth)
		self.PV = self.eta * self.GaussianI + (1.0 - self.eta) * self.LorentzI

	def SpecimenBroadening(self):
		self.SB = np.sinc((self.tth_list - self.tth) / 2.0 / np.pi)

	def initialize_spectrum(self, pdata):

		peak_tth 		= pdata.getTTh()
		mask 			= (peak_tth < self.S.spec_tth_max) & (peak_tth > self.S.spec_tth_min)
		peak_tth 		= peak_tth[mask]

		if(self.S.ndim == 1):
			self.spec_sim 	= np.zeros(self.S.nspec[0])

		elif(self.S.ndim == 2):
			self.spec_sim 	= np.zeros(self.S.nspec[1])

		self.spec_sim =  np.copy(self.B.background)

		for tth in peak_tth:

			self.CagliottiH(tth)
			self.MixingFact(tth)
			self.PseudoVoight(tth)
			self.spec_sim += self.PV

	def generate_tthlist(self):

		tthmin = self.S.spec_tth_min
		tthmax = self.S.spec_tth_max

		if(self.S.ndim == 1):
			nspec 	= self.S.nspec[0]

		elif(self.S.ndim == 2):
			nspec 	= self.S.nspec[1]

		self.tth_list = np.linspace(tthmin, tthmax, nspec)

	'''
		all the properties for rietveld class
	'''
	@property
	def U(self):
		return self._U

	@U.setter
	def U(self, Uinp):
		self._U = Uinp
		# self.CagliottiH()
		# self.Gaussian()
		# self.Lorentzian()
		# self.PseudoVoight()
		# return

	@property
	def V(self):
		return self._V

	@V.setter
	def V(self, Vinp):
		self._V = Vinp
		# self.CagliottiH()
		# self.Gaussian()
		# self.Lorentzian()
		# self.PseudoVoight()
		# return

	@property
	def W(self):
		return self._W

	@W.setter
	def W(self, Winp):
		self._W = Winp
		# self.CagliottiH()
		# self.Gaussian()
		# self.Lorentzian()
		# self.PseudoVoight()
		# return

	@property
	def eta1(self):
		return self._eta1

	@eta1.setter
	def eta1(self, val):
		self._eta1 = val
		# self.MixingFact()
		# self.PseudoVoight()
		# return

	@property
	def eta2(self):
		return self._eta2

	@eta2.setter
	def eta2(self, val):
		self._eta2 = val
		# self.MixingFact()
		# self.PseudoVoight()
		# return

	@property
	def eta3(self):
		return self._eta3

	@eta3.setter
	def eta3(self, val):
		self._eta3 = val
		# self.MixingFact()
		# self.PseudoVoight()
		# return

	@property
	def Imax(self):
		return self._Imax

	@Imax.setter
	def Imax(self, val):
		self._Imax = val

	''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	01/08/2020 SS 1.0 original
    >> @DETAILS:  	this is the spectrum class which takes in an input diffraction spectrum.
    				it could be either a 1d or 2d spectrum depending on the type of data.
    				the spectrum class will be a sub-class of the main Rietveld class
    	======================================================================================================== 
		======================================================================================================== 
	'''
	class Spectrum:

		def __init__(self, spectrum_arr, spec_tth_min, spec_tth_max):
			
			# make sure angles are in radians, otherwise throw warning
			Rietveld.checkangle(spec_tth_min, 'spec_tth_min')
			Rietveld.checkangle(spec_tth_max, 'spec_tth_max')

			# fill value and check dimensionality. only 1d or 2d allowed
			self._spec_arr 		= spectrum_arr

			if(self.ndim > 2):
				raise ValueError('incorrect number of dimensions in spectrum. \
								  should be 1d or 2d')

			# minimum and maximum angular range of the spectrum
			self._spec_tth_min 	= spec_tth_min
			self._spec_tth_max 	= spec_tth_max

		# define all properties of this class
		@property
		def spec_arr(self):
			return self._spec_arr

		@spec_arr.setter
		def spec_arr(self, val):
			self._spec_arr 		= val
			self._nspec 		= self._spec_arr.shape
			self._ndim 			= self._spec_arr.ndim
			Rietveld.Background.initialize_bkg()

		''' ndim can't be set, so will have no setter function 
			ndim tells if the spectrum is 1-d, 2-d etc.
		'''
		@property
		def ndim(self):
			return self.spec_arr.ndim

		''' nspec can't be set, so will have no setter function 
			nspec has information about the size of spectrum in 
			two-theta(1-D case) or eta/two-theta (2-D case)
		'''
		@property
		def nspec(self):
			return self.spec_arr.shape
		
		@property
		def spec_tth_min(self):
			return self._spec_tth_min
		
		@spec_tth_min.setter
		def spec_tth_min(self, val):
			Rietveld.checkangle(val, 'spec_tth_min')
			self._spec_tth_min = val

		@property
		def spec_tth_max(self):
			return self._spec_tth_max

		@spec_tth_max.setter
		def spec_tth_max(self, val):
			Rietveld.checkangle(val, 'spec_tth_max')
			self._spec_tth_max = val
		
	''' ======================================================================================================== 
		========================================================================================================

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	01/08/2020 SS 1.0 original
	>> @DETAILS:  	this is the background class which takes the spectrum and calculates
					the background using a variety of methods. default will be snip, but 
					polynomial, chebyshev polynomials etc. are also implemented

		======================================================================================================== 
		========================================================================================================
	'''
	class Background:

		def __init__(self, spec_arr, method='snip', snipw=8, numiter=2, deg=4):

			self._method 		= method
			self._snipw 		= snipw
			self._snip_niter 	= numiter
			self._deg 			= deg

			self.initialize_bkg(spec_arr)
			
		def polyfit(self, spectrum, deg=4):
			pass

		def chebfit(self, spectrum, deg=4):
			pass

		def initialize_bkg(self, spec_arr):

			if (self.method.lower() == 'snip'):
				self.background = snip1d(spec_arr, w=self.snipw, numiter=self.snip_niter)

			elif (self.method.lower() == 'poly'):
				self.background = polyfit(spec_arr, deg=self.deg)

			elif (self.method.lower() == 'cheb'):
				self.background = chebfit(spec_arr, deg=self.deg)

		@property
		def method(self):
			return self._method

		@method.setter
		def method(self, name):
			self._method = name
			self.initialize_bkg()
		
		@property
		def snipw(self):
			return self._snipw

		@snipw.setter
		def snipw(self, val):
			if (isinstance(val, int)):
				self._snipw = val
				self.initialize_bkg()
			else:
				warnings.warn('Not a integer. converting to nearest integer')
				self._snipw = int(round(val))
				self.initialize_bkg()
		
		@property
		def snip_niter(self):
			return self._snip_niter

		@snip_niter.setter
		def snip_niter(self, val):
			if (isinstance(val, int)):
				self._snip_niter = val
				self.initialize_bkg()
			else:
				warnings.warn('Not a integer. converting to nearest integer')
				self._snip_niter = int(round(val))
				self.initialize_bkg()
		
		@property
		def deg(self):
			return self._deg

		@deg.setter
		def deg(self, val):
			if (isinstance(val, int)):
				self._deg = val
				self.initialize_bkg()
			else:
				warnings.warn('input not a integer. converting to nearest integer')
				self._deg = int(round(val))
				self.initialize_bkg()
		


	''' ======================================================================================================== 
		========================================================================================================

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	01/08/2020 SS 1.0 original
	>> @DETAILS:  	this is the peak class which characterizes each peak. the overall 
					spectrum is a collection of such peaks
					the peak class will be a sub-class of the main Rietveld class

		======================================================================================================== 
		========================================================================================================
	'''
	class Peak:

		def __init__(self, tth):

			Rietveld.checkangle(tth, 'peak two theta')

			# 2theta for the peak
			self._tth = tth

			# range of angles for which the peak is calculated
			self._tth_min 	= Rietveld.Spectrum.spec_tth_min
			self._tth_max 	= Rietveld.Spectrum.spec_tth_max

			if (Rietveld.Spectrum.ndim == 1):
				self._tth_step 	= (self._tth_max - self._tth_min) / Rietveld.Spectrum.nspec[0]

			elif (Rietveld.Spectrum.ndim == 2):
				self._tth_step 	= (self._tth_max - self._tth_min) / Rietveld.Spectrum.nspec[1]	

			else:
				raise ValueError('incorrect number of dimensions in spectrum')

			''' maximum intensity, Imax = S * Mk * Lk * |Fk|**2 * Pk * Ak * Ek * ...
				S  = arbitrary scale factor
				Mk = multiplicity of reflection
				Lk = lorentz polarization factor
				Fk = structure factor
				Pk = preferrred orientation 
				Ak = absorption correction
				Ek = extinction correction 
				...
				...
				etc.
			'''
			self.Imax = 1.0

			'''
				get all the parameters for the peak 
			'''
			self.CagliottiH()
			self.MixingFact()

			'''
				get the actual peaks
			'''
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			self.SpecimenBroadening()

			'''
				set the parameter for background estimation
				the 5 times the ratio of halfwidth to step size was figured
				out empirically
			'''
			self.snipw = int(np.round(5.0 * self.H / self.tth_step))


		@property
		def tth(self):
			return self._tth

		@tth.setter
		def tth(self, val):
			self._tth = val
			self.CagliottiH()
			self.MixingFact()
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()

			return

		@property
		def tth_min(self):
			return self._tth_min

		@tth_min.setter
		def tth_min(self, val):
			self._tth_min = val
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			return

		@property
		def tth_max(self):
			return self._tth_max

		@tth_max.setter
		def tth_max(self, val):
			self._tth_max = val
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			return

		@property
		def tth_step(self):
			return self._tth_step

		@tth_step.setter
		def tth_step(self, val):
			self._tth_step = val
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			return

		# tth_list is read only
		@property
		def tth_list(self):
			return np.arange(self.tth_min, self.tth_max, self.tth_step)

	''' ======================================================================================================== 
		========================================================================================================

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	01/08/2020 SS 1.0 original
	>> @DETAILS:  	this is the refimenent class which takes the spectrum and the
					peak class and is responsible for the the optimization algorithms.
					the refine class will be a sub-class of the main Rietveld class

		======================================================================================================== 
		========================================================================================================
	'''
	class Refine:

		def __init__(self):
			pass