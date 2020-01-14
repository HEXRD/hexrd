import numpy as np
from hexrd import material
import warnings
from hexrd.imageutil import snip1d
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class Rietveld:

	''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	01/08/2020 SS 1.0 original
    >> @DETAILS:  	this is the main rietveld class and contains all the refinable parameters
    				for the analysis. the member classes are as follows (in order of initialization):

    				1. Spectrum  		contains the experimental spectrum
    				2. Background  		contains the background extracted from spectrum
    				3. Refine 			contains all the machinery for refinement
    	======================================================================================================== 
		======================================================================================================== 
	'''

	def __init__(self, pdata, inp_spectrum, 
				 spec_tth_min, spec_tth_max,
				 bkgmethod='spline', snipw=8, 
				 numiter=2, deg=8):

		'''
			these contain the main refinement parameters for the rietveld code
		'''

		# plane data
		self._pdata = pdata

		# Cagliotti parameters for the half width of the peak
		self._U = 1.0e-4
		self._V = 1.0e-4
		self._W = 1.0e-4

		# Pseudo-Voight mixing parameters
		self._eta1 = 4.0e-1
		self._eta2 = 1.0e-2
		self._eta3 = 1.0e-2

		# peak intensity
		sf 			= pdata.get_structFact()
		sf 			= sf / np.amax(sf)
		self._Imax 	= 1.0

		# initialize the spectrum class
		self.Spectrum  = self.Spectrum(inp_spectrum, spec_tth_min, spec_tth_max,
								method=bkgmethod, snipw=snipw, numiter=numiter, 
								deg=deg)

		self.generate_tthlist()

		# initialize simulated spectrum
		self.initialize_spectrum()

		x0 		= np.asarray([self.U,self.V,self.W,self.eta1,self.eta2,self.eta3])
		bounds 	= Bounds([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
						 [1e-2, 1e-2, 1e-2, 1.0, 1.0, 1.0]) 

		# initialize the refine class
		self.Optimize = self.Refine(x0, bounds)

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

	def initialize_spectrum(self):

		peak_tth 		= self.pdata.getTTh()
		mask 			= (peak_tth < self.Spectrum.spec_tth_max) & \
						  (peak_tth > self.Spectrum.spec_tth_min)
		peak_tth 		= peak_tth[mask]

		if(self.Spectrum.ndim == 1):
			self.spec_sim 	= np.zeros(self.Spectrum.nspec[0])

		elif(self.Spectrum.ndim == 2):
			self.spec_sim 	= np.zeros(self.Spectrum.nspec[1])

		for tth in peak_tth:

			self.CagliottiH(tth)
			self.MixingFact(tth)
			self.PseudoVoight(tth)
			self.spec_sim += self.PV

	def generate_tthlist(self):

		tthmin = self.Spectrum.spec_tth_min
		tthmax = self.Spectrum.spec_tth_max

		if(self.Spectrum.ndim == 1):
			nspec 	= self.Spectrum.nspec[0]

		elif(self.Spectrum.ndim == 2):
			nspec 	= self.Spectrum.nspec[1]

		self.tth_list = np.linspace(tthmin, tthmax, nspec)

	'''
		this is the function which evaluates the residual between 
		simulated and experimental spectra. this is the objective
		which needs to be minimized by the refine class. the input
		is a vector of values for different parameters in the model.

		@NOTE:
		for now it is limited to the U,V,W and eta1, eta2, eta3 just 
		for testing purpose

	'''
	def evalResidual(self, x0):

		self.U = x0[0]
		self.V = x0[1]
		self.W = x0[2]

		self.eta1 = x0[3]
		self.eta2 = x0[4]
		self.eta3 = x0[5]

		# return residual as percent
		residual = 100.0 * np.sum((self.spec_sim - self.Spectrum.spec_arr_nbkg)**2) / \
						   np.sum(self.Spectrum.spec_arr_nbkg**2)
		return residual

	def fit(self):

		res = minimize(self.evalResidual, self.Optimize.x0, \
					   method=self.Optimize.method, \
					   options=self.Optimize.options, \
					   bounds = self.Optimize.bounds)

		print(res.message+'\t'+'[ Exit status '+str(res.status)+' ]')
		print('\t minimum function value: '+str(res.fun))
		print('\t iterations: '+str(res.nit))
		print('\t function evaluations: '+str(res.nfev))
		print('\t gradient evaluations: '+str(res.njev))
		print('\t optimum values of parameters: '+str(res.x))
		return res

	'''
		all the properties for rietveld class
	'''
	@property
	def U(self):
		return self._U

	@U.setter
	def U(self, Uinp):
		self._U = Uinp
		self.initialize_spectrum()
		return

	@property
	def V(self):
		return self._V

	@V.setter
	def V(self, Vinp):
		self._V = Vinp
		self.initialize_spectrum()
		return

	@property
	def W(self):
		return self._W

	@W.setter
	def W(self, Winp):
		self._W = Winp
		self.initialize_spectrum()
		return

	@property
	def eta1(self):
		return self._eta1

	@eta1.setter
	def eta1(self, val):
		self._eta1 = val
		self.initialize_spectrum()
		return

	@property
	def eta2(self):
		return self._eta2

	@eta2.setter
	def eta2(self, val):
		self._eta2 = val
		self.initialize_spectrum()
		return

	@property
	def eta3(self):
		return self._eta3

	@eta3.setter
	def eta3(self, val):
		self._eta3 = val
		self.initialize_spectrum()
		return

	@property
	def Imax(self):
		return self._Imax

	@Imax.setter
	def Imax(self, val):
		self._Imax = val

	@property
	def pdata(self):
		return self._pdata
	
	@pdata.setter
	def pdata(self, inp_pdata):
		self._pdata = inp_pdata
		self.initialize_spectrum()

	''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	01/08/2020 SS 1.0 original
    				01/10/2020 SS 1.1 Background class absorbed in spectrum class
    >> @DETAILS:  	this is the spectrum class which takes in an input diffraction spectrum.
    				it could be either a 1d or 2d spectrum depending on the type of data.
    				the spectrum class will be a sub-class of the main Rietveld class

    	======================================================================================================== 
		======================================================================================================== 
	'''
	class Spectrum:

		def __init__(self, spectrum_arr, spec_tth_min, spec_tth_max,
					method='snip', snipw=8, numiter=2, deg = 8):
			
			# make sure angles are in radians, otherwise throw warning
			Rietveld.checkangle(spec_tth_min, 'spec_tth_min')
			Rietveld.checkangle(spec_tth_max, 'spec_tth_max')

			self._method 		= method
			self._snipw 		= snipw
			self._snip_niter 	= numiter
			self._deg 			= deg

			# minimum and maximum angular range of the spectrum
			self._spec_tth_min 	= spec_tth_min
			self._spec_tth_max 	= spec_tth_max

			if(spectrum_arr.ndim == 1):
				nspec 	= spectrum_arr.shape[0]

			elif(spectrum_arr.ndim == 2):
				nspec 	= spectrum_arr.shape[1]

			self.tth_list = np.linspace(spec_tth_min, spec_tth_max, nspec)

			# fill value and check dimensionality. only 1d or 2d allowed
			self.spec_arr 		= spectrum_arr

			if(self.ndim > 2):
				raise ValueError('incorrect number of dimensions in spectrum. \
								  should be 1d or 2d')


		# cubic spline fit of background using custom points chosen from plot
		def splinefit(self, x, y):
			cs = CubicSpline(x,y)
			self.background = cs(self.tth_list)

		def chebfit(self, spectrum, deg=4):
			pass

		def initialize_bkg(self):

			if (self.method.lower() == 'snip'):
				'''
					the snip method usually produces a pretty decent background, but
					it was observed that the estimated background has a small offset 
					thus we will add a small offset back to the background generated 
					here. this is done using a robust linear model
				'''
				background 			= snip1d(self.spec_arr, w=self.snipw, numiter=self.snip_niter)
				self.background 	= background

			elif (self.method.lower() == 'spline'):
				'''
					the cubic spline seems to be the ideal route in terms
					of determining the background intensity. this involves 
					selecting a small (~5) number of points from the spectrum,
					usually called the anchor points. a cubic spline interpolation
					is performed on this subset to estimate the overall background.
					scipy provides some useful routines for this
				'''
				self.selectpoints()
				x = self.points[:,0]
				y = self.points[:,1]

				self.splinefit(x, y)

			elif (self.method.lower() == 'poly'):
				self.background = polyfit(spec_arr, deg=self.deg)

			elif (self.method.lower() == 'cheb'):
				self.background = chebfit(spec_arr, deg=self.deg)

		def remove_bkg(self):

			self._spec_arr_nbkg = self.spec_arr - self.background
			mask = self._spec_arr_nbkg < 0.0
			self._spec_arr_nbkg[mask] = 0.0

		def selectpoints(self):

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_title('Select 5 points for background estimation')

			line, = ax.plot(self.tth_list, self.spec_arr, '-b', picker=5)  # 5 points tolerance
			plt.show()

			self.points = np.asarray(plt.ginput(5,timeout=-1, show_clicks=True))			
			plt.close()
		# define all properties of this class
		@property
		def spec_arr(self):
			return self._spec_arr

		@spec_arr.setter
		def spec_arr(self, val):
			self._spec_arr 		= val
			self._nspec 		= self._spec_arr.shape
			self._ndim 			= self._spec_arr.ndim
			self.initialize_bkg()
			self.remove_bkg()

		@property
		def spec_arr_nbkg(self):
			return self._spec_arr_nbkg

		# @spec_arr.setter
		# def spec_arr_nbkg(self, val):
		# 	assert(val.ndim == self.ndim), "dimensionality not equal"
		# 	assert(np.all(val.shape == self.nspec)), " number of dimensions do not match"
		# 	self._spec_arr_nbkg = val

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
		

		@property
		def background(self):
			return self._background

		@background.setter
		def background(self, val):
			assert(val.ndim == self.ndim), "dimensionality of background not equal to spectrum"
			assert(np.all(val.shape == self.nspec)), " shape of background does not match background"
			self._background = val
			self.remove_bkg()
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
		'''
			set up the minimization class
		'''
		def __init__(self,x0,bounds,method='SLSQP',ftol=1e-9,disp=False):

			self.bounds 		= bounds
			self.x0 			= x0

			self.method 		= 'SLSQP'
			self.ftol 			= 1e-9
			self.disp 			= False
			self.options 		= {'ftol':self.ftol, \
								   'disp':self.disp}