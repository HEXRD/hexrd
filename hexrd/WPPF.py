import numpy as np
import warnings
from hexrd.imageutil import snip1d
from hexrd.crystallography import PlaneData
from scipy.optimize import minimize, Bounds, shgo, least_squares
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import signal
from hexrd.valunits import valWUnit
import yaml
from os import path
import pickle


class Parameters:
	''' ======================================================================================================== 
	======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	05/18/2020 SS 1.0 original
	>> @DETAILS:  	this is the parameter class which handles all refinement parameters
		for both the Rietveld and the LeBail refimentment problems

		======================================================================================================== 
		======================================================================================================== 
	'''
	def __init__(self, name=None, vary=False, value=0.0, lb=-np.Inf, ub=np.Inf):

		self.param_dict = {}

		if(name is not None):
			self.add(name=name, vary=vary, value=value, lb=min, ub=max)

	def add(self, name, vary=False, value=0.0, lb=-np.Inf, ub=np.Inf):
		'''
			>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
			>> @DATE:     	05/18/2020 SS 1.0 original
			>> @DETAILS:  	add a single named parameter
		'''
		self[name] = Parameter(name=name, vary=vary, value=value, lb=lb, ub=ub)

	def add_many(self, names, varies, values, lbs, ubs):
		'''
			>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
			>> @DATE:     	05/18/2020 SS 1.0 original
			>> @DETAILS:  	load a list of named parameters 
		'''
		assert len(names)==len(varies),"lengths of tuples not consistent"
		assert len(names)==len(values),"lengths of tuples not consistent"
		assert len(names)==len(lbs),"lengths of tuples not consistent"
		assert len(names)==len(ubs),"lengths of tuples not consistent"

		for i,n in enumerate(names):
			self.add(n, vary=varies[i], value=values[i], lb=lbs[i], ub=ubs[i])

	def load(self, fname):
		'''
			>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
			>> @DATE:     	05/18/2020 SS 1.0 original
			>> @DETAILS:  	load parameters from yaml file
		'''
		with open(fname) as file:
			dic = yaml.load(file, Loader=yaml.FullLoader)

		for k in dic.keys():
			v = dic[k]
			self.add(k, value=v[0], lb=v[1], ub=v[2], vary=v[3])

	def dump(self, fname):
		'''
			>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
			>> @DATE:     	05/18/2020 SS 1.0 original
			>> @DETAILS:  	dump the class to a yaml looking file. name is the key and the list
							has [value, lb, ub, vary] in that order
		'''
		dic = {}
		for k in self.param_dict.keys():
			dic[k] =  [self[k].value,self[k].lb,self[k].ub,self[k].vary]

		with open(fname, 'w') as f:
			data = yaml.dump(dic, f, sort_keys=False)

	# def pretty_print(self):
	# 	'''
	# 		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	# 		>> @DATE:     	05/18/2020 SS 1.0 original
	# 		>> @DETAILS:  	print to the Parameter class to the terminal
	# 	'''
	# 	pass

	def __getitem__(self, key):
		if(key in self.param_dict.keys()):
			return self.param_dict[key]
		else:
			raise ValueError('variable with name not found')

	def __setitem__(self, key, parm_cls):

		if(key in self.param_dict.keys()):
			warnings.warn('variable already in parameter list. overwriting ...')
		if(isinstance(parm_cls, Parameter)):
			self.param_dict[key] = parm_cls
		else:
			raise ValueError('input not a Parameter class')

	def __iter__(self):
		self.n = 0
		return self

	def __next__(self):
		if(self.n < len(self.param_dict.keys())):
			res = list(self.param_dict.keys())[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration


	def __str__(self):
		retstr = 'Parameters{\n'
		for k in self.param_dict.keys():
			retstr += self[k].__str__()+'\n'

		retstr += '}'
		return retstr

class Parameter:
	''' ======================================================================================================== 
	======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	05/18/2020 SS 1.0 original
	>> @DETAILS:  	the parameters class (previous one) is a collection of this
					parameter class indexed by the name of each variable

		======================================================================================================== 
		======================================================================================================== 
	'''

	def __init__(self, name=None, vary=False, value=0.0, lb=-np.Inf, ub=np.Inf):

		self.name = name
		self.vary = vary
		self.value = value
		self.lb = lb
		self.ub = ub

	def __str__(self):
		retstr =  '< Parameter \''+self.name+'\'; value : '+ \
		str(self.value)+'; bounds : ['+str(self.lb)+','+ \
		str(self.ub)+' ]; vary :'+str(self.vary)+' >'

		return retstr

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		if(isinstance(name, str)):
			self._name = name

	@property
	def value(self):
		return self._value

	@value.setter
	def value(self, val):
		self._value = val

	@property
	def min(self):
		return self._min

	@min.setter
	def min(self, minval):
		self._min = minval

	@property
	def max(self):
		return self._max

	@max.setter
	def max(self, maxval):
		self._max = maxval

	@property
	def vary(self):
		return self._vary

	@vary.setter
	def vary(self, vary):
		if(isinstance(vary, bool)):
			self._vary = vary
	
class Spectrum:
	''' ======================================================================================================== 
	======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	05/18/2020 SS 1.0 original
	>> @DETAILS:  	spectrum class holds the a pair of x,y data, in this case, would be 
					2theta-intensity values

		======================================================================================================== 
		======================================================================================================== 
	'''
	def __init__(self, x=None, y=None, name=''):
		if x is None:
			self._x = np.linspace(10., 100., 500)
		else:
			self._x = x
		if y is None:
			self._y = np.log(self._x ** 2) - (self._x * 0.2) ** 2
		else:
			self._y = y
		self.name = name
		self.offset = 0
		self._scaling = 1
		self.smoothing = 0
		self.bkg_Spectrum = None

	@staticmethod
	def from_file(filename, skip_rows=0):
		try:
			if filename.endswith('.chi'):
				skip_rows = 4
			data = np.loadtxt(filename, skiprows=skip_rows)
			x = data.T[0]
			y = data.T[1]
			name = path.basename(filename).split('.')[:-1][0]
			return Spectrum(x, y, name)

		except ValueError:
			print('Wrong data format for spectrum file! - ' + filename)
			return -1

	def save(self, filename, header=''):
		data = np.dstack((self._x, self._y))
		np.savetxt(filename, data[0], header=header)

	def set_background(self, Spectrum):
		self.bkg_spectrum = Spectrum

	def reset_background(self):
		self.bkg_Spectrum = None

	def set_smoothing(self, amount):
		self.smoothing = amount

	def rebin(self, bin_size):
		"""
		Returns a new Spectrum which is a rebinned version of the current one.
		"""
		x, y = self.data
		x_min = np.round(np.min(x) / bin_size) * bin_size
		x_max = np.round(np.max(x) / bin_size) * bin_size
		new_x = np.arange(x_min, x_max + 0.1 * bin_size, bin_size)

		bins = np.hstack((x_min - bin_size * 0.5, new_x + bin_size * 0.5))
		new_y = (np.histogram(x, bins, weights=y)[0] / np.histogram(x, bins)[0])

		return Spectrum(new_x, new_y)

	@property
	def data(self):
		if self.bkg_Spectrum is not None:
			# create background function
			x_bkg, y_bkg = self.bkg_Spectrum.data

			if not np.array_equal(x_bkg, self._x):
				# the background will be interpolated
				f_bkg = interp1d(x_bkg, y_bkg, kind='linear')

				# find overlapping x and y values:
				ind = np.where((self._x <= np.max(x_bkg)) & (self._x >= np.min(x_bkg)))
				x = self._x[ind]
				y = self._y[ind]

				if len(x) == 0:
					# if there is no overlapping between background and Spectrum, raise an error
					raise BkgNotInRangeError(self.name)

				y = y * self._scaling + self.offset - f_bkg(x)
			else:
				# if Spectrum and bkg have the same x basis we just delete y-y_bkg
				x, y = self._x, self._y * self._scaling + self.offset - y_bkg
		else:
			x, y = self.original_data

		if self.smoothing > 0:
			y = gaussian_filter1d(y, self.smoothing)
		return x, y

	@data.setter
	def data(self, data):
		(x, y) = data
		self._x = x
		self._y = y
		self.scaling = 1
		self.offset = 0

	@property
	def original_data(self):
		return self._x, self._y * self._scaling + self.offset

	@property
	def x(self):
		return self._x

	@x.setter
	def x(self, new_value):
		self._x = new_value

	@property
	def y(self):
		return self._y

	@y.setter
	def y(self, new_y):
		self._y = new_y

	@property
	def scaling(self):
		return self._scaling

	@scaling.setter
	def scaling(self, value):
		if value < 0:
			self._scaling = 0
		else:
			self._scaling = value

	def limit(self, x_min, x_max):
		x, y = self.data
		return Spectrum(x[np.where((x_min < x) & (x < x_max))],
					   y[np.where((x_min < x) & (x < x_max))])

	def extend_to(self, x_value, y_value):
		"""
		Extends the current Spectrum to a specific x_value by filling it with the y_value. Does not modify inplace but
		returns a new filled Spectrum
		:param x_value: Point to which extend the Spectrum should be smaller than the lowest x-value in the Spectrum or
						vice versa
		:param y_value: number to fill the Spectrum with
		:return: extended Spectrum
		"""
		x_step = np.mean(np.diff(self.x))
		x_min = np.min(self.x)
		x_max = np.max(self.x)
		if x_value < x_min:
			x_fill = np.arange(x_min - x_step, x_value-x_step*0.5, -x_step)[::-1]
			y_fill = np.zeros(x_fill.shape)
			y_fill.fill(y_value)

			new_x = np.concatenate((x_fill, self.x))
			new_y = np.concatenate((y_fill, self.y))
		elif x_value > x_max:
			x_fill = np.arange(x_max + x_step, x_value+x_step*0.5, x_step)
			y_fill = np.zeros(x_fill.shape)
			y_fill.fill(y_value)

			new_x = np.concatenate((self.x, x_fill))
			new_y = np.concatenate((self.y, y_fill))
		else:
			return self

		return Spectrum(new_x, new_y)

	def plot(self, show=False, *args, **kwargs):
		plt.plot(self.x, self.y, *args, **kwargs)
		if show:
			plt.show()

	# Operators:
	def __sub__(self, other):
		orig_x, orig_y = self.data
		other_x, other_y = other.data

		if orig_x.shape != other_x.shape:
			# todo different shape subtraction of spectra seems the fail somehow...
			# the background will be interpolated
			other_fcn = interp1d(other_x, other_x, kind='linear')

			# find overlapping x and y values:
			ind = np.where((orig_x <= np.max(other_x)) & (orig_x >= np.min(other_x)))
			x = orig_x[ind]
			y = orig_y[ind]

			if len(x) == 0:
				# if there is no overlapping between background and Spectrum, raise an error
				raise BkgNotInRangeError(self.name)
			return Spectrum(x, y - other_fcn(x))
		else:
			return Spectrum(orig_x, orig_y - other_y)

	def __add__(self, other):
		orig_x, orig_y = self.data
		other_x, other_y = other.data

		if orig_x.shape != other_x.shape:
			# the background will be interpolated
			other_fcn = interp1d(other_x, other_x, kind='linear')

			# find overlapping x and y values:
			ind = np.where((orig_x <= np.max(other_x)) & (orig_x >= np.min(other_x)))
			x = orig_x[ind]
			y = orig_y[ind]

			if len(x) == 0:
				# if there is no overlapping between background and Spectrum, raise an error
				raise BkgNotInRangeError(self.name)
			return Spectrum(x, y + other_fcn(x))
		else:
			return Spectrum(orig_x, orig_y + other_y)

	def __rmul__(self, other):
		orig_x, orig_y = self.data
		return Spectrum(np.copy(orig_x), np.copy(orig_y) * other)

	def __eq__(self, other):
		if not isinstance(other, Spectrum):
			return False
		if np.array_equal(self.data, other.data):
			return True
		return False

class Phases:
	''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	05/20/2020 SS 1.0 original
	>> @DETAILS:  	simple class to handle different phases in the LeBail fit. contains planedata
					as member classes 
		======================================================================================================== 
		======================================================================================================== 
	'''
	def __init__(self, material_file=None, material_key=None):

		self.phase_dict = {}
		self.num_phases = 0
		if(material_file is not None):
			if(material_key is not None):
				self.add(material_file, material_key)

	def __str__(self):
		resstr = 'Phases in class:\n'
		for i,k in enumerate(self.phase_dict.keys()):
			resstr += '\t'+str(i)+'. '+k+'\n'
		return resstr

	def __getitem__(self, key):
		pass

	def __setitem__(self, key, value):
		pass

	def __getitem__(self, key):
		if(key in self.phase_dict.keys()):
			return self.phase_dict[key]
		else:
			raise ValueError('phase with name not found')

	def __setitem__(self, key, planedata_cls):

		if(key in self.phase_dict.keys()):
			warnings.warn('phase already in parameter list. overwriting ...')
		if(isinstance(planedata_cls, PlaneData)):
			self.phase_dict[key] = planedata_cls
		else:
			raise ValueError('input not a PlaneData class')

	def __iter__(self):
		self.n = 0
		return self

	def __next__(self):
		if(self.n < len(self.phase_dict.keys())):
			res = list(self.param_dict.keys())[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration

	def add(material_file, material_key):
		self[material_key] = load_planedata(material_file, material_key)
		self.num_phases += 1

	def add_many(material_file, material_keys):
		for k in material_keys:
			self[k] = load_planedata(material_file, k)
			self.num_phases += 1

	def load_pdata(material_file, material_key):
		pass
		#return pdata

class LeBail:
	''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	05/19/2020 SS 1.0 original
	>> @DETAILS:  	this is the main LeBail class and contains all the refinable parameters
					for the analysis. Since the LeBail method has no structural information 
					during refinement, the refinable parameters for this model will be:

					1. a, b, c, alpha, beta, gamma : unit cell parameters
					2. U, V, W : cagliotti paramaters
					3. 2theta_0 : Instrumental zero shift error
					4. eta1, eta2, eta3 : weight factor for gaussian vs lorentzian

					@NOTE: All angles are always going to be in degrees
		======================================================================================================== 
		======================================================================================================== 
	'''
	def __init__(self,expt_file=None,param_file=None):
		
		self.initialize_expt_spectrum(expt_file)
		self.initialize_parameters(param_file)

	def __str__(self):
		resstr = '<LeBail Fit class>\nParameters of the model are as follows:\n'
		resstr += self.params.__str__()
		return resstr

	def checkangle(ang, name):

		if(np.abs(ang) > 180.):
			warnings.warn(name + " : the absolute value of angles \
								seems to be large > 180 degrees")

	def initialize_parameters(self, param_file):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/19/2020 SS 1.0 original
		>> @DETAILS:  	initialize parameter list from file. if no file given, then initialize
						to some default values (lattice constants are for CeO2)
		'''
		params = Parameters()
		if(param_file is not None):
			if(path.exists(param_file)):
				params.load(param_file)
			else:
				raise FileError('parameter file doesn\'t exist.')
		else:
			'''
				first 6 are the lattice paramaters
				next three are cagliotti parameters
				next are the three gauss+lorentz mixing paramters
				final is the zero instrumental peak position error
			'''
			names  	= ('a','b','c','alpha','beta','gamma',\
					  'U','V','W','eta1','eta2','eta3','tth_zero')
			values 	= (5.415, 5.415, 5.415, 90., 90., 90., \
						0.5, 0.5, 0.5, 1e-3, 1e-3, 1e-3, 0.)

			lbs 		= (-np.Inf,) * len(names)
			ubs 		= (np.Inf,)  * len(names)
			varies 	= (False,)   * len(names)

			params.add_many(names,values=values,varies=varies,lbs=lbs,ubs=ubs)

		self.params = params

	def initialize_expt_spectrum(self, expt_file):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/19/2020 SS 1.0 original
		>> @DETAILS:  	load the experimental spectum of 2theta-intensity
		'''
		# self.spectrum_expt = Spectrum.from_file()
		if(expt_file is not None):
			if(path.exists(expt_file)):
				self.spectrum_expt = Spectrum.from_file(expt_file,skip_rows=0)
				''' also initialize statistical weights for the error calculation'''
				self.weights = 1.0 / np.sqrt(self.spectrum_expt.y)
			else:
				raise FileError('input spectrum file doesn\'t exist.')

	def CagliottiH(self, tth):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/20/2020 SS 1.0 original
		>> @DETAILS:  	calculates the cagiotti parameter for the peak width
		'''
		th 			= np.radians(0.5*tth)
		tanth 		= np.tan(th)
		self.Hcag 	= np.sqrt(self.U * tanth**2 + self.V * tanth + self.W)

	def MixingFact(self, tth):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/20/2020 SS 1.0 original
		>> @DETAILS:  	calculates the mixing factor eta
		'''
		self.eta = self.eta1 + self.eta2 * tth + self.eta3 * (tth*0.5)**2

		if(self.eta > 1.0):
			self.eta = 1.0

		elif(self.eta < 0.0):
			self.eta = 0.0

	def Gaussian(self, tth):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/20/2020 SS 1.0 original
		>> @DETAILS:  	this routine computes the gaussian peak profile
		'''
		beta = 0.5 * self.Hcag
		self.GaussianI = np.exp( -((self.tth_list - tth)/beta)**2 * np.log(2.) )

	def Lorentzian(self, tth):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/20/2020 SS 1.0 original
		>> @DETAILS:  	this routine computes the lorentzian peak profile
		'''
		w = 0.5 * self.Hcag
		self.LorentzI = 1. / ( 1. + ((self.tth_list - tth)/w)**2)

	def PseudoVoight(self, tth):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/20/2020 SS 1.0 original
		>> @DETAILS:  	this routine computes the pseudo-voight function as weighted 
						average of gaussian and lorentzian
		'''
		self.PV = np.zeros(self.tth_list.shape)
		self.PV = self.eta * self.GaussianI + \
				  (1.0 - self.eta) * self.LorentzI

	def calcRwp(self):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/19/2020 SS 1.0 original
		>> @DETAILS:  	this routine computes the weighted error between calculated and
						experimental spectra. goodness of fit is also calculated. the 
						weights are the inverse squareroot of the experimental intensities
		'''

		'''
		the err variable is the difference between simulated and experimental spectra
		'''
		err = (-self.spectrum_sim + self.spectrum_expt)

		''' weighted sum of square '''
		wss =  np.sum(self.weights * err**2)

		den = np.sum(self.weights * self.spectrum_sim**2)

		''' standard Rwp i.e. weighted residual '''
		Rwp = np.sqrt(wss/den)

		''' number of observations to fit i.e. number of data points '''
		N = self.spectrum_sim.shape[0]
		''' number of independent parameters in fitting '''
		P = 13
		Rexp = np.sqrt((N-P)/den)

		# Rwp and goodness of fit parameters
		self.Rwp = Rwp
		self.gofF = Rwp / Rexp

		return Rwp

	def Refine(self):
		'''
		>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
		>> @DATE:     	05/19/2020 SS 1.0 original
		>> @DETAILS:  	this routine performs the least squares refinement for all variables
						which are allowed to be varied.
		'''

		'''
		the err variable is the difference between simulated and experimental spectra
		'''

		x0 = []
		lb = []
		rb = []

		for k in iter(self.P):
			par = self.P[k]
			if(par.vary):
				x0.append(par.value)
				lb.append(par.lb)
				ub.append(par.ub)


		x0 = np.array(x0)
		lb = np.array(lb)
		ub = np.array(ub)
		res = least_squares(self.calcRwp,x0,bounds=(lb,ub))

		print(res.message+'\t'+'[ Exit status '+str(res.status)+' ]')
		# print('\t minimum function value: '+str(res.fun))
		# print('\t iterations: '+str(res.nit))
		# print('\t function evaluations: '+str(res.nfev))
		# print('\t gradient evaluations: '+str(res.njev))
		print('\t optimum values of parameters: '+str(res.x))

		self.initialize_refinedict()
		return res

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
	def Hcag(self):
		return self._Hcag

	@Hcag.setter
	def Hcag(self, val):
		self._Hcag = val

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
	def tth_list(self):
		return self.spectrum_expt._x

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

	def __init__(self, pdatalist, inp_spectrum, 
				 spec_tth_min, spec_tth_max,
				 unitcell,
				 bkgmethod='spline',
				 optmethod='SLSQP', 
				 parms_to_refine={},
				 snipw=8, numiter=2, deg=8,
				 pinkbeam_spec=None):

		'''
			these contain the main refinement parameters for the rietveld code
			if an empty dictionary is passed, then all refine options
			are set to true. otherwise the options to refine are set
			according to the passed values

			each key in the dictionary corresponds to the following list
			parms_to_refine[key] = [True/False, initial guess, lower bound, upper bound]
		'''

		# plane data
		self._pdata 	= pdatalist
		self._unitcell  = unitcell
		self._phasefrac = np.array( [1.0/len(pdatalist)] * len(pdatalist) )
		vol = []
		for uc in self.unitcell:
			vol.append(uc.vol)

		vol = np.array(vol)
		self._phasefrac /= vol

		# if (pinkbeam_spec is None):
		# 	e = 12.39841984 / self.pdata[0].wavelength
		# 	self._pinkbeam_spec = np.atleast_2d(np.array([e, 1.0]))
		# else:
		self._pinkbeam_spec = pinkbeam_spec

		# Cagliotti parameters for the half width of the peak
		self._U = np.radians(1e-2)
		self._V = np.radians(5e-2)
		self._W = np.radians(1e-2)

		# Pseudo-Voight mixing parameters
		self._eta1 = 1.
		self._eta2 = 1.
		self._eta3 = 1.

		# arbitrary scale factor
		self._scalef = 1.0

		self._Pmd = 1.#1.855
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

		self.parms_to_refine = parms_to_refine

		# initialize the spectrum class
		self.Spectrum  = self.Spectrum(inp_spectrum, spec_tth_min, spec_tth_max,
								method=bkgmethod, snipw=snipw, numiter=numiter, 
								deg=deg)

		self.generate_tthlist()

		# pink beam initialization
		self._applypb = False
		if(pinkbeam_spec is not None):
			self.pb = self.PinkBeam(self._pdata, pinkbeam_spec, self.tth_list)
			self._applypb = True

		# initialize simulated spectrum
		self.initialize_spectrum()  

		# initialize the refine class
		self.Optimize = self.Refine(self.refine_dict, 
									method=optmethod)

	def checkangle(ang, name):

		if(np.abs(ang) > 2.0 * np.pi):
			warnings.warn(name + " : the angles seems to be too large. \
							Please check if you've converted to radians.")

	def CagliottiH(self, tth):

		tanth 		= np.tan(0.5*tth)
		self.Hcag 	= np.sqrt(self.U * tanth**2 + self.V * tanth + self.W)

	def MixingFact(self, tth):
		self.eta = self.eta1 + self.eta2 * tth + self.eta3 * (tth*0.5)**2

		if(self.eta > 1.0):
			self.eta = 1.0

		elif(self.eta< 0.0):
			self.eta = 0.0

	def Gaussian(self, tth, L_eff):
		beta = 0.5 * self.Hcag #0.5 * self.Hcag * np.sqrt(np.pi / np.log(2.0)) 
		# + 0.94 * self.pdata[0].get_wavelength() * 0.1 / L_eff / np.cos(tth)

		self.GaussianI = np.exp(-((self.tth_list - tth)/beta)**2 * np.log(2.))
		# self.GaussianI = np.exp(-np.pi * (self.tth_list - tth)**2 / beta**2)
		# self.GaussianI /= np.trapz(self.GaussianI)

	def Lorentzian(self, tth, L_eff):
		w = 0.5 * self.Hcag
		# + 0.94 * self.pdata[0].get_wavelength() * 0.1 / L_eff / np.cos(tth)

		self.LorentzI = 1. / ( 1. + ((self.tth_list - tth)/w)**2)
		# self.LorentzI = (w**2 / (w**2 + (self.tth_list - tth)**2 ) )
		# self.LorentzI /= np.trapz(self.LorentzI)

	def PseudoVoight(self, tth, kernel=None):

		self.PV = np.zeros(self.tth_list.shape)
		if (self.applypb):
			# assert(kernel != None), 'Need to supply kernel for pink beam correction'
			for E, ww in self._pinkbeam_spec:
				tth_new = np.nan_to_num(np.arcsin(np.sin(tth)*self.pb.Emax/E))

			# 	# self.CrystalliteSizeFactor(tth_new, 88.)

				self.CagliottiH(tth_new)
				self.Gaussian(tth_new, 88.0)
				self.Lorentzian(tth_new, 88.0)

				peak_g = self.GaussianI
				peak_l = self.LorentzI

			# peak_g = np.convolve(kernel, self.GaussianI)
			# peak_l = np.convolve(kernel, self.LorentzI)

			# peak_g = signal.resample(peak_g, self.tth_list.shape[0])
			# peak_l = signal.resample(peak_l, self.tth_list.shape[0])

			# self.PV += self.eta * peak_g + \
			# 	(1.0 - self.eta) * peak_l

				self.PV += ww * (self.eta * peak_g + \
					(1.0 - self.eta) * peak_l)

		else:
			self.Gaussian(tth, 88.)
			self.Lorentzian(tth, 88.)
			# self.CrystalliteSizeFactor(tth, 88.)
			self.PV = self.eta * self.GaussianI + \
					(1.0 - self.eta) * self.LorentzI 

	def SpecimenBroadening(self):
		self.SB = np.sinc((self.tth_list - self.tth) / 2.0 / np.pi)

	def initialize_sf(self):

		self.sf = []
		for i, pdata in enumerate(self.pdata):
			hkl = pdata.getHKLs()
			sf 	= np.array([self.unitcell[i].CalcXRSF(g) for g in hkl])
			self.sf.append(sf[self.nanmask])

	def PolarizationFactor(self):

		self.LP = (1 + np.cos(self.tth_list)**2)/ \
		np.cos(0.5*self.tth_list)/np.sin(0.5*self.tth_list)**2

	def CrystalliteSizeFactor(self, tth, L_eff):
		# scherrer constant
		K = 1.0
		invcth = 1.0/np.cos(tth*0.5)
		self.CrystalliteSizeF = K * (self.pdata[0].get_wavelength()*10) * \
								invcth / L_eff
		self.CrystalliteSizeF = self.pb.lambda_mean

	def AbsorptionFactor(self, mu, thickness):
		# abs fact = 1 - exp(-2*mu*t/sin(theta))
		self.A = np.exp(-mu*thickness/np.cos(self.tth_list))

	# first pproximation will be March-Dollase model
	def TextureFactor(self, hkl, hkl_preferred, iphase):

		m = hkl.shape[1]
		self.tex = 0.0
		uc = self.unitcell[iphase]

		l_pref = uc.CalcLength(hkl_preferred, 'r')

		for i in range(m):
			l_g = uc.CalcLength(hkl[:,i], 'r')
			ca = uc.CalcDot(hkl_preferred, hkl[:,i], 'r')/l_pref/l_g
			ca = ca**2
			sa = 1. - ca
			self.tex += (self.Pmd**2 * ca + sa / self.Pmd)**-1.5

		self.tex /= m

	def initialize_Imax(self):

		self.Imax = []
		self.initialize_sf()

		for i,pdata in enumerate(self.pdata):

			m 			= pdata.getMultiplicity() 
			m 			= m.astype(np.float64)
			Imax 		= self.sf[i] * m[self.nanmask]
			self.Imax.append(Imax)

		Imax = np.array(Imax)
		Imax = 100.0 * Imax / np.amax(Imax)

	def initialize_spectrum(self):

		if(self.Spectrum.ndim == 1):
			self.spec_sim 	= np.zeros(self.Spectrum.nspec[0])

		elif(self.Spectrum.ndim == 2):
			self.spec_sim 	= np.zeros(self.Spectrum.nspec[1])


		for i,pdata in enumerate(self.pdata):

			peak_tth 		= pdata.getTTh()
			self.nanmask 	= ~np.isnan(peak_tth)
			peak_tth 		= peak_tth[self.nanmask]

			self.AbsorptionFactor(1./80., 50.)
			self.PolarizationFactor()

			self.initialize_Imax()

			mask 			= (peak_tth < self.Spectrum.spec_tth_max) & \
							  (peak_tth > self.Spectrum.spec_tth_min)
			peak_tth 		= peak_tth[mask]

			Imax 			= self.Imax[i][mask]

			pf 				= self.phasefrac[i]

			hklsym = pdata.getSymHKLs()

			for j in np.arange(peak_tth.shape[0]):

				tth = peak_tth[j]
				I 	= Imax[j]

				self.CagliottiH(tth)
				self.MixingFact(tth) 

				hkl = hklsym[j]
				hkl_preferred = np.array([1.,1.,1.])
				self.TextureFactor(hkl, hkl_preferred, i)

				if(self.applypb): 
					# if new peaks were introduced, then regenerate the kernel list
					if(peak_tth.shape[0] != len(self.pb.kernel[i])):
						self.pb = self.PinkBeam(self._pdata, self._pinkbeam_spec, self.tth_list)

					kernel = self.pb.kernel[i][j]
					self.PseudoVoight(tth, kernel)
					
					peak = np.convolve(kernel,self.PV)
					peak = signal.resample(peak,self.tth_list.shape[0])
					# peak_conv = peak_conv / np.amax(peak_conv)
				else:
					self.PseudoVoight(tth)
					peak = self.PV

				# peak = self.PV
				peak_int = np.trapz(peak)

				'''
				integrated intensity is product of 
				1. scale factor
				2. phase fraction
				3. Fhkl (already scaled by multiplicity)
				4. Absorption
				5. Lorentz polarization
				6. Extinction factor (ignored for now)
				7. Preferred orientation ignored for now)
				...
				...
				and so on
				'''
				if(peak_int != 0.):

					A = self.scalef * pf * I * self.LP * \
					self.A * self.tex / peak_int

					self.spec_sim += A * peak

		self.spec_sim += self.Spectrum.background


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

		ctr = 0
		if(self.refine_dict['lparms'][0]):
			for i, pdata in enumerate(self.pdata):
				lp = x0[np.sum(self.nlp[ctr:i+1]):np.sum(self.nlp[ctr:i+2])]
				ctr += np.sum(self.nlp[ctr:i+2])
				pdata.set_lparms(lp)
				self.initialize_spectrum()

		if(self.refine_dict['scale'][0]):
			self.scalef = x0[ctr]
			ctr += 1

		# this refines debye waller factors for atoms in unit cell
		if(self.refine_dict['B_factor'][0]):
			ndw = 0
			for uc in self.unitcell:
				ndw += uc.ATOM_ntype
			self.DW = x0[ctr:ctr+ndw]
			ctr += ndw
			

		if(self.refine_dict['cagliotti'][0]):
			self.U = x0[ctr]
			self.V = x0[ctr+1]
			self.W = x0[ctr+2]
			ctr += 3

		if(self.refine_dict['eta'][0]):
			self.eta1 = x0[ctr]
			self.eta2 = x0[ctr+1]
			self.eta3 = x0[ctr+2]
			ctr += 3

		if(self.refine_dict['texture'][0]):
			self.Pmd = x0[ctr]
			ctr += 1

		if(self.refine_dict['specimenbroad'][0]):
			pass

		if(self.refine_dict['strain'][0]):
			pass

		err = (-self.spec_sim[self.Spectrum.nzeromask] + \
			self.Spectrum.spec_arr[self.Spectrum.nzeromask])

		# weighted sum of square
		wss =  np.sum(self.Spectrum.weights * err**2)

		den = np.sum(self.Spectrum.weights * 
					self.Spectrum.spec_arr[self.Spectrum.nzeromask]**2)

		# standard Rwp i.e. weighted residual
		Rwp = np.sqrt(wss/den)

		N = self.Spectrum.spec_arr[self.Spectrum.nzeromask].shape[0]
		P = x0.shape[0]
		Rexp = np.sqrt((N-P)/den)

		# Rwp and goodness of fit parameters
		self.Rwp = Rwp
		self.gofF = Rwp / Rexp

		# also update initial values in refine class 
		self.initialize_refinedict()

		return Rwp

	'''
		define the various parameters that need to be refined along
		with their initial values and the lower and upper bounds
	'''
	def initialize_refinedict(self):

		refine_dict = {}

		''' 
			initial values of all the refinable parameters
		'''
		self.nlp = [0]
		for pdata in self.pdata:
			self.nlp.append(len(pdata.get_lparms()))

		self.nlp = np.asarray(self.nlp)

		x0_scale  = np.array([1.0])

		x0_lparms = np.zeros([np.sum(self.nlp),])
		for i, pdata in enumerate(self.pdata):
			x0_lparms[np.sum(self.nlp[0:i+1]):np.sum(self.nlp[0:i+2])] 	= pdata.get_lparms()

		x0_dw = []
		for uc in self.unitcell:
			for i in range(uc.ATOM_ntype):
				x0_dw.append(uc.ATOM_pos[i,4])
		x0_dw = np.array(x0_dw)

		x0_cag 		= np.array([self.U, self.V, self.W]) 
		x0_eta 		= np.array([self.eta1, self.eta2, self.eta3])
		x0_tex 		= np.array([self.Pmd])
		x0_sb 		= np.array([])
		x0_strain 	= np.array([])

		'''
			bounds for all refinable parameters
		'''
		bounds_scale 	= Bounds(np.array([-np.inf]), np.array([np.inf]))
		bounds_cag 		= Bounds(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
		bounds_eta 		= Bounds(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
		bounds_dw 		= Bounds(np.zeros(x0_dw.shape), np.ones(x0_dw.shape))
		bounds_tex 		= Bounds(np.array([0.]), np.array([5.]))

		# lattice parameters in +/- 10% of guess value
		lp_lb 			= 0.9 * x0_lparms 
		lp_ub 			= 1.1 * x0_lparms
		bounds_lparms 	= Bounds(lp_lb, lp_ub)

		bounds_sb 		= Bounds(np.array([]),np.array([]))
		bounds_strain 	= Bounds(np.array([]),np.array([]))


		if (self.parms_to_refine == {}):
			refine_dict = { 'lparms' : [True, x0_lparms, bounds_lparms],
							'scale'  : [True, x0_scale, bounds_scale],
							'B_factor':[True, x0_dw, bounds_dw],
							'cagliotti' : [True, x0_cag, bounds_cag], 
							'eta' : [True, x0_eta, bounds_eta], 
							'texture' : [True, x0_tex, bounds_tex],  
							'specimenbroad' : [True, x0_sb, bounds_sb], 
							'strain' : [True, x0_strain, bounds_strain]
							}
		else:
			refine_dict = { 'lparms' : [False, x0_lparms, bounds_lparms],
							'scale'  : [False, x0_scale, bounds_scale],
							'B_factor':[False, x0_dw, bounds_dw],
							'cagliotti' : [False, x0_cag, bounds_cag], 
							'eta' : [False, x0_eta, bounds_eta], 
							'texture' : [False, x0_tex, bounds_tex], 
							'specimenbroad' : [False, x0_sb, bounds_sb], 
							'strain' : [False, x0_strain, bounds_strain], 
							}
			for key, val in self.parms_to_refine.items():
				refine_dict[key][0] = val

		self.refine_dict = refine_dict
	
	def fit(self):

		# res = minimize(	self.evalResidual, self.Optimize.x0, \
		# 			   	method=self.Optimize.method, \
		# 			   	options=self.Optimize.options, \
		# 			   	bounds = self.Optimize.bounds)

		res = least_squares(self.evalResidual, self.Optimize.x0,\
							bounds = (self.Optimize.bounds.lb,self.Optimize.bounds.ub))

		print(res.message+'\t'+'[ Exit status '+str(res.status)+' ]')
		# print('\t minimum function value: '+str(res.fun))
		# print('\t iterations: '+str(res.nit))
		# print('\t function evaluations: '+str(res.nfev))
		# print('\t gradient evaluations: '+str(res.njev))
		print('\t optimum values of parameters: '+str(res.x))

		self.initialize_refinedict()
		return res

	'''
		all the properties for rietveld class
	'''
	@property
	def scalef(self):
		return self._scalef
	
	@scalef.setter
	def scalef(self, value):
		self._scalef = value
		self.initialize_spectrum()

	@property
	def phasefrac(self):
		return self._phasefrac
	
	@phasefrac.setter
	def phasefrac(self,val):
		assert(val.shape[0] == len(self._pdata)), "incorrect number of entries in phase fraction array"
		self.initialize_spectrum()

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
	def X(self):
		return self._X

	@property
	def Hcag(self):
		return self._Hcag

	@Hcag.setter
	def Hcag(self, val):
		self._Hcag = val

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
		if type(inp_pdata) is list:
			self._pdata = inp_pdata
			self.initialize_spectrum()
		else:
			print('input plane data has to be a list with len == # of phases')

	@property
	def unitcell(self):
		return self._unitcell
	
	@unitcell.setter
	def unitcell(self, inp_unitcell):
		if type(inp_unitcell) is list:
			self._unitcell = inp_unitcell
			self.initialize_spectrum()
		else:
			print('input unitcell has to be a list with len == # of phases')			

	@property
	def DW(self):
		B = []
		for uc in self.unitcell:
			B.append(self.unitcell.ATOM_pos[:,4])
		return B
	
	@DW.setter
	def DW(self, val):
		sz = 0
		for uc in self.unitcell:
			sz += uc.ATOM_ntype
		assert(val.shape[0] == sz), 'size of input in DW factor is not correct'
		ctr = 0
		for uc in self.unitcell:
			n = uc.ATOM_ntype
			uc.ATOM_pos[0:n,4] = val[ctr:ctr+n]
			ctr += n

		self.initialize_spectrum()

			# uc.ATOM_pos[0:n,5] = val[]

	@property
	def parms_to_refine(self):
		return self._parms_to_refine
	
	@parms_to_refine.setter
	def parms_to_refine(self,val):
		assert(type(val) == dict), 'parms_to_refine should be a dictionary'
		self._parms_to_refine = val
		self.initialize_refinedict() 
		self.Optimize = self.Refine(self.refine_dict)

	@property
	def refine_dict(self):
		return self._refine_dict
	
	@refine_dict.setter
	def refine_dict(self, val):
		assert(type(val) == dict), 'parms_to_refine should be a dictionary'
		self._refine_dict = val
		self.Optimize = self.Refine(self.refine_dict)

	@property
	def applypb(self):
		return self._applypb

	@applypb.setter
	def applypb(self,val):
		self._applypb = True
		self.initialize_spectrum()

	@property
	def Rwp(self):
		return self._Rwp

	@Rwp.setter
	def Rwp(self, value):
		self._Rwp = value
	
	@property
	def gofF(self):
		return self._gofF

	@gofF.setter
	def gofF(self, value):
		self._gofF = value
	
	# March-Dollase parameter
	@property
	def Pmd(self):
		return self._Pmd

	@Pmd.setter
	def Pmd(self, val):
		self._Pmd = val
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

			# return residual as percent
			# ignore points less tha 0.001
			eps = 1e-3
			self.nzeromask = self.spec_arr > eps
			self.weights = 1.0 / np.sqrt(self.spec_arr[self.nzeromask])

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
				x = np.radians(self.points[:,0])
				y = self.points[:,1]
				self.splinefit(x, y)
				plt.plot(np.degrees(self.tth_list), self.background, '--r')

			elif (self.method.lower() == 'poly'):
				self.background = polyfit(spec_arr, deg=self.deg)

			elif (self.method.lower() == 'cheb'):
				self.background = chebfit(spec_arr, deg=self.deg)

		def remove_bkg(self):

			self._spec_arr_nbkg = self.spec_arr - self.background
			# self._spec_arr_nbkg = self._spec_arr_nbkg / np.amax(np.abs(self._spec_arr_nbkg))

		def selectpoints(self):

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_title('Select 5 points for background estimation')

			line, = ax.plot(np.degrees(self.tth_list), self.spec_arr, '-b', picker=5)  # 5 points tolerance
			plt.show()

			self.points = np.asarray(plt.ginput(5,timeout=-1, show_clicks=True))			
			# plt.close()

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
		def __init__(self, refine_dict, method='SLSQP', ftol=1e-9, disp=False):

			'''
				initialize the class
			'''
			self.method 		= method
			self.ftol 			= 1e-9
			self.disp 			= False
			self.options 		= {'ftol':self.ftol, \
								   'disp':self.disp}
			self.refine_dict	= refine_dict 						

		def init_vals_bounds(self):

			'''
				set initial values and bounds for the different refinement parameters
			'''
			x0 = []
			lb = []
			ub = []

			for key, val in self.refine_dict.items():
				if(val[0]):
					x0 = x0 + list(val[1])
					lb = lb + list(val[2].lb)
					ub = ub + list(val[2].ub)

			bounds 				= Bounds(lb, ub)
			self.x0 			= np.array(x0)
			self.bounds 		= bounds

		@property
		def refine_dict(self):
			return self._refine_dict
		
		@refine_dict.setter
		def refine_dict(self, val):
			self._refine_dict = val
			self.init_vals_bounds()

	''' ======================================================================================================== 
		========================================================================================================

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	01/08/2020 SS 1.0 original
	>> @DETAILS:  	this is the class which deals with the pink beams present at DCS.
					input is the planedata and energy spectrum of the incident xray

		======================================================================================================== 
		========================================================================================================
	'''

	class PinkBeam:
		'''
			set up the pink beam class using following steps:

			1. calculate the distribution of two theta due to energy
			2. convolve the calculated distribution with the monochromatic
			   spectrum
			3. limit results between tth_min and tth_max
		'''
		def __init__(self, pdata, pinkbeam_spec, tth_list):

			self.hc = 12.39841984 # units of kev A
			self._pbspec = pinkbeam_spec
			self._tth_list = tth_list

			self.E  = self._pbspec[:,0]
			self.ww = self._pbspec[:,1]/np.trapz(self._pbspec[:,1])
			
			self.Emean = np.average(self.E, weights=self.ww)
			self.lambda_mean = self.hc / self.Emean

			self.kernel = []
			for planedata in pdata:
				self._tth = planedata.getTTh()
				self._tth_min = tth_list[0]
				self._tth_max = tth_list[-1]
				self._nspec = tth_list.shape[0]
				self.Emax = self.hc / planedata.wavelength
				self.E  = self.E[: : -1]
				self.ww = self.ww[: : -1]

				self.kernel.append(self.generate_flux_kernel())

		def generate_flux_kernel(self):

			kernel = []
			for ttheta in self._tth:
				if((ttheta > self._tth_min) and (ttheta < self._tth_max)):
					theta = np.nan_to_num(np.arcsin(np.sin(ttheta)*self.Emax/self.E))

					idx 	= np.intersect1d( np.where(theta > self._tth_min), 
							np.where(theta < self._tth_max) )

					theta 	= theta[idx]
					weights = self.ww[idx]

					if(theta.shape[0] != 0):
						cs 		= CubicSpline(theta, weights, bc_type='natural',extrapolate=False)
					else :
						cs = None
						print('No theta in specified range. returning None')

					kernel.append(np.nan_to_num(cs(self._tth_list)))

			return kernel	