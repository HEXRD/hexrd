import numpy as np
import warnings
from hexrd.imageutil import snip1d
from hexrd.crystallography import PlaneData
from scipy.optimize import minimize, Bounds, shgo, least_squares
import matplotlib.pyplot as plt
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
	