import numpy as np
import warnings
from hexrd.imageutil import snip1d
from scipy.optimize import minimize, Bounds, shgo, least_squares
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from hexrd.valunits import valWUnit
from scipy import signal
import yaml

class Parameters:
	''' ======================================================================================================== 
	======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	01/08/2020 SS 1.0 original
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

	def __str__(self):
		retstr = 'Parameters{\n'
		for k in self.param_dict.keys():
			retstr += self[k].__str__()+'\n'

		retstr += '}'
		return retstr

class Parameter:

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
	
class LeBail:
		''' ======================================================================================================== 
		======================================================================================================== 

	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	01/08/2020 SS 1.0 original
    >> @DETAILS:  	this is the main LeBail class and contains all the refinable parameters
    				for the analysis. the member classes are as follows (in order of initialization):

    				1. Spectrum  		contains the experimental spectrum
    				2. Background  		contains the background extracted from spectrum
    				3. Refine 			contains all the machinery for refinement
    	======================================================================================================== 
		======================================================================================================== 
	'''