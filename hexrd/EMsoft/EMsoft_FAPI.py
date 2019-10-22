#! /usr/bin/env python
# ============================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Saransh Singh <saransh1@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# Redistribution and use in source and binary forms, with or without modification, are 
# permitted provided that the following conditions are met:
#
# -  Redistributions of source code must retain the above copyright notice, this list 
#    of conditions and the following disclaimer.
# -  Redistributions in binary form must reproduce the above copyright notice, this 
#    list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
# -  Neither the names of Marc De Graef, Carnegie Mellon University nor the names 
#    of its contributors may be used to endorse or promote products derived from 
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import numpy as np
import sys
import warnings

# these are the modules that we shall write the API for
from EMsoft import constants
from EMsoft import rotations
from EMsoft import symmetry
from EMsoft import crystal
from EMsoft import quaternions
# from EMsoft import so3

class EMsoft_constants:

	def __init__(self):

		# some constants for atoms in periodic table
		self.atom_color 	= constants.atom_color
		self.atom_colors 	= constants.atom_colors
		self.atom_mtradii 	= constants.atom_mtradii
		self.atom_spradii 	= constants.atom_spradii
		self.atom_sym 		= constants.atom_sym
		self.atom_weights 	= constants.atom_weights

		# physical constants
		self.cAvogadro 		= constants.cAvogadro
		self.cBoltzmann 	= constants.cBoltzmann
		self.cCharge 		= constants.cCharge
		self.cJ2eV 			= constants.cJ2eV
		self.cLight 		= constants.cLight
		self.cMoment 		= constants.cMoment
		self.cPermea 		= constants.cPermea
		self.cPermit 		= constants.cPermit
		self.cPlanck 		= constants.cPlanck
		self.cRestmass 		= constants.cRestmass

		# math constants
		self.cPi 			= constants.cPi
		self.epsijk 		= constants.epsijk
		self.epsijkd 		= constants.epsijkd

		# some constants for fundamental zone
		self.fzoarray 		= constants.fzoarray
		self.fztarray 		= constants.fztarray
		self.icovertices 	= constants.icovertices

		# lambert constant type
		self.LambertParameters = constants.LambertParametersType()


class quaternions:

	def __init__(self, qu_in):
		self.qu = np.array(qu_in, dtype = np.float64)

	''' operation overload for the quaternion class

	defining add, subtract, multiplication and division of quaternions.

	--> addition and subtraction are like vector addition or subtraction
	--> multiplication represents a composition of the two quaternions
	--> division (q1/q2) represents q1 * inverseq2) = q1 * conjugate(q2)/norm(q2).
		as norm(q2) = 1, it is simply q1 * conjugate(q2)

	'''
	def __add__(self, qu_add):
		return(quaternions(self.qu+qu_add.qu))

	def __sub__(self, qu_add):
		return(quaternions(self.qu-qu_add.qu))

	def __truediv__(self, qu_div):
		return quaternions(quaternions.quat_div_d(self.qu,qu_div.qu))

	def __mul__(self, qu_mul):
		return quaternions(quaternions.quat_mult_d(self.qu,qu_mul.qu))

	''' define quaternion norm, normalization, conjugate,
	quaternion-quaternion and quaternion-vector operations'''

	# evaluate L2 norm of quaternion
	def norm(self):
		return quaternions.quat_norm_d(self.qu)

	# normalize the quaternion to have unit L2 norm
	def normalize(self):
		self.qu = self.qu/self.norm()

	# conjugate of a quaternion
	def conjugate(self):
		return quaternions(quaternions.quat_conjg_d(self.qu))

	# innerproduct of quaternion
	def innerprod(self, qu_inn):
		return quaternions.quat_innerproduct_d(self.qu, qu_inn.qu)

	# angle between two quaternions
	def angle(self, qu_ang):
		return quaternions.quat_angle_d(self.qu, qu_ang.qu)

	# vector rotation by quaternion
	def vecrot(self, v):
		return quaternions.quat_lp_d(self.qu, v)

	''' quaternion slerp i.e. smooth interpolation from one quaternion to
		another. the formula is given as follows:
		slerp = (sin((1-t)*omg)/sin(omg)) * qu1 + (sin(t*omg)/sin(omg)) * qu2

		--> t   = interpolation parameter t in range [0,1]
		--> omg = angle between qu1 and qu2 i.e. cos(omg) = innerproduct(qu1,qu2)
		--> qu1, qu2 are the two quaternions between which the interpolation is performed
	'''
	# n is the number of interpolation points we want
	def slerp(self, qu_slp, n):
		return quaternions(quaternions.quat_slerp_d(self.qu, qu_slp.qu, n))

class Typedefs:
	'''
	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	10/22/2019 SS 1.0 original
    >> @DETAILS:  	all the necessary interface for typedefs for the fortran library

	'''

	def __init__(self):
		pass

class Symmetry:
	'''
	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:     	10/09/2018 SS 1.0 original
       @DATE:		10/15/2018 SS 1.1 added space group handling
    >> @DETAILS:  	this is the symmetry class and handels all the "ROTATIONAL, POINT/SPACE GROUP"
	    		  	symmetries for any crystal.
	'''


	''' initialize all the symmetry matrices for the space group'''
	def __init__(self, cell):  #cell argument removed for now
		pass

		# self.SYM_PGnum = symmetry.sg2pg(cell.SGnum)

		# [self.FZtype, self.FZorder] = so3.getfztypeandorder(self.SYM_PGnum)

		# [self.Nqsym, self.Pm] = sym.generaterotationalsymmetry(self.SYM_PGnum)

		# [self.SYM_name, self.SYM_GENnum, self.centrosym, self.SYM_data, self.SYM_direc, \
		#  self.SYM_recip, self.SYM_NUMpt, self.SYM_MATnum, self.nonsymmorphic] 		= 	\
		# sym.generatesymmetry(cell.SGnum, 1, cell.dmt, cell.rmt) 

class rotation:

	''' 
	>> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:       02/27/2019 SS 1.0 original
	>> @DETAILS:    wrappers for all the rotation conversion routines. 
	the name convetion is *xx2yy* where xx is the original representation and
	yy is the desired representation. there are 8 different possibilities for 
	xx/yy resulting in 56 different conversion routines. all angles are in radians.
	legend as follows:

	eu = euler angle in Bunge (ZXZ) notations
	ro = rodrigues vector
	qu = quaternion
	cu = cubochoric coordinates
	ax = axis-angle pair
	om = orientation matrix
	ho = homochoric vector
	st = stereographic vector 
	'''
	def __init__(self):
		self.tol = 1.0e-6

	def check_eu(self, eu):

		eu = np.atleast_2d(eu)

		assert(eu.ndim == 2), 'dimension of array should be 2-d (n x 3).'
		assert(eu.shape[1] == 3), 'second dimension should be 3.'

		if(np.any(np.abs(eu) > 2.0*np.pi)):
			warnings.warn("The angles seems to be too large. Please check if you've converted to radians.")



	def check_ro(self, ro):

		ro = np.atleast_2d(ro)

		assert(ro.ndim == 2), 'dimension of array should be 2-d (n x 4).'
		assert(ro.shape[1] == 4), 'second dimension should be 4.'
		assert( np.all(np.linalg.norm(ro[:,0:3], axis=1) == 1) ), 'the unit vectors are not normal'

	def check_qu(self, qu):

		qu = np.atleast_2d(qu)

		assert(qu.ndim == 2), 'dimension of array should be 2-d (n x 4).'
		assert(qu.shape[1] == 4), 'second dimension should be 4.'
		assert( np.all(np.linalg.norm(qu, axis=1) == 1 ) ), 'the unit vectors are not normal'

	def check_cu(self, cu):

		cu = np.atleast_2d(cu)

		assert(cu.ndim == 2), 'dimension of array should be 2-d (n x 3).'
		assert(cu.shape[1] == 3), 'second dimension should be 3.'
		assert( np.all(np.abs(cu) < 0.5 * np.pi**(2./3.) ) ), 'coordinates outside of cubochoric box.'

	def check_ax(self, ax):

		ax = np.atleast_2d(ax)

		assert(ax.ndim == 2), 'dimension of array should be 2-d (n x 4).'
		assert(ax.shape[1] == 4), 'second dimension should be 4.'
		assert(np.all(np.linalg.norm(ax[:,0:3], axis=1) == 1 ) ), 'the unit vectors are not normal'
		if(np.any(np.abs(ax[:,3]) > 2.0*np.pi)):
			warnings.warn("angle seems to be too large. Please check if you've converted to radians.")

	def check_om(self, om):

		if(om.ndim == 2):
			om = np.atleast_3d(om).T
		else:
			om = np.atleast_3d(om)

		assert(om.ndim == 3), 'dimension of array should be 3-d (n x 3 x 3).'
		assert(om.shape[1:3] == (3,3) ), 'first two-dimensions should be 3 x 3.'

		diff = np.asarray([ np.sum( np.abs(np.linalg.inv(o) - o.T) ) for o in om ])
		assert(np.all(diff) < self.tol), "orientation matrices are not orthonormal"

	def check_ho(self, ho):

		ho = np.atleast_2d(ho)

		assert(ho.ndim == 2), 'dimension of array should be 2-d (n x 3).'
		assert(ho.shape[1] == 3), 'second dimension should be 3.'
		assert(np.any(np.linalg.norm(ho, axis=1) <= (0.75 * np.pi)**(1./3.) ) ), 'coordinates needs to be inside homochoric ball'

	def check_st(self, st):

		st = np.atleast_2d(st)

		assert(st.ndim == 2), 'dimension of array should be 2-d (n x 3).'
		assert(st.shape[1] == 3), 'second dimension should be 3.'
		assert(np.any(np.linalg.norm(st, axis=1) <= 1.0 ) ), 'coordinates needs to be inside unit ball'

	# 1. euler to orientation matrix
	def eu2om(self, eu):

		self.check_eu(eu)
		om = rotations.eu2om_d(eu)
		return om

	# 2. euler to axis-angle pair
	def eu2ax(self, eu):

		self.check_eu(eu)
		ax = rotations.eu2ax_d(eu)
		return ax

	# 3. euler to rodrigues vector
	def eu2ro(self, eu):
		self.check_eu(eu)
		ro = rotations.eu2ro_d(eu)
		return ro

	# 4.
	def eu2qu(self, eu):
		self.check_eu(eu)
		qu = rotations.eu2qu_d(eu)
		return qu

	# 5. orientation matrix to euler angles
	def om2eu(self, om):
		self.check_om(om)
		eu = rotations.om2eu_d(om)
		return eu

	# 6.
	def ax2om(self, ax):
		self.check_ax(ax)
		om = rotations.ax2om_d(ax)
		return om

	# 7.
	def qu2eu(self, qu):
		self.check_qu(qu)
		eu = rotations.qu2eu_d(qu)
		return eu

	# 8.
	def ax2ho(self, ax):
		self.check_ax(ax)
		ho = rotations.ax2ho_d(ax)
		return ho

	# 9.
	def ho2ax(self, ho):
		self.check_ho(ho)
		ax = rotations.ho2ax_d(ho)
		return ax

	# 10.
	def om2ax(self, om):
		self.check_om(om)
		ax = rotations.om2ax_d(om)
		return ax

	# 11.
	def ro2ax(self, ro):
		self.check_ro(ro)
		ax = rotations.ro2ax_d(ro)
		return ax

	# 12.
	def ax2ro(self, ax):
		self.check_ax(ax)
		ro = rotations.ax2ro_d(ax)
		return ro

	# 13.
	def ax2qu(self, ax):
		self.check_ax(ax)
		qu = rotations.ax2qu_d(ax)
		return qu

	# 14.
	def  ro2ho(self, ro):
		self.check_ro(ro)
		ho = rotations. ro2ho_d(ro)
		return ho

	# 15.
	def qu2om(self, qu):
		self.check_qu(qu)
		om = rotations.qu2om_d(qu)
		return om

	# 16. 
	def om2qu(self, om):
		self.check_om(om)
		qu = rotations.om2qu_d(om)
		return qu

	# 17.
	def qu2ax(self, qu):
		self.check_qu(qu)
		ax = rotations.qu2ax_d(qu)
		return qu

	# 18.
	def qu2ro(self, qu):
		self.check_qu(qu)
		ro = rotations.qu2ro_d(qu)
		return ro

	# 19.
	def qu2ho(self, qu):
		self.check_qu(qu)
		ho = rotations.qu2ho_d(qu)
		return ho

	# 20.
	def ho2cu(self, ho):
		self.check_ho(ho)
		cu = rotations.ho2cu_d(ho)
		return cu

	# 21.
	def cu2ho(self, cu):
		self.check_cu(cu)
		ho = rotations.cu2ho_d(cu)
		return ho

	# 22.
	def ro2eu(self, ro):
		self.check_ro(ro)
		eu = rotations.ro2eu_d(ro)
		return eu

	# 23.
	def eu2ho(self, eu):
		self.check_eu(eu)
		ho = rotations.eu2ho_d(eu)
		return ho

	# 24.
	def om2ro(self, om):
		self.check_om(om)
		ro = rotations.om2ro_d(om)
		return ro

	# 25.
	def om2ho(self, om):
		self.check_om(om)
		ho = rotations.om2ho_d(om)
		return ho

	# 26.
	def ax2eu(self, ax):
		self.check_ax(ax)
		eu = rotations.ax2eu_d(ax)
		return eu

	# 27.
	def ro2om(self, ro):
		self.check_ro(ro)
		om = rotations.ro2om_d(ro)
		return om

	# 28.
	def ro2qu(self, ro):
		self.check_ro(ro)
		qu = rotations.ro2qu_d(ro)
		return qu

	# 29.
	def ho2eu(self, ho):
		self.check_ho(ho)
		eu = rotations.ho2eu_d(ho)
		return eu

	# 30.
	def ho2om(self, ho):
		self.check_ho(ho)
		om = rotations.ho2om_d(ho)
		return om

	# 31.
	def ho2ro(self, ho):
		self.check_ho(ho)
		ro = rotations.ho2ro_d(ho)
		return ro

	# 32.
	def ho2qu(self, ho):
		self.check_ho(ho)
		qu = rotations.ho2qu_d(ho)
		return qu

	# 33.
	def eu2cu(self, eu):
		self.check_eu(eu)
		cu = rotations.eu2cu_d(eu)
		return cu

	# 34.
	def om2cu(self, om):
		self.check_om(om)
		cu = rotations.om2cu_d(om)
		return cu

	# 35.
	def ax2cu(self, ax):
		self.check_ax(ax)
		cu = rotations.ax2cu_d(ax)
		return cu

	# 36.
	def ro2cu(self, ro):
		self.check_ro(ro)
		cu = rotations.ro2cu_d(ro)
		return cu

	# 37.
	def qu2cu(self, qu):
		self.check_cu(cu)
		cu = rotations.qu2cu_d(qu)
		return cu

	# 38.
	def cu2eu(self, cu):
		self.check_cu(cu)
		eu = rotations.cu2eu_d(cu)
		return eu

	# 39.
	def cu2om(self, cu):
		self.check_cu(cu)
		om = rotations.cu2om_d(cu)
		return om

	# 40.
	def cu2ax(self, cu):
		self.check_cu(cu)
		ax = rotations.cu2ax_d(cu)
		return ax

	# 41.
	def cu2ro(self, cu):
		self.check_cu(cu)
		ro = rotations.cu2ro_d(cu)
		return ro

	# 42.
	def cu2qu(self, cu):
		self.check_cu(cu)
		qu = rotations.cu2qu_d(cu)
		return qu

	# 43.
	def om2st(self, om):
		self.check_om(om)
		st = rotations.om2st_d(om)
		return st

	# 44.
	def ax2st_d(self, ax):
		self.check_ax(ax)
		st = rotations.ax2st_d(ax)
		return st

	# 45.
	def ro2st_d(self, ro):
		self.check_ro(ro)
		st = rotations.ro2st_d(ro)
		return st

	# 46.
	def ho2st_d(self, ho):
		self.check_ho(ho)
		st = rotations.ho2st_d(ho)
		return st

	# 47.
	def cu2st_d(self, cu):
		self.check_cu(cu)
		st = rotations.cu2st_d(cu)
		return st

	# 48.
	def eu2st_d(self, eu):
		self.check_eu(eu)
		st = rotations.eu2st_d(eu)
		return st

	# 49.
	def qu2st_d(self, qu):
		self.check_qu(qu)
		st = rotations.qu2st_d(qu)
		return st

	# 50.
	def st2om_d(self, st):
		self.check_st(st)
		om = rotations.st2om_d(st)
		return om

	# 51.
	def st2eu_d(self, st):
		self.check_st(st)
		eu = rotations.st2eu_d(st)
		return eu

	# 52.
	def st2qu_d(self, st):
		self.check_st(st)
		qu = rotations.st2qu_d(st)
		return qu

	# 53.
	def st2ax_d(self, st):
		self.check_st(st)
		ax = rotations.st2ax_d(st)
		return ax

	# 54.
	def st2ro_d(self, st):
		self.check_st(st)
		ro = rotations.st2ro_d(st)
		return ro

	# 55.
	def st2ho_d(self, st):
		self.check_st(st)
		ho = rotations.st2ho_d(st)
		return ho

	# 56.
	def st2cu_d(self, st):
		self.check_st(st)
		cu = rotations.st2cu_d(st)
		return cu

class crystal:
	'''
	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	10/09/2018 SS 1.0 original
	   @DATE:		10/15/2018 SS 1.1 added space group handling
	>> @DETAILS:  	this is the crystal class 

	'''

	def __init__(self):
		pass

