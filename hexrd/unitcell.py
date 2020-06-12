import numpy as np
from hexrd import constants
from hexrd import symmetry
import warnings
import h5py
from pathlib import Path
from scipy.interpolate import interp1d

class unitcell:
	'''
	>> @AUTHOR:   	Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	>> @DATE:     	10/09/2018 SS 1.0 original
	   @DATE:		10/15/2018 SS 1.1 added space group handling
	>> @DETAILS:  	this is the unitcell class 

	'''

	# initialize using the EMsoft unitcell type
	# need lattice parameters and space group data from HDF5 file
	def __init__(self, lp, sgnum, atomtypes, atominfo, dmin, beamenergy, sgsetting=0):

		self.pref  = 0.4178214

		self.atom_ntype = atomtypes.shape[0]
		self.atom_type  = atomtypes 
		self.atom_pos   = atominfo
		self.dmin 		= dmin

		# set some default values for 
		# a,b,c,alpha,beta,gamma
		self._a = 1.
		self._b = 1.
		self._c = 1.
		self._alpha = 90.
		self._beta = 90.
		self._gamma = 90.

		if(lp[0].unit == 'angstrom'):
			self.a = lp[0].value * 0.1
		elif(lp[0].unit == 'nm'):
			self.a = lp[0].value
		else:
			raise ValueError('unknown unit in lattice parameter')

		if(lp[1].unit == 'angstrom'):
			self.b = lp[1].value * 0.1
		elif(lp[1].unit == 'nm'):
			self.b = lp[1].value
		else:
			raise ValueError('unknown unit in lattice parameter')

		if(lp[2].unit == 'angstrom'):
			self.c = lp[2].value * 0.1
		elif(lp[2].unit == 'nm'):
			self.c = lp[2].value
		else:
			raise ValueError('unknown unit in lattice parameter')

		if(lp[3].unit == 'degrees'):
			self.alpha = lp[3].value
		elif(lp[3].unit == 'radians'):
			self.alpha = np.degrees(lp[3].value)

		if(lp[4].unit == 'degrees'):
			self.beta = lp[4].value
		elif(lp[4].unit == 'radians'):
			self.beta = np.degrees(lp[4].value)

		if(lp[5].unit == 'degrees'):
			self.gamma = lp[5].value
		elif(lp[5].unit == 'radians'):
			self.gamma = np.degrees(lp[5].value) 

		'''
		initialize interpolation from table for anomalous scattering
		'''
		self.InitializeInterpTable()

		'''
		sets x-ray energy
		calculate wavelength 
		also calculates anomalous form factors for xray scattering
		'''
		self.voltage = beamenergy * 1000.0
		'''
		calculate symmetry
		'''
		self.sgsetting = sgsetting
		self.sgnum = sgnum

		'''
		asymmetric positions due to space group symmetry
		used for structure factor calculations
		'''
		self.CalcPositions()

	def CalcWavelength(self):
		# wavelength in nm
		self.wavelength = 		constants.cPlanck * \
							constants.cLight /  \
							constants.cCharge / \
							self.voltage
		self.wavelength *= 1e9
		self.CalcAnomalous()

	def calcmatrices(self):

		a = self.a
		b = self.b
		c = self.c

		alpha = np.radians(self.alpha)
		beta  = np.radians(self.beta)
		gamma = np.radians(self.gamma)

		ca = np.cos(alpha);
		cb = np.cos(beta);
		cg = np.cos(gamma);
		sa = np.sin(alpha);
		sb = np.sin(beta);
		sg = np.sin(gamma);
		tg = np.tan(gamma);

		'''
			direct metric tensor
		'''
		self._dmt = np.array([[a**2, a*b*cg, a*c*cb],\
							 [a*b*cg, b**2, b*c*ca],\
							 [a*c*cb, b*c*ca, c**2]])
		self._vol = np.sqrt(np.linalg.det(self.dmt))

		if(self.vol < 1e-5):
			warnings.warn('unitcell volume is suspiciously small')

		'''
			reciprocal metric tensor
		'''
		self._rmt = np.linalg.inv(self.dmt)

		'''
			direct structure matrix
		'''
		self._dsm = np.array([[a, b*cg, c*cb],\
							 [0., b*sg, -c*(cb*cg - ca)/sg],
							 [0., 0., self.vol/(a*b*sg)]])

		'''
			reciprocal structure matrix
		'''
		self._rsm = np.array([[1./a, 0., 0.],\
							 [-1./(a*tg), 1./(b*sg), 0.],
							 [b*c*(cg*ca - cb)/(self.vol*sg), a*c*(cb*cg - ca)/(self.vol*sg), a*b*sg/self.vol]])

	''' transform between any crystal space to any other space.
 		choices are 'd' (direct), 'r' (reciprocal) and 'c' (cartesian)'''
	def TransSpace(self, v_in, inspace, outspace):
		if(inspace == 'd'):
			if(outspace == 'r'):
				v_out = np.dot(v_in, self.dmt)
			elif(outspace == 'c'):
				v_out = np.dot(self.dsm, v_in)
			else:
				raise ValueError('inspace in ''d'' but outspace can''t be identified')

		elif(inspace == 'r'):
			if(outspace == 'd'):
				v_out = np.dot(v_in, self.rmt)
			elif(outspace == 'c'):
				v_out = np.dot(self.rsm, v_in)
			else:
				raise ValueError('inspace in ''r'' but outspace can''t be identified')

		elif(inspace == 'c'):
			if(outspace == 'r'):
				v_out = np.dot(v_in, self.rsm)
			elif(outspace == 'd'):
				v_out = np.dot(v_in, self.dsm)
			else:
				raise ValueError('inspace in ''c'' but outspace can''t be identified')

		else:
			raise ValueError('incorrect inspace argument')

		return v_out

	''' calculate dot product of two vectors in any space 'd' 'r' or 'c' '''
	def CalcDot(self, u, v, space):

		if(space == 'd'):
			dot = np.dot(u,np.dot(self.dmt,v))
		elif(space == 'r'):
			dot = np.dot(u,np.dot(self.rmt,v))
		elif(space == 'c'):
			dot = np.dot(u,v)
		else:
			raise ValueError('space is unidentified')

		return dot

	''' calculate dot product of two vectors in any space 'd' 'r' or 'c' '''
	def CalcLength(self, u, space):

		if(space =='d'):
			vlen = np.sqrt(np.dot(u, np.dot(self.dmt, u)))
		elif(space =='r'):
			vlen = np.sqrt(np.dot(u, np.dot(self.rmt, u)))
		elif(spec =='c'):
			vlen = np.linalg.norm(u)
		else:
			raise ValueError('incorrect space argument')

		return vlen

	''' normalize vector in any space 'd' 'r' or 'c' '''
	def NormVec(self, u, space):
		ulen = self.CalcLength(u, space)
		return u/ulen

	''' calculate angle between two vectors in any space'''
	def CalcAngle(self, u, v, space):

		ulen = self.CalcLength(u, space)
		vlen = self.CalcLength(v, space)

		dot  = self.CalcDot(u, v, space)/ulen/vlen
		angle = np.arccos(dot)

		return angle

	''' calculate cross product between two vectors in any space.
	
	cross product of two vectors in direct space is a vector in 
	reciprocal space
	
	cross product of two vectors in reciprocal space is a vector in 
	direct space
	
	the outspace specifies if a conversion needs to be made 

	 @NOTE: iv is the switch (0/1) which will either turn division 
	 by volume of the unit cell on or off.'''
	def CalcCross(self, p, q, inspace, outspace, vol_divide=False):
		iv = 0
		if(vol_divide):
			vol = self.vol
		else:
			vol = 1.0

		pxq = np.array([p[1]*q[2]-p[2]*q[1],\
						p[2]*q[0]-p[0]*q[2],\
						p[0]*q[1]-p[1]*q[0]])

		if(inspace == 'd'):
			'''
			cross product vector is in reciprocal space
			and can be converted to direct or cartesian space
			'''
			pxq *= vol

			if(outspace == 'r'):
				pass
			elif(outspace == 'd'):
				pxq = self.TransSpace(pxq, 'r', 'd')
			elif(outspace == 'c'):
				pxq = self.TransSpace(pxq, 'r', 'c')
			else:
				raise ValueError('inspace is ''d'' but outspace is unidentified')

		elif(inspace == 'r'):
			'''
			cross product vector is in direct space and 
			can be converted to any other space
			'''
			pxq /= vol
			if(outspace == 'r'):
				pxq = self.TransSpace(pxq, 'd', 'r')
			elif(outspace == 'd'):
				pass
			elif(outspace == 'c'):
				pxq = self.TransSpace(pxq, 'd', 'c')
			else:
				raise ValueError('inspace is ''r'' but outspace is unidentified')

		elif(inspace == 'c'):
			'''
			cross product is already in cartesian space so no 
			volume factor is involved. can be converted to any
			other space too
			'''
			if(outspace == 'r'):
				pxq = self.TransSpace(pxq, 'c', 'r')
			elif(outspace == 'd'):
				pxq = self.TransSpace(pxq, 'c', 'd')
			elif(outspace == 'c'):
				pass
			else:
				raise ValueError('inspace is ''c'' but outspace is unidentified')

		else:
			raise ValueError('inspace is unidentified')

		return pxq

	def GenerateRecipPGSym(self):

		self.SYM_PG_r = self.SYM_PG_d[0,:,:]
		self.SYM_PG_r = np.broadcast_to(self.SYM_PG_r,[1,3,3])

		for i in range(1,self.npgsym):
			g = self.SYM_PG_d[i,:,:]
			g = np.dot(self.dmt,np.dot(g,self.rmt))
			g = np.broadcast_to(g,[1,3,3])
			self.SYM_PG_r = np.concatenate((self.SYM_PG_r,g))

	def CalcOrbit(self):
		'''
		calculate the equivalent position for the space group
		symmetry
		'''
		pass

	def CalcPositions(self):
		'''
		calculate the asymmetric positions in the fundamental unitcell
		used for structure factor calculations
		'''
		numat = []
		asym_pos = []

		# using the wigner-seitz notation
		for i in range(self.atom_ntype):

			n = 1
			r = self.atom_pos[i,0:3]
			r = np.hstack((r, 1.))

			asym_pos.append(np.broadcast_to(r[0:3],[1,3]))
			
			for symmat in self.SYM_SG:
				# get new position
				rnew = np.dot(symmat, r)

				# reduce to fundamental unitcell with fractional
				# coordinates between 0-1
				rr = rnew[0:3]
				rr = np.modf(rr)[0]
				rr[rr < 0.] += 1.
				rr[np.abs(rr) < 1.0E-6] = 0.

				# check if this is new
				isnew = True
				for j in range(n):
					if(np.sum(np.abs(rr - asym_pos[i][j,:])) < 1E-4):
						isnew = False
						break

				# if its new add this to the list
				if(isnew):
					asym_pos[i] = np.vstack((asym_pos[i],rr))
					n += 1

			numat.append(n)

		self.numat = np.array(numat)
		self.asym_pos = asym_pos

	def CalcDensity(self):
		''' 
		calculate density, average atomic weight (avA) 
		and average atomic number(avZ)
		'''
		self.avA = 0.0
		self.avZ = 0.0

		for i in range(self.atom_ntype):
			'''
			atype is atom type i.e. atomic number
			numat is the number of atoms of atype
			atom_pos(i,4) has the occupation factor
			'''
			atype = self.atom_type[i]
			numat = self.numat[i]
			occ   = self.atom_pos[i,3]
			avA  += numat * constants.atom_weights[atype-1] * occ # -1 due to 0 indexing in python
			avZ  += numat * atype


		self.density = avA / (self.vol * 1.0E-21 * constants.cAvogadro)
		
		av_natom = np.dot(self.numat, self.atom_pos[:,3])

		self.avA = avA / av_natom
		self.avZ = avZ / np.sum(self.numat)

	''' calculate the maximum index of diffraction vector along each of the three reciprocal
	 basis vectors '''
	def CalcMaxGIndex(self):
		while (1.0 / self.CalcLength(np.array([self.ih, 0, 0], dtype=np.float64), 'r') > self.dmin):
			self.ih = self.ih + 1
		while (1.0 / self.CalcLength(np.array([0, self.ik, 0], dtype=np.float64), 'r') > self.dmin):
			self.ik = self.ik + 1
		while (1.0 / self.CalcLength(np.array([0, 0, self.il], dtype=np.float64),'r') > self.dmin):
			self.il = self.il + 1

	def InitializeInterpTable(self):

		self.f1 = {}
		self.f2 = {}
		self.f_anam = {}

		fid = h5py.File(str(Path(__file__).resolve().parent)+'/Anomalous.h5','r')

		for i in range(0,self.atom_ntype):

			Z    = self.atom_type[i]
			elem = constants.ptableinverse[Z]
			gid = fid.get('/'+elem)
			data = gid.get('data')

			self.f1[elem] = interp1d(data[:,7], data[:,1])
			self.f2[elem] = interp1d(data[:,7], data[:,2])

		fid.close()

	def CalcAnomalous(self):

		for i in range(self.atom_ntype):

			Z = self.atom_type[i]
			elem = constants.ptableinverse[Z]
			f1 = self.f1[elem](self.wavelength)
			f2 = self.f2[elem](self.wavelength)
			frel = constants.frel[elem]
			Z = constants.ptable[elem]
			self.f_anam[elem] = np.complex(f1+frel-Z, f2)

	def CalcXRFormFactor(self, Z, s):

		'''
		we are using the following form factors for x-aray scattering:
		1. coherent x-ray scattering, f0 tabulated in Acta Cryst. (1995). A51,416-431
		2. Anomalous x-ray scattering (complex (f'+if")) tabulated in J. Phys. Chem. Ref. Data, 24, 71 (1995)
		and J. Phys. Chem. Ref. Data, 29, 597 (2000).
		3. Thompson nuclear scattring, fNT tabulated in Phys. Lett. B, 69, 281 (1977).

		the anomalous scattering is a complex number (f' + if"), where the two terms are given by
		f' = f1 + frel - Z
		f" = f2

		f1 and f2 have been tabulated as a function of energy in Anomalous.h5 in hexrd folder

		overall f = (f0 + f' + if" +fNT)
		'''
		elem = constants.ptableinverse[Z]
		sfact = constants.scatfac[elem]
		fe = sfact[5]
		fNT = constants.fNT[elem]
		frel = constants.frel[elem]
		f_anomalous = self.f_anam[elem]

		for i in range(5):
			fe += sfact[i] * np.exp(-sfact[i+6]*s)

		return (fe+fNT+f_anomalous)


	def CalcXRSF(self, hkl):

		'''
		the 1E-2 is to convert to A^-2
		since the fitting is done in those units
		'''
		s =  0.25 * self.CalcLength(hkl, 'r')**2 * 1E-2
		sf = np.complex(0.,0.)

		for i in range(0,self.atom_ntype):

			Z   = self.atom_type[i]
			ff  = self.CalcXRFormFactor(Z,s)
			ff *= self.atom_pos[i,3] * np.exp(-self.atom_pos[i,4]*s)

			for j in range(self.asym_pos[i].shape[0]):
				arg =  2.0 * np.pi * np.sum( hkl * self.asym_pos[i][j,:] )
				sf  = sf + ff * np.complex(np.cos(arg),-np.sin(arg))

		return np.abs(sf)**2

	''' calculate bragg angle for a reflection. returns Nan if
		the reflections is not possible for the voltage/wavelength
	'''
	def CalcBraggAngle(self, hkl):
		glen = self.CalcLength(hkl, 'r')
		sth  = self.mlambda * glen * 0.5
		return np.arcsin(sth)

	'''
		set some properties for the unitcell class. only the lattice
		parameters, space group and asymmetric positions can change,
		but all the dependent parameters will be automatically updated
	'''

	# lattice constants as properties
	@property
	def a(self):
		return self._a

	@a.setter
	def a(self, val):
		self._a = val
		self.calcmatrices()
		self.ih = 1
		self.ik = 1
		self.il = 1
		self.CalcMaxGIndex()

	@property
	def b(self):
		return self._b

	@b.setter
	def b(self, val):
		self._b = val
		self.calcmatrices()
		self.ih = 1
		self.ik = 1
		self.il = 1
		self.CalcMaxGIndex()

	@property
	def c(self):
		return self._c

	@c.setter
	def c(self, val):
		self._c = val
		self.calcmatrices()
		self.ih = 1
		self.ik = 1
		self.il = 1
		self.CalcMaxGIndex()	

	@property
	def alpha(self):
		return self._alpha

	@alpha.setter
	def alpha(self, val):
		self._alpha = val
		self.calcmatrices()
		self.ih = 1
		self.ik = 1
		self.il = 1
		self.CalcMaxGIndex()

	@property
	def beta(self):
		return self._beta

	@beta.setter
	def beta(self, val):
		self._beta = val
		self.calcmatrices()
		self.ih = 1
		self.ik = 1
		self.il = 1
		self.CalcMaxGIndex()

	@property
	def gamma(self):
		return self._gamma

	@gamma.setter
	def gamma(self, val):
		self._gamma = val
		self.calcmatrices()
		self.ih = 1
		self.ik = 1
		self.il = 1
		self.CalcMaxGIndex()

	@property 
	def voltage(self):
		return self._voltage

	@voltage.setter
	def voltage(self,v):
		self._voltage = v
		self.CalcWavelength()

	@property
	def wavelength(self):
		return self._mlambda

	@wavelength.setter
	def wavelength(self,mlambda):
		self._mlambda = mlambda	

	# space group number
	@property
	def sgnum(self):
		return self._sym_sgnum
	
	@sgnum.setter
	def sgnum(self, val):
		if(not(isinstance(val, int))):
			raise ValueError('space group should be integer')
		if(not( (val >= 1) and (val <= 230) ) ):
			raise ValueError('space group number should be between 1 and 230.')

		self._sym_sgnum = val

		self.SYM_SG, self.SYM_PG_d = \
		symmetry.GenerateSGSym(self.sgnum, self.sgsetting)
		self.nsgsym = self.SYM_SG.shape[0]
		self.npgsym = self.SYM_PG_d.shape[0]

		self.GenerateRecipPGSym()
		self.CalcPositions()

	@property
	def atom_pos(self):
		return self._atom_pos

	@atom_pos.setter
	def atom_pos(self, val):
		self._atom_pos = val

	@property
	def B_factor(self):
		return self._atom_pos[:,4]

	@B_factor.setter
	def B_factor(self, val):
		if (val.shape[0] != self.atom_ntype):
			raise ValueError('Incorrect shape for B factor')
		self._atom_pos[:,4] = val
	
		# val = np.atleast_2d(val)
		# if( (val.shape[1] != 5) ):
		# 	raise ValueError('Incorrect shape for atomic positions')
		# sz = val.shape[0]
		# if(sz != self.cell.atom_ntype):
		# 	raise ValueError('different number of atom types than previously specified.')
		# self.cell.atom_pos[0:sz,:] = val[:,:]
		# symmetry.calcpositions(self.cell, 'v')

	# asymmetric positions in unit cell
	@property
	def asym_pos(self):
		return self._asym_pos

	@asym_pos.setter
	def asym_pos(self, val):
		assert(type(val) == list), 'input type to asymmetric positions should be list'
		self._asym_pos = val

	@property
	def numat(self):
		return self._numat
	
	@numat.setter
	def numat(self, val):
		assert(val.shape[0] == self.atom_ntype),'shape of numat is not consistent'
		self._numat = val
	
	# different atom types; read only
	@property
	def Z(self):
		sz = self.atom_ntype
		return self.atom_type[0:atom_ntype]
	
	# direct metric tensor is read only
	@property
	def dmt(self):
		return self._dmt

	# reciprocal metric tensor is read only
	@property
	def rmt(self):
		return self._rmt

	# direct structure matrix is read only
	@property
	def dsm(self):
		return self._dsm

	# reciprocal structure matrix is read only
	@property
	def rsm(self):
		return self._rsm
	
	@property
	def vol(self):
		return self._vol