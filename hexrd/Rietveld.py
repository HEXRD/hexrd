import numpy as np
from hexrd.EMsoft import EMsoft_FAPI
import warnings
from hexrd.imageutil import snip1d

class Rietveld:

	def __init__(self):
		pass

	def Refine(self):
		pass

	class Peak:

		def __init__(self, tth):
			if(np.abs(tth) > 2.0 * np.pi):
				warnings.warn(" The angles in the exponential map seems to be too large. \
							Please check if you've converted to radians.")
			# 2theta for the peak
			self._tth = tth
			
			# Cagliotti parameters for the half width of the peak
			self._U = 1.0e-4
			self._V = 1.0e-4
			self._W = 1.0e-4

			# Pseudo-Voight mixing parameters
			self._eta1 = 4.0e-1
			self._eta2 = 1.0e-2
			self._eta3 = 1.0e-2

			# range of angles for which the peak is calculated
			self._tth_min = 0.0
			self._tth_max = np.pi / 2.0
			self._tth_step = np.radians(0.1)

			''' maximum intensity, Imax = S * Mk * Lk * |Fk|**2 * Pk * Ak * Ek
				S  = arbitrary scale factor
				Mk = multiplicity of reflection
				Lk = lorentz polarizatin factor
				Fk = structure factor
				Pk = preferrred orientation 
				Ak = absorption correction
				Ek = extinction correction 
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

		def CagliottiH(self):
			tanth = np.tan(0.5 * self.tth)
			self.H = np.sqrt(self.U * tanth**2 + self.V * tanth + self.W)

			'''
				set the parameter for background estimation
				the 5 times the ratio of halfwidth to step size was figured
				out empirically
			'''
			self.snipw = int(np.round(5.0 * self.H / self.tth_step))

		def MixingFact(self):
			self.eta = self.eta1 + self.eta2 * self.tth + self.eta3 * self.tth**2

			if(self.eta > 1.0):
				self.eta = 1.0

			elif(self.eta< 0.0):
				self.eta = 0.0

		def Gaussian(self):
			beta = 0.5 * self.H * np.sqrt(np.pi / np.log(2.0))
			self.GaussianI = self.Imax * np.exp(-np.pi * (self.tth_list - self.tth)**2 / beta**2)

		def Lorentzian(self):
			w = 0.5 * self.H
			self.LorentzI = self.Imax * (w**2 / (w**2 + (self.tth_list - self.tth)**2 ) )

		def PseudoVoight(self):
			self.PV = self.eta * self.GaussianI + (1.0 - self.eta) * self.LorentzI


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


		@property
		def U(self):
			return self._U

		@U.setter
		def _set_U(self, Uinp):
			self._U = Uinp
			self.CagliottiH()
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			return

		@property
		def V(self):
			return self._V

		@V.setter
		def V(self, Vinp):
			self._V = Vinp
			self.CagliottiH()
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			return

		@property
		def W(self):
			return self._W

		@W.setter
		def W(self, Winp):
			self._W = Winp
			self.CagliottiH()
			self.Gaussian()
			self.Lorentzian()
			self.PseudoVoight()
			return

		@property
		def eta1(self):
			return self._eta1

		@eta1.setter
		def _set_eta1(self, val):
			self._eta1 = val
			self.MixingFact()
			self.PseudoVoight()
			return

		@property
		def eta2(self):
			return self._eta2

		@eta2.setter
		def eta2(self, val):
			self._eta2 = val
			self.MixingFact()
			self.PseudoVoight()
			return

		@property
		def eta3(self):
			return self._eta3

		@eta3.setter
		def eta3(self, val):
			self._eta3 = val
			self.MixingFact()
			self.PseudoVoight()
			return


	class Background:

		def __init__(self, spectrum, w=8, numiter=2):
			self.background = snip1d(spectrum, w=w, numiter=numiter)
			

