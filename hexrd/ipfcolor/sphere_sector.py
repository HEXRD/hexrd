class sector
'''
	@AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	@DATE    10/28/2020 SS 1.0 original
	@DETAIL  this class is used to store spherical patch for a given point group.
			 the class also has methods to compute the color of a direction by
			 computing the hue, saturation and lightness values in [0,1]. these
			 values can be converted to rgb for display with the well known 
			 conversion formula.


	All the methodology and equations have been taken from the paper:
	Orientations – perfectly colored, G. Nolze and R. Hielscher, J. Appl. Cryst. (2016). 49, 1786–1802

'''
def __init__(self, vertex, center):
	pass

def check_hemisphere(self):
	pass

def calc_azi_rho(di3):
	'''
	@AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	@DATE    10/28/2020 SS 1.0 original
	@PARAM   dir3 direction in fundamental sector. behavior is undefined if
		     direction is outside the fundamental sector
	@DETAIL  this function is used to calculate the azimuthal angle of the
			 direction inside a spherical patch. this is computed as the angle
			 with the vector defined by the first vertex and barycenter and the
			 vector defined by direction and barycenter.

	'''
	pass

def calc_pol_theta(dir3):
	'''
	@AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	@DATE    10/28/2020 SS 1.0 original
	@PARAM   dir3 direction in fundamental sector. behavior is undefined if
		     direction is outside the fundamental sector
	@DETAIL  this function is used to calculate the polar angle of the
			 direction inside a spherical patch. this is computed as the scaled
			 angular distance between direction and barycenter. the scaling is such
			 that the boundary is 90 degrees from barycenter.

	'''
	pass

def calc_hue(self, rho):
	'''
	@AUTHOR  Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
	@DATE    10/28/2020 SS 1.0 original
	@PARAM   dir3 direction in fundamental sector. behavior is undefined if
		     direction is outside the fundamental sector
	@DETAIL  this function is used to calculate the hue based on angular coordinates
			 of direction within the spherical patch. the velocity

	'''

	pass

def calc_saturation(self):
	pass

def calc_lightness(self, theta):
	pass