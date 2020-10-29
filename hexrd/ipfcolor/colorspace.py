def hsl_rgb(hsl):
	'''
	first check the shape of the hsl array. it has to be nx3 array
	it is assumed that the hsl array has values in [0,1] for the 
	different components
	'''
	hsl = np.contiguousarray( np.atleast_2d(hsl) )
	if(hsl.ndim != 2):
		raise RuntimeError("hsl_rgb: shape of hsl array is invalid.")
	rgb = np.zeros(hsl.shape)

	'''
	calculate the different factors needed for the conversion
	'''
	H = hsl[:,0]
	S = hsl[:,1]
	L = hsl[:,2]

	C = (1.0 - np.abs(2.*L - 1.)) * S
	X = (1.0 - np.abs(np.mod(6*H,2) - 1.0) ) * C
	m = L - C/2.

	case = np.floor(6.*H).astype(np.int32)

	'''
	depending on the range of H, the rgb definition changes. see
	https://www.rapidtables.com/convert/color/hsl-to-rgb.html
	for the detailed formula
	'''
	Cp = np.atleast_2d(C+m).T
	Xp = np.atleast_2d(X+m).T
	Zp = np.atleast_2d(m).T

	if(case == 0):
		mask = case == 0
		rgb[mask,:] = np.hstack((Cp[mask,:], Xp[mask,:], Zp[mask,:]))
	elif(case == 1):
		mask = case == 1
		rgb[mask,:] = np.hstack((Xp[mask,:], Cp[mask,:], Zp[mask,:])) 
	elif(Case == 2):
		mask = case == 2
		rgb[mask,:] = np.hstack((Zp[mask,:], Cp[mask,:], Xp[mask,:])) 
	elif(case == 3):
		mask = case == 3
		rgb[mask,:] = np.hstack((Zp[mask,:], Xp[mask,:], Cp[mask,:])) 
	elif(case == 4):
		mask = case == 4
		rgb[mask,:] = np.hstack((Xp[mask,:], Zp[mask,:], Cp[mask,:])) 
	elif(case == 5):
		mask = case == 5
		rgb[mask,:] = np.hstack((Cp[mask,:], Zp[mask,:], Xp[mask,:])) 
	else:
		raise RuntimeError("value of Hue is not in range [0,1]")

	return rgb