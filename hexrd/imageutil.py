import numpy as np
from scipy import ndimage


'''
from astropy import convolution

def snip1d(y, w=4, numiter=3):
    """
    Return SNIP-estimated baseline-background for given spectrum y.
    """
    zfull = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    bkg = np.zeros_like(zfull)
    for k, z in enumerate(zfull):
        b = z
        for i in range(numiter):
            for p in range(w, 0, -1):
                kernel = np.zeros(p*2 + 1)
                kernel[0] = kernel[-1] = 1./2.
                b = np.minimum(
                    b,
                    convolution.convolve(z, kernel, boundary='extend')
                )
            z = b
        bkg[k, :] = (np.exp(np.exp(b) - 1) - 1)**2 - 1
    return bkg
'''


def snip1d(y, w=4, numiter=2, mask=None):
    """
    Return SNIP-estimated baseline-background for given spectrum y.
    """
    if mask is not None:
        for ir, rmask in enumerate(mask):
            if np.sum(~rmask) > 0:
                y[ir, rmask] = np.median(y[ir, ~rmask])
    z = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    b = z
    for i in range(numiter):
        for p in range(w, 0, -1):
            kernel = np.zeros(p*2 + 1)
            kernel[0] = kernel[-1] = 1./2.
            b = np.minimum(
                b,
                ndimage.convolve1d(z, kernel, mode='reflect')
            )
        z = b
    bkg = (np.exp(np.exp(b) - 1) - 1)**2 - 1
    return bkg


def snip1d_quad(y, w=4, numiter=2):
    """Return SNIP-estimated baseline-background for given spectrum y.

    Adds a quadratic kernel convolution in parallel with the linear kernel."""
    convolve1d = ndimage.convolve1d
    kernels = []
    for p in range(w, 1, -2):
        N = p * 2 + 1
        # linear kernel
        kern1 = np.zeros(N)
        kern1[0] = kern1[-1] = 1./2.

        # quadratic kernel
        kern2 = np.zeros(N)
        kern2[0] = kern2[-1] = -1./6.
        kern2[p/2] = kern2[3*p/2] = 4./6.
        kernels.append([kern1, kern2])

    z = b = np.log(np.log(y + 1) + 1)
    for i in range(numiter):
        for (kern1, kern2) in zip(kernels):
            c = np.maximum(convolve1d(z, kern1, mode='nearest'),
                           convolve1d(z, kern2, mode='nearest'))
            b = np.minimum(b, c)
        z = b

    return np.exp(np.exp(b) - 1) - 1
