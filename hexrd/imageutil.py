import numpy as np
from scipy import ndimage

from hexrd import convolution


def snip1d(y, w=4, numiter=2, threshold=0):
    """
    Return SNIP-estimated baseline-background for given spectrum y.

    !!!: threshold values get marked as NaN in convolution
    """
    mask = y <= threshold
    zfull = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    bkg = np.zeros_like(zfull)
    for k, z in enumerate(zfull):
        if np.all(mask[k]):
            bkg[k, :] = np.nan
        else:
            b = z
            for i in range(numiter):
                for p in range(w, 0, -1):
                    kernel = np.zeros(p*2 + 1)
                    kernel[0] = kernel[-1] = 1./2.
                    b = np.minimum(
                        b,
                        convolution.convolve(
                            z, kernel, boundary='extend', mask=mask[k]
                        )
                    )
                z = b
            bkg[k, :] = (np.exp(np.exp(b) - 1) - 1)**2 - 1
    nan_idx = np.isnan(bkg)
    bkg[nan_idx] = threshold
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
