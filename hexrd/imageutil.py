import numpy as np
from scipy import signal, ndimage

from hexrd import convolution


def fast_snip1d(y, w=4, numiter=2):
    """
    """
    bkg = np.zeros_like(y)
    zfull = np.log(np.log(np.sqrt(y + 1.) + 1.) + 1.)
    for k, z in enumerate(zfull):
        b = z
        for i in range(numiter):
            for p in range(w, 0, -1):
                kernel = np.zeros(p*2 + 1)
                kernel[0] = 0.5
                kernel[-1] = 0.5
                b = np.minimum(b, signal.fftconvolve(z, kernel, mode='same'))
            z = b
        bkg[k, :] = (np.exp(np.exp(b) - 1.) - 1.)**2 - 1.
    return bkg


def snip1d(y, w=4, numiter=2, threshold=None):
    """
    Return SNIP-estimated baseline-background for given spectrum y.

    !!!: threshold values get marked as NaN in convolution
    !!!: mask in astropy's convolve is True for masked; set to NaN
    """
    # scal input
    zfull = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    bkg = np.zeros_like(zfull)

    # handle mask
    if threshold is not None:
        mask = y <= threshold
    else:
        mask = np.zeros_like(y, dtype=bool)

    # step through rows
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
                            z, kernel, boundary='extend', mask=mask[k],
                            nan_treatment='interpolate', preserve_nan=True
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
        kern2[int(p/2)] = kern2[int(3*p/2)] = 4./6.
        kernels.append([kern1, kern2])

    z = b = np.log(np.log(y + 1) + 1)
    for i in range(numiter):
        for (kern1, kern2) in kernels:
            c = np.maximum(convolve1d(z, kern1, mode='nearest'),
                           convolve1d(z, kern2, mode='nearest'))
            b = np.minimum(b, c)
        z = b

    return np.exp(np.exp(b) - 1) - 1


def snip2d(y, w=4, numiter=2, order=1):
    """
    Return estimate of 2D-array background by "clipping" peak-like structures.

    2D adaptation of the peak-clipping component of the SNIP algorithm.

    Parameters
    ----------
    y : 2-D input array
    w : integer (default 4)
        kernel size (maximum kernel extent actually = 2 * w * order + 1)
    numiter : integer (default 2)
        number of iterations
    order : integer (default 1)
        maximum order of filter kernel, either 1 (linear) or 2 (quadratic)

    Returns
    -------
    out : 2-D array with SNIP-estimated background of y

    References!!!
    -----
    [1] C.G. Ryan et al, "SNIP, A statistics-sensitive background treatment
          for the quantitative analysis of PIXE spectra in geoscience
          applications," Nucl. Instr. and Meth. B 34, 396 (1988).
    [2] M. Morhac et al., "Background elimination methods for multidimensional
          coincidence gamma-ray spectra," Nucl. Instr. and Meth. A 401, 113
          (1997).
    """
    maximum, minimum = np.fmax, np.fmin

    # create list of kernels
    kernels = []
    for p in range(w, 0, -1):  # decrement window starting from w
        N = 2 * p * order + 1  # size of filter kernels
        p1 = order * p

        # linear filter kernel
        kern1 = np.zeros((N, N))  # initialize a kernel with all zeros
        xx, yy = np.indices(kern1.shape)  # x-y indices of kernel points
        ij = np.round(
            np.hypot(xx - p1, yy - p1)
        ) == p1  # select circular shape
        kern1[ij] = 1 / ij.sum()  # normalize so sum of kernel elements is 1
        kernels.append([kern1])

        if order >= 2:  # add quadratic filter kernel
            p2 = p1 // 2
            kern2 = np.zeros_like(kern1)
            radii, norms = (p2, 2 * p2), (4/3, -1/3)
            for radius, norm in zip(radii, norms):
                ij = np.round(np.hypot(xx - p1, yy - p1)) == radius
                kern2[ij] = norm / ij.sum()
            kernels[-1].append(kern2)

    # convolve kernels with input array
    z = b = np.log(np.log(y + 1) + 1)  # perform convolutions in logspace
    for i in range(numiter):
        for kk in kernels:
            if order > 1:
                c = maximum(ndimage.convolve(z, kk[0], mode='nearest'),
                            ndimage.convolve(z, kk[1], mode='nearest'))
            else:
                c = ndimage.convolve(z, kk[0], mode='nearest')
            b = minimum(b, c)
        z = b

    return np.exp(np.exp(b) - 1) - 1
