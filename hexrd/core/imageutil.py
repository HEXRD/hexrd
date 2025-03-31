from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

import numpy as np
from scipy import signal, ndimage

from skimage.feature import blob_dog, blob_log
from skimage.exposure import rescale_intensity

from hexrd.core import convolution
from hexrd.core.constants import fwhm_to_sigma


# =============================================================================
# BACKGROUND REMOVAL
# =============================================================================

def _scale_image_snip(y, offset, invert=False):
    """
    Log-Log scale image for snip

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    offset : TYPE
        DESCRIPTION.
    invert : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Notes
    -----
    offset should be <= min of the original image

    """
    if invert:
        return (np.exp(np.exp(y) - 1.) - 1.)**2 + offset
    else:
        return np.log(np.log(np.sqrt(y - offset) + 1.) + 1.)


def fast_snip1d(y, w=4, numiter=2):
    """
    """
    bkg = np.zeros_like(y)
    min_val = np.nanmin(y)
    zfull = _scale_image_snip(y, min_val, invert=False)
    for k, z in enumerate(zfull):
        b = z
        for i in range(numiter):
            for p in range(w, 0, -1):
                kernel = np.zeros(p*2 + 1)
                kernel[0] = 0.5
                kernel[-1] = 0.5
                b = np.minimum(b, signal.convolve(z, kernel, mode='same'))
            z = b
        bkg[k, :] = _scale_image_snip(b, min_val, invert=True)
    return bkg


def snip1d(y, w=4, numiter=2, threshold=None, max_workers=os.cpu_count()):
    """
    Return SNIP-estimated baseline-background for given spectrum y.

    !!!: threshold values get marked as NaN in convolution
    !!!: mask in astropy's convolve is True for masked; set to NaN
    """
    # scale input
    bkg = np.zeros_like(y)
    min_val = np.nanmin(y)
    zfull = _scale_image_snip(y, min_val, invert=False)

    # handle mask
    if threshold is not None:
        mask = y <= threshold
    else:
        mask = np.zeros_like(y, dtype=bool)

    # step through rows
    tasks = enumerate(zip(zfull, mask))
    f = partial(_run_snip1d_row, numiter=numiter, w=w, min_val=min_val)

    if max_workers > 1:
        # Parallelize over tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for k, result in executor.map(f, tasks):
                bkg[k, :] = result
    else:
        # Run the tasks in this process
        for task in tasks:
            k, result = f(task)
            bkg[k, :] = result

    nan_idx = np.isnan(bkg)
    bkg[nan_idx] = threshold
    return bkg


def _run_snip1d_row(task, numiter, w, min_val):
    k, (z, mask) = task

    if np.all(mask):
        return k, np.nan

    b = z
    for i in range(numiter):
        for p in range(w, 0, -1):
            kernel = np.zeros(p*2 + 1)
            kernel[0] = kernel[-1] = 1./2.
            b = np.minimum(
                b,
                convolution.convolve(
                    z, kernel, boundary='extend', mask=mask,
                    nan_treatment='interpolate', preserve_nan=True
                )
            )
        z = b
    return k, _scale_image_snip(b, min_val, invert=True)


def snip1d_quad(y, w=4, numiter=2):
    """Return SNIP-estimated baseline-background for given spectrum y.

    Adds a quadratic kernel convolution in parallel with the linear kernel."""
    min_val = np.nanmin(y)
    kernels = []
    for p in range(w, 1, -2):
        N = p * 2 + 1
        # linear kernel
        kern1 = np.zeros(N)
        kern1[0] = kern1[-1] = 1./2.

        # quadratic kernel
        kern2 = np.zeros(N)
        kern2[0] = kern2[-1] = -1./6.
        kern2[int(p/2.)] = kern2[int(3.*p/2.)] = 4./6.
        kernels.append([kern1, kern2])

    z = b = _scale_image_snip(y, min_val, invert=False)
    for i in range(numiter):
        for (kern1, kern2) in kernels:
            c = np.maximum(ndimage.convolve1d(z, kern1, mode='nearest'),
                           ndimage.convolve1d(z, kern2, mode='nearest'))
            b = np.minimum(b, c)
        z = b

    return _scale_image_snip(b, min_val, invert=True)


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
    min_val = np.nanmin(y)

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

    # convolve kernels with input array (in log space)
    z = b = _scale_image_snip(y, min_val, invert=False)
    for i in range(numiter):
        for kk in kernels:
            if order > 1:
                c = maximum(ndimage.convolve(z, kk[0], mode='nearest'),
                            ndimage.convolve(z, kk[1], mode='nearest'))
            else:
                c = ndimage.convolve(z, kk[0], mode='nearest')
            b = minimum(b, c)
        z = b

    return _scale_image_snip(b, min_val, invert=True)


# =============================================================================
# FEATURE DETECTION
# =============================================================================


def find_peaks_2d(img, method, method_kwargs):
    if method == 'label':
        # labeling mask
        structureNDI_label = ndimage.generate_binary_structure(2, 1)

        # First apply filter if specified
        filter_fwhm = method_kwargs['filter_radius']
        if filter_fwhm:
            filt_stdev = fwhm_to_sigma * filter_fwhm
            img = -ndimage.filters.gaussian_laplace(
                img, filt_stdev
            )

        labels_t, numSpots_t = ndimage.label(
            img > method_kwargs['threshold'],
            structureNDI_label
            )
        coms_t = np.atleast_2d(
            ndimage.center_of_mass(
                img,
                labels=labels_t,
                index=np.arange(1, np.amax(labels_t) + 1)
                )
            )
    elif method in ['blob_log', 'blob_dog']:
        # must scale map
        # TODO: we should so a parameter study here
        scl_map = rescale_intensity(img, out_range=(-1, 1))

        # TODO: Currently the method kwargs must be explicitly specified
        #       in the config, and there are no checks
        # for 'blob_log': min_sigma=0.5, max_sigma=5,
        #                 num_sigma=10, threshold=0.01, overlap=0.1
        # for 'blob_dog': min_sigma=0.5, max_sigma=5,
        #                 sigma_ratio=1.6, threshold=0.01, overlap=0.1
        if method == 'blob_log':
            blobs = np.atleast_2d(
                blob_log(scl_map, **method_kwargs)
            )
        else:  # blob_dog
            blobs = np.atleast_2d(
                blob_dog(scl_map, **method_kwargs)
            )
        numSpots_t = len(blobs)
        coms_t = blobs[:, :2]

    return numSpots_t, coms_t
