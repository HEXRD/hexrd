"""Stats for imageseries"""


import numpy as np
import logging

from psutil import virtual_memory

from hexrd.imageseries.process import ProcessedImageSeries as PIS

# Default Buffer: 100 MB
#STATS_BUFFER = 419430400    # 50 GE frames
#STATS_BUFFER = 838860800    # 100 GE frames
vmem = virtual_memory()
STATS_BUFFER = int(0.5*vmem.available)

def max(ims, nframes=0):
    nf = _nframes(ims, nframes)
    imgmax = ims[0]
    for i in range(1, nf):
        imgmax = np.maximum(imgmax, ims[i])
    return imgmax

def average(ims, nframes=0):
    """return image with average values over all frames"""
    nf = _nframes(ims, nframes)
    avg = np.array(ims[0], dtype=float)
    for i in range(1, nf):
        avg += ims[i]
    return avg/nf

def median(ims, nframes=0):
    """return image with median values over all frames"""
    # use percentile since it has better performance
    return percentile(ims, 50, nframes=nframes)

def percentile(ims, pct, nframes=0):
    """return image with given percentile values over all frames"""
    # could be done by rectangle by rectangle if full series
    # too  big for memory
    nf = _nframes(ims, nframes)
    dt = ims.dtype
    (nr, nc) = ims.shape
    nrpb  = _rows_in_buffer(nframes, nf*nc*dt.itemsize)

    # now build the result a rectangle at a time
    img = np.zeros_like(ims[0])
    for rr in _row_ranges(nr, nrpb):
        rect = np.array([[rr[0], rr[1]], [0, nc]])
        pims = PIS(ims, [('rectangle', rect)])
        img[rr[0]:rr[1], :] = np.percentile(_toarray(pims, nf), pct, axis=0)
    return img

#
# ==================== Utilities
#
def _nframes(ims, nframes):
    """number of frames to use: len(ims) or specified number"""
    mynf = len(ims)
    return np.min((mynf, nframes)) if nframes > 0 else mynf

def _toarray(ims, nframes):
    ashp = (nframes,) + ims.shape
    a = np.zeros(ashp, dtype=ims.dtype)
    for i in range(nframes):
        logging.info('frame: %s', i)
        a[i] = ims[i]

    return a

def _row_ranges(n, m):
    """return row ranges, representing m rows or remainder, until exhausted"""
    i = 0
    while i < n:
        imax = i+m
        if imax <= n:
            yield (i, imax)
        else:
            yield (i, n)
        i = imax

def _rows_in_buffer(ncol, rsize):
    """number of rows in buffer

    NOTE: Use ceiling to make sure at it has at least one row"""
    return int(np.ceil(STATS_BUFFER/rsize))
