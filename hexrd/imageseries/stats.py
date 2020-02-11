"""Stats for imageseries"""


import numpy as np
import logging

from hexrd.imageseries.process import ProcessedImageSeries as PIS

# Default Buffer: 100 MB
#STATS_BUFFER = 419430400    # 50 GE frames
STATS_BUFFER = 838860800    # 100 GE frames


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


def median(ims, nframes=0, chunk=None):
    """return image with median values over all frames

    chunk -- a tuple: (i, n, img), where i is the current chunk (0-based), n is total number of chunks,
             and img is the current state of the image and will be updated on return

    For example, to use 50 chunks, call 50 times consecutively with (0, 50, img) ... (49, 50, img)
    Each time, a number of rows of the image is updated.
"""
    # use percentile since it has better performance
    if chunk is None:
        return percentile(ims, 50, nframes=nframes)

    nf = _nframes(ims, nframes)
    nrows, ncols = ims.shape
    i = chunk[0]
    nchunk = chunk[1]
    img = chunk[2]
    r0, r1 = _chunk_ranges(nrows, nchunk, i)
    rect = np.array([[r0, r1], [0, ncols]])
    pims = PIS(ims, [('rectangle', rect)])
    img[r0:r1, :] = np.median(_toarray(pims, nf), axis=0)

    return img


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


def _chunk_ranges(nrows, nchunk, chunk):
    """Return start and end row for current chunk

    nrows -- total number of rows (row indices are 0-based)
    nchunk -- number of chunks
    chunk -- current chunk (0-based, i.e. ranges from 0 to nchunk - 1)
"""
    csize = nrows//nchunk
    rem = nrows % nchunk
    if chunk < rem:
        r0 = (chunk)*(csize + 1)
        r1 = r0 + csize + 1
    else:
        r0 = chunk*csize + rem
        r1 = r0 + csize

    return r0, r1


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
