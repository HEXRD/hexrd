"""aggregate statistics for imageseries

The functions here operate on the frames of an imageseries.
Because there may be an large number of images, the images
are processed in chunks, some number of rows at a time.
All of the functions take a keyword arguments of "chunk" and "nframes".
If "nframes" is greater than 0, it means use only that many frames of
the imageseries; otherwise use all the frames.
The "chunks" argument may be either None or a 3-tuple (i, n, img), where:

 i is the current chunk to compute,
 n is the total number of chunks, and
 img is the current image with same shape as the imageseries

If chunks is not None, then the the function needs to be called n times,
and after each call an intermediate image is returned, the last being complete.

If chunks is None, then only one call is needed, and the number of chunks is
determined by the STATS_BUFFER variable.

For example, to find the median image for an imageseries "ims" using "nchunk" chunks:

    img = np.zeros(ims.shape)
    for i in range(nchunk):
        img = op(ims, chunk=(i, nchunk, img))
        print("%d/%d" % (i, nchunk), img)

"""
import numpy as np

from psutil import virtual_memory

# Default Buffer: half of available memory
vmem = virtual_memory()
STATS_BUFFER = int(0.5*vmem.available)
del vmem


def max(ims, chunk=None, nframes=0):
    return _run_chunks(np.max, ims, chunk, nframes)


def min(ims, chunk=None, nframes=0):
    return _run_chunks(np.min, ims, chunk, nframes)


def median(ims, chunk=None, nframes=0):
    return _run_chunks(np.median, ims, chunk, nframes)


def percentile(ims, pctl, chunk=None, nframes=0):
    return _run_chunks(np.percentile, ims, chunk, nframes, *(pctl,))


def average(ims, chunk=None, nframes=0):
    return _run_chunks(np.average, ims, chunk, nframes)


# ==================== Utilities
#
def _nframes(ims, nframes):
    """number of frames to use: len(ims) or specified number"""
    mynf = len(ims)
    return np.min((mynf, nframes)) if nframes > 0 else mynf


def _toarray(ims, nframes, r0, r1):
    _, nc = ims.shape
    ashp = (nframes, r1 - r0, nc)
    a = np.zeros(ashp, dtype=ims.dtype)
    for i in range(nframes):
        a[i] = ims[i][r0:r1, :]

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


def _chunk_op(op, ims, nf, chunk, *args):
    """run operation on one chunk of image data
    ims -- the imageseries
    nf -- total number of frames to use
    chunk -- tuple of (i, n, img)
    args -- args to pass to op
"""
    nrows, ncols = ims.shape
    i = chunk[0]
    nchunk = chunk[1]
    img = chunk[2]
    r0, r1 = _chunk_ranges(nrows, nchunk, i)
    a = _toarray(ims, nf, r0, r1)
    img[r0:r1, :] = op(a, *args, axis=0)

    return img


def _run_chunks(op, ims, chunk, nframes, *args):
    """run chunked operation"""
    nf = _nframes(ims, nframes)
    if chunk is None:
        dt = ims.dtype
        (nr, nc) = ims.shape
        mem = nf*nr*nc*dt.itemsize
        nchunks = 1 + mem // STATS_BUFFER
        img = np.zeros((nr, nc), dtype=dt)
        for i in range(nchunks):
            chunk = (i, nchunks, img)
            img = _chunk_op(op, ims, nf, chunk, *args)
    else:
        img = _chunk_op(op, ims, nf, chunk, *args)

    return img
