"""aggregate statistics for imageseries

The functions here operate on the frames of an imageseries and return a
single aggregate image. For each function, there is a corresponding iterable
that allows you to run the function in smaller bits; the bits are either groups
of frames or groups of rows, depending on the function. The iterable returns a
sequence of images, the last being the final result.

For example:

#  Using the standard function call
img = stats.average(ims)

# Using the iterable with 10 chunks
for img in stats.average_iter(ims, 10):
    # update progress bar
    pass

NOTE:
* Perhaps we should rename min -> minimum and max -> maximum to avoid
  conflicting with the python built-ins
"""
import numpy as np

from psutil import virtual_memory

# Default Buffer Size: half of available memory
vmem = virtual_memory()
STATS_BUFFER = int(0.5*vmem.available)
del vmem


def max(ims, nframes=0):
    """maximum over frames"""
    nf = _nframes(ims, nframes)
    img = ims[0]
    for i in range(1, nf):
        img = np.maximum(img, ims[i])
    return img


def max_iter(ims, nchunk, nframes=0):
    """iterator for max function"""
    nf = _nframes(ims, nframes)
    stops = _chunk_stops(nf, nchunk)
    s0 = 0
    stop = stops[s0]
    img = ims[0]
    if stop == 0:
        if nf > 1:
            s0, stop = 1, stops[1]
        yield img

    for i in range(1, nf):
        img = np.maximum(img, ims[i])
        if i >= stop:
            yield img
            if (i + 1) < nf:
                s0 += 1
                stop = stops[s0]


def min(ims, nframes=0):
    """minimum over frames"""
    nf = _nframes(ims, nframes)
    img = ims[0]
    for i in range(1, nf):
        img = np.minimum(img, ims[i])
    return img


def min_iter(ims, nchunk, nframes=0):
    """iterator for min function"""
    nf = _nframes(ims, nframes)
    stops = _chunk_stops(nf, nchunk)
    s0, stop = 0, stops[0]
    img = ims[0]
    if stop == 0:
        if nf > 1:
            s0, stop = 1, stops[1]
        yield img

    for i in range(1, nf):
        img = np.minimum(img, ims[i])
        if i >= stop:
            yield img
            if (i + 1) < nf:
                s0 += 1
                stop = stops[s0]


def average(ims, nframes=0):
    """average over frames"""
    nf = _nframes(ims, nframes)
    img = ims[0]
    for i in range(1, nf):
        img += ims[i]
    return (img/nf).astype(np.float32)


def average_iter(ims, nchunk, nframes=0):
    """average over frames

    Note: average returns a float even if images are uint
"""
    nf = _nframes(ims, nframes)
    stops = _chunk_stops(nf, nchunk)
    s0, stop = 0, stops[0]
    img = ims[0]
    if stop == 0:
        if nf > 1:
            s0, stop = 1, stops[1]
        yield img

    for i in range(1, nf):
        img += ims[i]
        if i >= stop:
            if (i + 1) < nf:
                s0 += 1
                stop = stops[s0]
            yield (img/(i+1)).astype(np.float32)


def percentile(ims, pctl, nframes=0):
    """percentile function over frames

    ims - the imageseries
    pctl - the percentile
    nframes - the number of frames to use (default/0 = all)
    """
    nf = _nframes(ims, nframes)
    return np.percentile(_toarray(ims, nf), pctl, axis=0).astype(np.float32)


def percentile_iter(ims, pctl, nchunks,  nframes=0, use_buffer=True):
    """iterator for percentile function"""
    nf = _nframes(ims, nframes)
    nr, nc = ims.shape
    stops = _chunk_stops(nr, nchunks)
    r0 = 0
    img = np.zeros(ims.shape)
    buffer = _alloc_buffer(ims, nf) if use_buffer else None
    for s in stops:
        r1 = s + 1
        img[r0:r1] = np.percentile(
            _toarray(ims, nf, rows=(r0, r1), buffer=buffer),
            pctl, axis=0
        )
        r0 = r1
        yield img.astype(np.float32)


def median(ims, nframes=0):
    return percentile(ims, 50, nframes=nframes)


def median_iter(ims, nchunks, nframes=0, use_buffer=True):
    return percentile_iter(
        ims, 50, nchunks, nframes=nframes, use_buffer=use_buffer
    )


# ==================== Utilities
#
def _nframes(ims, nframes):
    """number of frames to use: len(ims) or specified number"""
    mynf = len(ims)
    return np.min((mynf, nframes)) if nframes > 0 else mynf


def _chunk_stops(n, nchunks):
    """Return yield points

    n -- number of items to be chunked (e.g. frames/rows)
    nchunks -- number of chunks
"""
    if nchunks > n:
        raise ValueError("number of chunks cannot exceed number of items")
    csize = n//nchunks
    rem = n % nchunks
    pieces = csize*np.ones(nchunks, dtype=int)
    pieces[:rem] += 1
    pieces[0] += -1

    return np.cumsum(pieces)


def _toarray(ims, nframes, rows=None, buffer=None):
    """generate array for either whole imageseries or subset of rows

    ims - imageseries
    nframes - number of frames to use

    OPTIONAL
    rows - if None, use all rows, otherwise a 2-tuple of row indices
    buffer - if None, get images directly from ims, otherwise
             use buffer to store frames of the imageseries to
             prevent repeated frame accesses; buffer only applies
             if rows is not None
    """
    nr, nc = ims.shape
    use_buffer = buffer is not None
    if rows is None: # use all
        use_buffer, nbf = False, 0
        r0, r1 = 0, nr
        ashp = (nframes, nr, nc)
    else:
        r0, r1 = rows
        ashp = (nframes, r1 - r0, nc)

    nbf = 0
    if use_buffer:
        nbf = len(buffer)
        if r0 == 0:
            # copy as many frames as possible into buffer
            print("copying images to buffer")
            for i in range(nbf):
                buffer[i] = ims[i]

    a = np.empty(ashp, dtype=ims.dtype)
    if use_buffer:
        a[:nbf] = buffer[:nbf, r0:r1, :]
    for i in range(nbf, nframes):
        a[i] = ims[i][r0:r1]

    return a


def _alloc_buffer(ims, nf):
    """Allocate buffer to save as many full frames as possible"""
    print("allocating buffer")
    shp, dt = ims.shape, ims.dtype
    framesize = shp[0]*shp[1]*dt.itemsize
    nf = np.minimum(nf, np.floor(STATS_BUFFER/framesize).astype(int))
    bshp = (nf,) +  shp

    return np.empty(bshp, dt)
