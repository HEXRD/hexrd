"""Class for processing individual frames"""
import copy

import numpy as np
import scipy

from .baseclass import ImageSeries


class ProcessedImageSeries(ImageSeries):
    """imageseries based on existing one with image processing options

    Parameters
    ----------
    imser: ImageSeries
       an existing imageseries
    oplist: list
       list of processing operations; each option is a (key, data) pair,
       with key specifying the operation to perform using specified data
    frame_list: list of ints or None, default = None
       specify subset of frames by list; if None, then all frames are used
    """
    FLIP = 'flip'
    DARK = 'dark'
    RECT = 'rectangle'
    ADD = 'add'
    GAUSS_LAPLACE = 'gauss_laplace'

    def __init__(self, imser, oplist, **kwargs):

        self._imser = imser
        self._meta = copy.deepcopy(imser.metadata)
        self._oplist = oplist
        self._frames = kwargs.pop('frame_list', None)
        self._hasframelist = (self._frames is not None)
        if self._hasframelist:
            self._update_omega()
        self._opdict = {}

        self.addop(self.DARK, self._subtract_dark)
        self.addop(self.FLIP, self._flip)
        self.addop(self.RECT, self._rectangle)
        self.addop(self.ADD, self._add)
        self.addop(self.GAUSS_LAPLACE, self._gauss_laplace)

    def __getitem__(self, key):
        if isinstance(key, int):
            idx = key
            rest = []
        else:
            # Handle fancy indexing
            idx = key[0]
            rest = key[1:]

        idx = self._get_index(idx)

        if rest:
            arg = tuple([idx, *rest])
        else:
            arg = idx

        return self._process_frame(arg)

    def _get_index(self, key):
        return self._frames[key] if self._hasframelist else key

    def __len__(self):
        return len(self._frames) if self._hasframelist else len(self._imser)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def _process_frame(self, key):
       # note: key refers to original imageseries
        oplist = self.oplist

        # when rectangle is the first operation we can try to call the
        # optimized version. If the adapter provides one it should be
        # significantly faster if not it will fallback to the same
        # implementation that _rectangle provides.
        if oplist and oplist[0][0] == self.RECT:
            region = oplist[0][1]
            if isinstance(key, int):
                idx = key
                rest = []
            else:
                # Handle fancy indexing
                idx = key[0]
                rest = key[1:]

            img = self._rectangle_optimized(idx, region)

            if rest:
                img = img[*rest]

            # remove the first operation since we already used it
            oplist = oplist[1:]
        else:
            img = self._imser[key]

        for k, d in oplist:
            func = self._opdict[k]
            img = func(img, d)

        return img

    def _subtract_dark(self, img, dark):
        # need to check for values below zero
        # !!! careful, truncation going on here;necessary to promote dtype?

        # This has been performance tested with the following:
        # 1. return np.where(img > dark, img - dark, 0)
        # 2. return np.clip(img - dark, a_min=0, a_max=None)
        # 3. return (img - dark).clip(min=0)
        # 4. ret = img - dark
        #    ret[ret < 0] = 0
        #    return ret
        # Method 1 was the slowest, and method 4 was the fastest, perhaps
        # because it creates fewer copies of the data.
        ret = img - dark
        ret[ret < 0] = 0
        return ret

    def _rectangle_optimized(self, img_key, r):
        return self._imser.get_region(img_key, r)

    def _rectangle(self, img, r):
        # restrict to rectangle
        return img[r[0][0]:r[0][1], r[1][0]:r[1][1]]

    def _flip(self, img, flip):
        if flip in ('y', 'v'):  # about y-axis (vertical)
            pimg = img[:, ::-1]
        elif flip in ('x', 'h'):  # about x-axis (horizontal)
            pimg = img[::-1, :]
        elif flip in ('vh', 'hv', 'r180'):  # 180 degree rotation
            pimg = img[::-1, ::-1]
        elif flip in ('t', 'T'):  # transpose (possible shape change)
            pimg = img.T
        elif flip in ('ccw90', 'r90'):  # rotate 90 (possible shape change)
            pimg = img.T[::-1, :]
        elif flip in ('cw90', 'r270'):  # rotate 270 (possible shape change)
            pimg = img.T[:, ::-1]
        else:
            pimg = img

        return pimg

    def _add(self, img, addend):
        # To avoid overflow/underflow, always convert to float
        if not np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)
        return img + addend

    def _gauss_laplace(self, img, sigma):
        return scipy.ndimage.gaussian_laplace(img, sigma)

    def _update_omega(self):
        """Update omega if there is a framelist"""
        if "omega" in self.metadata:
            omega = self.metadata["omega"]
            self.metadata["omega"] = omega[self._frames]
    #
    # ==================== API
    #

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape

    @property
    def metadata(self):
        # this is a modifiable copy of metadata of the original imageseries
        return self._meta

    def addop(self, key, func):
        """Add operation to processing options

        *key* - string to use to specify this op
        *func* - function to call for this op: f(data)
        """
        self._opdict[key] = func

    @property
    def oplist(self):
        """list of operations to apply"""
        return self._oplist
