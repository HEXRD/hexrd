"""Class for processing individual frames"""
import copy

import numpy as np

from .baseclass import ImageSeries


class ProcessedImageSeries(ImageSeries):
    """Images series with mapping applied to frames"""
    FLIP = 'flip'
    DARK = 'dark'
    RECT = 'rectangle'

    _opdict = {}

    def __init__(self, imser, oplist, **kwargs):
        """imsageseries based on existing one with image processing options

        *imser* - an existing imageseries
        *oplist* - list of processing operations;
                   a list of pairs (key, data) pairs, with key specifying the
                   operation to perform using specified data

        *keyword args*
        'frame_list' - specify subset of frames by list

        """
        self._imser = imser
        self._meta = copy.deepcopy(imser.metadata)
        self._oplist = oplist
        self._frames = kwargs.pop('frame_list', None)
        self._hasframelist = (self._frames is not None)

        self.addop(self.DARK, self._subtract_dark)
        self.addop(self.FLIP, self._flip)
        self.addop(self.RECT, self._rectangle)

    def __getitem__(self, key):
        return self._process_frame(self._get_index(key))

    def _get_index(self, key):
        return self._frames[key] if self._hasframelist else key

    def __len__(self):
        return len(self._frames) if self._hasframelist else len(self._imser)

    def __iter__(self):
        if self._hasframelist:
            return (self._imser[i] for i in self._frames)

        return self._imser.__iter__()

    def _process_frame(self, key):
        # note: key refers to original imageseries
        img = np.copy(self._imser[key])
        for k, d in self.oplist:
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

    def _rectangle(self, img, r):
        # restrict to rectangle
        return img[r[0, 0]:r[0, 1], r[1, 0]:r[1, 1]]

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

    @classmethod
    def addop(cls, key, func):
        """Add operation to processing options

        *key* - string to use to specify this op
        *func* - function to call for this op: f(data)
        """
        cls._opdict[key] = func

    @property
    def oplist(self):
        """list of operations to apply"""
        return self._oplist

    pass  # end class
