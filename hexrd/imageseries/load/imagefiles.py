"""Adapter class for list of image files
"""


# import sys
import os
# import logging
import glob

# # Put this before fabio import and reset level if you
# # want to control its import warnings.
# logging.basicConfig(level=logging.INFO)

import numpy as np
import fabio
import yaml

from . import ImageSeriesAdapter
from .metadata import yamlmeta
from ..imageseriesiter import ImageSeriesIterator


class ImageFilesImageSeriesAdapter(ImageSeriesAdapter):
    """collection of image files"""

    format = 'image-files'

    def __init__(self, fname):
        """Constructor for image files image series

        Parameters
        ----------
        fname: string | Path
           name of YAML file or bytestring of YAML contents
        """
        self._fname = fname
        self._load_yml()
        self._process_files()

    # @memoize
    def __len__(self):
        if self._maxframes_tot > 0:
            return min(self._nframes, self._maxframes_tot)
        else:
            return self._nframes

    def __getitem__(self, key):
        if self.singleframes:
            frame = None
            filename = self._files[key]
        else:
            (fnum, frame) = self._file_and_frame(key)
            filename = self.infolist[fnum].filename

        data = self._load_data(filename, frame)

        if self._dtype is not None:
            # !!! handled in self._process_files
            try:
                dinfo = np.iinfo(self._dtype)
            except(ValueError):
                dinfo = np.finfo(self._dtype)
            if np.max(data) > dinfo.max:
                raise RuntimeError("specified dtype will truncate image")
            return np.array(data, dtype=self._dtype)
        else:
            return data

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __str__(self):
        s = """==== imageseries from file list
    fabio class: %s
number of files: %s
        nframes: %s
          dtype: %s
          shape: %s
  single frames: %s
     """ % (self.fabioclass, len(self._files), len(self),
            self.dtype, self.shape, self.singleframes)
        return s

    @property
    def fname(self):
        return self._fname

    def _load_yml(self):
        EMPTY = 'empty-frames'
        MAXTOTF = 'max-total-frames'
        MAXFILF = 'max-file-frames'
        DTYPE = 'dtype'
        #
        # Check whether fname is a pathlib Path, a filename or YAML content.
        # If it has multiple lines, we consider it to be YAML content,
        # otherwise a file name.
        #
        is_str = isinstance(self.fname, str)
        nlines = len(self.fname.splitlines()) if is_str else 1
        if nlines > 1:
            d = yaml.safe_load(self.fname)
        else:
            with open(self._fname, "r") as f:
                d = yaml.safe_load(f)

        imgsd = d['image-files']
        dname = imgsd['directory']
        fglob = imgsd['files']
        self._files = []
        for g in fglob.split():
            self._files += glob.glob(os.path.join(dname, g))
        # !!! must sort due to the non-determinate nature of glob on various
        #     filesystems.  See https://github.com/HEXRD/hexrd/issues/263
        self._files.sort()
        self.optsd = d['options'] if 'options' else None
        self._empty = self.optsd[EMPTY] if EMPTY in self.optsd else 0
        self._maxframes_tot = self.optsd[MAXTOTF] \
            if MAXTOTF in self.optsd else 0
        self._maxframes_file = self.optsd[MAXFILF] \
            if MAXFILF in self.optsd else 0
        self._dtype = np.dtype(self.optsd[DTYPE]) \
            if DTYPE in self.optsd else None

        self._meta = yamlmeta(d['meta'])  # , path=imgsd)

    def _process_files(self):
        kw = {'empty': self._empty, 'max_frames': self._maxframes_file}
        fcl = None
        shp = None
        dtp = None
        nf = 0
        self._singleframes = True
        infolist = []
        for imgf in self._files:
            info = FileInfo(imgf, **kw)
            infolist.append(info)
            shp = self._checkvalue(shp, info.shape,
                                   "inconsistent image shapes")
            if self._dtype is not None:
                dtp = self._dtype

            else:
                dtp = self._checkvalue(
                    dtp, info.dtype,
                    "inconsistent image dtypes")
            fcl = self._checkvalue(fcl, info.fabioclass,
                                   "inconsistent image types")
            nf += info.nframes
            if info.nframes > 1:
                self._singleframes = False

        self._nframes = nf
        self._shape = shp
        self._dtype = dtp
        self._fabioclass = fcl
        self._infolist = infolist

    # from make_imageseries_h5
    @staticmethod
    def _checkvalue(v, vtest, msg):
        """helper: ensure value set conistently"""
        if v is None:
            val = vtest
        else:
            if vtest != v:
                raise ValueError(msg)
            else:
                val = v

        return val

    def _file_and_frame(self, key):
        """for multiframe images"""
        # allow for negatives (just use [nframes + key])
        nf = len(self)
        if key < -nf or key >= nf:
            msg = "frame out of range: %s" % key
            raise LookupError(msg)
        k = key if key >= 0 else (nf + key)

        frame = -nf - 1
        fnum = 0
        for info in self.infolist:
            if k < info.nframes:
                frame = k + info.empty
                break
            else:
                k -= info.nframes
                fnum += 1

        return fnum, frame

    # ======================================== API

    @property
    def metadata(self):
        """(read-only) Image sequence metadata

        Currently returns none
        """
        return self._meta

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def infolist(self):
        return self._infolist

    @property
    def fabioclass(self):
        return self._fabioclass

    @property
    def singleframes(self):
        """indicates whether all files are single frames"""
        return self._singleframes

    def _load_data(self, filename, frame=None):
        """Load data from a file, including processing if needed"""
        with fabio.open(filename) as img:
            if frame is None:
                data = img.data
            else:
                # Fabio does some arithmetic with the frame number.
                # This can cause overflows if np.uint32 is used, so
                # make sure we convert to a Python int before passing to fabio.
                data = img.getframe(int(frame)).data

        return _process_data(filename, data)


class FileInfo(object):
    """class for managing individual file information"""

    def __init__(self, filename, **kwargs):
        self.filename = filename
        with fabio.open(filename) as img:
            self._fabioclass = img.classname
            self._imgframes = img.nframes
            self.dat = _process_data(filename, img.data)

        d = kwargs.copy()
        self._empty = d.pop('empty', 0)
        # user may set max-frames to 0, indicating use all frames
        self._maxframes = d.pop('max_frames', 0)
        if self._maxframes == 0:
            self._maxframes = self._imgframes
        if self._empty >= self._imgframes:
            msg = "more empty frames than images: %s" % self.filename
            raise ValueError(msg)

    def __str__(self):
        s = """==== image file
       name: %s
fabio class: %s
     frames: %s
      dtype: %s
      shape: %s\n""" % (self.filename, self.fabioclass,
                        self.nframes, self.dtype, self.shape)

        return s

    @property
    def empty(self):
        return self._empty

    @property
    def shape(self):
        return self.dat.shape

    @property
    def dtype(self):
        return self.dat.dtype

    @property
    def fabioclass(self):
        return self._fabioclass

    @property
    def nframes(self):
        return min(self._maxframes, self._imgframes - self.empty)


def _process_data(filename, data):
    # Perform any necessary processing on the data before returning it
    # For example, gel files need to be decompressed.
    process_funcs = {
        '.gel': _process_gel_data,
    }

    ext = os.path.splitext(filename)[-1]
    if ext in process_funcs:
        data = process_funcs[ext](data)

    return data


GEL_SCALE_FACTOR = 2.9452155399372724e-07


def _process_gel_data(array):
    """Convert a gel data array to regular image data"""
    # An inversion seems to be necessary for our examples
    array = np.invert(array)
    return array.astype(np.float64)**2 * GEL_SCALE_FACTOR
