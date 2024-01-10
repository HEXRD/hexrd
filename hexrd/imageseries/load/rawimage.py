""" Adapter class for raw image reader
"""
import warnings

import numpy as np
import yaml

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator


class RawImageSeriesAdapter(ImageSeriesAdapter):
    """collection of images in HDF5 format"""

    format = 'raw-image'

    def __init__(self, fname, **kwargs):
        """Image data in custom format

        *fname* - filename of the data file
        *kwargs* - keyword arguments (none required)
        """
        self.fname = fname
        with open(fname, "r") as f:
            y = yaml.safe_load(f)

        self.fname = y['filename']
        self.dtype = self._get_dtype(y['scalar'])
        self._shape = tuple((int(si) for si in y['shape'].split()))
        self.framesize = self._shape[0]*self._shape[1]
        self.skipbytes = y['skip']
        self._len = self._get_length()
        self._meta = dict()
        self.kwargs = kwargs

        # Prepare to read
        self.iframe = -1

    def _get_dtype(self, scalar):
        numtype = scalar['type']
        bytes_ = scalar['bytes']
        signed = scalar['signed']
        if scalar['endian'] == "little":
            little = True
        elif scalar['endian'] == "big":
            little = False
        else:
            raise ValueError('endian must be "big" for "little"')

        return np.dtype(self.typechars(numtype, bytes_, signed, little))

    def _get_length(self):
        """Read file and determine length"""
        iframe = 0
        with open(self.fname, "r") as f:
            _ = np.fromfile(f, np.byte, count=self.skipbytes)
            # now keep reading frames until EOF
            moreframes = True
            while moreframes:
                _ = np.fromfile(f, self.dtype, count=self.framesize)
                if len(_) == self.framesize:
                    iframe += 1
                else:
                    moreframes = False

        return iframe

    @staticmethod
    def typechars(numtype, bytes_=4, signed=False, little=True):
        """Return byte-type for data type and endianness

        numtype (str) - "i", "f", "d", "b"  for int, float, double or bool
        bytes - number of bytes: 1,2,4, or 8 for ints only
        signed (bool) - true for signed ints, false for unsigned
        little (bool) - true for little endian
        """
        intbytes = {
            1: "b",
            2: "h",
            4: "i",
            8: "l"
        }

        typechar = {
            "f": "f",
            "d": "d",
            "b": "?"
        }

        if numtype == "i":
            char = intbytes[bytes_]
            if not signed:
                char = char.upper()
        else:
            char = typechar[numtype]

        return "<"+char if little else ">"+char

    def __len__(self):
        return self._len

    def __iter__(self):
        return ImageSeriesIterator(self)

    def __getitem__(self, key):
        if self.iframe < 0:
            self.f = open(self.fname, "r")
            _ = np.fromfile(self.f, np.byte, count=self.skipbytes)
            self.iframe = 0
        if key == 0:
            self.f.seek(0, 0)
            _ = np.fromfile(self.f, np.byte, count=self.skipbytes) #dcp fix to make sure bytes are properly skipped
            self.iframe = 0
        if key != self.iframe:
            msg = "frame %d not available, series must be read in sequence!"
            raise ValueError(msg % key)
        frame = np.fromfile(self.f, self.dtype, count=self.framesize)
        self.iframe += 1
        if self.iframe == len(self):
            self.iframe = -1
            self.f.close()

        return frame.reshape(self.shape)

    @property
    def shape(self):
        """shape of individual image frame"""
        return self._shape

    @property
    def metadata(self):
        """imageseries metadata"""
        return self._meta
