""" Adapter class for raw image reader"""
import os
import threading

import numpy as np
import yaml

from . import ImageSeriesAdapter
from ..imageseriesiter import ImageSeriesIterator


class RawImageSeriesAdapter(ImageSeriesAdapter):
    """collection of images in HDF5 format"""

    format = 'raw-image'

    def __init__(self, fname, **kwargs):
        """Image data in custom format

        Parameters
        ----------
        fname: string or Path
           name of input YAML file describing the format
        """
        self.fname = fname
        with open(fname, "r") as f:
            y = yaml.safe_load(f)

        self.fname = y['filename']
        self.dtype = self._get_dtype(y['scalar'])
        self._shape = tuple((int(si) for si in y['shape'].split()))
        self._frame_size = self._shape[0] * self._shape[1]
        self._frame_bytes = self._frame_size * self.dtype.itemsize
        self._frame_read_lock = threading.Lock()
        self.skipbytes = y['skip']
        self._len = self._get_length()
        self._meta = dict()
        self.kwargs = kwargs

        # Open file for reading.
        self.f = open(self.fname, "r")

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
        nbytes = os.path.getsize(self.fname)
        if nbytes % self._frame_bytes == self.skipbytes:
            nframes = (nbytes - self.skipbytes) // self._frame_bytes
        else:
            msg = (
                f"Total number of bytes ({nbytes}) does not work with "
                f"skipbytes ({self.skipbytes}) and "
                f"_frame_size ({self._frame_size}). Check the skipbytes value."
            )
            raise ValueError(msg)

        return nframes

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
        count = key * self._frame_bytes + self.skipbytes

        # Ensure reading a frame the file is thread-safe
        with self._frame_read_lock:
            self.f.seek(count, 0)
            frame = np.fromfile(self.f, self.dtype, count=self._frame_size)

        return frame.reshape(self.shape)

    @property
    def shape(self):
        """shape of individual image frame"""
        return self._shape

    @property
    def metadata(self):
        """imageseries metadata"""
        return self._meta
