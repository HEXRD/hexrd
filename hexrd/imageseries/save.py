"""Write imageseries to various formats"""

import abc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import threading
import warnings

import numpy as np
import h5py
import yaml

from hexrd.matrixutil import extract_ijv

MAX_NZ_FRACTION = 0.1    # 10% sparsity trigger for frame-cache write


# =============================================================================
# METHODS
# =============================================================================


def write(ims, fname, fmt, **kwargs):
    """write imageseries to file with options

    *ims* - an imageseries
    *fname* - name of file or an h5py file for writing HDF5
    *fmt* - a format string
    *kwargs* - options specific to format
    """
    wcls = _Registry.getwriter(fmt)
    w = wcls(ims, fname, **kwargs)
    w.write()


# Registry
class _RegisterWriter(abc.ABCMeta):

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        _Registry.register(cls)


class _Registry(object):
    """Registry for imageseries writers"""
    writer_registry = dict()

    @classmethod
    def register(cls, wcls):
        """Register writer class"""
        if wcls.__name__ != 'Writer':
            cls.writer_registry[wcls.fmt] = wcls

    @classmethod
    def getwriter(cls, name):
        """return instance associated with name"""
        return cls.writer_registry[name]
    #
    pass  # end class


class Writer(object, metaclass=_RegisterWriter):
    """Base class for writers"""
    fmt = None

    def __init__(self, ims, fname, **kwargs):
        self._ims = ims
        self._shape = ims.shape
        self._dtype = ims.dtype
        self._nframes = len(ims)
        self._meta = ims.metadata
        self._fname = fname
        self._opts = kwargs

        if isinstance(fname, h5py.File):
            filename = fname.filename
        else:
            filename = fname

        # split filename into components
        tmp = os.path.split(filename)
        self._fname_dir = tmp[0]
        tmp = os.path.splitext(tmp[1])
        self._fname_base = tmp[0]
        self._fname_suff = tmp[1]

    pass  # end class


class WriteH5(Writer):
    fmt = 'hdf5'
    dflt_gzip = 1
    dflt_chrows = 0
    dflt_shuffle = True

    def __init__(self, ims, fname, **kwargs):
        """Write imageseries in HDF5 file

           Required Args:
           path - the path in HDF5 file

           Options:
           gzip - 0-9; 0 turns off compression; 4 is default
           chunk_rows - number of rows per chunk; default is all
           """
        Writer.__init__(self, ims, fname, **kwargs)
        self._path = self._opts['path']

    #
    # ======================================== API
    #
    def write(self):
        """Write imageseries to HDF5 file"""
        if isinstance(self._fname, h5py.File):
            f = self._fname
        else:
            f = h5py.File(self._fname, "w")

        g = f.create_group(self._path)
        s0, s1 = self._shape

        ds = g.create_dataset('images', (self._nframes, s0, s1), self._dtype,
                              **self.h5opts)

        for i in range(self._nframes):
            ds[i, :, :] = self._ims[i]

        # add metadata
        for k, v in list(self._meta.items()):
            if np.issubdtype(v.dtype, 'U'):
                # HDF5 can't handle unicode strings.
                # Turn it into a regular string.
                v = v.astype('S')

            g.attrs[k] = v

    @property
    def h5opts(self):
        d = {}

        # shuffle
        shuffle = self._opts.pop('shuffle', self.dflt_shuffle)
        d['shuffle'] = shuffle

        # compression
        compress = self._opts.pop('gzip', self.dflt_gzip)
        if compress > 9:
            raise ValueError('gzip compression cannot exceed 9: %s' % compress)
        if compress > 0:
            d['compression'] = 'gzip'
            d['compression_opts'] = compress

        # chunk size
        s0, s1 = self._shape
        chrows = self._opts.pop('chunk_rows', self.dflt_chrows)
        if chrows < 1 or chrows > s0:
            chrows = s0
        d['chunks'] = (1, chrows, s1)

        return d

    pass  # end class


class WriteFrameCache(Writer):
    """info from yml file"""
    fmt = 'frame-cache'

    def __init__(self, ims, fname, **kwargs):
        """write yml file with frame cache info

        kwargs has keys:

        cache_file - name of array cache file
        meta - metadata dictionary
        """
        Writer.__init__(self, ims, fname, **kwargs)
        self._thresh = self._opts['threshold']
        cf = kwargs['cache_file']
        if os.path.isabs(cf):
            self._cache = cf
        else:
            cdir = os.path.dirname(fname)
            self._cache = os.path.join(cdir, cf)
        self._cachename = cf
        self.max_workers = kwargs.get('max_workers', None)

    def _process_meta(self, save_omegas=False):
        d = {}
        for k, v in list(self._meta.items()):
            if isinstance(v, np.ndarray) and save_omegas:
                # Save as a numpy array file
                # if file does not exist (careful about directory)
                #    create new file

                cdir = os.path.dirname(self._cache)
                b = self._fname_base
                fname = os.path.join(cdir, "%s-%s.npy" % (b, k))
                if not os.path.exists(fname):
                    np.save(fname, v)

                # add trigger in yml file
                d[k] = "! load-numpy-array %s" % fname
            else:
                d[k] = v

        return d

    def _write_yml(self):
        datad = {'file': self._cachename, 'dtype': str(self._ims.dtype),
                 'nframes': len(self._ims), 'shape': list(self._ims.shape)}
        info = {'data': datad, 'meta': self._process_meta(save_omegas=True)}
        with open(self._fname, "w") as f:
            yaml.safe_dump(info, f)

    def _write_frames(self):
        """also save shape array as originally done (before yaml)"""
        buff_size = self._ims.shape[0]*self._ims.shape[1]
        arrd = {}

        ncpus = multiprocessing.cpu_count()
        max_workers = ncpus if self.max_workers is None else self.max_workers
        num_workers = min(max_workers, len(self._ims))

        row_buffers = np.empty((num_workers, buff_size), dtype=np.uint16)
        col_buffers = np.empty((num_workers, buff_size), dtype=np.uint16)
        val_buffers = np.empty((num_workers, buff_size), dtype=self._ims.dtype)
        buffer_ids = {}
        assign_buffer_lock = threading.Lock()

        def assign_buffer_id():
            with assign_buffer_lock:
                buffer_ids[threading.get_ident()] = len(buffer_ids)

        def extract_data(i):
            buffer_id = buffer_ids[threading.get_ident()]
            rows = row_buffers[buffer_id]
            cols = col_buffers[buffer_id]
            vals = val_buffers[buffer_id]

            # FIXME: in __init__() of ProcessedImageSeries:
            # 'ProcessedImageSeries' object has no attribute '_adapter'

            # wrapper to find (sparse) pixels above threshold
            count = extract_ijv(self._ims[i], self._thresh,
                                rows, cols, vals)

            # check the sparsity
            #
            # FIXME: formalize this a little better
            # ???: maybe set a hard limit of total nonzeros for the imageseries
            # ???: could pass as a kwarg on open
            fullness = count / float(buff_size)
            if fullness > MAX_NZ_FRACTION:
                sparseness = 100.*(1 - fullness)
                msg = "frame %d is %4.2f%% sparse (cutoff is 95%%)" \
                    % (i, sparseness)
                warnings.warn(msg)

            arrd[f'{i}_row'] = rows[:count].copy()
            arrd[f'{i}_col'] = cols[:count].copy()
            arrd[f'{i}_data'] = vals[:count].copy()

        kwargs = {
            'max_workers': num_workers,
            'initializer': assign_buffer_id,
        }
        with ThreadPoolExecutor(**kwargs) as executor:
            executor.map(extract_data, range(len(self._ims)))

        arrd['shape'] = self._ims.shape
        arrd['nframes'] = len(self._ims)
        arrd['dtype'] = str(self._ims.dtype).encode()
        arrd.update(self._process_meta())
        np.savez_compressed(self._cache, **arrd)

    def write(self, output_yaml=False):
        """writes frame cache for imageseries

        presumes sparse forms are small enough to contain all frames
        """
        self._write_frames()
        if output_yaml:
            self._write_yml()
