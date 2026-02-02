"""Write imageseries to various formats"""

import abc
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing
import os
import threading
import warnings

import numpy as np
import h5py
import hdf5plugin
import yaml

from hexrd.core.matrixutil import extract_ijv
from hexrd.core.utils.hdf5 import unwrap_dict_to_h5

logger = logging.getLogger(__name__)

MAX_NZ_FRACTION = 0.1  # 10% sparsity trigger for frame-cache write


# =============================================================================
# METHODS
# =============================================================================


def write(ims, fname, fmt, **kwargs):
    """write imageseries to file with options

    Parameters
    ----------
    ims: Imageseries
       the imageseries to write
    fname: str
       the name of the HDF5 file to write
    fmt: str
       format name of the imageseries
    kwargs: dict
       options specific to format
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


class Writer(object, metaclass=_RegisterWriter):
    """Base class for writers

    Parameters
    ----------
    ims: Imageseries
       the imageseries to write
    fname: str
       the name of the HDF5 file to write
    kwargs: dict
       options specific to format
    """

    fmt: str | None = None

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

    @property
    def fname(self):
        return self._fname

    @property
    def fname_dir(self):
        return self._fname_dir

    @property
    def opts(self):
        return self._opts


class WriteH5(Writer):
    """Write imageseries in HDF5 file

    Parameters
    ----------
    ims: Imageseries
       the imageseries to write
    fname: str
       the name of the HDF5 file to write
    path: str, required
       the path in HDF5 file
    gzip: int 0-9
       0 turns off compression, default=1
    chunk_rows: int
       number of rows per chunk; default is all
    shuffle: bool
       shuffle HDF5 data
    """

    fmt = 'hdf5'
    dflt_gzip = 1
    dflt_chrows = 0
    dflt_shuffle = True

    def __init__(self, ims, fname, **kwargs):
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

        ds = g.create_dataset(
            'images', (self._nframes, s0, s1), self._dtype, **self.h5opts
        )

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


class WriteFrameCache(Writer):
    """write frame cache imageseries

    The default write option is to save image data and metadata all
    in a single npz file. The original option was to have a YAML file
    as the primary file and a specified cache file for the images; this
    method is deprecated.

    Parameters
    ----------
    ims: Imageseries instance
       the imageseries to write
    fname: str or Path
       name of file to write;
    threshold: float
       threshold value for image, at or below which values are zeroed
    cache_file: str or Path, optional
       name of the npz file to save the image data, if not given in the
       `fname` argument; for YAML format (deprecated), this is required
    style: str, type of file to use for saving. options are:
       - 'npz' for saving in a numpy compressed file
       - 'fch5' for saving in the HDF5-based frame-cache format
    max_workers: int, optional
       The max number of worker threads for multithreading. Defaults to
       the number of CPUs.
    """

    fmt = 'frame-cache'

    def __init__(self, ims, fname, style='npz', **kwargs):
        Writer.__init__(self, ims, fname, **kwargs)
        self._thresh = self._opts['threshold']
        self._cache, self.cachename = self._set_cache()

        ncpus = multiprocessing.cpu_count()
        self.max_workers = kwargs.get('max_workers', ncpus)
        supported_formats = ['npz', 'fch5']
        if style not in supported_formats:
            raise TypeError(
                f"Unknown file style for writing framecache: {style}. "
                f"Supported formats are {supported_formats}"
            )
        self.style = style

        self.hdf5_compression = hdf5plugin.Blosc(cname="zstd", clevel=5)

    def _set_cache(self):
        cf = self.opts.get('cache_file')

        if cf is None:
            cachename = cache = self.fname
        else:
            if os.path.isabs(cf):
                cache = cf
            else:
                cdir = os.path.dirname(self.fname)
                cache = os.path.join(cdir, cf)
            cachename = cf

        return cache, cachename

    @property
    def cache(self):
        return self._cache

    def _process_meta(self, save_omegas=False):
        d = {}
        for k, v in list(self._meta.items()):
            if isinstance(v, dict):
                logger.warning(
                    'NPZ files do not support nested metadata. '
                    f'The metadata key "{k}" will not be written out.'
                )
                continue

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
        datad = {
            'file': self._cachename,
            'dtype': str(self._ims.dtype),
            'nframes': len(self._ims),
            'shape': list(self._ims.shape),
        }
        info = {'data': datad, 'meta': self._process_meta(save_omegas=True)}
        with open(self._fname, "w") as f:
            yaml.safe_dump(info, f)

    def _write_frames(self):
        if self.style == 'npz':
            self._write_frames_npz()
        elif self.style == 'fch5':
            self._write_frames_fch5()

    def _check_sparsity(self, frame_id, count, buff_size):
        # check the sparsity
        #
        # FIXME: formalize this a little better
        # ???: maybe set a hard limit of total nonzeros for the imageseries
        # ???: could pass as a kwarg on open
        fullness = count / float(buff_size)
        if fullness > MAX_NZ_FRACTION:
            sparseness = 100.0 * (1 - fullness)
            msg = "frame %d is %4.2f%% sparse (cutoff is 95%%)" % (
                frame_id,
                sparseness,
            )
            warnings.warn(msg)

    def _write_frames_npz(self):
        """also save shape array as originally done (before yaml)"""
        buff_size = self._ims.shape[0] * self._ims.shape[1]
        arrd = {}

        num_workers = min(self.max_workers, len(self._ims))

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

            # wrapper to find (sparse) pixels above threshold
            count = extract_ijv(self._ims[i], self._thresh, rows, cols, vals)

            self._check_sparsity(i, count, buff_size)

            arrd[f'{i}_row'] = rows[:count].copy()
            arrd[f'{i}_col'] = cols[:count].copy()
            arrd[f'{i}_data'] = vals[:count].copy()

        kwargs = {
            'max_workers': num_workers,
            'initializer': assign_buffer_id,
        }
        with ThreadPoolExecutor(**kwargs) as executor:
            # Evaluate the results via `list()`, so that if an exception is
            # raised in a thread, it will be re-raised and visible to the user.
            list(executor.map(extract_data, range(len(self._ims))))

        arrd['shape'] = self._ims.shape
        arrd['nframes'] = len(self._ims)
        arrd['dtype'] = str(self._ims.dtype).encode()
        arrd.update(self._process_meta())
        np.savez_compressed(self.cache, **arrd)

    def _write_frames_fch5(self):
        """Write framecache into an hdf5 file. The file will use three
        datasets for the framecache:
        - 'data': (m,1) array holding the datavalues of all frames. `m` is
          evaluated upon runtime
        - 'indices': (m,2) array holding the row& col information for the
          values in data. 'data' together within 'indices' represent tha data
          using the CSR format for sparse matrices.
        - 'frame_ids`: (2*nframes)  holds the range that the i-th frame
          occupies in the above arrays. i.e. the information of the i-th frame
          can be accessed using:

          data_i = data[frame_ids[2*i]:frame_ids[2*i+1]] and
          indices_i = indices[frame_ids[2*i]:frame_ids[2*i+1]]
        """
        max_frame_size = self._ims.shape[0] * self._ims.shape[1]
        nframes = len(self._ims)
        shape = self._ims.shape
        data_dtype = self._ims.dtype

        frame_indices = np.empty((2 * nframes,), dtype=np.uint64)
        data_dataset = None
        indices_dataset = None
        file_position = 0
        total_size = 0

        common_lock = threading.Lock()
        thread_local = threading.local()

        # creating an array in memory will fail if data is too big or threshold
        # too low, so we write to the file while iterating the frames
        with h5py.File(self.cache, "w") as h5f:
            h5f.attrs['HEXRD_FRAMECACHE_VERSION'] = 1
            h5f["shape"] = shape
            h5f["nframes"] = nframes
            h5f["dtype"] = str(np.dtype(self._ims.dtype)).encode("utf-8")
            metadata = h5f.create_group("metadata")
            unwrap_dict_to_h5(metadata, self._meta.copy())

            def initialize_buffers():
                thread_local.data = np.empty((max_frame_size, 1), dtype=self._ims.dtype)
                thread_local.indices = np.empty((max_frame_size, 2), dtype=np.uint16)

            def single_array_write_thread(i):
                nonlocal file_position, total_size
                im = self._ims[i]
                row_slice = thread_local.indices[:, 0]
                col_slice = thread_local.indices[:, 1]
                data_slice = thread_local.data[:, 0]
                count = extract_ijv(im, self._thresh, row_slice, col_slice, data_slice)

                self._check_sparsity(i, count, max_frame_size)

                # get the range this thread is doing to write into the file
                start_file = 0
                end_file = 0
                with common_lock:
                    start_file = file_position
                    file_position += count
                    end_file = file_position
                    total_size += end_file - start_file
                # write within the appropriate ranges
                data_dataset[start_file:end_file, :] = thread_local.data[:count, :]
                indices_dataset[start_file:end_file, :] = thread_local.indices[
                    :count, :
                ]
                frame_indices[2 * i] = start_file
                frame_indices[2 * i + 1] = end_file

            kwargs = {
                "max_workers": self.max_workers,
                "initializer": initialize_buffers,
            }

            data_dataset = h5f.create_dataset(
                "data",
                shape=(nframes * max_frame_size, 1),
                dtype=data_dtype,
                compression=self.hdf5_compression,
            )
            indices_dataset = h5f.create_dataset(
                "indices",
                shape=(nframes * max_frame_size, 2),
                dtype=np.uint16,
                compression=self.hdf5_compression,
            )
            with ThreadPoolExecutor(**kwargs) as executor:
                # Evaluate the results via `list()`, so that if an exception is
                # raised in a thread, it will be re-raised and visible to
                # the user.
                list(executor.map(single_array_write_thread, range(nframes)))

                # update the sizes of the dataset to match the amount of data
                # that have been actually written
                data_dataset.resize(total_size, axis=0)
                indices_dataset.resize(total_size, axis=0)

            h5f.create_dataset(
                "frame_ids",
                data=frame_indices,
                compression=self.hdf5_compression,
            )

    def write(self, output_yaml=False):
        """writes frame cache for imageseries

        presumes sparse forms are small enough to contain all frames
        """
        self._write_frames()
        if output_yaml:
            warnings.warn(
                "YAML output for frame-cache is deprecated", DeprecationWarning
            )
            self._write_yml()
