"""Optional direct FCH5 input for benchmarks with strict host RAM constraints.

RLIMIT_AS bounds allocations, but not the kernel's filesystem page cache. Linux
O_DIRECT bypasses that cache. HDF5's file-object driver supplies arbitrary byte
requests; this adapter aligns them without changing their returned contents.
Unsupported filesystems fail rather than silently reverting to cached I/O.
"""

import io
import mmap
import os


class DirectFile(io.RawIOBase):
    alignment = 4096

    def __init__(self, filename):
        if not hasattr(os, 'O_DIRECT') or not hasattr(os, 'preadv'):
            raise OSError('Direct image reads require Linux O_DIRECT and preadv')
        self.fd = os.open(filename, os.O_RDONLY | os.O_DIRECT)
        self.size = os.fstat(self.fd).st_size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        self._checkClosed()
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.size
        elif whence != os.SEEK_SET:
            raise ValueError('invalid seek mode')
        if offset < 0:
            raise ValueError('negative seek position')
        self.position = offset
        return offset

    def readinto(self, target):
        self._checkClosed()
        target = memoryview(target).cast('B')
        wanted = min(len(target), max(0, self.size - self.position))
        if not wanted:
            return 0
        first = self.position // self.alignment * self.alignment
        skip = self.position - first
        size = (skip + wanted + self.alignment - 1) // self.alignment * self.alignment
        # Anonymous mmap supplies an aligned buffer and counts against RLIMIT_AS.
        with mmap.mmap(-1, size) as buffer:
            read = os.preadv(self.fd, [buffer], first)
            count = max(0, min(wanted, read - skip))
            target[:count] = buffer[skip : skip + count]
        self.position += count
        return count

    def close(self):
        if not self.closed:
            fd = getattr(self, 'fd', None)
            if fd is not None:
                os.close(fd)
                self.fd = None
            super().close()


def open_direct_hdf5(filename, **kwargs):
    import h5py

    class DirectHdf5(h5py.File):
        def close(self):
            try:
                super().close()
            finally:
                raw.close()

    raw = DirectFile(filename)
    try:
        return DirectHdf5(raw, 'r', driver='fileobj', **kwargs)
    except BaseException:
        raw.close()
        raise
