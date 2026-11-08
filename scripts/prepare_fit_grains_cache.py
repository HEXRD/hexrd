#!/usr/bin/env python3
"""Prepare an experimental fit-only FCH5 cache with bounded memory on Linux.

Only for nearest-neighbor fitting at the supplied threshold or higher. Keep
the original input for other analyses; existing readers do not enforce this
restriction. Requires canonical, sorted, unique sparse coordinates.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

# Two workers and the parent have independent hard allocation limits. Set these
# before numerical imports; do not turn a small preprocessing job into 8 GiB.
resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024**2,) * 2)
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402 -- install allocation/thread limits before numerical imports
import hdf5plugin  # noqa: E402
import numpy as np  # noqa: E402

from hexrd.hedm._direct_io import open_direct_hdf5  # noqa: E402


def initialize(source, threshold):
    global image_file, bounds, height, width, cutoff
    image_file = open_direct_hdf5(source, rdcc_nbytes=4 * 1024**2)
    bounds = image_file['frame_ids'][()].reshape(-1, 2)
    height, width = map(int, image_file['shape'][()])
    cutoff = threshold


def extract(frame):
    first, last = map(int, bounds[frame])
    previous, values, coordinates = -1, [], []
    for start in range(first, last, image_file['data'].chunks[0]):
        stop = min(last, start + image_file['data'].chunks[0])
        data = image_file['data'][start:stop, 0]
        ij = image_file['indices'][start:stop, :]
        keys = ij[:, 0].astype('i8') * width + ij[:, 1]
        if (np.any(ij < 0) or np.any(ij[:, 0] >= height)
                or np.any(ij[:, 1] >= width) or keys[0] <= previous
                or np.any(keys[1:] <= keys[:-1])):
            raise ValueError('Coalesce and sort sparse coordinates before filtering')
        previous = int(keys[-1])
        keep = (data > cutoff) | np.signbit(data) | ~np.isfinite(data)
        values.append(data[keep])
        coordinates.append(ij[keep])
    return (frame, last - first,
            np.concatenate(values) if values else np.empty(0, image_file['data'].dtype),
            np.concatenate(coordinates) if coordinates else
            np.empty((0, 2), image_file['indices'].dtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error('threshold must be finite and nonnegative')
    if args.output.exists() or args.output.resolve() == args.source.resolve():
        parser.error('output must be a new file, separate from the source')
    started, kept, total = time.perf_counter(), 0, 0
    temporary = args.output.with_name(args.output.name + '.incomplete')
    with open_direct_hdf5(args.source, rdcc_nbytes=4 * 1024**2) as source:
        if args.threshold < source.attrs.get('FIT_ONLY_THRESHOLD', args.threshold):
            parser.error('source was already filtered at a higher threshold')
        if (source['data'].ndim != 2 or source['data'].shape[1] != 1
                or source['indices'].shape != (len(source['data']), 2)
                or source['data'].chunks is None or source['indices'].chunks is None
                or source['data'].dtype.kind not in 'iuf'
                or source['indices'].dtype.kind not in 'iu'):
            parser.error('requires chunked FCH5 data (N, 1) and indices (N, 2)')
        frame_bounds = source['frame_ids'][()].reshape(-1, 2)
        if (len(frame_bounds) != int(source['nframes'][()])
                or np.any(frame_bounds < 0)
                or np.any(frame_bounds > len(source['data']))
                or np.any(frame_bounds[:, 0] > frame_bounds[:, 1])):
            parser.error('frame bounds must refer to valid source entries')
        output_bounds = np.empty_like(frame_bounds)
        ids = iter(np.argsort(frame_bounds[:, 0], kind='stable').tolist())
        with h5py.File(temporary, 'x', rdcc_nbytes=4 * 1024**2) as output:
            output.attrs.update(source.attrs)
            for name in ('shape', 'nframes', 'dtype', 'metadata'):
                source.copy(name, output)
            compression = hdf5plugin.Blosc(cname='zstd', clevel=5)
            arrays = [output.create_dataset(
                name, shape=(0, columns), maxshape=(None, columns),
                dtype=source[name].dtype, chunks=source[name].chunks, **compression
            ) for name, columns in (('data', 1), ('indices', 2))]
            with ProcessPoolExecutor(
                max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize, initargs=(str(args.source), args.threshold)
            ) as pool:
                pending = deque(pool.submit(extract, frame)
                                for frame in (next(ids, None) for _ in range(4))
                                if frame is not None)
                processed = pending_bytes = 0
                while pending:
                    frame, count, data, ij = pending.popleft().result()
                    end = kept + len(data)
                    output_bounds[frame] = kept, end
                    if len(data):
                        for array, value in zip(arrays, (data[:, None], ij)):
                            array.resize(end, axis=0)
                            array[kept:end] = value
                    kept, total, processed = end, total + count, processed + 1
                    pending_bytes += data.nbytes + ij.nbytes
                    following = next(ids, None)
                    if following is not None:
                        pending.append(pool.submit(extract, following))
                    if processed % 100 == 0 or pending_bytes >= 32 * 1024**2:
                        output.flush()
                        fd = output.id.get_vfd_handle()
                        os.fdatasync(fd)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        pending_bytes = 0
                        print(f'{processed}/{len(output_bounds)} frames', flush=True)
            output.create_dataset('frame_ids', data=output_bounds.ravel(), **compression)
            output.attrs['FIT_ONLY_THRESHOLD'] = args.threshold
            output.attrs['FIT_ONLY_SOURCE'] = str(args.source.resolve())
            output.attrs['FIT_ONLY_NOTE'] = __doc__
            output.flush()
            fd = output.id.get_vfd_handle()
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    temporary.rename(args.output)
    print(json.dumps(dict(seconds=time.perf_counter() - started,
                          source_entries=total, retained_entries=kept,
                          output_bytes=args.output.stat().st_size)))


if __name__ == '__main__':
    main()
