from typing import Generator

import h5py
import numpy as np

from hexrd.imageseries import ImageSeries

from spot_finder import SpotFinder


def find_spots(
    ims_dict: dict[str, ImageSeries],
    finder: SpotFinder,
) -> Generator[tuple[str, int, np.ndarray], None, None]:
    # Find spots and return a generator containing each spots table
    # You could use this with the 'function' imageseries, for example,
    # for live streaming.
    # The generator yields (det_key, frame_index, spots_array)
    num_frames = len(next(iter(ims_dict.values())))
    for frame_index in range(num_frames):
        for det_key, ims in ims_dict.items():
            img = ims[frame_index]
            spots = finder.find_spots(img)
            spots = np.vstack(spots) if spots else np.empty((0,))
            yield det_key, frame_index, spots


def write_spots(
    ims_dict: dict[str, ImageSeries],
    finder: SpotFinder,
    file: h5py.File,
):
    # Write the spots to an h5py.File
    for det_key, frame_idx, spots in find_spots(ims_dict, finder):
        path = f'{det_key}/{frame_idx}'
        file[path] = spots
