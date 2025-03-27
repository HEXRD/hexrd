import logging

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import center_of_mass

LOGGER = logging.getLogger(__name__)


@dataclass
class Spot:
    i: float
    '''The row pixel coordinate of the spot: img[i, j]'''
    j: float
    '''The column pixel coordinate of the spot: img[i, j]'''
    w: int
    '''The full width of the bounding box of the spot'''
    bounding_box: tuple[int, int, int, int]
    '''The bounding box of the spot: (rmin, cmin, rmax, cmax)'''
    max: float
    '''The maximum pixel value in the spot'''
    sum: float
    '''The sum of all pixel values in the spot'''

    def __array__(self) -> np.ndarray:
        return np.array([
            self.i,
            self.j,
            self.w,
            *self.bounding_box,
            self.max,
            self.sum,
        ])

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Spot':
        a = array
        return Spot(
            a[0],
            a[1],
            int(a[2]),
            tuple(a[3:7].tolist()),
            a[7],
            a[8],
        )


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax


class SpotFinder:
    def __init__(
        self,
        threshold: float = 25.0,
        minimum_separation: float = 1,
        min_area: int = 1,
        max_area: int = 1000,
    ):
        self.threshold = threshold
        self.minimal_separation = minimum_separation
        self.min = min_area
        self.max = max_area

    def find_spots(self, img: np.ndarray) -> list[Spot]:
        '''
        Finds blobs in the image using the configured parameters via either blob_log or blob_dog

        Returns the blobs found in the image as coordinates and widths

        Returns: A list of the spots found in the image
        '''
        binary_image = img > self.threshold

        _n, all_labels, stats, _binary_centroids = (
            cv2.connectedComponentsWithStats(
                binary_image.astype(np.uint8), connectivity=8
            )
        )
        spots: list[Spot] = []
        for l, (lj, li, dj, di, area) in enumerate(stats):
            if l == 0:
                continue
            if not (self.min <= area <= self.max):
                continue
            pixels = img[li : li + di, lj : lj + dj]
            labels = all_labels[li : li + di, lj : lj + dj]
            mask = labels == l

            i: int
            j: int
            i, j = center_of_mass(pixels, labels, l)  # type: ignore

            i += li
            j += lj

            spots.append(
                Spot(
                    i,
                    j,
                    max(di, dj),
                    (li, lj, li + di, lj + dj),
                    np.max(pixels[mask]),
                    np.sum(pixels[mask]),
                )
            )
        LOGGER.debug('Detected %d blobs', len(spots))
        return spots
