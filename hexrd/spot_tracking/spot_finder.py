"""Spot finding

NOTE: you must have OpenCV installed to import this file
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import center_of_mass

from .spot import Spot

LOGGER = logging.getLogger(__name__)


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
