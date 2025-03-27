from dataclasses import dataclass

import numpy as np


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
