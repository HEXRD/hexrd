"""Small example: load a saved matrix and apply it to Patrick's Eiger test image.

Run from the workspace root. The two paths identify matching example inputs.
"""

import numpy as np
from hexrd.core import imageseries
from hexrd.powder.cake import CakeMatrix

images = imageseries.open(
    '/home/patrick/data/hexrd/caking/test_dataset.h5',
    format='hdf5',
    path='/entry/data',
    dataname='data',
)
operator = CakeMatrix.load(
    'hexrd-wip-branches/cake-data-research/presentation/full-image.hexrdcake'
)
C = operator.matrix
assert images.shape == operator.composite_shape
frame = images[0]
cake = (C @ frame.ravel()).reshape(operator.shape)
cake[operator.mask] = np.nan
profile = np.nanmean(cake, axis=0)
