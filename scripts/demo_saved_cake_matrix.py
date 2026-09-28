"""Small example: load a saved matrix and apply it to Patrick's Eiger test image.

Run from the workspace root. The two paths identify matching example inputs.
"""

import numpy as np
from hexrd.core import imageseries
from hexrd.core.cake import CakeMatrix

images = imageseries.open(
    '/home/patrick/data/hexrd/caking/test_dataset.h5',
    format='hdf5',
    path='/entry/data',
    dataname='data',
)
operator = CakeMatrix.load(
    'hexrd-wip-branches/cake-data-research/presentation/coarse.hexrdcake'
)
x = operator.gather_active_composite_pixels(images[0])
del images
cake = (operator.matrix @ x).reshape(operator.shape)
cake[operator.mask] = np.nan
profile = np.nanmean(cake, axis=0)
