#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:30:53 2019

@author: bernier2
"""

import os

import numpy as np

from hexrd import imageseries

from skimage import io


# dirs
working_dir = '/Users/Shared/APS/PUP_AFRL_Feb19'
image_dir = os.path.join(working_dir, 'image_data')

samp_name = 'ceria_cal'
scan_number = 0

tif_file_template = samp_name + '_%06d-%s.tif'

raw_data_dir_template = os.path.join(
    image_dir,
    'raw_images_%s_%06d-%s.yml'
)
yml_string = """
image-files:
  directory: %s
  files: "%s"

options:
  empty-frames: 0
  max-frames: 0
meta:
  panel: %s
"""


ims = imageseries.open(
    os.path.join(image_dir, 'ceria_cal.h5'),
    'hdf5',
    path='/imageseries'
)

metadata = ims.metadata
det_keys = np.array(metadata['panels'], dtype=str)

for i, det_key in enumerate(det_keys):
    yml_file = open(
        raw_data_dir_template % (samp_name, scan_number, det_key),
        'w'
    )
    tiff_fname = tif_file_template % (scan_number, det_key)
    print(yml_string % (image_dir, tiff_fname, det_key),
          file=yml_file)
    io.imsave(
        os.path.join(image_dir, tiff_fname),
        ims[i]
    )
