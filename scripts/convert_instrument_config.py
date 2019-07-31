#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:33:57 2019

@author: joel
"""

import yaml
from hexrd import rotations
from hexrd.transforms import xfcapi

icfg = yaml.safe_load(open('/Users/joel/Downloads/Hydra_Feb19.yml', 'r'))
               
for k, d in icfg['detectors'].items():
    tilt_angles = np.r_[d['transform']['tilt_angles']]
    rmat_d = xfcapi.makeDetectorRotMat(tilt_angles)
    phi, n = rotations.angleAxisOfRotMat(rmat_d)
    print("%s: [%.8e, %.8e, %.8e]" % tuple([k, *(phi*n.flatten())]))
