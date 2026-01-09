# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Saransh Singh <saransh1@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================

from hexrd.core import constants
import numpy as np

eps = constants.sqrt_epsf


def hsl2rgb(hsl):
    '''
    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       10/29/2020 SS 1.0 original
    >> @DETAILS:    routines to convert between hsl to rgb representations, the
                    input and output will have all components in the range [0,1]
    '''

    '''
    first check the shape of the hsl array. it has to be nx3 array
    it is assumed that the hsl array has values in [0,1] for the 
    different components
    '''
    hsl = np.atleast_2d(hsl)
    hsl[np.abs(hsl) < eps] = 0.0
    hsl[np.abs(hsl - np.ones(hsl.shape)) < eps] = 1.0

    if (hsl.min() < 0.0) or (hsl.max() > 1.0):
        raise RuntimeError("value of not in range [0,1]. normalizing before conversion")

    if hsl.ndim != 2:
        raise RuntimeError("hsl_rgb: shape of hsl array is invalid.")
    rgb = np.zeros(hsl.shape)

    '''
    calculate the different factors needed for the conversion
    '''
    H = hsl[:, 0]
    S = hsl[:, 1]
    L = hsl[:, 2]

    C = (1.0 - np.abs(2.0 * L - 1.0)) * S
    X = (1.0 - np.abs(np.mod(6 * H, 2) - 1.0)) * C
    m = L - C / 2.0

    case = np.floor(6.0 * H).astype(np.int32)

    '''
    depending on the range of H, the rgb definition changes. see
    https://www.rapidtables.com/convert/color/hsl-to-rgb.html
    for the detailed formula
    '''
    Cp = np.atleast_2d(C + m).T
    Xp = np.atleast_2d(X + m).T
    Zp = np.atleast_2d(m).T

    mask = np.logical_or((case == 0), (case == 6))
    rgb[mask, :] = np.hstack((Cp[mask, :], Xp[mask, :], Zp[mask, :]))

    mask = case == 1
    rgb[mask, :] = np.hstack((Xp[mask, :], Cp[mask, :], Zp[mask, :]))

    mask = case == 2
    rgb[mask, :] = np.hstack((Zp[mask, :], Cp[mask, :], Xp[mask, :]))

    mask = case == 3
    rgb[mask, :] = np.hstack((Zp[mask, :], Xp[mask, :], Cp[mask, :]))

    mask = case == 4
    rgb[mask, :] = np.hstack((Xp[mask, :], Zp[mask, :], Cp[mask, :]))

    mask = case == 5
    rgb[mask, :] = np.hstack((Cp[mask, :], Zp[mask, :], Xp[mask, :]))

    '''
        catch all cases where rgb values are out of [0,1] bounds
    '''
    rgb[rgb < 0.0] = 0.0
    rgb[rgb > 1.0] = 1.0
    return rgb


def rgb2hsl(rgb):
    '''
    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       10/29/2020 SS 1.0 original
    >> @DETAILS:    routines to convert between rgb to hsl representations, the
                    input and output will have all components in the range [0,1]
    '''
    '''
    first check the shape of the rgb array. it has to be nx3 array
    it is assumed that the hsl array has values in [0,1] for the 
    different components
    '''
    rgb = np.atleast_2d(rgb)
    if rgb.ndim != 2:
        raise RuntimeError("hsl_rgb: shape of hsl array is invalid.")
    hsl = np.zeros(rgb.shape)

    '''
    calculate different factors needed for the conversion
    '''
    Cmax = rgb.max(axis=1)
    Cmin = rgb.min(axis=1)
    delta = Cmax - Cmin

    L = 0.5 * (Cmax + Cmin)

    # catching cases where delta is close to zero
    zmask = np.abs(delta) < eps
    hsl[zmask, 1] = 0.0

    # depending on whether r,g or b is maximum, the hue is
    # assigned different values, we will deal with those cases here

    rmask = rgb[:, 0] == Cmax
    rmask = np.logical_and(rmask, np.logical_not(zmask))
    hsl[rmask, 0] = np.mod((rgb[rmask, 1] - rgb[rmask, 2]) / delta[rmask], 6) / 6.0

    gmask = rgb[:, 1] == Cmax
    gmask = np.logical_and(gmask, np.logical_not(zmask))
    hsl[gmask, 0] = (
        np.mod((rgb[gmask, 2] - rgb[gmask, 0]) / delta[gmask] + 2.0, 6) / 6.0
    )

    bmask = rgb[:, 2] == Cmax
    bmask = np.logical_and(bmask, np.logical_not(zmask))
    hsl[bmask, 0] = (
        np.mod((rgb[bmask, 0] - rgb[bmask, 1]) / delta[bmask] + 4.0, 6) / 6.0
    )

    hsl[np.logical_not(zmask), 1] = delta[np.logical_not(zmask)] / (
        1.0 - np.abs(2 * L[np.logical_not(zmask)] - 1.0)
    )

    hsl[:, 2] = L

    '''
        catch cases where hsl is out of [0,1] bounds
    '''
    hsl[hsl < 0.0] = 0.0
    hsl[hsl > 1.0] = 1.0
    return hsl
