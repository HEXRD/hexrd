# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
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
import numpy as np
from scipy import constants as sc

# pi related
pi = np.pi
piby2 = 0.5 * pi
piby3 = pi / 3.
piby4 = 0.25 * pi
piby6 = pi / 6.

# misc radicals
sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)
sqrt3by2 = 0.5 * sqrt3

# fwhm
sigma_to_fwhm = 2.*np.sqrt(2.*np.log(2.))

# tolerancing
epsf = np.finfo(float).eps      # ~2.2e-16
ten_epsf = 10 * epsf            # ~2.2e-15
sqrt_epsf = np.sqrt(epsf)       # ~1.5e-8

# for angles
period_dict = {'degrees': 360.0, 'radians': 2*pi}
angular_units = 'radians'  # module-level angle units
d2r = pi / 180.
r2d = 180. / pi

# identity arrays
identity_3x3 = np.eye(3)  # (3, 3) identity
identity_6x1 = np.r_[1., 1., 1., 0., 0., 0.]

# basis vectors
lab_x = np.r_[1., 0., 0.]  # X in the lab frame
lab_y = np.r_[0., 1., 0.]  # Y in the lab frame
lab_z = np.r_[0., 0., 1.]  # Z in the lab frame

zeros_3 = np.zeros(3)
zeros_3x1 = np.zeros((3, 1))
zeros_6x1 = np.zeros((6, 1))

# reference beam direction and eta=0 ref in LAB FRAME for standard geometry
beam_vec = -lab_z
eta_vec = lab_x


# for energy/wavelength conversions
def keVToAngstrom(x):
    return (1e7*sc.c*sc.h/sc.e) / np.array(x, dtype=float)


def _readenv(name, ctor, default):
    try:
        import os
        res = os.environ[name]
        del os
    except KeyError:
        del os
        return default
    else:
        try:
            return ctor(res)
        except:
            import warnings
            warnings.warn("environ %s defined but failed to parse '%s'" %
                          (name, res), RuntimeWarning)
            del warnings
            return default


# 0 = do NOT use numba
# 1 = use numba (default)
USE_NUMBA = _readenv("HEXRD_USE_NUMBA", int, 1)
if USE_NUMBA:
    try:
        import numba
    except ImportError:
        print("*** Numba not available, processing may run slower ***")
        USE_NUMBA = False

del _readenv
