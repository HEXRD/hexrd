#! /usr/bin/env python
# ============================================================
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
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999.
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
# ============================================================

import numpy as np

# Minimal subset of constants needed for xrd_transforms.

# tolerance
epsf = np.finfo(float).eps      # ~2.2e-16
sqrt_epsf = np.sqrt(epsf)       # ~1.5e-8


# identity 3x3 matrix
identity_3x3 = np.eye(3)

# basis vectors
lab_x = np.r_[1., 0., 0.] # X in the lab frame
lab_y = np.r_[0., 1., 0.] # Y in the lab frame
lab_z = np.r_[0., 0., 1.] # Z in the lab frame

# reference beam direction and eta=0 ref in LAB FRAME for standard geometry
# WARNING: In some parts of the code it is assumed that the beam rotation
#          matrix is assumed to be the identity when using the standard
#          beam_vec and eta_vec. If this changes code must be reviewed!
beam_vec = -lab_z
eta_vec  = lab_x


_period_dict = {
    'degrees': 360.0,
    'radians': 2.0*np.pi,
}

def period_for_unit(units=None):

    # default units is radians
    units = units if units is not None else 'radians'
    try:
        return _period_dict(units)
    except KeyError:
        raise ValueError("Expecting 'radians', 'degrees' or None for default")

