# ============================================================
# Copyright (c) 2021, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Saransh Singh <saransh1@llnl.gov>/Joel Bernier
# <bernier2@llnl.gov> and others.
# LLNL-CODE-819716.
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
# ============================================================

"""
this function contains some helper functions for the WPPF module
the functions which are common to both the Rietveld and LeBail
classes are put here to minimize code duplication. Some examples
include initialize background, generate_default_parameter list etc.
"""


def _generate_default_parameters_pseudovoight(params):
    """
    generate some default values of peak profile
    for the Thompson et. al. model. A total of
    18 parameters are genrated which includes the
    following:
    4 -> cagliotti + Scherrer broadening
    5 -> lorentzian width with stacking fault
    2 to 15 -> anisotropic hkl broadening depending on
    symmetry
    1 --> width_mixing of anisotropic broadening
    """
    p = {"U": [1e-2, 0., 1., True],
         "V": [1e-2, 0., 1., True],
         "W": [1e-2, 0., 1., True],
         "P": [0., 0., 1., False],
         "X": [1e-2, 0., 1., True],
         "Y": [1e-2, 0., 1., True],
         "Xe": [0., 0., 1., False],
         "Ye": [0., 0., 1., False],
         "Xs": [0., 0., 1., False],
         "eta_w": [0.5, 0., 1., False]
         }

    for k, v in p.items():
        params.add(name=k,
                   value=v[0],
                   lb=v[1],
                   ub=v[2],
                   varies=v[3])

def _add_Shkl_terms(params,
                    mat):
    """
    add the SHKL terms in the anisotropic peak
    broadening contribution. this depends on the
    lattice type. details can be found in 
    P. Stephens, J. Appl. Cryst. (1999). 32, 281-289

    @NOTE: the rhombohedral lattices are assumed to be in
    the hexagonal setting
    """
    latticetype = mat.latticeType
    sgnum = mat.sgnum
    name = mat.name

    if latticetype == "cubic":
        valid_shkl = 
        params.add
    elif latticetype == "hexagonal":
        pass
    elif latticetype == "trigonal":
        pass
    elif latticetype == "tetragonal":
        pass
    elif latticetype == "orthorhombic":
        pass
    elif latticetype == "monoclinic":
        pass
    elif latticetype == "triclinic":
        pass

def _generate_default_parameters_LeBail(mat):
    """
    @author:  Saransh Singh, Lawrence Livermore National Lab
    @date:    03/12/2021 SS 1.0 original
    @details: generate a default parameter class given a list/dict/
    single instance of material class
    """
    params = Parameters()
    _generate_default_parameters_pseudovoight(params)

    if isinstance(mat, Phases_LeBail):
        """
        phase file
        """
        for p in mat:
            m = mat[p]
            _add_Shkl_terms(params, m)
            _add_lp_to_params(params, m)

    elif isinstance(mat, Material):
        """
        just an instance of Materials class
        this part initializes the lattice parameters in the
        """
        _add_Shkl_terms(params, m)
        _add_lp_to_params(params, mat)

    elif isinstance(mat, list):
        """
        a list of materials class
        """
        for m in mat:
            _add_Shkl_terms(params, m)
            _add_lp_to_params(params, m)

    elif isinstance(mat, dict):
        """
        dictionary of materials class
        """
        for k, m in mat.items():
            _add_Shkl_terms(params, m)
            _add_lp_to_params(params, m)

    else:
        msg = (f"_generate_default_parameters: "
               f"incorrect argument. only list, dict or "
               f"Material is accpeted.")
        raise ValueError(msg)

    return params


def _generate_default_parameters_Rietveld(mat):
    """
    @author:  Saransh Singh, Lawrence Livermore National Lab
    @date:    03/12/2021 SS 1.0 original
    @details: generate a default parameter class given a list/dict/
    single instance of material class
    """
    params = Parameters()
    names = ["U", "V", "W", "X",
             "Y", "scale", "zero_error"]
    values = 5*[1e-3]
    values.append(0.)
    values.append(1.)
    lbs = 6*[0.]
    lbs.append(-1.)
    ubs = 5*[1.]
    ubs.append(1e3)
    ubs.append(1.)
    varies = 7*[False]

    params.add_many(names, values=values,
                    varies=varies, lbs=lbs, ubs=ubs)

    if isinstance(mat, Phases_Rietveld):
        """
        phase file
        """
        for p in mat:
            m = mat[p]
            _add_atominfo_to_params(params, m)

    elif isinstance(mat, Material):
        """
        just an instance of Materials class
        this part initializes the lattice parameters in the
        """
        _add_atominfo_to_params(params, mat)

    elif isinstance(mat, list):
        """
        a list of materials class
        """
        for m in mat:
            _add_atominfo_to_params(params, m)

    elif isinstance(mat, dict):
        """
        dictionary of materials class
        """
        for k, m in mat.items():
            _add_atominfo_to_params(params, m)

    else:
        msg = (f"_generate_default_parameters: "
               f"incorrect argument. only list, dict or "
               f"Material is accpeted.")
        raise ValueError(msg)

    return params

_shkl_name = ["s400", "s040", "s004", "s220", "s202", "s022",
              "s310", "s103", "s031", "s130", "s301", "s013",
              "s211", "s121", "s112"]

"""
function to take care of equality constraints
"""
def _fill_shkl(x, eq_const):
    """
    fill all values of shkl when only reduced set
    is passed
    """
    x_ret = {}
    for n in _shkl_name:
        if n in x:
            x_ret[n] = x[n]
        else:
            x_ret[n] = 0.0
    if not eq_const:
        pass

    else:
        for c in eq_const:
            n = _shkl_name[c[0]]
            neq = _shkl_name[c[1]]
            x_ret[neq] = c[2]*x_ret[n]

    return x_ret
"""
this dictionary structure holds information for the shkl
coefficeints needed for anisotropic broadening of peaks
first component of list are the required shkl components
second component of ist are the equality constraints with
a weight factor (sometimes theres a factor of 2 or 3.)
"""
_rqd_shkl = {
"cubic": [(0, 3),
          ((0,1,1.),(0,2,1.),(3,4,1.),(3,5,1.))],
"hexagonal": [(0, 2, 4),
((0,1,1.),(0,6,2.),(0,9,2.),(0,3,3.),
(4,5,1.),(4,14,1.))],
"trigonal": [(0, 2, 4, 10), 
((0,1,1.),(0,6,2.),(0,9,2.),(0,3,3.),
(4,5,1.),(4,14,1.),
(10,8,-1.),(10,12,1.5),(10,13,-1.5))],
"tetragonal": [(0, 2, 3, 4),((0,1,1.),(4,5,1.))],
"orthorhombic": [tuple(np.arange(6)),()],
"monoclinic": [tuple(np.arange(6))+(7, 10, 13),()],
"triclinic": [tuple(np.arange(15)),()]
}