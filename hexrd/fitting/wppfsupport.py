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
