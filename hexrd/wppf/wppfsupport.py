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
from hexrd.symbols import pstr_spacegroup
from hexrd.wppf.parameters import Parameters
from hexrd.wppf.phase import Phases_LeBail, Phases_Rietveld
from hexrd.material import Material
from hexrd.unitcell import _rqpDict
import numpy as np
from hexrd import constants

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
    p = {"zero_error":[0., -1., 1., False],
         "U": [1e-5, 0., 1e-2, True],
         "V": [1e-5, 0., 1e-2, True],
         "W": [1e-5, 0., 1e-2, True],
         "P": [0., 0., 1e-2, False],
         "X": [1e-5, 0., 1e-2, True],
         "Y": [1e-5, 0., 1e-2, True],
         "Xe": [0., 0., 1e-2, False],
         "Ye": [0., 0., 1e-2, False],
         "Xs": [0., 0., 1e-2, False],
         "eta_w": [0.5, 0., 1., False]
         }

    for k, v in p.items():
        params.add(name=k,
                   value=v[0],
                   lb=v[1],
                   ub=v[2],
                   vary=v[3])

# def _add_asymmetry_terms(params):
#     """
#     adds the terms in the asymmetry of
#     2-theta shifts
#     """
#     p = {"trns":,[0.0, ]}

def _add_Shkl_terms(params,
                    mat,
                    return_dict=None):
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
    mname = mat.name
    hmsym = pstr_spacegroup[sgnum-1].strip()
    trig_ptype = False

    if latticetype == "trigonal" and hmsym[0] == "P":
        """
        this is a trigonal group so the hexagonal
        constants are valid
        """
        latticetype = "haxagonal"
        trig_ptype = True

    rqd_index = _rqd_shkl[latticetype][0]
    valid_shkl = [_shkl_name[i] for i in rqd_index]

    if return_dict is None:

        for s in valid_shkl:
            n = f"{mname}_{s}"
            params.add(name=n,
                       value = 0.0,
                       lb = -1.0,
                       ub=1.0,
                       vary=False)
        params.add(name="eta_fwhm",
               value = 1.0,
               lb = 0.0,
               ub=1.0,
               vary=False)
    else:
        res = {}
        for s in valid_shkl:
            res[s] = 0.0
        return res, trig_ptype

def _add_lp_to_params(params, 
                      mat):
    """
    03/12/2021 SS 1.0 original
    given a material, add the required
    lattice parameters
    """
    lp = mat.lparms
    rid = _rqpDict[mat.latticeType][0]
    lp = [lp[i] for i in rid]
    name = [_lpname[i] for i in rid]
    phase_name = mat.name
    for n, l in zip(name, lp):
        nn = phase_name+'_'+n
        """
        is n is a,b,c, it is one of the length units
        else it is an angle
        """
        if(n in ['a', 'b', 'c']):
            params.add(nn, value=l, lb=l-0.05,
                       ub=l+0.05, vary=True)
        else:
            params.add(nn, value=l, lb=l-1.,
                       ub=l+1., vary=True)

def _add_atominfo_to_params(params, mat):
    """
    03/12/2021 SS 1.0 original
    given a material, add the required
    lattice parameters, atom positions,
    occupancy, DW factors etc.
    """
    phase_name = mat.name
    atom_pos = mat.atom_pos[:, 0:3]
    occ = mat.atom_pos[:, 3]
    atom_type = mat.atom_type

    atom_label = _getnumber(atom_type)

    for i in range(atom_type.shape[0]):

        Z = atom_type[i]
        elem = constants.ptableinverse[Z]

        nn = f"{phase_name}_{elem}{atom_label[i]}_x"
        params.add(
            nn, value=atom_pos[i, 0],
            lb=0.0, ub=1.0,
            vary=False)

        nn = f"{phase_name}_{elem}{atom_label[i]}_y"
        params.add(
            nn, value=atom_pos[i, 1],
            lb=0.0, ub=1.0,
            vary=False)

        nn = f"{phase_name}_{elem}{atom_label[i]}_z"
        params.add(
            nn, value=atom_pos[i, 2],
            lb=0.0, ub=1.0,
            vary=False)

        nn = f"{phase_name}_{elem}{atom_label[i]}_occ"
        params.add(nn, value=occ[i],
                   lb=0.0, ub=1.0,
                   vary=False)

        if(mat.aniU):
            U = mat.U
            for j in range(6):
                nn = f("{phase_name}_{elem}{atom_label[i]}"
                       f"_{nameU[j]}")
                params.add(
                    nn, value=U[i, j],
                    lb=-1e-3,
                    ub=np.inf,
                    vary=False)
        else:
            nn = f"{phase_name}_{elem}{atom_label[i]}_dw"
            params.add(
                nn, value=mat.U[i],
                lb=0.0, ub=np.inf,
                vary=False)

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
        _add_Shkl_terms(params, mat)
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
    params = _generate_default_parameters_LeBail(mat)
    params.add(name="scale",
        value=1.0,
        lb=0.0,
        ub=1e9,
        vary=True)

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
_lpname = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
_nameU = ['U11', 'U22', 'U33', 'U12', 'U13', 'U23']

"""
function to take care of equality constraints
"""
def _fill_shkl(x, eq_const):
    """
    fill all values of shkl when only reduced set
    is passed
    """
    x_ret = np.zeros([15,])
    for ii,n in enumerate(_shkl_name):
        if n in x:
            x_ret[ii] = x[n]
        else:
            x_ret[ii] = 0.0
    if not eq_const:
        pass
    else:
        for c in eq_const:
            # n = _shkl_name[c[0]]
            # neq = _shkl_name[c[1]]
            x_ret[c[1]] = c[2]*x_ret[c[0]]

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
"orthorhombic": [tuple(range(6)),()],
"monoclinic": [tuple(range(6))+(7, 10, 13),()],
"triclinic": [tuple(range(15)),()]
}

def _getnumber(arr):

    res = np.ones(arr.shape)
    for i in range(arr.shape[0]):
        res[i] = np.sum(arr[0:i+1] == arr[i])
    res = res.astype(np.int32)

    return res

background_methods = {
    'spline': None,

    'chebyshev': [
        {
            'label': 'Chebyshev Polynomial Degree',
            'type': int,
            'min': 0,
            'max': 99,
            'tooltip': 'The polynomial degree used '
            'for polynomial fit.',
        }
    ],
    'snip': [
        {
            'label': 'Snip Width',
            'type': float,
            'min': 0.,
            'tooltip': 'Maximum width of peak to retain for '
            'background estimation (in degrees).'
        },
        {
            'label': 'Snip Num Iterations',
            'type': int,
            'min': 1,
            'max': 99,
            'tooltip': 'number of snip iterations.'
        }
    ],
}