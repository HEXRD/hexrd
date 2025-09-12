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
import copy
import warnings

import lmfit

from hexrd.material.symbols import pstr_spacegroup
from hexrd.wppf.phase import Phases_LeBail, Phases_Rietveld
from hexrd.material import Material
from hexrd.material.unitcell import _rqpDict
import hexrd
import numpy as np
from hexrd import constants

def _generate_default_parameters_pseudovoight(params):
    """
    generate some default values of peak profile
    for the Thompson et. al. model. A total of
    6 parameters are genrated which includes the
    following:
    3 -> cagliotti (instrumental broadening)
    """
    p = {"zero_error":[0., -1., 1., False],
         "trns":[0.0, -1.0, 1.0, False],
         "shft":[0.0,-1.0,1.0,False],
         "U": [81.5, 0., np.inf, False],
         "V": [1.0337, 0., np.inf, False],
         "W": [5.18275, 0., np.inf, False]
         }

    for k, v in p.items():
        params.add(
            name=k,
            value=v[0],
            min=v[1],
            max=v[2],
            vary=v[3],
        )

def _add_phase_dependent_parameters_pseudovoight(params,
                                                 mat):
    """
    add the particle size broadening term
    P : Gaussian scherrer broadening
    X : Lorentzian scherrer broadening
    Y : Lorentzian microstrain broadening
    """
    name = mat.name
    p = {"P": [0., 0., np.inf, False],
     "X": [0.5665, 0., np.inf, False],
     "Y": [1.90994, 0., np.inf, False]
     }

    for k, v in p.items():
        pname = f"{name}_{k}"
        params.add(
            name=pname,
            value=v[0],
            min=v[1],
            max=v[2],
            vary=v[3],
        )

def _add_pvfcj_parameters(params):
    p = {"HL":[1e-3,1e-7,1e-1,False],
         "SL":[1e-3,1e-7,1e-1,False]
         }
    for k, v in p.items():
        params.add(
            name=k,
            value=v[0],
            min=v[1],
            max=v[2],
            vary=v[3],
        )

def _add_pvpink_parameters(params):
    p = {"alpha0":[14.4, -100., 100., False],
         "alpha1":[0., -100., 100., False],
         "beta0":[3.016, -100., 100., False],
         "beta1":[-2.0, -100., 100., False]
         }
    for k, v in p.items():
        params.add(
            name=k,
            value=v[0],
            min=v[1],
            max=v[2],
            vary=v[3],
        )

def _add_chebyshev_background(params,
                              degree,
                              init_val):
    """
    add coefficients for chebyshev background
    polynomial. The initial values will be the
    same as determined by WPPF.chebyshevfit
    routine
    """
    for d in range(degree+1):
        n = f"bkg_{d}"
        params.add(
            name=n,
            value=init_val[d],
            min=-np.inf,
            max=np.inf,
            vary=False,
        )

def _add_stacking_fault_parameters(params,
                                   mat):
    """
    add stacking fault parameters for cubic systems only
    """
    phase_name = mat.name
    if mat.sgnum == 225:
        sf_alpha_name = f"{phase_name}_sf_alpha"
        twin_beta_name = f"{phase_name}_twin_beta"
        params.add(sf_alpha_name, value=0., min=0.,
                   max=1., vary=False)
        params.add(twin_beta_name, value=0., min=0.,
                   max=1., vary=False)

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
    mname = mat.name
    valid_shkl, eq_const, rqd_index, trig_ptype = _required_shkl_names(mat)

    if return_dict is None:

        for s in valid_shkl:
            n = f"{mname}_{s}"
            params.add(
                name=n,
                value=0.0,
                min=0.0,
                max=np.inf,
                vary=False,
            )

        ne = f"{mname}_eta_fwhm"
        params.add(
            name=ne,
            value=0.5,
            min=0.0,
            max=1.0,
            vary=False,
        )
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
            params.add(nn, value=l, min=l-0.025,
                       max=l+0.025, vary=False)
        else:
            params.add(nn, value=l, min=l-1.,
                       max=l+1., vary=False)

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
            min=0.0, max=1.0,
            vary=False)
        nn = f"{phase_name}_{elem}{atom_label[i]}_y"
        params.add(
            nn, value=atom_pos[i, 1],
            min=0.0, max=1.0,
            vary=False)
        nn = f"{phase_name}_{elem}{atom_label[i]}_z"
        params.add(
            nn, value=atom_pos[i, 2],
            min=0.0, max=1.0,
            vary=False)
        nn = f"{phase_name}_{elem}{atom_label[i]}_occ"
        params.add(nn, value=occ[i],
                   min=0.0, max=1.0,
                   vary=False)
        if(mat.aniU):
            U = mat.U
            for j in range(6):
                nn = (f"{phase_name}_{elem}{atom_label[i]}"
                       f"_{_nameU[j]}")
                params.add(
                    nn,
                    value=U[i, j],
                    min=-1e-3,
                    max=np.inf,
                    vary=False,
                )
        else:
            nn = f"{phase_name}_{elem}{atom_label[i]}_dw"
            params.add(
                nn, value=mat.U[i],
                min=0.0, max=np.inf,
                vary=False)

def _generate_default_parameters_amorphous_model(
                                        params,
                                        amorphous_model):
    '''
    add refinement parameters for amorphous models
    '''
    if amorphous_model is None:
        return

    for key in amorphous_model.scale:
        params.add(f'{key}_amorphous_scale',
            value=amorphous_model.scale[key],
            min=0,
            max=np.inf,
            vary=False,
        )

        if amorphous_model.model_type == "experimental":
            params.add(f'{key}_amorphous_shift',
                value=amorphous_model.shift[key],
                min=-np.inf,
                max=np.inf,
                vary=False,
            )
        else:
            params.add(f'{key}_amorphous_center',
                value=amorphous_model.center[key],
                min=-np.inf,
                max=np.inf,
                vary=False,
            )

        if amorphous_model.model_type == "split_gaussian":
            params.add(f'{key}_amorphous_fwhm_l',
                value=amorphous_model.fwhm[key][0],
                min=0,
                max=np.inf,
                vary=False,
            )

            params.add(f'{key}_amorphous_fwhm_r',
                value=amorphous_model.fwhm[key][1],
                min=0,
                max=np.inf,
                vary=False,
            )

        elif amorphous_model.model_type == "split_pv":
            params.add(f'{key}_amorphous_fwhm_g_l',
                value=amorphous_model.fwhm[key][0],
                min=0,
                max=np.inf,
                vary=False,
            )

            params.add(f'{key}_amorphous_fwhm_l_l',
                value=amorphous_model.fwhm[key][1],
                min=0,
                max=np.inf,
                vary=False,
            )

            params.add(f'{key}_amorphous_fwhm_g_r',
                value=amorphous_model.fwhm[key][2],
                min=0,
                max=np.inf,
                vary=False,
            )

            params.add(f'{key}_amorphous_fwhm_l_r',
                value=amorphous_model.fwhm[key][3],
                min=0,
                max=np.inf,
                vary=False,
            )


def _generate_default_parameters_LeBail(mat,
                                        peakshape,
                                        bkgmethod,
                                        init_val=None,
                                        amorphous_model=None):
    """
    @author:  Saransh Singh, Lawrence Livermore National Lab
    @date:    03/12/2021 SS 1.0 original
    @details: generate a default parameter class given a list/dict/
    single instance of material class
    """
    # Sanitize the material names
    mat = _sanitize_material_names(mat)

    params = lmfit.Parameters()
    _generate_default_parameters_pseudovoight(params)

    if peakshape == 0:
        _add_pvfcj_parameters(params)
    elif peakshape == 1:
        pass
    elif peakshape == 2:
        _add_pvpink_parameters(params)
    else:
        msg = (f"_generate_default_parameters_LeBail: "
            f"unknown peak shape.")
        raise ValueError(msg)

    if "chebyshev" in bkgmethod:
        deg = bkgmethod["chebyshev"]
        if not (init_val is None):
            if len(init_val) < deg+1:
                msg = (f"size of init_val and degree "
                       f"of polynomial are not consistent. "
                       f"setting initial guess to zero.")
                warnings.warn(msg)
                init_val = np.zeros([deg+1,])
        else:
            init_val = np.zeros([deg+1,])

        _add_chebyshev_background(params,
                                  deg,
                                  init_val)

    for m in _mat_list(mat):
        _add_phase_dependent_parameters_pseudovoight(params, m)
        _add_Shkl_terms(params, m)
        _add_lp_to_params(params, m)
        _add_stacking_fault_parameters(params, m)

    _generate_default_parameters_amorphous_model(params,
                                                 amorphous_model)

    return params


def _add_phase_fractions(mat, params):
    """
     @author:  Saransh Singh, Lawrence Livermore National Lab
     @date:    04/01/2021 SS 1.0 original
     @details: ass phase fraction to params class
     given a list/dict/single instance of material class
    """
    pf_list = []
    mat_list = _mat_list(mat)
    if isinstance(mat, Phases_Rietveld):
        # Use phase fraction setting
        pf_list = mat.phase_fraction
    else:
        # Otherwise, evenly distribute among materials
        pf_list = [1 / len(mat_list)] * len(mat_list)

    all_names = [f'{x.name}_phase_fraction' for x in mat_list]
    for name, pf in zip(all_names, pf_list):
        params.add(
            name=name,
            value=pf,
            min=0.0,
            max=1.0,
            vary=False,
        )

    if all_names:
        # Make the final one an expression using the other ones.
        # This is so that they will always sum to 1.
        fixed_name = all_names[-1]
        if len(all_names) == 1:
            expr = '1'
        else:
            others = ' - '.join(all_names[:-1])
            expr = f'1 - {others}'

        params[fixed_name].expr = expr


def _add_extinction_parameters(mat, params):
    return params


def _add_absorption_parameters(mat, params):
    return params

def _add_texture_model_parameters(texture_model, params):
        if texture_model is not None:
            for k, hm in texture_model.items():
                if hm is not None:
                    hm.get_parameters(params,
                                      vary=False)
        return

def _generate_default_parameters_Rietveld(mat,
                                          peakshape,
                                          bkgmethod,
                                          init_val=None,
                                          amorphous_model=None,
                                          texture_model=None):
    """
    @author:  Saransh Singh, Lawrence Livermore National Lab
    @date:    03/12/2021 SS 1.0 original
    @details: generate a default parameter class given a list/dict/
    single instance of material class
    """
    mat = _sanitize_material_names(mat)

    params = _generate_default_parameters_LeBail(mat,
                                                 peakshape,
                                                 bkgmethod,
                                                 init_val,
                                                 amorphous_model=amorphous_model)

    params.add(name="scale",
               value=1.0,
               min=0.0,
               max=np.inf,
               vary=False)

    params.add(name="Ph",
               value=1.0,
               min=0.0,
               max=1.0,
               vary=False)

    _add_phase_fractions(mat, params)
    _add_extinction_parameters(mat, params)
    _add_absorption_parameters(mat, params)
    _add_texture_model_parameters(texture_model, params)

    for m in _mat_list(mat):
        _add_atominfo_to_params(params, m)

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
            x_ret[c[1]] = c[2]*x_ret[c[0]]

    return x_ret


def _required_shkl_names(mat):
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
    eq_constraints = _rqd_shkl[latticetype][1]
    valid_shkl = [_shkl_name[i] for i in rqd_index]

    return valid_shkl, eq_constraints, rqd_index, trig_ptype


def _add_texture_coefficients(crystal_sym, sample_sym, name, degree):
    """
    add the texture coefficients for a particular phase
    given its laue group. the crystal sym decides what the
    symmetry of the crystal is and what coefficients to add. the
    sample symmetry decides what the sample symmtry is. allowed ones
    are
    triclinic : -1
    monoclinic: 2/m
    orthorhombic: mmm
    cylindrical: inf/mmm

    if cylindrical symmetry is used, then the total coefficients used
    are drastically reduced
    """
    pass


def _add_texture_parameters(mat, degree):
    """
    @SS 06/22/2021 1.0 original
    this routine adds the texture coefficients to the wppf
    parameter list based on the material list and the
    degree of harmonic coefficients passed. Also required is the
    assumed sample symmetry. The same sample symmetry will be used
    for each of the phases.
    """
    for m in _mat_list(mat):
        _add_atominfo_to_params(params, m)


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
    "triclinic": [tuple(range(15)),()],
}


def _getnumber(arr):

    res = np.ones(arr.shape)
    for i in range(arr.shape[0]):
        res[i] = np.sum(arr[0:i+1] == arr[i])
    res = res.astype(np.int32)

    return res


def _add_detector_geometry(params, instr):
    """
    this function adds the geometry of the
    detector as a parameter to the LeBail class
    such that those can be refined as well
    """
    if isinstance(instr, hexrd.instrument.HEDMInstrument):
        for key,det in instr.detectors.items():
            tvec = det.tvec
            tilt = det.tilt
            pnametvec = [f"{key}_tvec{i}" for i in range(3)]
            pnametilt = [f"{key}_tilt{i}" for i in range(3)]
            [params.add(name=pnametvec[i],value=tvec[i]) for i in range(3)]
            [params.add(name=pnametilt[i],value=tilt[i]) for i in range(3)]
    else:
        msg = "input is not an HEDMInstrument class"
        raise ValueError(msg)


def _add_background(params,lineouts,bkgdegree):
    for k in lineouts:
        pname = [f"{k}_bkg_C{ii}" for ii in range(bkgdegree)]
        shape = len(pname)
        [params.add(name=pname[i],value=0.0) for i in range(shape)]


def striphkl(g):
    return str(g)[1:-1].replace(" ","")


def _add_intensity_parameters(params,hkls,Icalc,prefix):
    """
    this routine adds the Icalc values as refinable
    parameters in the params parameter class
    """
    for p in Icalc:
        for k in Icalc[p]:
            shape = Icalc[p][k].shape[0]

            pname = [f"{prefix}_{p}_{k}_I{striphkl(g)}"
                     for i,g in zip(range(shape),hkls[p][k])]
            [params.add(name=pname[i],
                        value=Icalc[p][k][i],
                        min=0.0) for i in range(shape)]


def _sanitize_material_names(mats):
    if isinstance(mats, Material):
        if '-' in mats.name:
            mats = copy.deepcopy(mats)
            mats.name = mats.name.replace('-', '_')
    elif isinstance(mats, (list, dict)):
        mats = copy.deepcopy(mats)
        if isinstance(mats, list):
            mats_iter = mats
        else:
            mats_iter = mats.values()

        for mat in mats_iter:
            if '-' in mat.name:
                mat.name = mat.name.replace('-', '_')

    return mats


def _mat_list(mat):
    # Make a list of unique materials depending on
    # what the argument of `mat` is.
    if isinstance(mat, Phases_LeBail):
        return list(mat.phase_dict.values())
    elif isinstance(mat, Phases_Rietveld):
        # Get the first wavelength name
        k = next(iter(next(iter(mat.phase_dict.values()))))
        return [d[k] for d in mat.phase_dict.values()]
    elif isinstance(mat, Material):
        return [mat]
    elif isinstance(mat, list):
        return mat
    elif isinstance(mat, dict):
        return list(mat.values())

    msg = "incorrect argument. only list, dict or Material is accpeted."
    raise ValueError(msg)


background_methods = {
    'spline': None,

    'chebyshev': [
        {
            'label': 'Chebyshev Polynomial Degree',
            'type': int,
            'min': 0,
            'max': 99,
            'value': 3,
            'tooltip': 'The polynomial degree used '
            'for polynomial fit.',
        }
    ],
    'snip1d': [
        {
            'label': 'Snip Width',
            'type': float,
            'min': 0.,
            'value': 1.0,
            'tooltip': 'Maximum width of peak to retain for '
            'background estimation (in degrees).'
        },
        {
            'label': 'Snip Num Iterations',
            'type': int,
            'min': 1,
            'max': 99,
            'value':2,
            'tooltip': 'number of snip iterations.'
        }
    ],
}
