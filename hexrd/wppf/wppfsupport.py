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
from hexrd.material.symbols import pstr_spacegroup
from hexrd.wppf.parameters import Parameters
from lmfit import Parameters as Parameters_lmfit
from hexrd.wppf.phase import Phases_LeBail, Phases_Rietveld
from hexrd.material import Material
from hexrd.material.unitcell import _rqpDict
import hexrd
import numpy as np
from hexrd import constants
import warnings

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
         "trns":[0.0, -1.0, 1.0, False],
         "shft":[0.0,-1.0,1.0,False],
         "U": [81.5, 0., 5000, False],
         "V": [1.0337, 0., 5000, False],
         "W": [5.18275, 0., 5000, False],
         "P": [0., 0., 5000, False],
         "X": [0.5665, 0., 100., False],
         "Y": [1.90994, 0., 100., False],
         "Xe": [0., 0., 1, False],
         "Ye": [0., 0., 1, False],
         "Xs": [0., 0., 1, False]
         }

    for k, v in p.items():
        if isinstance(params, Parameters):
            params.add(name=k,
                       value=v[0],
                       lb=v[1],
                       ub=v[2],
                       vary=v[3])
        elif isinstance(params, Parameters_lmfit):
            params.add(name=k,
                       value=v[0],
                       min=v[1],
                       max=v[2],
                       vary=v[3])

def _add_pvfcj_parameters(params):
    p = {"HL":[1e-3,1e-7,1e-1,False],
         "SL":[1e-3,1e-7,1e-1,False]
         }
    for k, v in p.items():
        if isinstance(params, Parameters):
            params.add(name=k,
                       value=v[0],
                       lb=v[1],
                       ub=v[2],
                       vary=v[3])
        elif isinstance(params, Parameters_lmfit):
            params.add(name=k,
                       value=v[0],
                       min=v[1],
                       max=v[2],
                       vary=v[3])

def _add_pvpink_parameters(params):
    p = {"alpha0":[14.4, -100., 100., False],
         "alpha1":[0., -100., 100., False],
         "beta0":[3.016, -100., 100., False],
         "beta1":[-2.0, -100., 100., False]
         }
    for k, v in p.items():
        if isinstance(params, Parameters):
            params.add(name=k,
                       value=v[0],
                       lb=v[1],
                       ub=v[2],
                       vary=v[3])
        elif isinstance(params, Parameters_lmfit):
            params.add(name=k,
                       value=v[0],
                       min=v[1],
                       max=v[2],
                       vary=v[3])

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
        if isinstance(params, Parameters):
            params.add(name=n,
                   value=init_val[d],
                   lb=-np.inf,
                   ub=np.inf,
                   vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(name=n,
                   value=init_val[d],
                   min=-np.inf,
                   max=np.inf,
                   vary=False)

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
    valid_shkl,\
    eq_const,\
    rqd_index, \
    trig_ptype = \
    _required_shkl_names(mat)

    if return_dict is None:

        for s in valid_shkl:
            n = f"{mname}_{s}"
            ne = f"{mname}_eta_fwhm"

            if isinstance(params, Parameters):
                params.add(name=n,
                       value=0.0,
                       lb=0.0,
                       ub=np.inf,
                       vary=False)
            elif isinstance(params, Parameters_lmfit):
                params.add(name=n,
                       value=0.0,
                       min=0.0,
                       max=np.inf,
                       vary=False)
        if isinstance(params, Parameters):
            params.add(name=ne,
                   value=0.5,
                   lb=0.0,
                   ub=1.0,
                   vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(name=ne,
                   value=0.5,
                   min=0.0,
                   max=1.0,
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
            if isinstance(params, Parameters):
                params.add(nn, value=l, lb=l-0.05,
                           ub=l+0.05, vary=False)
            elif isinstance(params, Parameters_lmfit):
                params.add(nn, value=l, min=l-0.05,
                           max=l+0.05, vary=False)
        else:
            if isinstance(params, Parameters):
                params.add(nn, value=l, lb=l-1.,
                           ub=l+1., vary=False)
            elif isinstance(params, Parameters_lmfit):
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
        if isinstance(params, Parameters):
            params.add(
                nn, value=atom_pos[i, 0],
                lb=0.0, ub=1.0,
                vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(
                nn, value=atom_pos[i, 0],
                min=0.0, max=1.0,
                vary=False)
        nn = f"{phase_name}_{elem}{atom_label[i]}_y"
        if isinstance(params, Parameters):
            params.add(
                nn, value=atom_pos[i, 1],
                lb=0.0, ub=1.0,
                vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(
                nn, value=atom_pos[i, 1],
                min=0.0, max=1.0,
                vary=False)
        nn = f"{phase_name}_{elem}{atom_label[i]}_z"
        if isinstance(params, Parameters):
            params.add(
                nn, value=atom_pos[i, 2],
                lb=0.0, ub=1.0,
                vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(
                nn, value=atom_pos[i, 2],
                min=0.0, max=1.0,
                vary=False)
        nn = f"{phase_name}_{elem}{atom_label[i]}_occ"
        if isinstance(params, Parameters):
            params.add(nn, value=occ[i],
                       lb=0.0, ub=1.0,
                       vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(nn, value=occ[i],
                       min=0.0, max=1.0,
                       vary=False)
        if(mat.aniU):
            U = mat.U
            for j in range(6):
                nn = (f"{phase_name}_{elem}{atom_label[i]}"
                       f"_{_nameU[j]}")
                if isinstance(params, Parameters):
                    params.add(
                        nn, value=U[i, j],
                        lb=-1e-3,
                        ub=np.inf,
                        vary=False)
                elif isinstance(params, Parameters_lmfit):
                    params.add(
                        nn, value=U[i, j],
                        min=-1e-3,
                        max=np.inf,
                        vary=False)
        else:
            nn = f"{phase_name}_{elem}{atom_label[i]}_dw"
            if isinstance(params, Parameters):
                params.add(
                    nn, value=mat.U[i],
                    lb=0.0, ub=np.inf,
                    vary=False)
            elif isinstance(params, Parameters_lmfit):
                params.add(
                    nn, value=mat.U[i],
                    min=0.0, max=np.inf,
                    vary=False)
def _generate_default_parameters_LeBail(mat,
                                        peakshape,
                                        bkgmethod,
                                        init_val=None,
                                        ptype="wppf"):
    """
    @author:  Saransh Singh, Lawrence Livermore National Lab
    @date:    03/12/2021 SS 1.0 original
    @details: generate a default parameter class given a list/dict/
    single instance of material class
    """
    if ptype == "wppf":
        params = Parameters()
    elif ptype == "lmfit":
        params = Parameters_lmfit()
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

    if isinstance(mat, Phases_LeBail):
        """
        phase file
        """
        for p in mat:
            m = mat[p]
            _add_Shkl_terms(params, m)
            _add_lp_to_params(params, m)

    elif isinstance(mat, Phases_Rietveld):
        """
        phase file
        """
        for p in mat:
            m = mat[p]
            k = list(m.keys())
            mm = m[k[0]]
            _add_Shkl_terms(params, mm)
            _add_lp_to_params(params, mm)

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

def _add_phase_fractions(mat, params):
    """
     @author:  Saransh Singh, Lawrence Livermore National Lab
     @date:    04/01/2021 SS 1.0 original
     @details: ass phase fraction to params class
     given a list/dict/single instance of material class
    """
    if isinstance(mat, Phases_Rietveld):
        """
        phase file
        """
        pf = mat.phase_fraction
        for ii,p in enumerate(mat):
            name=f"{p}_phase_fraction"
            if isinstance(params, Parameters):
                params.add(
                    name=name, value=pf[ii],
                    lb=0.0, ub=1.0,
                    vary=False)
            elif isinstance(params, Parameters_lmfit):
                params.add(
                    name=name, value=pf[ii],
                    min=0.0, max=1.0,
                    vary=False)
    elif isinstance(mat, Material):
        """
        just an instance of Materials class
        this part initializes the lattice parameters in the
        """
        p = mat.name
        name=f"{p}_phase_fraction"
        if isinstance(params, Parameters):
            params.add(
                name=name, value=1.0,
                lb=0.0, ub=1.0,
                vary=False)
        elif isinstance(params, Parameters_lmfit):
            params.add(
                name=name, value=1.0,
                min=0.0, max=1.0,
                vary=False)
    elif isinstance(mat, list):
        """
        a list of materials class
        """
        pf = [1./len(mat)]*len(mat)
        for ii,m in enumerate(mat):
            p = m.name
            name=f"{p}_phase_fraction"
            if isinstance(params, Parameters):
                params.add(
                    name=name, value=pf[ii],
                    lb=0.0, ub=1.0,
                    vary=False)
            elif isinstance(params, Parameters_lmfit):
                params.add(
                    name=name, value=pf[ii],
                    min=0.0, max=1.0,
                    vary=False)
    elif isinstance(mat, dict):
        """
        dictionary of materials class
        """
        pf = [1./len(mat)]*len(mat)
        for ii, (k,m) in enumerate(mat.items()):
            p = m.name
            name=f"{p}_phase_fraction"
            if isinstance(params, Parameters):
                params.add(
                    name=name, value=pf[ii],
                    lb=0.0, ub=1.0,
                    vary=False)
            elif isinstance(params, Parameters_lmfit):
                params.add(
                    name=name, value=pf[ii],
                    min=0.0, max=1.0,
                    vary=False)
    else:
        msg = (f"_generate_default_parameters: "
               f"incorrect argument. only list, dict or "
               f"Material is accpeted.")
        raise ValueError(msg)

def _add_extinction_parameters(mat, params):
    return params

def _add_absorption_parameters(mat, params):
    return params

def _generate_default_parameters_Rietveld(mat,
                                          peakshape,
                                          bkgmethod,
                                          init_val=None,
                                          ptype="wppf"):
    """
    @author:  Saransh Singh, Lawrence Livermore National Lab
    @date:    03/12/2021 SS 1.0 original
    @details: generate a default parameter class given a list/dict/
    single instance of material class
    """
    params = _generate_default_parameters_LeBail(mat,
                                                 peakshape,
                                                 bkgmethod,
                                                 init_val,
                                                 ptype=ptype)

    if ptype == "wppf":
        params.add(name="scale",
                   value=1.0,
                   lb=0.0,
                   ub=np.inf,
                   vary=False)

        params.add(name="Ph",
                   value=1.0,
                   lb=0.0,
                   ub=1.0,
                   vary=False)

    elif ptype == "lmfit":
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

    if isinstance(mat, Phases_Rietveld):
        """
        phase file
        """
        for p in mat:
            m = mat[p]
            k = list(m.keys())
            mm = m[k[0]]
            _add_atominfo_to_params(params, mm)

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
    if isinstance(mat, Phases_Rietveld):
        """
        phase file
        """
        for p in mat:
            m = mat[p]
            k = list(m.keys())
            mm = m[k[0]]
            _add_atominfo_to_params(params, mm)

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
            if isinstance(params, Parameters):
                [params.add(name=pnametvec[i],value=tvec[i]) for i in range(3)]
                [params.add(name=pnametilt[i],value=tilt[i]) for i in range(3)]
            elif isinstance(params, Parameters_lmfit):
                [params.add(name=pnametvec[i],value=tvec[i]) for i in range(3)]
                [params.add(name=pnametilt[i],value=tilt[i]) for i in range(3)]
    else:
        msg = "input is not an HEDMInstrument class"
        raise ValueError(msg)

def _add_background(params,lineouts,bkgdegree):
    for k in lineouts:
        pname = [f"{k}_bkg_C{ii}" for ii in range(bkgdegree)]
        shape = len(pname)
        if isinstance(params, Parameters):
            [params.add(name=pname[i],value=0.0) for i in range(shape)]
        elif isinstance(params, Parameters_lmfit):
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
            if isinstance(params, Parameters):
                [params.add(name=pname[i],
                    value=Icalc[p][k][i],
                    lb=0.0) for i in range(shape)]
            elif isinstance(params, Parameters_lmfit):
                [params.add(name=pname[i],
                    value=Icalc[p][k][i],
                    min=0.0) for i in range(shape)]
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
