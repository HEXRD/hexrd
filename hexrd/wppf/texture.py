import copy
from warnings import warn
# 3rd party imports
import h5py
from lmfit import Parameters, Minimizer
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.special import sph_harm_y
from matplotlib import pyplot as plt

"""
===============================================================================

>> @AUTHOR: Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
>> @DATE: 06/14/2021 SS 1.0 original

>> @DETAILS: this module deals with the texture computations for the wppf
    package.  two texture models employed are the March-Dollase model and the
    spherical harmonics model. the spherical harmonics model uses the axis
    distribution function for computing the scale factors for each reflection.
    maybe it makes sense to use symmetrized FEM mesh functions for 2-sphere for
    this. probably use the two theta-eta array from the instrument class to
    precompute the values of k_l^m(y) and we already know what the reflections
    are so k_l^m(h) can also be pre-computed.

>> @PARAMETERS: symmetry of the mesh
===============================================================================
"""

# FIXME: these are available in hexrd.constants @saransh13
I3 = np.eye(3)

Xl = np.ascontiguousarray(I3[:, 0].reshape(3, 1))     # X in the lab frame
Yl = np.ascontiguousarray(I3[:, 1].reshape(3, 1))     # Z in the lab frame
Zl = np.ascontiguousarray(I3[:, 2].reshape(3, 1))     # Z in the lab frame

bVec_ref = -Zl
eta_ref = Xl

class mesh_s2:
    """
    this class deals with the basic functions of the s2 mesh. the
    class is initialized just based on the symmetry. the main functions
    are the interpolation of harmonic values for a given set of points
    and the number of invariant harmonics up to a maximum degree. this
    is the main class used for computing the general axis distribution
    function.
    """
    def __init__(self,
                 symmetry):

        data = importlib.resources.open_binary(hexrd.resources, "surface_harmonics.h5")
        with h5py.File(data, 'r') as fid:

            gname = f"{symmetry}"
            self.symmetry = symmetry
            self.ncrd = fid[gname].attrs["ncrd"]
            self.neqv = fid[gname].attrs["neqv"]
            self.nindp = fid[gname].attrs["nindp"]

            dname = f"/{symmetry}/harmonics"
            self.harmonics = np.array(fid[dname])

            dname = f"/{symmetry}/eqv"
            self.eqv = np.array(fid[dname]).astype(np.int32)

            dname = f"/{symmetry}/crd"

            pts = np.array(fid[dname])
            n = np.linalg.norm(pts, axis=1)
            self.points = pts/np.tile(n, [3,1]).T

            points_st = self.points[:,:2]/np.tile(
                (1.+np.abs(self.points[:,2])),[2,1]).T

            self.mesh = Delaunay(points_st, qhull_options="QJ")

    def _get_simplices(self,
                       points):
        """
        this function is used to get the index of the simplex
        in which the point lies. this is the first step to
        calculating the barycentric coordinates, which are the
        weights to linear interpolation.
        """

        n = np.linalg.norm(points, axis=1)
        points = points/np.tile(n, [3,1]).T

        points_st = points[:,:2]/np.tile(
                (1.+np.abs(points[:,2])),[2,1]).T

        simplices = self.mesh.find_simplex(points_st)

        """
        there might be some points which end up being slighty outside
        the circle due to numerical precision. just scale those points
        by a number close to one to bring them closer to the origin.
        the program should ideally never get here
        """
        mask = (simplices == -1)
        simplices[mask] = self.mesh.find_simplex(points_st[mask,:]*0.995)

        if -1 in simplices:
            msg = (f"some points seem to not be in the "
            f"mesh. please check input")
            raise RuntimeError(msg)

        return simplices

    def _get_barycentric_coordinates(self,
                                    points):

        """
        get the barycentric coordinates of points
        this is used for linear interpolation of
        the harmonic function on the mesh

        points is nx3 shaped

        first make sure that the points are all normalized
        to unit length then take the stereographic projection
        """
        n = np.linalg.norm(points, axis=1)
        points = points/np.tile(n, [3,1]).T

        points_st = points[:,:2]/np.tile(
                (1.+points[:,2]),[2,1]).T

        """
        next get the simplices. a value of -1 is returned
        when the point can not be found inside any simplex.
        those points will be handled separately
        """
        simplices = self._get_simplices(points)

        """
        we have the simplex index for each point now. move on to
        computing the barycentric coordinates. the logic is simple
        inverse of the T matrix is in mesh.transforms, we use that
        to compute T^{-1}(r-r3)
        see wikipedia for more details
        """
        bary_center = [self.mesh.transform[simplices[i],:2].dot(
        (np.transpose(points_st[i,:] -
        self.mesh.transform[simplices[i],2])))
        for i in np.arange(points.shape[0])]

        bary_center = np.array(bary_center)

        """
        the fourth coordinate is 1 - sum of the other three
        """
        bary_center = np.hstack((bary_center,
        1. - np.atleast_2d(bary_center.sum(axis=1)).T))

        bary_center[np.abs(bary_center)<1e-9] = 0.

        return np.array(bary_center), simplices

    def _get_equivalent_node(self,
                             node_id):
        """
        given the index of the node, find out the
        equivalent node. if the node is already one
        of the independent nodes, then the same value
        is returned. Note that the index is all 1 based
        in the surface harmonic file and all 0 based in
        the scipy Delaunay function
        """
        node_id = node_id
        mask = node_id+1 > self.nindp
        eqv_node_id = np.zeros(node_id.shape).astype(np.int32)

        """
        for the independent node, return the same value
        """
        eqv_node_id[~mask] = node_id[~mask]

        """
        for the other node, calculate the equivalent index
        using the eqv array
        """
        if not np.all(mask == False):
            eqv_id = np.array([np.where(self.eqv[:,0] == i+1)
                for i in node_id[mask]]).astype(np.int32)
            eqv_id = np.squeeze(eqv_id)
            eqv_node_id[mask] = self.eqv[eqv_id,1]-1

        return eqv_node_id


    def _get_harmonic_values(self,
                            points_inp):
        """
        this is the main function which compute the
        value of the harmonic function at a given set
        of points using linear interpolation of the
        values on the fem mesh. the sequence of function
        calls is as follows:

        1. compute the barycentric coordinates
        2. compute the equivalent nodes
        3. perform weighted mean

        the value of harmonics upto the nodal degree of
        freedom is return. the user can then select how many
        to use and where to truncate

        Note that if z component is negative, an inversion
        symmetry is automatically applied to the point
        """
        points = copy.deepcopy(points_inp)
        # mask = points[:,-1] < 0.
        # points[mask,:] = -points[mask,:]
        bary_center, simplex_id = self._get_barycentric_coordinates(points)
        node_id = self.mesh.simplices[simplex_id]

        eqv_node_id = np.array([self._get_equivalent_node(nid)
            for nid in node_id]).astype(np.int32)

        fval = np.array([self.harmonics[nid,:] for nid in eqv_node_id])
        nharm = self.harmonics.shape[1]

        fval_points = np.array([np.sum(np.tile(bary_center[i,:],[nharm,1]).T*
            fval[i,:,:],axis=0) for i in range(points.shape[0])])

        return np.atleast_2d(fval_points)

    def num_invariant_harmonic(self,
                           max_degree):

        """
        >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
        >> @DATE:       06/29/2021 SS 1.0 original

        >> @DETAILS:    this function computes the number of symmetrized harmonic coefficients
                        for a given degree. this is essential for computing the number of
                        terms that is in the summation of the general axis distribution function.
                        the generating function of the dimension of the invariant subspace is
                        given by Meyer and Polya and enumerated in
                        "ON THE SYMMETRIES OF SPHERICAL HARMONICS", Burnett Meyer
        >> @PARAMETERS:  symmetry, degree
        """

        """
        first get the polynomials in the
        denominator
        """
        if self.symmetry == "cylindrical":
            v = []
            for i in range(0,max_degree+1,2):
                v.append([i,1])
            v = np.array(v).astype(np.int32)
            return v
        else:
            powers = Polya[self.symmetry]
            den = powers["denominator"]

            polyd = []
            if not den:
                pass
            else:
                for m in den:
                    polyd.append(self.denominator(m, max_degree))

            num = powers["numerator"]
            polyn = []
            if not num:
                pass
            else:
                nn = [x[0] for x in num]
                mmax = max(nn)
                coef = np.zeros([mmax+1,])
                coef[0] = 1.
                for m in num:
                    coef[m[0]] = m[1]
                    polyn.append(Polynomial(coef))

            """
            multiply all of the polynomials in the denominator
            with the ones in the numerator
            """
            poly = Polynomial([1.])
            for pd in polyd:
                if not pd:
                    break
                else:
                    poly = poly*pd

            for pn in polyn:
                if not pn:
                    break
                else:
                    poly = poly*pn

            poly = poly.truncate(max_degree+1)
            idx = np.nonzero(poly.coef)
            v = poly.coef[idx]

            return np.vstack((idx,v)).T.astype(np.int32)

    def denominator(self,
                    m,
                    max_degree):
        """
        this function computes the Maclaurin expansion
        of the function 1/(1-t^m)
        this is just the binomial expansion with negative
        exponent

        1/(1-x^m) = 1 + x^m + x^2m + x^3m ...
        """
        coeff = np.zeros([max_degree+1, ])
        ideg = 1+int(max_degree/m)
        for i in np.arange(ideg):
            idx = i*m
            coeff[idx] = 1.0

        return Polynomial(coeff)


class harmonic_model:
    """
    this class brings all the elements together to compute the
    texture model given the sample and crystal symmetry.
    """
    def __init__(self,
                pole_figures,
                sample_symmetry,
                max_degree):

        self.pole_figures = pole_figures
        self.crystal_symmetry = pole_figures.material.sg.laueGroup_international
        self.sample_symmetry = sample_symmetry
        self.max_degree = max_degree
        self.mesh_crystal = mesh_s2(self.crystal_symmetry)
        self.mesh_sample = mesh_s2(self.sample_symmetry)

        self.init_harmonic_values()
        self.itercounter = 0

        ncoeff = self._num_coefficients()
        self.coeff = np.zeros([ncoeff,])

    def init_harmonic_values(self):
        """
        once the harmonic model is initialized, initialize
        the values of the harmonic functions for different
        values of hkl and sample directions. the hkl are the
        keys and the sample_dir are the values of the sample_dir
        dictionary.
        """
        self.V_c_allowed = {}
        self.V_s_allowed = {}
        self.allowed_degrees = {}

        pole_figures = self.pole_figures
        for ii in np.arange(pole_figures.num_pfs):
            key = str(pole_figures.hkls[ii,:])[1:-1].replace(" ", "")
            hkl = np.atleast_2d(pole_figures.hkls_c[ii,:])
            sample_dir = pole_figures.gvecs[key]

            self.V_c_allowed[key], self.V_s_allowed[key] = \
            self._compute_harmonic_values_grid(hkl,
                                               sample_dir)
        self.allowed_degrees = self._allowed_degrees()

    def init_equiangular_grid(self):
        """
        this function initializes sample directions
        for an 5x5 equiangular grid of points. the harmonic
        functions will be calculated using the
        calc_pole_figures function instead of here.
        """
        angs = []
        for tth in np.arange(0,91,5):
            for eta in np.arange(0, 360, 5):
                angs.append([np.radians(tth), np.radians(eta), 0.])
                if tth == 0:
                    break
        angs = np.array(angs)

        return angs

    def init_coeff_params(self):
        """
        this function initializes a lmfit
        parameter class with all the coefficients.
        this will be passed to the minimize
        routine
        """
        params = Parameters()
        self.coeff_loc = {}
        self.coeff_loc_inv = {}
        ctr = 0
        for ii,(k,v) in enumerate(self.allowed_degrees.items()):
            if k > 0:
                for icry in np.arange(v[0]):
                    for isamp in np.arange(v[1]):
                        vname = f"c_{k}_{icry}_{isamp}"
                        self.coeff_loc[vname] = ctr
                        self.coeff_loc_inv[ctr] = vname
                        idx = self.coeff_loc[vname]
                        ctr += 1
                        params.add(vname,value=self.coeff[idx],
                        vary=True,min=-np.inf,max=np.inf)

        if hasattr(self, 'phon'):
            val = self.phon
        else:
            val = 0.0

        params.add("phon",value=val,vary=True,min=0.0)

        return params

    def set_coeff_from_param(self, params):
        """
        this function takes the the values in the
        parameters and sets the values of the coefficients
        """
        self.coeff = np.zeros([len(self.coeff_loc),])
        for ii,k in enumerate(params):
            if k.lower() != "phon":
                loc = self.coeff_loc[k]
                self.coeff[loc] = params[k].value

        self.phon = params["phon"].value

    def residual_function(self, params):
        """
        calculate the residual between input pole figure
        data and generalized axis distribution function
        """
        self.set_coeff_from_param(params)
        pf_recalc = self.recalculate_pole_figures()

        for ii,(k,v) in enumerate(self.pole_figures.pfdata.items()):
            inp_intensity = np.squeeze(v[:,3])
            calc_intensity = np.squeeze(pf_recalc[k])
            diff = (inp_intensity-calc_intensity)
            if ii == 0:
                residual = diff
                vals = inp_intensity
                weights = 1./np.sqrt(inp_intensity)
                weights[np.isnan(weights)] = 0.0
            else:
                residual = np.hstack((residual,diff))
                vals = np.hstack((vals,inp_intensity))
                ww = 1./np.sqrt(inp_intensity)
                ww[np.isnan(ww)] = 0.0
                weights = np.hstack((weights,ww))

        err = (weights*residual)**2
        wss = np.sum(err)
        den = np.sum((weights*vals)**2)

        Rwp = np.sqrt(wss/den)
        if np.mod(self.itercounter,100) == 0:
            msg = f"iteration# {self.itercounter}, Rwp error = {Rwp*100} %"
            print(msg)
        self.itercounter += 1
        return err

    def _compute_harmonic_values_grid(self,
                                      hkl,
                                      sample_dir):
        """
        compute the dictionary of invariant harmonic values for
        a given set of sample directions and hkls
        """
        ninv_c = self.mesh_crystal.num_invariant_harmonic(
        self.max_degree)

        ninv_s = self.mesh_sample.num_invariant_harmonic(
                 self.max_degree)

        V_c = self.mesh_crystal._get_harmonic_values(hkl)
        V_s = self.mesh_sample._get_harmonic_values(sample_dir)

        """
        some degrees for which the crystal symmetry has
        fewer terms than sample symmetry or vice versa
        needs to be weeded out
        """
        V_c_allowed = {}
        V_s_allowed = {}
        for i in np.arange(0,self.max_degree+1,2):
            if i in ninv_c[:,0] and i in ninv_s[:,0]:

                istc, ienc = self._index_of_harmonics(i, "crystal")
                V_c_allowed[i] = V_c[:,istc:ienc]

                ists, iens = self._index_of_harmonics(i, "sample")
                V_s_allowed[i] = V_s[:,ists:iens]

        return V_c_allowed, V_s_allowed

    def compute_texture_factor(self,
                               coeff):
        """
        first check if the size of the coef vector is
        consistent with the max degree argumeents for
        crystal and sample.
        """
        ncoeff = coeff.shape[0]+1

        ncoeff_inv = self._num_coefficients()
        if ncoeff < ncoeff_inv:
            msg = (f"inconsistent number of entries in "
                   f"coefficients based on the degree of "
                   f"crystal and sample harmonic degrees. "
                   f"needed {ncoeff_inv}, found {ncoeff}")
            raise ValueError(msg)
        elif ncoeff > ncoeff_inv:
            msg = (f"more coefficients passed than required "
                   f"based on the degree of crystal and "
                   f"sample harmonic degrees. "
                   f"needed {ncoeff_inv}, found {ncoeff}. "
                   f"ignoring extra terms.")
            warn(msg)
            coeff = coeff[:ncoeff_inv]

        tex_fact = {}
        for g in self.pole_figures.hkls:
            key = str(g)[1:-1].replace(" ","")
            nsamp = self.pole_figures.gvecs[key].shape[0]
            tex_fact[key] = np.zeros([nsamp,])

            tex_fact[key] = self._compute_sum(nsamp,
                                              coeff,
                                              self.allowed_degrees,
                                              self.V_c_allowed[key],
                                              self.V_s_allowed[key])
        return tex_fact

    def _index_of_harmonics(self,
                            deg,
                            c_or_s):
        """
        calculate the start and end index of harmonics
        of a given degree and crystal or sample symmetry
        returns the start and end index
        """
        ninv_c = self.mesh_crystal.num_invariant_harmonic(
                 self.max_degree)

        ninv_s = self.mesh_sample.num_invariant_harmonic(
                 self.max_degree)

        ninv_c_csum = np.r_[0,np.cumsum(ninv_c[:,1])]
        ninv_s_csum = np.r_[0,np.cumsum(ninv_s[:,1])]

        if c_or_s.lower() == "crystal":
            idx = np.where(ninv_c[:,0] == deg)[0]
            return int(ninv_c_csum[idx]), int(ninv_c_csum[idx+1])
        elif c_or_s.lower() == "sample":
            idx = np.where(ninv_s[:,0] == deg)[0]
            return int(ninv_s_csum[idx]), int(ninv_s_csum[idx+1])
        else:
            msg = f"unknown input to c_or_s"
            raise ValueError(msg)

    def _compute_sum(self,
                    nsamp,
                    coeff,
                    allowed_degrees,
                    V_c_allowed,
                    V_s_allowed):
        """
        compute the degree by degree sum in the
        generalized axis distribution function
        """
        tmp = copy.deepcopy(allowed_degrees)
        del tmp[0]
        nn = np.cumsum(np.array([tmp[k][0]*tmp[k][1]
            for k in tmp]))
        ncoeff_csum = np.r_[0,nn]

        val = np.ones([nsamp,])+self.phon
        for i,(k,v) in enumerate(tmp.items()):
            deg = k
            kc = V_c_allowed[deg]

            ks = V_s_allowed[deg].T

            mu = kc.shape[1]
            nu = ks.shape[0]

            ist = ncoeff_csum[i]
            ien = ncoeff_csum[i+1]

            C = np.reshape(coeff[ist:ien],[mu, nu])

            val = val + np.squeeze(np.dot(kc,np.dot(C,ks))*4*np.pi/(2*k+1))

        return val

    def _num_coefficients(self):
        """
        utility function to compute the number of
        independent coefficients required for the
        given maximum degree of harmonics
        """
        ninv_c = self.mesh_crystal.num_invariant_harmonic(
         self.max_degree)

        ninv_s = self.mesh_sample.num_invariant_harmonic(
                 self.max_degree)
        ncoef_inv = 0

        for i in np.arange(0,self.max_degree+1,2):
            if i in ninv_c[:,0] and i in ninv_s[:,0]:
                idc = int(np.where(ninv_c[:,0] == i)[0])
                ids = int(np.where(ninv_s[:,0] == i)[0])

                ncoef_inv += ninv_c[idc,1]*ninv_s[ids,1]
        return ncoef_inv

    def _allowed_degrees(self):
        """
        utility function to get the allowed degrees
        and the corresponding number of harmonics for
        crystal and sample symmetry
        """
        ninv_c = self.mesh_crystal.num_invariant_harmonic(
         self.max_degree)

        ninv_s = self.mesh_sample.num_invariant_harmonic(
                 self.max_degree)

        """
        some degrees for which the crystal symmetry has
        fewer terms than sample symmetry or vice versa
        needs to be weeded out
        """
        allowed_degrees = {}

        for i in np.arange(0,self.max_degree+1,2):
            if i in ninv_c[:,0] and i in ninv_s[:,0]:
                idc = int(np.where(ninv_c[:,0] == i)[0])
                ids = int(np.where(ninv_s[:,0] == i)[0])

                allowed_degrees[i] = [ninv_c[idc,1], ninv_s[ids,1]]

        return allowed_degrees

    def refine(self):

        params = self.init_coeff_params()
        fdict = {'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6,
         'verbose': 0, 'max_nfev': 20000, 'method':'trf',
         'jac':'3-point'}

        fitter = Minimizer(self.residual_function, params)

        res = fitter.least_squares(**fdict)
        params_res = res.params
        self.set_coeff_from_param(params_res)

    def recalculate_pole_figures(self):
        """
        this function calculates the pole figures for
        the sample directions in the pole figure used
        to initialize the model. this can be used to
        test how well the harmonic model fit the data.
        """
        return self.compute_texture_factor(self.coeff)



    def calc_pole_figures(self,
                      hkls,
                      grid="equiangular"):
        """
        given a set of hkl, coefficients and maximum degree of
        harmonic function to use for both crystal and sample
        symmetries, compute the pole figures for full coverage.
        the default grid is the equiangular grid, but other
        options include computing it on the s2 mesh or modified
        lambert grid or custom (theta, phi) coordinates.

        this uses the general axis distributiin function. other
        formalisms such as direct pole figure inversion is also
        possible using quadratic programming, but for that explicit
        pole figure operators are needed. the axis distributuion
        function is easy to integrate with the rietveld method.
        """

        """
        first check if the dimensions of coef is consistent with
        the maximum degrees of the harmonics
        """

        """
        check if class already exists with the same
        hkl values. if it does, then do nothing.
        otherwise initialize a new instance
        """
        init = True
        if hasattr(self, "pf_equiangular"):
            if self.pf_equiangular.hkls.shape == hkls.shape:
                if np.sum(np.abs(hkls - self.pf_equiangular.hkls)) < 1e-6:
                    init = False

        if init:

            angs = self.init_equiangular_grid()
            mat = self.pole_figures.material
            bHat_l = self.pole_figures.bHat_l
            eHat_l = self.pole_figures.eHat_l
            chi = self.pole_figures.chi

            t = angs[:,0]
            r = angs[:,1]
            st = np.sin(t)
            ct = np.cos(t)
            sr = np.sin(r)
            cr = np.cos(r)
            pfdata = {}
            for g in hkls:
                key = str(g)[1:-1].replace(" ","")
                xyz = np.vstack((st*cr,st*sr,ct)).T
                v = angs[:,2]
                pfdata[key] = np.vstack((xyz.T,v)).T

            args   = (mat, hkls, pfdata)
            kwargs = {"bHat_l":bHat_l,
                      "eHat_l":eHat_l,
                      "chi":chi}
            self.pf_equiangular = pole_figures(*args, **kwargs)

        model = harmonic_model(self.pf_equiangular,
                               self.sample_symmetry,
                               self.max_degree)

        model.coeff = self.coeff
        model.phon = self.phon

        pf = model.recalculate_pole_figures()
        pfdata = {}
        for k,v in self.pf_equiangular.pfdata.items():
            pfdata[k] = np.hstack((
                        np.degrees(model.pole_figures.angs[k]),
                        np.atleast_2d(pf[k]).T ))

        return pfdata

    def calc_inverse_pole_figures(self,
                                  sample_dir="ND",
                                  grid="equiangular",
                                  resolution = 5.0):
        """
        given a sample direction such as TD, RD and ND,
        calculate the distribution of crystallographic
        axes aligned with that direction. this is sometimes
        more useful than the regular pole figures especially
        for cylindrical sample symmetry where the PFs are
        usually "Bulls Eye" type. this follows the same
        flow as the pole figure calculation.
        sample_dir can have "RD", "TD" and "ND" as string
        arguments or can also have a nx3 array
        grid can be "equiangular" or "FEM"
        resolution in degrees. only used for equiangular grid

        instead of sampling asymmetric unit of stereogram,
        we will sample the entire northern hemisphere. the
        resulting IPDF will have symmetry of crystal
        """
        ipf = inverse_pole_figures(sample_dir,
                                   sampling=grid,
                                   resolution=resolution)
        vc, vs = self._compute_ipdf_mesh_vals(ipf)
        ipdf = self._compute_ipdf(ipf,vc,vs)
        angs = ipf.angs
        return np.hstack((angs,np.atleast_2d(ipdf).T))

    def _compute_ipdf_mesh_vals(self,
                    ipf):
        """
        compute the inverse pole density function.
        """
        ninv_c = self.mesh_crystal.num_invariant_harmonic(
        self.max_degree)

        ninv_s = self.mesh_sample.num_invariant_harmonic(
                 self.max_degree)

        V_c = self.mesh_crystal._get_harmonic_values(ipf.crystal_dir)
        V_s = self.mesh_sample._get_harmonic_values(ipf.sample_dir)

        """
        some degrees for which the crystal symmetry has
        fewer terms than sample symmetry or vice versa
        needs to be weeded out
        """
        V_c_allowed = {}
        V_s_allowed = {}
        for i in np.arange(0,self.max_degree+1,2):
            if i in ninv_c[:,0] and i in ninv_s[:,0]:

                istc, ienc = self._index_of_harmonics(i, "crystal")
                V_c_allowed[i] = V_c[:,istc:ienc]

                ists, iens = self._index_of_harmonics(i, "sample")
                V_s_allowed[i] = V_s[:,ists:iens]

        return V_c_allowed,V_s_allowed


    def _compute_ipdf(self,
                      ipf,
                      vc,
                      vs):
        """
        compute the generalized axis distribution
        function sum for the coefficients
        """
        allowed_degrees = self.allowed_degrees
        tmp = copy.deepcopy(allowed_degrees)
        del tmp[0]
        nn = np.cumsum(np.array([tmp[k][0]*tmp[k][1]
            for k in tmp]))
        ncoeff_csum = np.r_[0,nn]
        nsamp = ipf.sample_dir.shape[0]
        ncryst = ipf.crystal_dir.shape[0]
        coeff = self.coeff

        val = np.ones([ncryst,])+self.phon
        for i,(k,v) in enumerate(tmp.items()):
            deg = k
            kc = vc[deg]

            ks = vs[deg].T

            mu = kc.shape[1]
            nu = ks.shape[0]

            ist = ncoeff_csum[i]
            ien = ncoeff_csum[i+1]

            C = np.reshape(coeff[ist:ien],[mu, nu])

            val = val + np.squeeze(np.dot(kc,np.dot(C,ks))*4*np.pi/(2*k+1))

        return val

    def write_pole_figures(self, pfdata):
        """
        take output of the calc_pole_figures routine and write
        it out as text files
        """
        for k,v in pfdata.items():
            fname = f"pf_{k}.txt"
            np.savetxt(fname, v, fmt="%10.4f", delimiter="\t")


    @property
    def phon(self):
        return self._phon

    @phon.setter
    def phon(self, val):
        self._phon = val


    @property
    def J(self):
        tmp = copy.deepcopy(self.allowed_degrees)
        del tmp[0]
        nn = np.cumsum(np.array([tmp[k][0]*tmp[k][1]
            for k in tmp]))
        ncoeff_csum = np.r_[0,nn]
        J = 1.0
        for ii,k in enumerate(tmp):
            ist = ncoeff_csum[ii]
            ien = ncoeff_csum[ii+1]
            J += np.sum(self.coeff[ist:ien]**2)/(2*k+1)

        return J


class pole_figures:
    """
    this class deals with everything related to pole figures.
    pole figures can be initialized in a number of ways. the most
    basic being the (x,y,z, intensities) array. There are
    other formats which will be slowly added to this class. a list of
    hkl and a material/material_rietveld class is also supplied along
    with the (angle, intensity) info to get a class holding all the
    information.
    @DATE 10/06/2021 SS changes input from angles to unit vectors
                        and added routines to compute the angles
    """
    def __init__(self,
                 material,
                 hkls,
                 pfdata,
                 bHat_l=bVec_ref,
                 eHat_l=eta_ref,
                 chi=0.):
        """
        material Either a Material object of Material_Rietveld object
        hkl      reciprocal lattice vectors for which pole figures are
                 available (size nx3)
        pfdata   dictionary containing (tth, eta, omega, intensities)
                 array for each hkl specified as last input. key is hkl
                 and value are the arrays with size m x 4. m for each hkl
                 can be different. A length check is performed during init
        bHat_l   unit vector of xray beam in the lab frame. default value
                 is -Z direction
        eHat_l   direction which defines zero azimuth. default value is
                 x direction
        chi      inclination of sample frame about x direction. default
                 value is 0, corresponding to no tilt.
        """
        self.hkls = hkls
        self.material = material
        self.convert_hkls_to_cartesian()

        self.bHat_l = bHat_l
        self.eHat_l = eHat_l
        self.chi = chi

        if hkls.shape[0] != len(pfdata):
            msg = (f"pole figure initialization.\n"
                f"# reciprocal reflections = {hkls.shape[0]}.\n"
                f"# of entries in pfdata = {len(pfdata)}.")
            raise RuntimeError(msg)

        self.pfdata = pfdata

    def convert_hkls_to_cartesian(self):
        """
        this routine converts hkls in the crystallographic frame to
        the cartesian frame and normalizes them
        """
        self.hkls_c = np.atleast_2d(np.zeros(self.hkls.shape))

        for ii, g in enumerate(self.hkls):
            v = self.material.TransSpace(g, "r", "c")
            v = v/np.linalg.norm(v)
            self.hkls_c[ii,:] = v

    def write_data(self, prefix):
        """
        write out the data in text format
        the prefix goes in front of the names
        name will be "<prefix>_hkl.txt"
        """
        for k in self.pfdata:
            fname = f"{prefix}_{k}.txt"
            angs = np.degrees(self.angs[k])
            intensities = np.atleast_2d(self.intensities[k]).T
            data = np.hstack((angs,intensities))
            np.savetxt(fname, data, delimiter="\t")

    def stereographic_radius(self):
        sr = {}
        for h, angs in self.angs.items():
            t = angs[:,0]
            cth = np.cos(t)
            sth = np.sin(t)
            sr[h] = sth/(1 + cth)
        return sr

    def plot_pf(self,
                filled=False,
                grid=False,
                cmap='jet',
                colorbar=True,
                colorbar_label='m.r.d.',
                show=True):
        '''first get the layout of the subplot
        '''
        n = self.num_pfs
        nrows = int(n/3)+1
        if nrows == 1:
            ncols = np.min((3, n))
        else:
            ncols = 3

        self.fig, self.ax = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={'projection': 'polar'},
            figsize=(12,4*nrows))
        self.ax = np.atleast_2d(self.ax)

        [ax.set_axis_off() for ax in self.ax.flatten()]
        [ax.set_yticklabels([]) for ax in self.ax.flatten()]
        [ax.set_xticklabels([]) for ax in self.ax.flatten()]
        [ax.grid(False) for ax in self.ax.flatten()]

        for ii, h in enumerate(self.pfdata):
            nr = int(ii/3)
            nc = int(np.mod(ii, 3))
            rho = self.angs[h][:,1]
            r = self.stereo_radius[h]
            I = self.intensities[h]
            if filled:
                pf = self.ax[nr][nc].tricontourf(rho, r, I, 
                    levels=20, cmap=cmap)
            else:
                pf = self.ax[nr][nc].tricontour(rho, r, I, 
                    levels=20, cmap=cmap)
            self.ax[nr][nc].set_yticklabels([])
            self.ax[nr][nc].grid(grid)
            self.ax[nr][nc].set_title(f'({h})')
            plt.colorbar(pf, label=colorbar_label)

        if show:
            self.fig.show()

    @property
    def num_pfs(self):
        """ number of pole figures (read only) """
        return len(self.pfdata)

    """
    the pfdata property returns the pole figure data
    in the form of a dictionary with keys as the hkl
    values and the value as the (tth, eta, omega) array.
    """
    @property
    def pfdata(self):
        return self._pfdata

    @pfdata.setter
    def pfdata(self, val):
        self._pfdata = {}
        self._intensities = {}
        self._gvecs = {}
        self._angs = {}
        for k,v in val.items():
            norm = np.linalg.norm(v[:,0:3],axis=1)
            v[:,0:3] = v[:,0:3]/np.tile(norm, [3,1]).T
            t = np.arccos(v[:,2])
            rho = np.arctan2(v[:,1],v[:,0])

            self._gvecs[k] = v[:,0:3]
            self._pfdata[k] = v
            self._intensities[k] = v[:,3]
            self._angs[k] = np.vstack((t, rho)).T
            self._stereo_radius = self.stereographic_radius()

    @property
    def gvecs(self):
        return self._gvecs

    @property
    def angs(self):
        return self._angs

    @property
    def intensities(self):
        return self._intensities

    @property
    def stereo_radius(self):
        return self._stereo_radius
    

class inverse_pole_figures:
    """
    this class deals with everything related to inverse pole figures.

    """
    def __init__(self,
                 sample_dir,
                 sampling="equiangular",
                 resolution=5.0):
        """
        this is the initialization of the class. the inputs are
        1. laue_sym for laue symmetry
        2. sample_dir the sample direction for pole figure
        3. sampling sampling strategy for the IPF stereogram.
           optios are "equiangular" and "FEM"
        4. resolution grid resolution in degrees. only valid for
           equiangular sampling type
        """
        self.sample_dir = sample_dir
        # self.laue_sym = laue_sym
        if sampling == "equiangular":
            self.resolution = resolution
        self.sampling = sampling

    def initialize_crystal_dir(self,
                               samplingtype,
                               resolution=5.0):
        """
        this function prepares the unit vectors
        of the stereogram
        """
        if samplingtype.lower() == "equiangular":
            angs = []
            for tth in np.arange(0,91,resolution):
                for eta in np.arange(0, 360, resolution):
                    angs.append([np.radians(tth), np.radians(eta)])
                    if tth == 0:
                        break
            angs = np.array(angs)
            self.crystal_dir = np.zeros([angs.shape[0],3])
            for i,a in enumerate(angs):
                t, r = a
                st = np.sin(t)
                ct = np.cos(t)
                sr = np.sin(r)
                cr = np.cos(r)
                self.crystal_dir[i,:] = np.array([st*cr,st*sr,ct])

        if samplingtype.lower() == "fem":
            msg = "sampling type FEM not implemented yet."
            raise ValueError(msg)

    @property
    def sample_dir(self):
        """ sample direction for IPF """
        return self._sample_dir

    @sample_dir.setter
    def sample_dir(self, val):
        # interpret sample_dir input
        # sample_dir size = nx3
        if isinstance(val, str):
            if val.upper() == "RD":
                self._sample_dir = np.atleast_2d([1.,0.,0.])
                self._sample_dir_name = "RD"
            elif val.upper() == "TD":
                self._sample_dir = np.atleast_2d([0.,1.,0.])
                self._sample_dir_name = "TD"
            elif val.upper() == "ND":
                self._sample_dir = np.atleast_2d([0.,0.,1.])
                self._sample_dir_name = "ND"
            else:
                msg = f"unknown direction."
                raise ValueError(msg)
        elif isinstance(val, np.array):
            v = np.atleast_2d(val)
            if v.shape[1] != 3:
                msg = (f"incorrect shape for sample_dir input.\n"
                       f"expected nx3, got {val.shape[0]}x{val.shape[1]}")
                raise ValueError(msg)
            self._sample_dir = v
            self._sample_dir_name = "array"

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        if val < 1.0:
            msg = (f"the resolution appears to be very fine.\n"
                f"Are you sure the value is in degrees?")
            warn(msg)
        self._resolution = val

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, val):
        if val.lower() == "equiangular":
            self.initialize_crystal_dir("equiangular",
                                     resolution=self.resolution)

        elif val.lower() == "fem":
            self.initialize_crystal_dir("FEM")

    @property
    def angs(self):
        polar = np.arccos(self.crystal_dir[:,2])
        az = np.arctan2(self.crystal_dir[:,1],self.crystal_dir[:,0])
        return np.degrees(np.vstack((polar,az)).T)

'''
in this section we will list the constants we need for symmetrizing
the spherical harmonics for the cubic crystals (sgnum 195-230). 
For cyclic and dihedral symmetries, the symmetrization is done by
selection rules, so they don't need to be explicitly written here

we will list all constants up to degree 16. If the user asks for
anything more than that, we will need to compute it on the fly
using the spherical python package. However, this will build an 
extra dependency, so not sure if we need that
'''
Blmn = {
    'td':{
        4:np.array([[0.4564354645876384,0.0,
                     0.7637626158259735,0.0,
                     0.4564354645876382,]]),
        6:np.array([[-0.3952847075210473,0.0,
                     0.5863019699779287,0.0,
                     0.5863019699779286,0.0,
                    -0.3952847075210476,],
                    [-0.23587862860606446,0.5307650734434042,
                    0.349864545721299,-0.28370586538444675,
                    0.349864545721299,0.5307650734434044,
                    -0.23587862860606465,]]),
        8:np.array([[0.4114253678777401,0.0,
                    0.270030862433661,0.0,
                    0.7180703308172532,0.0,
                    0.270030862433661,0.0,
                    0.4114253678777393,]]),
        10:np.array([[-0.39365513190875756,-0.040358697757753324,
                    0.10745401012934847,-0.033908088412492564,
                    0.547910094463344,0.033650228484593495,
                    0.547910094463344,-0.0339080884124925,
                    0.10745401012934841,-0.04035869775775318,
                    -0.3936551319087571,],
                     [0.11765319612066327,0.47226403896469205,
                    -0.03211518586433684,0.39678115689893184,
                    -0.16375595940491486,-0.3937637659075811,
                    -0.163755959404915,0.39678115689893145,
                    -0.03211518586433705,0.4722640389646918,
                    0.117653196120663,]]),
        12:np.array([[0.35435463302663495,-0.17379154613641162,
                    0.13422630916783754,0.2754451663660531,
                    0.2490795461687716,-0.07009627867210125,
                    0.6055804920306865,-0.07009627867210133,
                    0.24907954616877184,0.27544516636605293,
                    0.13422630916783768,-0.1737915461364115,
                    0.3543546330266347,],
                     [0.029964757460644757,-0.34851935846790966,
                    0.20072357823323558,0.5523742368897112,
                    -0.06998760599808057,-0.1405701866223871,
                    0.12243483258685284,-0.1405701866223871,
                    -0.06998760599808042,0.5523742368897113,
                    0.20072357823323567,-0.3485193584679096,
                    0.029964757460644646,],
                     [-0.14236359629542855,0.21023745347413406,
                    0.4399875511085443,-0.33320890248081825,
                    -0.33754122137204073,0.0847962024255695,
                    -0.057526585048038516,0.0847962024255693,
                    -0.33754122137204073,-0.3332089024808181,
                    0.4399875511085443,0.21023745347413386,
                    -0.14236359629542847,]]),
        14:np.array([[-0.4081984524254106,0.0,
                    0.014263608268363568,0.0,
                    0.1757378418625312,0.0,
                    0.5498061329724928,0.0,
                    0.5498061329724929,0.0,
                    0.1757378418625312,0.0,
                    0.01426360826836375,0.0,
                    -0.4081984524254106,],
                     [-0.2609562260878263,-0.31517901621833544,
                    0.009118548495201403,-0.2595707050634366,
                    0.11234703052100681,-0.24189163409933004,
                    0.3514842662630142,0.32894254900247105,
                    0.3514842662630142,-0.24189163409933007,
                    0.11234703052100689,-0.25957070506343644,
                    0.009118548495201627,-0.31517901621833566,
                    -0.2609562260878262,]]),
        16:np.array([[0.4082607490228619,0.0,
                    0.004724991415765843,0.0,
                    0.08080978644090049,0.0,
                    0.3744089876279586,0.0,
                    0.6108821877615943,0.0,
                    0.37440898762795843,0.0,
                    0.08080978644090046,0.0,
                    0.004724991415765664,0.0,
                    0.4082607490228623,],
                     [0.0002761390670381201,-0.36909867621231496,
                    0.20752737777828104,0.1813395395452281,
                    0.12139515667567446,0.4656650692573661,
                    -0.1280742029731327,-0.17983149224440206,
                    0.12240364673979474,-0.1798314922444023,
                    -0.12807420297313307,0.46566506925736645,
                    0.12139515667567449,0.18133953954522805,
                    0.20752737777828129,-0.369098676212315,
                    0.00027613906703817444,],
                     [-0.07283664719091107,0.09028739951989459,
                    0.488682129558481,-0.04435853204267701,
                    0.2718108984316556,-0.11390907326450131,
                    -0.3695065787821054,0.04398963971680427,
                    0.17877535778204223,0.04398963971680411,
                    -0.36950657878210535,-0.11390907326450159,
                    0.2718108984316559,-0.04435853204267742,
                    0.4886821295584812,0.09028739951989452,
                    -0.0728366471909116,]]),
        },
    'oh': {
        2:np.array([[1.]]),
        4:np.array([[0.4564354645876381,0.7637626158259734,
                     0.4564354645876386,],]),
        6:np.array([[0.6614378277661478,-0.35355339059327384,
                     0.6614378277661475,],]),
        8:np.array([[0.4114253678777395,0.2700308624336608,
                     0.7180703308172537,0.2700308624336609,
                     0.41142536787773965,],]),
        10:np.array([[0.49344663676362555,0.41457809879442475,
                     -0.41142536787773953,0.41457809879442487,
                     0.4934466367636257,],]),
        12:np.array([[0.4084475818062013,0.0410768724615144,
                     0.34173954767501197,0.6552821453699549,
                     0.34173954767501197,0.04107687246151456,
                     0.4084475818062011,],[0.4078044975385258,
                     0.006250066404837358,0.3579150356801124,
                     0.6411758817850054,0.3579150356801124,
                     0.006250066404837514,0.4078044975385257,],]),
        14:np.array([[-0.4216820546434637,-0.347282980795201,
                     -0.3236299246438747,0.440096461964117,
                     -0.32362992464387447,-0.3472829807952011,
                     -0.421682054643464,],]),
        16:np.array([[0.4082607490228622,0.004724991415765745,
                      0.08080978644090052,0.3744089876279585,
                      0.6108821877615942,0.37440898762795843,
                      0.08080978644090042,0.00472499141576571,
                      0.40826074902286186,],
                      [0.33277795799251847, -0.2935570355266691,
                      -0.10802732705903098,0.4890944822748237,
                      0.3231092124465864,0.4890944822748239,
                      -0.10802732705903116,-0.2935570355266692,
                      0.3327779579925184,],],)
    }
}
