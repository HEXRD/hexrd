import numpy as np
from scipy.spatial import Delaunay
import hexrd.resources
import importlib.resources
import h5py
from numpy.polynomial.polynomial import Polynomial
from warnings import warn
from hexrd.transforms.xfcapi import anglesToGVec
from hexrd.wppf import phase
from lmfit import Parameters, Minimizer
"""
========================================================================================================
========================================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       06/14/2021 SS 1.0 original

    >> @DETAILS:    this module deals with the texture computations for the wppf package.
    two texture models employed are the March-Dollase model and the spherical harmonics
    model. the spherical harmonics model uses the axis distribution function for computing
    the scale factors for each reflection. maybe it makes sense to use symmetrized FEM mesh
    functions for 2-sphere for this. probably use the two theta-eta array from the instrument
    class to precompute the values of k_l^m(y) and we already know what the reflections are
    so k_l^m(h) can also be pre-computed.

    >> @PARAMETERS:  symmetry symmetry of the mesh
========================================================================================================
========================================================================================================   
"""
I3 = np.eye(3)

Xl = np.ascontiguousarray(I3[:, 0].reshape(3, 1))     # X in the lab frame
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

            dname = f"/{symmetry}/crd"

            pts = np.array(fid[dname])
            n = np.linalg.norm(pts, axis=1)
            self.points = pts/np.tile(n, [3,1]).T

            points_st = self.points[:,:2]/np.tile(
                (1.+self.points[:,2]),[2,1]).T

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
                (1.+points[:,2]),[2,1]).T

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
                            points):
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
        mask = points[:,-1] < 0.
        points[mask,:] = -points[mask,:]
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
        self.phon = 0.0
        self.itercounter = 0

        ncoeff = self._num_coefficients()
        self.coeff = np.zeros([ncoeff,])
        self.coeff[0] = 1.0

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
        ctr = 0
        for ii,(k,v) in enumerate(self.allowed_degrees.items()):
          for icry in np.arange(v[0]):
              for isamp in np.arange(v[1]):
                  vname = f"c_{k}_{icry}_{isamp}"
                  self.coeff_loc[vname] = ctr
                  ctr += 1
                  params.add(vname,value=self.coeff[ii],vary=True)
        return params

    def set_coeff_from_param(self, params):
        """
        this function takes the the values in the 
        parameters and sets the values of the coefficients
        """
        for ii,k in enumerate(params):
            loc = self.coeff_loc[k]
            self.coeff[loc] = params[k].value

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
            diff = (inp_intensity-calc_intensity-self.phon)
            if ii == 0:
                residual = diff
                vals = inp_intensity
            else:
                residual = np.hstack((residual,diff))
                vals = np.hstack((vals,inp_intensity))

        Rp = np.linalg.norm(residual)/np.linalg.norm(vals)
        if np.mod(self.itercounter,100) == 0:
            msg = f"iteration# {self.itercounter}, Rp error = {Rp*100} %"
            print(msg)
        self.itercounter += 1
        return residual

    def phon_residual_function(self, params):
        """
        calculate the residual in estimating the uniform
        portion of the odf to make it into a sharp odf.
        same method as mtex
        """
        for ii,(k,v) in enumerate(self.pole_figures.pfdata.items()):
            inp_intensity = np.squeeze(v[:,3])
            diff = (inp_intensity - params["phon"].value)
            if ii == 0:
                residual = diff
                vals = inp_intensity
            else:
                residual = np.hstack((residual,diff))
                vals = np.hstack((vals,inp_intensity))
        return residual

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
        ncoeff = coeff.shape[0]
        
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
            coeff = coef[:ncoeff_inv]
 
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
        nn = np.cumsum(np.array([allowed_degrees[k][0]*allowed_degrees[k][1] 
            for k in allowed_degrees]))
        ncoeff_csum = np.r_[0,nn]

        val = np.zeros([nsamp,])
        for i,(k,v) in enumerate(allowed_degrees.items()):
            deg = k

            kc = V_c_allowed[deg]

            ks = V_s_allowed[deg].T

            mu = kc.shape[1]
            nu = ks.shape[0]

            ist = ncoeff_csum[i]
            ien = ncoeff_csum[i+1]
            C = np.reshape(coeff[ist:ien],[mu, nu])

            val = val + np.dot(kc,np.dot(C,ks))

        val = val
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
        """
        this is the function which updates the harmonic coefficients
        to fit better to the input pole figure data. 
        """
        """
        first determine the phon
        """
        # params = Parameters()
        # params.add("phon",value=0.0)
        # fdict = {'ftol': 1e-4, 'xtol': 1e-4, 
        #  'gtol': 1e-4,'max_nfev': 100}
        # # fdict = {'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6,
        # #  'verbose': 0, 'max_nfev': 100, 'method':'trf',
        # #  'jac':'2-point'}
        # fitter = Minimizer(self.phon_residual_function, params)
        # res = fitter.leastsq(**fdict)
        # params_phon_res = res.params
        # self.phon = params_phon_res["phon"].value

        # msg = f"setting uniform portion to {self.phon}"
        # print(msg)

        params = self.init_coeff_params()
        fdict = {'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4,
         'verbose': 0, 'max_nfev': 10000, 'method':'trf',
         'jac':'2-point'}
        # fdict = {'ftol': 1e-4, 'xtol': 1e-4, 
        #  'gtol': 1e-4,'max_nfev': 10000}
        fitter = Minimizer(self.residual_function, params)

        res = fitter.least_squares(**fdict)
        # res = fitter.leastsq(**fdict)
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
            zc = np.atleast_2d(np.zeros([angs.shape[0],])).T
            angs = np.append(angs,zc,axis=1)
            mat  = self.pole_figures.material
            bHat_l = self.pole_figures.bHat_l
            eHat_l = self.pole_figures.eHat_l
            chi = self.pole_figures.chi
            pfdata = {}
            for g in hkls:
                key = str(g)[1:-1].replace(" ","")
                pfdata[key] = angs

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
            pfdata[k] = np.hstack((np.degrees(v[:,0:2]), 
                self.phon+np.atleast_2d(pf[k]).T ))

        return pfdata

    def write_pole_figures(self, pfdata):
        """
        take output of the calc_pole_figures routine and write
        it out as text files
        """
        for k,v in pfdata.items():
            fname = f"pf_{k}.txt"
            np.savetxt(fname, v, fmt="%10.4f", delimiter="\t")


class pole_figures:
    """
    this class deals with everything related to pole figures. 
    pole figures can be initialized in a number of ways. the most
    basic being the (tth, eta, omega, intensities) array. There are
    other formats which will be slowly added to this class. a list of 
    hkl and a material/material_rietveld class is also supplied along 
    with the (angle, intensity) info to get a class holding all the 
    information. 
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
        self.convert_angs_to_gvecs()

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


    def convert_angs_to_gvecs(self):
        """
        this routine converts angular coordinates in (tth, eta, omega)
        to g-vectors in the lab frame
        """
        self.gvecs = {}
        for k,v in self.pfdata.items():
            angs = v[:,0:3]
            if np.abs(angs).max() > 2.0*np.pi:
                msg = f"angles seem to be large. converting to radians."
                print(msg)
                angs = np.atleast_2d(np.radians(angs))
                self.pfdata[k][:,0:3] = angs

            self.gvecs[k] = anglesToGVec(angs, 
                                         bHat_l=self.bHat_l,
                                         eHat_l=self.eHat_l,
                                         chi=self.chi)

    @property
    def num_pfs(self):
        """ number of pole figures (read only) """
        return len(self.pfdata)

Polya = {
        "m35":
        {"numerator":[],
        "denominator":[6, 10]},

        "532":
        {"numerator":[[15, 1.]],
        "denominator":[6, 10]},

        "m3m":
        {"numerator":[],
        "denominator":[4, 6]},

        "432":
        {"numerator":[[9, 1.]],
        "denominator":[4, 6]},

        "1":
        {"numerator":[[1, 1.]],
        "denominator":[1, 1]},

        "-1":
        {"numerator":[[2, 3.]],
        "denominator":[2, 2]}

        }