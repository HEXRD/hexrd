import numpy as np
from scipy.spatial import Delaunay
import hexrd.resources
import importlib.resources
import h5py
from numpy.polynomial.polynomial import Polynomial
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

            dname = f"/{symmetry}/harmonics"
            self.harmonics = np.array(fid[dname])

            dname = f"/{symmetry}/eqv"
            self.eqv = np.array(fid[dname]).astype(np.int32)

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
            msg = (f"some points seem to not be in the"
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
        to unit length

        then take the stereographic projection
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
            eqv_id = np.array([np.where(self.eqv[:,0] == i) 
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
        """
        bary_center, simplex_id = self._get_barycentric_coordinates(points)
        node_id = self.mesh.simplices[simplex_id]

        eqv_node_id = np.array([self._get_equivalent_node(nid) 
            for nid in node_id]).astype(np.int32)

        fval = np.array([self.harmonics[nid,:] for nid in eqv_node_id])
        nharm = self.harmonics.shape[1]

        fval_points = np.array([np.sum(np.tile(bary_center[i,:],[nharm,1]).T*
            fval[i,:,:],axis=0) for i in range(points.shape[0])])

        return fval_points

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

        poly = poly.truncate(max_degree)
        idx = np.nonzero(poly.coef)
        v = poly.coef[idx]

        return np.vstack((idx,v)).T

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
                crystal_symmetry,
                sample_symmetry,
                max_degree_crystal,
                max_degree_sample):

        self.crystal_symmetry = crystal_symmetry
        self.sample_symmetry = sample_symmetry
        self.max_degree_sample = max_degree_sample
        self.max_degree_crystal = max_degree_crystal

        self.mesh_crystal = mesh_s2(self.crystal_symmetry)
        self.mesh_sample = mesh_s2(self.sample_symmetry)




    def calc_pole_figures(self, 
                          hkls, 
                          coef,
                          max_degree_crystal,
                          max_degree_sample):
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
        pass

Polya = {
        "m35":
        {"numerator":[],
        "denominator":[6, 10]},

        "532":
        {"numerator":[[15,1.]],
        "denominator":[6, 10]},

        "m3m":
        {"numerator":[],
        "denominator":[4, 6]},

        "432":
        {"numerator":[[9,1.]],
        "denominator":[4, 6]},

        "1":
        {"numerator":[[1,1.]],
        "denominator":[1, 1]},

        "-1":
        {"numerator":[[2,3.]],
        "denominator":[2, 2]}

        }











