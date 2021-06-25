import numpy as np
from scipy.spatial import Delaunay
import hexrd.resources
import importlib.resources
import h5py

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

    >> @PARAMETERS:     
"""
class texture:
    """
    ========================================================================================================
    ========================================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       06/23/2021 SS 1.0 original

    >> @DETAILS:    this is the main LeBail class and contains all the refinable parameters
                    for the analysis. Since the LeBail method has no structural information
                    during refinement, the refinable parameters for this model will be:

                    1. a, b, c, alpha, beta, gamma : unit cell parameters
                    2. U, V, W : cagliotti paramaters
                    3. 2theta_0 : Instrumental zero shift error
                    4. eta1, eta2, eta3 : weight factor for gaussian vs lorentzian

                    @NOTE: All angles are always going to be in degrees

    >> @PARAMETERS  expt_spectrum: name of file or numpy array or Spectrum class of experimental intensity
                    params: yaml file or dictionary or Parameter class
                    phases: yaml file or dictionary or Phases_Lebail class
                    wavelength: dictionary of wavelengths
                    bkgmethod: method to estimate background. either spline or chebyshev fit
                    or filename or numpy array (last two options added 01/22/2021 SS)
                    Intensity_init: if set to none, then some power of 10 is used. User has option
                    to pass in dictionary of structure factors. must ensure that the size of structure
                    factor matches the possible reflections (added 01/22/2021 SS)
    ========================================================================================================
    ========================================================================================================
    """

    def __init__(self,
                 symmetry):

        data = importlib.resources.open_binary(hexrd.resources, "surface_harmonics.h5")
        with h5py.File(data, 'r') as fid:
            
            dname = f"/{symmetry}/crd"
            pts = np.array(fid[dname])
            n = np.linalg.norm(pts,axis=1)
            self.points = pts/np.tile(n,[3,1]).T

            dname = f"/{symmetry}/harmonics"
            self.harmonics = np.array(fid[dname])

            self.mesh = Delaunay(self.points, qhull_options="QJ")

    def _get_barycentric_coordinates(self,
                                    points):

        """
        get the barycentric coordinates of points
        this is used for linear interpolation of
        the harmonic function on the mesh
        """
