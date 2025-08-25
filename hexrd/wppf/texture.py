import copy
from warnings import warn
# 3rd party imports
import h5py
from lmfit import Parameters, Minimizer
import numpy as np
from hexrd import constants
from scipy.special import sph_harm_y
from matplotlib import pyplot as plt
from hexrd.rotations import rotMatOfExpMap
"""
===============================================================================

>> @AUTHOR: Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
>> @DATE: 06/14/2021 SS 1.0 original

>> @DETAILS: this module deals with the texture computations for the wppf
    package.  two texture models employed are the March-Dollase model and the
    spherical harmonics model. the spherical harmonics model uses the axis
    distribution function for computing the scale factors for each reflection.
    maybe it makes sense to use symmetrized spherical harmonics. probably use 
    the two theta-eta array from the instrument class to precompute the values 
    of k_l^m(y) and we already know what the reflections are so k_l^m(h) can 
    also be pre-computed.
===============================================================================
"""


bvec_ref = constants.beam_vec
eta_ref  = constants.lab_x

class harmonic_model:
    """
    this class brings all the elements together to compute the
    texture model given the sample and crystal symmetry.
    """
    def __init__(self):
        pass

    @property
    def phon(self):
        return self._phon

    @phon.setter
    def phon(self, val):
        self._phon = val

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
                 ssym='axial',
                 bvec=bvec_ref,
                 evec=eta_ref,
                 sample_normal=-constants.lab_z,
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

        self.bvec = bvec
        self.etavec = evec
        self.sample_normal = sample_normal
        self.chi = chi
        self.ssym = ssym

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
            sr[h] = sth/(1 + np.abs(cth))
        return sr

    def plot_pf(self,
                filled=False,
                grid=False,
                cmap='jet',
                colorbar=True,
                colorbar_label='m.r.d.',
                show=True,
                recalculated=False):
        '''
        # FIXME: there is a bug during plots when the
        0/2pi or -pi/pi points are not filled in for some
        set of inputs. need to come up with a fix for this

        08/25/25 SS added a fix which seems to work for some
        cases. needs more extensive testing
        '''

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

            # since there is a discontinuity at +/- pi azimuth
            # we will add some extra points there for the plots

            # first get the points which have rho values of +/- pi
            mask = np.isclose(np.abs(rho), np.pi)
            # add these same points at -pi
            rho = np.concatenate((rho, -rho[mask]))

            # next add points at r and intensity
            r = np.concatenate((r, r[mask]))
            I = np.concatenate((I, I[mask]))

            if recalculated:
                I = self.intensities_recalc[h]
                I = np.concatenate((I, I[mask]))
            if filled:
                pf = self.ax[nr][nc].tricontourf(rho, r, 
                                    I, levels=20, cmap=cmap)
            else:
                pf = self.ax[nr][nc].tricontour(rho, r,
                                    I, levels=20, cmap=cmap)
            self.ax[nr][nc].set_yticklabels([])
            self.ax[nr][nc].grid(grid)
            self.ax[nr][nc].set_title(f'({h})')
            plt.colorbar(pf, label=colorbar_label)

        if show:
            self.fig.show()

    def calc_residual(self, params):
        '''get the difference between the
        input pole figures and the calculated
        pole figures
        '''
        measured = np.empty(0)
        for h, v in self.intensities.items():
            measured = np.concatenate((measured, v))

        calculated = np.empty(0)
        for h, v in self.intensities.items():
            term = np.ones_like(v)

            for ell in np.arange(2, self.ell_max+1, 2):
                pre = 4*np.pi/(2*ell+1)
                cmat = self.get_c_matrix(params, ell)
                sph_s_mat = self.get_sph_s_matrix(h, ell)
                sph_c_mat = self.get_sph_c_matrix(h, ell)
                t = pre*np.dot(sph_s_mat, np.dot(
                                   cmat, sph_c_mat))
                term += np.squeeze(t)

            calculated = np.concatenate((calculated, term))

        return np.sqrt(self.weights*(measured - calculated)**2)

    def recalculated_pf(self, params):
        self.intensities_recalc = {}

        for h, v in self.intensities.items():
            term = np.ones_like(v)

            for ell in np.arange(2, self.ell_max+1, 2):
                pre = 4*np.pi/(2*ell+1)
                cmat = self.get_c_matrix(params, ell)
                sph_s_mat = self.get_sph_s_matrix(h, ell)
                sph_c_mat = self.get_sph_c_matrix(h, ell)
                t = pre*np.dot(sph_s_mat, np.dot(
                                   cmat, sph_c_mat))
                term += np.squeeze(t)

            self.intensities_recalc[h] = term

    def calc_new_pole_figure(self, hkls):
        pass

    def calculate_harmonic_coefficients(self,
                                        ell_max,
                                        ):

        self.ell_max = ell_max
        param = get_parameters(ell_max,
                               csym=self.csym,
                               ssym=self.ssym)
        '''precompute the spherical harmonics for
        the given hkls and sample directions
        '''

        self.sph_c = {}
        self.sph_s = {}

        for ii, (h, v) in enumerate(self.hkl_angles.items()):
            '''needs special attention for monoclininc case
                since the 2-fold axis is aligned with b*
            '''
            theta = np.array([v[0]])
            phi = np.array([v[1]])
            self.sph_c[h] = {}
            self.sph_s[h] = {}
            for ell in np.arange(2, ell_max+1, 2):
                Ylm = calc_sym_sph_harm(ell, 
                                        theta,
                                        phi,
                                        sym=self.csym)
                for jj in np.arange(Ylm.shape[1]):
                    kname = f'c_{ell}{jj}'
                    self.sph_c[h][kname] = Ylm[:,jj]

            '''use the rotated angles to account for the
            misalignment between sample normal and beam
            propagation vector
            '''
            theta = self.rotated_angs[h][:,0]
            phi   = self.rotated_angs[h][:,1]
            for ell in np.arange(2, ell_max+1, 2):
                Ylm = calc_sym_sph_harm(ell, 
                                        theta,
                                        phi,
                                        sym=self.ssym)
                for jj in np.arange(Ylm.shape[1]):
                    kname = f's_{ell}{jj}'
                    self.sph_s[h][kname] = Ylm[:,jj]

        params = get_parameters(self.ell_max)
        fdict = {'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6,
         'verbose': 2, 'max_nfev': 20000, 'method':'trf',
         'jac':'3-point'}

        fitter = Minimizer(self.calc_residual, params)

        self.res = fitter.least_squares(**fdict)

    def get_c_matrix(self, params, ell):
        nc = get_num_sym_harm(ell, 
                            sym=self.csym)
        ns = get_num_sym_harm(ell,
                            sym=self.ssym)

        cmat = np.zeros([ns, nc])
        for ii in np.arange(ns):
            for jj in np.arange(nc):
                pname = f'c_{ell}{ii}{jj}'
                cmat[ii, jj] = params[pname].value
        return cmat

    def get_sph_s_matrix(self, h, ell):

        ns = get_num_sym_harm(ell,
                            sym=self.ssym)
        ngrid = self.angs[h].shape[0]
        smat = np.zeros((ngrid, ns))
        Ylm = self.sph_s[h]

        for ii in np.arange(ns):
            yname = f's_{ell}{ii}'
            smat[:,ii] = Ylm[yname]

        return smat

    def get_sph_c_matrix(self, h, ell):

        nc = get_num_sym_harm(ell,
                            sym=self.csym)
        ngrid = 1
        cmat = np.zeros((nc, ngrid))
        Ylm = self.sph_c[h]

        for ii in np.arange(nc):
            yname = f'c_{ell}{ii}'

            cmat[ii, :] = Ylm[yname]

        return cmat

    def calc_ref_frame(self):
        an = np.arccos(np.dot(
                self.bvec,
                self.sample_normal))
        ax = np.cross(self.bvec,
                      self.sample_normal)
        norm = np.linalg.norm(ax)
        if norm == 0:
            self._ref_frame_rmat = np.eye(3)
        else:
            ax = ax/norm
            self._ref_frame_rmat = (
                rotMatOfExpMap(an*ax)
                )

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
        self._weights = np.empty(0)
        self._gvecs = {}
        self._angs = {}
        self._rotated_angs = {}
        self._hkl_angles = {}
        for ii, (k,v) in enumerate(val.items()):

            norm = np.linalg.norm(v[:,0:3],axis=1)
            v[:,0:3] = v[:,0:3]/np.tile(norm, [3,1]).T

            # sanitize v[:, 2] for arccos operation
            mask = np.abs(v[:,2]) > 1.
            v[mask,2] = np.sign(v[mask,2])

            t = np.arccos(v[:,2])
            rho = np.arctan2(v[:,1],v[:,0])

            vr = np.dot(self.ref_frame_rmat, v[:,0:3].T).T
            
            # sanitize vr[:, 2] for arccos operation
            mask = np.abs(vr[:,2]) > 1.
            vr[mask,2] = np.sign(vr[mask,2])
            
            tr = np.arccos(vr[:,2])
            rhor = np.arctan2(vr[:,1],vr[:,0])

            self._gvecs[k] = v[:,0:3]
            self._pfdata[k] = v
            self._intensities[k] = v[:,3]
            self._weights = np.concatenate((self._weights,v[:,3]))
            self._angs[k] = np.vstack((t, rho)).T
            self._rotated_angs[k] = np.vstack((tr, rhor)).T
            self._stereo_radius = self.stereographic_radius()
            self._hkl_angles[k] = np.array([np.arccos(
                self.hkls_c[ii, 2]),
                np.arctan2(self.hkls_c[ii, 1], self.hkls_c[ii, 0])])
        self._weights = 1/self._weights
        self._weights = np.nan_to_num(self._weights)

    @property
    def gvecs(self):
        return self._gvecs

    @property
    def angs(self):
        return self._angs

    @property
    def rotated_angs(self):
        return self._rotated_angs

    @property
    def intensities(self):
        return self._intensities

    @property
    def weights(self):
        return self._weights

    @property
    def stereo_radius(self):
        return self._stereo_radius

    @property
    def hkl_angles(self):
        return self._hkl_angles

    @property
    def csym(self):
        return self.material.sg.laueGroup

    @property
    def ssym(self):
        return self._ssym

    @ssym.setter
    def ssym(self, v):
        if isinstance(v, str):
            self._ssym = v
        else:
            msg = f'unknown sample symmetry type'
            raise ValueError(msg)

    @property
    def J(self):
        if hasattr(self, 'res'):
            J = 1.
            for ell in np.arange(2, self.ell_max+1, 2):
                nc = get_num_sym_harm(ell, sym=self.csym)
                ns = get_num_sym_harm(ell, sym=self.ssym)
                for ii in np.arange(ns):
                    for jj in np.arange(nc):
                        pname = f'c_{ell}{ii}{jj}'
                        pre = 1/(2*ell+1)
                        J += pre*(
                        self.res.params[pname].value**2)
        else:
            msg = f'coefficeints have not been computed yet'
            warnings.warn(msg)
        return J

    @property
    def bvec(self):
        return self._bvec

    @property
    def etavec(self):
        return self._etavec

    @bvec.setter
    def bvec(self, bHat_l):
        self._bvec = bHat_l
        if hasattr(self, '_sample_normal'):
            self.calc_ref_frame()

    @etavec.setter
    def etavec(self, eHat_l):
        self._etavec = eHat_l

    @property
    def sample_normal(self):
        return self._sample_normal

    @sample_normal.setter
    def sample_normal(self, val):
        self._sample_normal = val
        if hasattr(self, '_bvec'):
            self.calc_ref_frame()

    @property
    def ref_frame_rmat(self):
        return self._ref_frame_rmat


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
    'th':{
        2:np.array([[1.]]),
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

'''
=================================================================
=================================================================
utility functions for texture computation
=================================================================
=================================================================
'''

def calc_sym_sph_harm(deg, 
                      theta,
                      phi,
                      sym='oh'):
    '''
    get symmetrized harmonics with given degree
    and set of coefficients

    Parameters
    ----------
    deg : int
        degree of the symmetrized harmonic.
    coeff : numpy.ndarray
        coefficeints to get symmetrized harmonic
        only for cubic symmetry
        size is nx1
    theta : numpy.ndarray
        polar angle in radians
        size is nx1
    phi : numpy.ndarray
        co-latiture in radians
        size is nx1

    Returns
    -------
    numpy.ndarray
        spherical harmonics
    '''
    if sym == 'oh':
        nfold = 4
    elif sym == 'th':
        nfold = 2

    if sym in ['th', 'oh']:
        coeff = Blmn[sym][deg]

        num_sym = coeff.shape[0]
        n = int(deg/nfold)

        Intensity = np.zeros((theta.shape[0], num_sym)).astype(complex)

        for ii in np.arange(num_sym):
            for jj in np.arange(-n, n+1):
                Intensity[:,ii] += (coeff[ii, jj+n]*
                    sph_harm_y(deg, nfold*jj, theta, phi))

    elif sym == 'axial':
        Intensity = np.zeros((theta.shape[0], 1)).astype(complex)
        Intensity[:,0] = sph_harm_y(deg, 0, theta, phi)

    return np.real(Intensity)

def get_num_sym_harm(ell,
                     sym='oh'):
    if ell < 0:
        msg = f'the degree is negative'
        raise ValueError(msg)
    if ell > 16:
        msg = f'maximum degree available is 16'
        raise ValueError(msg)
    elif sym in ['th', 'oh']:
        return Blmn[sym][ell].shape[0]
    elif sym == 'axial':
        return 1

def get_total_sym_harm(ell_max,
                       sym='oh'):
    if ell_max < 0:
        msg = f'the degree is negative'
        raise ValueError(msg)
    if ell_max > 16:
        msg = f'maximum degree available is 16'
        raise ValueError(msg)
    elif sym in ['th', 'oh']:
        num = 0
        for ell in np.arange(2, ell_max+1, 2):
            num += Blmn[sym][ell].shape[0]
    elif sym == 'axial':
        num = int(ell_max/2)

    return num

def get_parameters(ell_max,
                   csym='oh',
                   ssym='axial'
                   ):
    '''
    make lmfit parameter class for a given
    symmetry and maximum degree
    '''
    params = Parameters()

    if ell_max < 0:
        msg = f'the degree is negative'
        raise ValueError(msg)
    if ell_max > 16:
        msg = f'maximum degree available is 16'
        raise ValueError(msg)

    for ell in np.arange(2, ell_max+1, 2):
        nc = get_num_sym_harm(ell, sym=csym)
        ns = get_num_sym_harm(ell, sym=ssym)
        
        for ii in np.arange(ns):
            for jj in np.arange(nc):

                pname = f'c_{ell}{ii}{jj}'
                params.add(name=pname, 
                           value=0.,
                           vary=True)

    return params


