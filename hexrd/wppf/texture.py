import copy
from warnings import warn
from hexrd.rotations import rotMatOfExpMap
from hexrd.transforms.xfcapi import anglesToGVec
from hexrd import constants
from hexrd.wppf.phase import Material_Rietveld

# 3rd party imports
import h5py
from lmfit import Parameters, Minimizer
import numpy as np
from scipy.special import sph_harm_y
from matplotlib import pyplot as plt

"""
===============================================================================

>> @AUTHOR: Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
>> @DATE: 06/14/2021 SS 1.0 original
   @DATE: 09/09/2025 SS 1.0 rewrite without FEM

>> @DETAILS: this module deals with the texture computations for the wppf
    package.  two texture models employed are the March-Dollase model and the
    spherical harmonics model. the spherical harmonics model uses the axis
    distribution function for computing the scale factors for each reflection.
    maybe it makes sense to use symmetrized spherical harmonics.
===============================================================================
"""

'''
some constants that are used in the module
'''
bvec_ref = constants.beam_vec
eta_ref  = constants.lab_x

SYMLIST = [
'ci' ,
'c2h',
'd2h',
'c4h',
'd4h',
's6' ,
'd3d',
'c6h',
'd6h',
'th',
'oh', # end of crystal symmtries
'monoclinic',
'orthorhombic',
'triclinic',
'axial'] # end of sample symmetries

'''

=================================================================
utility functions for texture computation
=================================================================

quick overview of different laue group symmetries

laue_1 = 'ci'  # triclinic
laue_2 = 'c2h' # monoclinic
laue_3 = 'd2h' # Orthorhombic
laue_4 = 'c4h' # cyclic tetragonal
laue_5 = 'd4h' # dihedral tetragonal
laue_6 = 's6'  # cyclic trigonal + inversion
laue_7 = 'd3d' # dihedral trigonal
laue_8 = 'c6h' # cyclic
laue_9 = 'd6h' # dihedral hexagonal
laue_10 = 'th' # cubic low
laue_11 = 'oh' # cubic high

'''
def check_symmetry(sym, 
                   symtype='either'):
    '''
    helper functio to check crystal and sample
    symmetry
    '''
    if symtype == 'crystal':
        return sym in SYMLIST[0:11]
    elif symtype == 'sample':
        return sym in SYMLIST[11:]
    elif symtype == 'either':
        return sym in SYMLIST
    return False

def check_degrees(ell,
                  sym):

    if ell < 0:
        msg = f'the degree is negative'
        raise ValueError(msg)
    if np.mod(ell, 2) != 0:
        msg = (f'only even degrees allowed due to'
               f'inversion symmetry')
        raise ValueError(msg)
    if sym in ['td', 'oh']:
        if ell > 16:
            msg = f'maximum degree available is 16'
            raise ValueError(msg)
    return True

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
    if not check_symmetry(sym):
        msg = (f'unknown crystal/sample symmetry. '
               f'It should be one of {symlist[0:11]} '
               f'for crystal symmetries and {symlist[11:]} '
               f'for sample symmetries')
        raise ValueError(msg)

    if check_degrees(deg, sym=sym):
        pass

    # cubic and tetragonal
    if sym in ['oh', 'c4h', 'd4h']:
        nfold = 4

    # hexagonal
    elif sym in ['c6h', 'd6h']:
        nfold = 6

    # cubic low, orthorhombic and monoclinic
    elif sym in ['th', 'c2h', 'd2h', 
            'orthorhombic', 'monoclinic']:
        nfold = 2

    # trigonal
    elif sym in ['s6', 'd3d']:
        nfold = 3

    # triclinic
    elif sym in ['ci', 'triclinic']:
        nfold = 1

    # cylindrical symmetry
    elif sym == 'axial':
        nfold = np.inf

    n = int(deg/nfold)

    if sym in ['th', 'oh']:
        coeff = Blmn[sym][deg]

        num_sym = coeff.shape[0]

        Intensity = np.zeros((theta.shape[0], num_sym)).astype(complex)

        for ii in np.arange(num_sym):
            for jj in np.arange(-n, n+1):
                Intensity[:,ii] += (coeff[ii, jj+n]*
                    sph_harm_y(deg, nfold*jj, theta, phi))
        Intensity = np.real(Intensity)

    elif sym == 'axial':
        Intensity = np.zeros((theta.shape[0], 1))
        Intensity[:,0] = np.real(sph_harm_y(deg, 0,
                                            theta, phi))

    elif sym in ['d2h', 'd3d', 'd4h', 'd6h', 'orthorhombic']:
        '''this is the dihedral symmetry with the 
        following selection rules:
        1. l = 2l'
        2. m = [0, nfold*m'] for all allowed m's
        '''
        num_sym = n+1
        Intensity = np.zeros((theta.shape[0], 
                            num_sym))
        for ii in np.arange(0, n+1):
            if ii == 0:
                Intensity[:,ii] = np.real(sph_harm_y(
                                             deg, nfold*ii,
                                             theta, phi))
            else:
                '''we need special attention for laue 
                group d3d and d6h. for this cases the 
                real/imaginary component flips for
                m = (2k+1)*nfold*m'
                '''
                Intensity[:,ii] = np.real((
                    sph_harm_y(deg, -nfold*ii,
                               theta, phi) + 
                    sph_harm_y(deg, nfold*ii,
                               theta, phi)
                    )/np.sqrt(2.))
                if sym in ['d3d',] and np.mod(ii, 2) != 0:
                    Intensity[:,ii] = np.imag((
                        sph_harm_y(deg, -nfold*ii,
                                   theta, phi) + 
                        sph_harm_y(deg, nfold*ii,
                                   theta, phi)
                        )/np.sqrt(2.))
    elif sym in ['ci', 'c2h', 's6', 'c4h', 'c6h', 'monoclinic', 'triclinic']:
        '''this is the cyclic symmetry with the 
        following selection rules:
        1. l = 2l'
        2. m = 2m' for all allowed m's
        '''
        num_sym = 2*n+1
        Intensity = np.zeros((theta.shape[0], 
                            num_sym))
        for ii in np.arange(-n, n+1):
            if ii < 0:
                Intensity[:,ii+n] = np.imag(sph_harm_y(
                                    deg, nfold*ii,
                                    theta, phi))
            else:
                Intensity[:,ii+n] = np.real(sph_harm_y(
                                    deg, nfold*ii,
                                    theta, phi))

    return Intensity

def get_num_sym_harm(ell,
                     sym='oh'):

    if check_degrees(ell, sym=sym):
        pass

    # crystal symmetries
    if sym in ['th', 'oh']:
        return Blmn[sym][ell].shape[0]
    elif sym == 'ci':
        return 2*ell+1
    elif sym == 'c2h':
        return 2*int(ell/2)+1
    elif sym == 'd2h':
        return int(ell/2) + 1
    elif sym == 'c4h':
        return 2*int(ell/4)+1
    elif sym == 'd4h':
        return int(ell/4)+1
    elif sym == 's6':
        return 2*int(ell/3)+1
    elif sym == 'd3h':
        return int(ell/3)+1
    elif sym == 'c6h':
        return 2*int(ell/6)+1
    elif sym == 'd6h':
        return int(ell/6)+1

    # sample symmetries
    elif sym == 'axial':
        return 1
    elif sym == 'monoclinic':
        return 2*int(ell/2)+1
    elif sym == 'orthorhombic':
        return int(ell/2) + 1
    elif sym == 'triclinic':
        return 2*ell + 1

def get_total_sym_harm(ell_max,
                       sym='oh'):
    num = 0
    for ell in np.arange(2, ell_max+1, 2):
        num += get_num_sym_harm(ell, sym=sym)

    return num

def get_parameters(ell_max,
                   csym='oh',
                   ssym='axial'
                   params=None):
    '''
    make lmfit parameter class for a given
    symmetry and maximum degree
    '''
    if params is None:
        params = Parameters()

    else:
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

class harmonic_model:
    """
    this class brings all the elements together to compute the
    texture model given the sample and crystal symmetry. the model
    will be part of the Rietveld class and give the modification in
    the integrated intensity of each hkl reflection at a particular
    azimuthal angle.

    Parameters
    ----------
    material : material_rietveld class
        rietveld material class
    ssym : str
        string encoding sample symmetry. four allowed
        options -- 'monoclinic', 'orthorhombic', 'axial'
        and 'triclinic'
    ell_max: int
        maximum degree of spherical harmonic to use for
        texture computation
    params: lmfit.Parameters
        Parameter class for the refinement
    """
    
    def __init__(self,
                material=None,
                ssym='axial',
                ell_max=10):

        self.material = material
        self.ssym = ssym
        self.ell_max = ell_max

        @property
        def csym(self):
            return self._csym

        @property
        def ssym(self):
            return self._ssym

        @material.setter
        def material(self, mat):
            if isinstance(mat, Material_Rietveld):
                self._material = mat
                self._csym = mat.sg.laueGroup

        @ssym.setter
        def ssym(self, val):
            if check_symmetry(val, 
                    symtype='sample'):
                self._ssym = val
            else:
                msg = f'unknown sample symmetry'
                raise ValueError(msg)

        @property
        def ell_max(self):
            return self._ell_max
        
        @ell_max.setter
        def ell_max(self, val):
            if check_degrees(val, self.csym):
                self._ell_max = val

        def J(self, params):
            '''this is the texture index for the harmonic
            model. A value of 1 is a random texture and any
            value > 1 is preferred orientation.
            '''
            J = 1.
            for ell in np.arange(2, self.ell_max+1, 2):
                nc = get_num_sym_harm(ell, sym=self.csym)
                ns = get_num_sym_harm(ell, sym=self.ssym)
                for ii in np.arange(ns):
                    for jj in np.arange(nc):
                        pname = f'c_{ell}{ii}{jj}'
                        pre = 1/(2*ell+1)
                        J += pre*params[pname].value**2
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
        self.hkls_c = self.convert_hkls_to_cartesian(self.hkls)

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

    def convert_hkls_to_cartesian(self,
                                  hkls):
        """
        this routine converts hkls in the crystallographic frame to
        the cartesian frame and normalizes them
        """
        hkls_c = np.atleast_2d(np.zeros(hkls.shape))

        for ii, g in enumerate(hkls):
            v = self.material.TransSpace(g, "r", "c")
            v = v/np.linalg.norm(v)
            hkls_c[ii,:] = v
        return hkls_c

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

    def stereographic_radius(self,
                             new=False):
        sr = {}
        angs = self.angs
        if new:
            angs = self.angs_new
        for h, angs in angs.items():
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

    def plot_new_pf(self,
                filled=False,
                grid=False,
                cmap='jet',
                colorbar=True,
                colorbar_label='m.r.d.',
                show=True):

        '''first get the layout of the subplot
        '''
        n = self.num_pfs_new
        nrows = int(n/3)+1
        if nrows == 1:
            ncols = np.min((3, n))
        else:
            ncols = 3

        self.fig_new, self.ax_new = plt.subplots(
            nrows=nrows, ncols=ncols,
            subplot_kw={'projection': 'polar'},
            figsize=(12,4*nrows))
        self.ax_new = np.atleast_2d(self.ax_new)

        [ax.set_axis_off() for ax in self.ax_new.flatten()]
        [ax.set_yticklabels([]) for ax in self.ax_new.flatten()]
        [ax.set_xticklabels([]) for ax in self.ax_new.flatten()]
        [ax.grid(False) for ax in self.ax_new.flatten()]

        for ii, h in enumerate(self.angs_new):
            nr = int(ii/3)
            nc = int(np.mod(ii, 3))
            rho = self.angs_new[h][:,1]
            r = self.stereo_radius_new[h]
            I = self.intensities_new[h]

            # since there is a discontinuity at +/- pi azimuth
            # we will add some extra points there for the plots

            # first get the points which have rho values of +/- pi
            mask = np.isclose(np.abs(rho), np.pi)
            if np.any(mask):
                # add these same points at -/+ pi
                rho = np.concatenate((rho, -rho[mask]))

                # next add points at r and intensity
                r = np.concatenate((r, r[mask]))
                I = np.concatenate((I, I[mask]))

            if filled:
                pf = self.ax_new[nr][nc].tricontourf(rho, r, 
                                    I, levels=20, cmap=cmap)
            else:
                pf = self.ax_new[nr][nc].tricontour(rho, r,
                                    I, levels=20, cmap=cmap)
            self.ax_new[nr][nc].set_yticklabels([])
            self.ax_new[nr][nc].grid(grid)
            self.ax_new[nr][nc].set_title(f'({h})')
            plt.colorbar(pf, label=colorbar_label)

        if show:
            self.fig_new.show()

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

    def recalculated_pf(self,
                        params,
                        new=False):

        if not new:
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
        else:
            self.intensities_new = {}

            for h, v in self.angs_new.items():
                term = np.ones_like(v[:,0])

                for ell in np.arange(2, self.ell_max+1, 2):
                    pre = 4*np.pi/(2*ell+1)
                    cmat = self.get_c_matrix(params, ell)
                    sph_s_mat = self.get_sph_s_matrix(h, ell, gridtype='new')
                    sph_c_mat = self.get_sph_c_matrix(h, ell, gridtype='new')
                    t = pre*np.dot(sph_s_mat, np.dot(
                                       cmat, sph_c_mat))
                    term += np.squeeze(t)

                self.intensities_new[h] = term

    def calc_new_pole_figure(self,
                             hkls,
                             pfgrid=None,
                             plot=False):
        '''calculate pole figure for new poles
        given by hkls. the data will be computed 
        on the same grid as the input grid. this
        is done so that the spherical harmonics 
        don't have to be recomputed. 

        by default, we will pick the pf grid with
        the most points i.e. densest gridding

        hkls has the shape nx3
        '''
        if not hasattr(self, 'res'):
            msg = (f'harmonic coefficients have not '
                    f'been computed yet')
            raise RuntimeError(msg)

        self.num_pfs_new = hkls.shape[0]
        self.hkls_c_new = self.convert_hkls_to_cartesian(
                                    hkls)
        self.calc_new_pfdata(hkls,
                             pfgrid=pfgrid)

        if plot:
            self.plot_new_pf(
                        filled=True,
                        grid=False,
                        cmap='jet',
                        colorbar=True,
                        colorbar_label='m.r.d.',
                        show=True)

    def calc_pf_rings(self,
                      eta_min=-np.pi,
                      eta_max=np.pi,
                      eta_step=np.radians(0.1)):
        '''this functin computes the  intensity variation 
        along the debye-scherrer rings for each hkl in the material.
        the intensity variation around the ring is computed between 
        eta_min and eta_max. Default values are (-pi,pi) which gives 
        the full ring. eta_step is the angular step size in azimuth. 
        Default value is 0.1 degrees for eta_step
        '''
        eta_grid = np.arange(eta_min, 
                             eta_max,
                             eta_step)

        '''initialize all the dictionaries which will store the data
        '''
        if not hasattr(self, 'intensities_rings'):
            self.intensities_rings = {}
        if not hasattr(self, 'angs_rings'):
            self.angs_rings = {}
        if not hasattr(self, 'rotated_angs_rings'):
            self.rotated_angs_rings = {}
        if not hasattr(self, 'hkl_angles_rings'):
            self.hkl_angles_rings = {}

        '''also pre-compute the spherical harmonics
        '''
        if not hasattr(self, 'sph_c_rings'):
            self.sph_c_rings = {}
        if not hasattr(self, 'sph_s_rings'):
            self.sph_s_rings = {}

        hkls   = self.material.hkls
        tth    = np.radians(self.material.getTTh(self.material.wavelength))
        hkls_c = self.convert_hkls_to_cartesian(hkls)

        for ii, (t, h, hc) in enumerate(zip(tth, hkls, hkls_c)):
            hstr = str(h).strip('[').strip(']').replace(" ","")

            angs = np.vstack((t*np.ones_like(eta_grid),
                             eta_grid,
                             np.zeros_like(eta_grid))).T

            pfgrid = anglesToGVec(angs,
                 bHat_l=self.bvec,
                 eHat_l=self.etavec)

            norm = np.linalg.norm(pfgrid, axis=1)
            v = pfgrid/np.tile(norm, [3,1]).T

            # sanitize v[:, 2] for arccos operation
            mask = np.abs(v[:,2]) > 1.
            v[mask, 2] = np.sign(v[mask, 2])

            tt = np.arccos(v[:, 2])
            rho = np.arctan2(v[:,1], v[:,0])

            vr = np.dot(self.ref_frame_rmat, v[:,0:3].T).T

            # sanitize vr[:, 2] for arccos operation
            mask = np.abs(vr[:,2]) > 1.
            vr[mask, 2] = np.sign(vr[mask, 2])
            
            tr = np.arccos(vr[:, 2])
            rhor = np.arctan2(vr[:,1], vr[:,0])

            self.angs_rings[hstr] = np.vstack((tt, rho)).T
            self.rotated_angs_rings[hstr] = np.vstack((tr, rhor)).T
            self.hkl_angles_rings[hstr] = np.array([np.arccos(hc[2]),
                                                    np.arctan2(hc[1], 
                                                               hc[0])])

            theta = np.array([self.hkl_angles_rings[hstr][0]])
            phi   = np.array([self.hkl_angles_rings[hstr][1]])

            self.sph_c_rings[hstr] = {}

            for ell in np.arange(2, self.ell_max+1, 2):
                Ylm = calc_sym_sph_harm(ell, 
                                        theta,
                                        phi,
                                        sym=self.csym)
                for jj in np.arange(Ylm.shape[1]):
                    kname = f'c_{ell}{jj}'
                    self.sph_c_rings[hstr][kname] = Ylm[:,jj]

            self.sph_s_rings[hstr] = {}
            theta_samp = self.rotated_angs_rings[hstr][:,0]
            phi_samp   = self.rotated_angs_rings[hstr][:,1]

            for ell in np.arange(2, self.ell_max+1, 2):
                Ylm = calc_sym_sph_harm(ell, 
                                        theta_samp,
                                        phi_samp,
                                        sym=self.ssym)
                for jj in np.arange(Ylm.shape[1]):
                    kname = f's_{ell}{jj}'
                    self.sph_s_rings[hstr][kname] = Ylm[:,jj]

            '''perform the series sum if coefficeints are calculated
            '''
            if hasattr(self, 'res'):
                term = np.ones_like(v[:,0])

                for ell in np.arange(2, self.ell_max+1, 2):
                    pre = 4*np.pi/(2*ell+1)
                    cmat = self.get_c_matrix(self.res.params, ell)
                    sph_s_mat = self.get_sph_s_matrix(hstr,
                                                      ell,
                                                      gridtype='rings')
                    sph_c_mat = self.get_sph_c_matrix(hstr,
                                                      ell,
                                                      gridtype='rings')
                    t = pre*np.dot(sph_s_mat, np.dot(
                                       cmat, sph_c_mat))
                    term += np.squeeze(t)

                self.intensities_rings[hstr] = term

            else:
                msg = (f'coefficeints have not been computed yet. '
                       f'consider running'
                       f'"pole_figures.calculate_harmonic_coefficients" first')
                raise RuntimeError(msg)

    def calc_new_pfdata(self,
                        hkls,
                        pfgrid=None):
        '''this routine computes the new pfdata which will
        will be used in computing pole figures for a new hkl
        pole
        '''
        if not hasattr(self, 'intensities_new'):
            self.intensities_new = {}
        if not hasattr(self, 'angs_new'):
            self.angs_new = {}
        if not hasattr(self, 'rotated_angs_new'):
            self.rotated_angs_new = {}
        if not hasattr(self, 'hkl_angles_new'):
            self.hkl_angles_new = {}

        '''also pre-compute the spherical harmonics
        '''
        if not hasattr(self, 'sph_c_new'):
            self.sph_c_new = {}
        if not hasattr(self, 'sph_s_new'):
            self.sph_s_new = {}

        '''get the densest grid and associated data
        if the user does not specify a grid of points

        otherwise use the grid data provided by the
        user to generate the angs and the values of the
        spherical harmonics
        
        '''
        if pfgrid is None:
            dg = self.densest_grid
            angs = self.angs[dg]
            rotated_angs = self.rotated_angs[dg]

        else:
            norm = np.linalg.norm(pfgrid,axis=1)
            v = pfgrid/np.tile(norm, [3,1]).T

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

            angs = np.vstack((t, rho)).T
            rotated_angs = np.vstack((tr, rhor)).T

        hkls_c = self.convert_hkls_to_cartesian(hkls)

        for ii, h in enumerate(hkls):

            hstr = str(h).strip('[').strip(']').replace(" ","")
            self.angs_new[hstr] = angs.copy()
            self.rotated_angs[hstr] = rotated_angs.copy()
            self.stereo_radius_new = self.stereographic_radius(new=True)
            self.hkl_angles_new[hstr] = np.array([np.arccos(
                                         hkls_c[ii, 2]),
                                         np.arctan2(hkls_c[ii, 1], 
                                                    hkls_c[ii, 0])])

            h = hstr
            v = self.hkl_angles_new[hstr]
            '''needs special attention for monoclininc case
                since the 2-fold axis is aligned with b*
            '''
            theta = np.array([v[0]])
            phi = np.array([v[1]])
            '''is the requested data on a new grid?
            '''
            self.sph_c_new[h] = {}

            for ell in np.arange(2, self.ell_max+1, 2):
                Ylm = calc_sym_sph_harm(ell, 
                                        theta,
                                        phi,
                                        sym=self.csym)
                for jj in np.arange(Ylm.shape[1]):
                    kname = f'c_{ell}{jj}'
                    self.sph_c_new[h][kname] = Ylm[:,jj]

            if pfgrid is None:
                self.sph_s_new[h] = self.sph_s[dg].copy()

            else:
                self.sph_s_new[h] = {}
                theta_samp = self.rotated_angs[h][:,0]
                phi_samp   = self.rotated_angs[h][:,1]

                for ell in np.arange(2, self.ell_max+1, 2):
                    Ylm = calc_sym_sph_harm(ell, 
                                            theta_samp,
                                            phi_samp,
                                            sym=self.ssym)
                    for jj in np.arange(Ylm.shape[1]):
                        kname = f's_{ell}{jj}'
                        self.sph_s_new[h][kname] = Ylm[:,jj]


        self.recalculated_pf(self.res.params, new=True)

    def calculate_harmonic_coefficients(self,
                                        ell_max,
                                        ):

        self.ell_max = ell_max
        params = get_parameters(ell_max,
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

    def get_sph_s_matrix(self,
                         h,
                         ell,
                         gridtype=None):

        if gridtype is None:
            ngrid = self.angs[h].shape[0]
            Ylm = self.sph_s[h]

        elif gridtype == 'new':
            ngrid = self.angs_new[h].shape[0]
            Ylm = self.sph_s_new[h]

        elif gridtype == 'rings':
            ngrid = self.angs_rings[h].shape[0]
            Ylm = self.sph_s_rings[h]

        ns = get_num_sym_harm(ell,
                            sym=self.ssym)
        smat = np.zeros((ngrid, ns))

        for ii in np.arange(ns):
            yname = f's_{ell}{ii}'
            smat[:,ii] = Ylm[yname]

        return smat

    def get_sph_c_matrix(self,
                         h,
                         ell,
                         gridtype=None):

        nc = get_num_sym_harm(ell,
                            sym=self.csym)
        ngrid = 1
        cmat = np.zeros((nc, ngrid))

        if gridtype is None:
            Ylm = self.sph_c[h]
        elif gridtype == 'new':
            Ylm = self.sph_c_new[h]
        elif gridtype == 'rings':
            Ylm = self.sph_c_rings[h]

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

    @property
    def densest_grid(self):
        sz = 0
        dg = ''
        for h, v in self.pfdata.items():
            if v.shape[0] > sz:
                dg = h
                sz = v.shape[0]
        return dg

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
        2:np.array([[0., 1., 0.]]),
        4:np.array([[0.4564354645876387,0.0,
        0.7637626158259737,0.0,
        0.4564354645876387,]]),
        6:np.array([[-0.39528470752104744,0.0,
                0.5863019699779289,0.0,
                0.5863019699779287,0.0,
                -0.3952847075210474,],
                 [0.0,0.6614378277661476,
                0.0,-0.35355339059327384,
                0.0,0.6614378277661477,
                0.0,]]),
        8:np.array([[0.4114253678777394,0.0,
                0.27003086243366087,0.0,
                0.7180703308172529,0.0,
                0.270030862433661,0.0,
                0.4114253678777394,]]),
        10:np.array([[-0.32179907341941766,-0.3026705127825632,
                0.08783983261480693,-0.25429409464318103,
                0.4478970205736254,0.25236027118158605,
                0.44789702057362524,-0.25429409464318115,
                0.08783983261480709,-0.3026705127825629,
                -0.3217990734194177,],
                 [0.24992195050936286,-0.3897180314346175,
                -0.06821990525403415,-0.3274286386826981,
                -0.34785462810576917,0.32493865092120033,
                -0.34785462810576934,-0.3274286386826982,
                -0.06821990525403382,-0.38971803143461725,
                0.24992195050936322,]]),
        12:np.array([[-0.10774660850851898,0.35440527769341207,
                0.042183172840390366,-0.5617029300070009,
                -0.11564076236589332,0.14294418606853107,
                -0.15291919171429025,0.1429441860685311,
                -0.1156407623658934,-0.5617029300070007,
                0.042183172840390734,0.35440527769341246,
                -0.10774660850851901,],
                 [0.3411000537498401,0.10028705753178294,
                -0.2620437799555088,-0.15894665684441117,
                0.42787445167677257,0.04044931809534165,
                0.4357742152904641,0.040449318095341594,
                0.4278744516767727,-0.15894665684441128,
                -0.2620437799555087,0.100287057531783,
                0.3411000537498399,],
                 [0.19715691408140235,0.02017705548403037,
                0.561512087862538,-0.03197895713646226,
                -0.09548336992239026,0.00813812026783474,
                0.5200389527975391,0.00813812026783466,
                -0.09548336992239025,-0.0319789571364624,
                0.5615120878625383,0.02017705548403026,
                0.19715691408140232,]]),
        14:np.array([[-0.28955062552019833,-0.2972304474122511,
                0.010117717673205564,-0.2447888749918319,
                0.12465750846560059,-0.22811657797300322,
                0.38999831765926807,0.3102101852656362,
                0.3899983176592681,-0.228116577973003,
                0.1246575084656004,-0.24478887499183172,
                0.010117717673205809,-0.297230447412251,
                -0.2895506255201985,],
                 [0.28772627934090717,-0.2991150553540337,
                -0.010053969858649543,-0.2463409739167105,
                -0.12387208985743604,-0.22956296517271021,
                -0.387541089533881,0.31217709203398175,
                -0.3875410895338812,-0.22956296517271035,
                -0.12387208985743622,-0.2463409739167103,
                -0.010053969858649592,-0.29911505535403393,
                0.28772627934090733,]]),
        16:np.array([[0.16618905573198167,-0.3654438510445056,
                -0.059213072462354854,0.17954391047429719,
                -0.002851898776857607,0.4610540410294765,
                0.19021431714221818,-0.17805079589901174,
                0.2127310570059948,-0.17805079589901207,
                0.19021431714221823,0.4610540410294766,
                -0.002851898776857436,0.1795439104742971,
                -0.05921307246235514,-0.36544385104450566,
                0.16618905573198162,],
                 [0.30796148329941286,0.1641675465982823,
                -0.26096265404956615,-0.08065612050929621,
                -0.09371337438188433,-0.20711830435427173,
                0.44600261056672585,0.07998537189520905,
                0.305305250454509,0.07998537189520934,
                0.4460026105667255,-0.20711830435427164,
                -0.09371337438188454,-0.08065612050929624,
                -0.26096265404956615,0.16416754659828256,
                0.3079614832994129,],
                 [0.21028019819195082,0.04839013590249786,
                0.4381584188768272,-0.02377425205947472,
                0.29639275890473565,-0.06105033000294009,
                -0.07659622573459653,0.02357654174909582,
                0.5707783675995487,0.02357654174909576,
                -0.07659622573459626,-0.06105033000294006,
                0.29639275890473576,-0.023774252059474876,
                0.4381584188768272,0.048390135902498174,
                0.2102801981919509,]]),
        },
    'oh': {
        2:np.array([[1.]]),
        4:np.array([[-0.4564354645876382,-0.7637626158259734,
        -0.4564354645876386,]]),
        6:np.array([[0.6614378277661479,-0.3535533905932739,
                0.6614378277661476,]]),
        8:np.array([[-0.4114253678777393,-0.2700308624336607,
                -0.7180703308172534,-0.2700308624336608,
                -0.4114253678777394,]]),
        10:np.array([[-0.4934466367636259,-0.41457809879442503,
                0.4114253678777398,-0.41457809879442514,
                -0.49344663676362605,]]),
        12:np.array([[-0.40844758180620133,-0.04107687246151441,
                -0.34173954767501197,-0.6552821453699549,
                -0.34173954767501197,-0.04107687246151456,
                -0.40844758180620117,],
                 [0.0,0.6197216133464934,
                -0.2979605473965942,0.23308639662726535,
                -0.29796054739659417,0.6197216133464934,
                0.0,]]),
        14:np.array([[-0.421682054643464,-0.3472829807952012,
                -0.32362992464387486,0.4400964619641172,
                -0.32362992464387463,-0.34728298079520126,
                -0.4216820546434642,]]),
        16:np.array([[0.40826074902286213,0.004724991415765744,
                0.0808097864409005,0.3744089876279584,
                0.6108821877615941,0.3744089876279583,
                0.0808097864409004,0.004724991415765709,
                0.4082607490228618,],
                 [0.0,-0.5133889064323339,
                -0.3001812382731627,0.3174660719799944,
                -0.3017891584619822,0.3174660719799949,
                -0.30018123827316284,-0.5133889064323339,
                0.0,]]),
    }
}