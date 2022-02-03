"""
Calculation of x-ray diffraction scattering angle for PXRDIP and TARDIS.

Calculates the nominal, sample-weighted, and pinhole-weighted twotheta
  scattering angle for PXRDIP and TARDIS diffraction experiments.

Reference:
[1] J. R. Rygg et al, "X-ray diffraction at the National Ignition Facility,"
    Rev. Sci. Instrum. 91, 043902 (2020). https://doi.org/10.1063/1.5129698

file: pinhole_2theta.py
revisions:
    2022-02-02, v1.5: refactor for standalone script, additional comments,
                        prepare for commit to https://github.com/HEXRD/hexrd
    2021-11-18, v1.4: option to increase twotheta precision for output table
    2020-12-21, v1.3: additional metadata in report file and filename
    2020-09-21, v1.2: added ReportFileWriter to automate file writing
    2020-08-22, v1.1: minor updates
    2020-08-09, v1.0: migrated from earlier scripts
"""
# written in python 3.7; not tested with earlier versions.
__author__ = "J. R. Rygg"
__version__ = "1.5"

# header of text file report
HEADER = (
'This file contains tables of the evaluated pinhole twotheta correction values\n'
'  described in section III.C.2 of ref (1) for parameters defined below.\n'
'\n'
'  (1) J. R. Rygg et al, "X-ray Diffraction at the National Ignition Facility,"\n'
'      Rev. Sci. Instrum. 91, 043902 (2020). https://doi.org/10.1063/1.5129698\n'
'\n[definitions]\n'
'twotheta_n: nominal twotheta, Ref (1) Eq. 5.\n'
'twotheta_p: pinhole twotheta, Ref (1) section III.C.2 and Eq 6.\n'
'twotheta_JHE: J. Eggert approximation of twotheta_p, "In my PXRDIP and\n'
'    TARDIS Igor analysis package, I approximated this correction by\n'
'    assuming that the pinhole diffraction arises from the center of the\n'
'    pinhole wall opposite the image-plate pixel."\n'
)

# imports
import datetime
import io
import os.path as osp

import numpy as np
import numpy.dual
from numpy import pi, sin, cos, tan, radians

import matplotlib as mpl
import matplotlib.pyplot as plt

DPI_SCREEN = 150

# reset mpl rc to defaults and adjust settings for image generation
mpl.rcdefaults()
mpl.rcParams.update({'backend': 'Qt5Agg' })
mpl.rc('font',**{'family':'sans-serif', 'sans-serif':['Arial'], 'size':12})
mpl.rc('mathtext', default='regular') # same font for mathtext as normal text
mpl.rc('image', cmap='viridis', origin='lower') #
mpl.rc('lines', linewidth=0.75) #
mpl.rc('xtick', top=True, bottom=True)
mpl.rc('ytick', left=True, right=True)

# linear attenuation coefficient [um^-1] for pinhole, xsf pairs
MU_P_LOOKUP = { # (pinhole_mat, xs_mat): linear attenuation coefficient [um^-1]
    ('Ta', 'Fe'): 1/2.33, ('Ta', 'Cu'): 1/4.11, ('Ta', 'Ge'): 1/2.69, # 1/mic
    ('W', 'Fe'):  1/1.94, ('W', 'Cu'):  1/3.42, ('W', 'Ge'):  1/2.25, # 1/mic
    ('Pt', 'Fe'): 1/1.50, ('Pt', 'Cu'): 1/2.63, ('Pt', 'Ge'): 1/4.38, # 1/mic
}

# baseline parameters
OMEGA_BASELINE=dict(mat_x='Cu', r_x=24.24, alpha=45, phi_x=0,
                    mat_p='Ta', d_p=0.3, h_p=0.075, centered_ph=True)
EP_BASELINE   =dict(mat_x='Cu', r_x=24.24, alpha=22.5, phi_x=0,
                    mat_p='Ta', d_p=0.3, h_p=0.075, centered_ph=True)
NIF_BASELINE  =dict(mat_x='Ge', r_x=32, alpha=30, phi_x=0,
                    mat_p='Ta', d_p=0.4, h_p=0.1, centered_ph=True)

# output path for image and report text file
OUTPUT_PATH = osp.dirname(__file__)

# ============================ Helper Functions ===============================
def zenith(vv, v0=(0,0,1), uom='radians'):
    """Return zenith angle between vector array vv and reference vector(s) v0.

    Sometimes known as the polar angle, inclination angle, or colatitude.
    expects the 3-vector to be contained along last axis.
    """
    try:
        np.broadcast(vv, v0)
    except ValueError: # if vv and v0 as given cannot be broadcast to eachother
        shapev, shape0 = np.shape(vv), np.shape(v0)
        shapev_new = shapev[:-1] + (1,) * (len(shape0) - 1) + (shapev[-1],)
        shape0_new = (1,) * (len(shapev) - 1) + shape0
        vv = np.reshape(vv, shapev_new)
        v0 = np.reshape(v0, shape0_new)

    # normalization factor along last axis
    norm = np.dual.norm(vv, axis=-1) * np.dual.norm(v0, axis=-1)

    # np.clip eliminates rounding errors for (anti)parallel vectors
    out = np.arccos(np.clip(np.sum(vv * v0, -1) / norm, -1, 1))
    if 'deg' in uom:
        return out * 180 / pi
    return out


def azimuth(vv, v0=None, v1=None, uom='radians'):
    """Return azimuthal angle btwn vv and v0, with v1 defining phi=0."""
    if v0 is None: v0 = np.array((0, 0, 1), dtype=float) # z unit vector
    if v1 is None: v1 = np.array((1, 0, 0), dtype=float) # x unit vector

    with np.errstate(divide='ignore', invalid='ignore'):
        n0 = np.cross(v0, v1)
        n0 /= np.dual.norm(n0, axis=-1)[...,np.newaxis]
        nn = np.cross(v0, vv)
        nn /= np.dual.norm(nn, axis=-1)[...,np.newaxis]

    azi = np.arccos(np.sum(nn * n0, -1))
    if len(np.shape(azi)) > 0:
        azi[np.dot(vv, n0) < 0] *= -1
        azi[np.isnan(azi)] = 0 # arbitrary angle where vv is (anti)parallel to v0
    else:
        if np.isnan(azi):
            return 0
        elif np.dot(vv, v0) < 1 and azi > 0:
            azi *= -1

    if 'deg' in uom:
        return azi * 180 / pi
    return azi


def calc_twotheta_nominal(alpha, beta, phi_x=0, phi_d=0): # Eq. (5)
    """Nominal twotheta [radians] given alpha, beta, phi_x, phi_d.
    
    optional: alpha, phi_x taken from first argument if is XraySource instance
    """
    if isinstance(alpha, XraySource):
        alpha, phi_x = alpha.alpha, alpha.phi
    
    # Rygg2020, Eq. (5)
    cos_2qn = cos(alpha)*cos(beta) - sin(alpha)*sin(beta) * cos(phi_d - phi_x)
    return np.arccos(np.clip(cos_2qn, -1, 1))


def calc_twotheta_sample_minus_nominal(qq_n, beta, r_x, z_s): # Eq. (8)
    """Return (sample twotheta - nominal twotheta) [radians].
    
    See Rygg2020, Eq. (8)
    """
    return np.arctan(sin(qq_n) / (r_x/z_s * cos(beta) - cos(qq_n)))


def calc_twotheta_JHE(X, P, D):
    """J. Eggert approximation to pinhole twotheta.
    
    Quote from J. Eggert describing approximation: "In my PXRDIP and TARDIS
      Igor analysis package, I approximated this correction by assuming that
      the pinhole diffraction arises from the center of the pinhole wall
      opposite the image-plate pixel."
    """
    v_x = X.v_x  if isinstance(X, XraySource) else X
    d_p = P.diam if isinstance(P, Pinhole) else P
    v_d = D.v_d  if isinstance(D, Detector) else D
    
    b = d_p / 2 # impact parameter
    dw = np.zeros_like(v_d) # offset vector of sample interaction region
    dw[...,0] = b * cos(D.phi + pi)
    dw[...,1] = b * sin(D.phi + pi)
        
    return zenith(v_d - dw, v0=(v_x + dw))


def calc_twotheta_pinhole(X, P, D, Np=120, showdetailview=False):
    """Return pinhole twotheta [rad] and effective scattering volume [mm3].
    
    X: XraySource instance, or position tuple
    P: Pinhole instance, or (material, diameter, thickness) tuple
    D: Detector insteance, or tuple of (r_d, beta, phi_d)
    Np: number of pinhole phi elements for integration
    showdetailview: show detailed report, and plots if one detector element
    """
    #------ Determine geometric parameters ------
    if isinstance(X, XraySource):
        r_x, alpha, phi_x = X.r, X.alpha, X.phi
        mat_x = X.mat
    else: # assume tuple of r_x, alpha, phi_x
        r_x, alpha, phi_x = X
        mat_x = "Cu"
    if isinstance(P, Pinhole):
        mat_p, d_p, h_p = P.mat, P.d, P.h
    else: # assume tuple of (material, diameter, thickness)
        mat_p, d_p, h_p = P

    mu_p = MU_P_LOOKUP.get((mat_p, mat_x), 1/3.0) # attenuation coefficient
    mu_p = 1000 * mu_p # convert to mm**-1

    if isinstance(D, Detector):
        r_d, beta, phi_d = D.r_d, D.beta, D.phi_d
    elif len(D) == 3: # assume tuple of r_d, beta, phi_d
        r_d, beta, phi_d = D
    else: # handle other inputs
        print("unexpected form for 'D' in calc_twotheta_pinhole")

    # reshape so D grids are on axes indices 2 and 3 [1 x 1 x Nu x Nv]
    r_d   = np.atleast_2d(r_d)[None, None, :, :]
    beta  = np.atleast_2d(beta)[None, None, :, :]
    phi_d = np.atleast_2d(phi_d)[None, None, :, :]

    #------ define pinhole grid ------
    Nz = max(3, int(Np * h_p / (pi * d_p))) # approximately square pinhole surface elements
    dphi = 2 * pi / Np # [rad] phi interval
    dl = d_p * dphi # [mm] azimuthal distance increment
    dz = h_p / Nz # [mm] axial distance increment
    dA = dz * dl # [mm^2] area element
    dV_s = dA * mu_p**-1 # [mm^3] volume of surface element
    dV_e = dl * mu_p**-2 # [mm^3] volume of edge element
    
    phi_vec = np.arange(dphi/2, 2*pi, dphi)
    z_vec = np.arange(-h_p/2 - dz/2, h_p/2 + dz/1.999, dz) # includes elements for X and D edges
    z_vec[0] = -h_p/2 # X-side edge (negative z)
    z_vec[-1] = h_p/2 # D-side edge (positive z)
    phi_i, z_i = np.meshgrid(phi_vec, z_vec) # [Nz x Np]
    phi_i = phi_i[:, :, None, None]    # [Nz x Np x 1 x 1]
    z_i   = z_i[:, :, None, None]      #   axes 0,1 => P; axes 2,3 => D
    
    #------ calculate twotheta_i [a.k.a. qq_i], for each grid element ------
    bx, bd = (d_p / (2 * r_x),  d_p / (2 * r_d))
    sin_a,    cos_a,   tan_a  = sin(alpha), cos(alpha), tan(alpha)
    sin_b,    cos_b,   tan_b  = sin(beta),  cos(beta),  tan(beta)
    sin_phii, cos_phii        = sin(phi_i), cos(phi_i)
    cos_dphi_x = cos(phi_i - phi_x + pi) # [Nz x Np x Nu x Nv]
    cos_dphi_d = cos(phi_i - phi_d + pi)

    arctan2 = np.arctan2
    alpha_i = arctan2(np.sqrt(sin_a**2 + 2*bx*sin_a*cos_dphi_x + bx**2), cos_a + z_i/r_x)
    beta_i  = arctan2(np.sqrt(sin_b**2 + 2*bd*sin_b*cos_dphi_d + bd**2), cos_b - z_i/r_d)
    phi_xi = arctan2(sin_a*sin(phi_x) - bx*sin_phii, sin_a*cos(phi_x) - bx*cos_phii)
    phi_di = arctan2(sin_b*sin(phi_d) - bd*sin_phii, sin_b*cos(phi_d) - bd*cos_phii)

    arg = cos(alpha_i) * cos(beta_i) - sin(alpha_i) * sin(beta_i) * cos(phi_di - phi_xi)
    qq_i = np.arccos(np.clip(arg, -1, 1)) # scattering angle for each P to each D
    
    #------ calculate effective volumes: 1 (surface), 2 (Xedge), 3 (Dedge) -----
    sec_psi_x = 1 / (sin_a * cos_dphi_x)
    sec_psi_d = 1 / (sin_b * cos_dphi_d)
    sec_alpha = 1 / cos_a
    sec_beta  = 1 / cos_b
    tan_eta_x = np.where(cos_dphi_x[0]  <= 0, 0, cos_a * cos_dphi_x[0])
    tan_eta_d = np.where(cos_dphi_d[-1] <= 0, 0, cos_b * cos_dphi_d[-1])

    V_i = dV_s / (sec_psi_x + sec_psi_d)  # [mm^3]
    V_i[0]  = dV_e / (sec_psi_d[0]  * (sec_alpha + sec_psi_d[0] * tan_eta_x)) # X-side edge (z = -h_p / 2)
    V_i[-1] = dV_e / (sec_psi_x[-1] * (sec_beta + sec_psi_x[-1] * tan_eta_d)) # D-side edge (z = +h_p / 2)

    #------ visibility of each grid element ------
    is_seen = np.logical_and(z_i >  h_p/2 - d_p/tan_b * cos_dphi_d, # pinhole surface
                             z_i < -h_p/2 + d_p/tan_a * cos_dphi_x)
    is_seen[0]  = np.where(h_p/d_p * tan_b < cos_dphi_d[0], 1, 0)  # X-side edge
    is_seen[-1] = np.where(h_p/d_p * tan_a < cos_dphi_x[-1], 1, 0) # D-side edge
    
    #------ weighted sum over elements to obtain average ------
    V_i *= is_seen # zero weight to elements with no view of both X and D
    V_p = np.nansum(V_i, axis=(0,1)) # [Nu x Nv] <= detector
    qq_p = np.nansum(V_i * qq_i, axis=(0,1)) / V_p # [Nu x Nv] <= detector

    if not showdetailview:
        return qq_p, V_p

    #====== Print and plot detailed view ======
    print("=== Pinhole.calc_twotheta_pinhole: detailed view ===")
    print("{} PH with (d={:.3f} mm, h={:.3f} mm, mu_p=1/({:.2f} um))".format(mat_p, d_p, h_p, 1000/mu_p))
    print("{} XS at (r={:.1f} mm, alpha={:.1f} deg, phi={:.0f})".format(mat_x, float(r_x), np.degrees(float(alpha)), np.degrees(float(phi_x))))
    print("number of detector elements =", np.size(beta))
    print("number of pinhole elements =", np.size(phi_i))
    print("number of scattering calculations =", np.size(qq_i))
    if np.size(beta) == 1:
        print("detector pixel at (r={:.1f} mm, beta={:.0f} deg, phi={:.0f})".format(float(r_d), np.degrees(float(beta)), np.degrees(float(phi_d))))
        qq_n = np.squeeze(calc_twotheta_nominal(alpha, beta, phi_x, phi_d))
        qq_p = np.squeeze(qq_p)
#        qq_JHE = np.squeeze(calc_twotheta_JHE(X, d_p, D))

        print("nominal twotheta = {:7.4f} deg".format(np.degrees(qq_n)))
        print("pinhole twotheta = {:7.4f} deg".format(np.degrees(qq_p)))
        print("twotheta correc. = {:7.4f} deg".format(np.degrees(qq_p - qq_n)))
        print("max 2q deviation = {:7.4f} deg".format(np.degrees(np.max(np.squeeze(qq_i)) - qq_n)))

#            qq_JHE = self.calc_twotheta_pinhole_JHE(X, (r_d, beta, phi_d))
#            print("JHE approx = {:7.4f} deg".format(np.degrees(qq_JHE - qq_n)))

        print("signal volume (total) = {:.2g} um3".format(1e9*np.sum(V_i)))
        print("signal volume (total) = {:.1f}% of surface ref volume (pi d h / mu)".format(100*np.sum(V_i)/(pi*d_p*h_p/mu_p)))
        print("signal contribution from surface = {:.1f}%".format(100*np.sum(V_i[1:-1,:])/np.sum(V_i)))
        
        extent = (0,360,-h_p/2,h_p/2)
        fig = plt.figure()
        
        ax1 = plt.subplot(311)
        im = ax1.imshow(np.degrees(np.squeeze(qq_i[1:-1,:] - qq_n)), aspect='auto',
                       vmin=-0.6, vmax=0.6, cmap='coolwarm',extent=extent)
        plt.colorbar(im, label=r"$2\theta_p - 2\theta_n$ [deg]")
        ax1.plot(180/pi*(np.squeeze(phi_d)+pi), 0, 'wD')
#            cb.set_label("2theta_p - 2theta_n [deg]")
        
        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        im = ax2.imshow(1e9*np.squeeze(V_i), aspect='auto',extent=extent)
#            im = ax.imshow(np.squeeze(V_i[1:-1,:]), aspect='auto',extent=extent)
        plt.colorbar(im, label=r"$V_i$ [um$^3$]")
        ax2.plot(180/pi*(np.squeeze(phi_d)+pi), 0, 'wD')

        ax3 = plt.subplot(313, sharex=ax1, sharey=ax1, xlabel="phi_p")
        im = ax3.imshow(np.squeeze(is_seen), aspect='auto', extent=extent)
#            im = ax.imshow(np.squeeze(is_seen[1:-1,:]), aspect='auto', extent=extent)
#            print(np.shape(phi_vec), np.shape(h_p/2 - d_p/tan_b * cos_dphi_d))
        ax3.plot(180/pi*phi_vec, np.squeeze(h_p/2 - d_p/tan_b * cos(phi_vec - phi_d + pi)))
        ax3.plot(180/pi*phi_vec, np.squeeze(-h_p/2 + d_p/tan_a * cos(phi_vec - phi_x + pi)))
        ax3.plot(180/pi*(np.squeeze(phi_d)+pi), 0, 'wD')
        ax3.axis(extent)
        fig.tight_layout()
    
    print("=== /end detail view ===")
    return qq_p, V_p


# ============================ Utility Classes ===============================
class XraySource():
    """X-ray source (XS), e.g. due to He-alpha emission of a mid-Z metal.

    Parameters
    ----------
    mat   : backlighter foil material (e.g. Fe, Cu, Ge)
    r     : distance of XS from origin (i.e. center of pinhole) [mm]
    alpha : zenith angle of XS from (negative) pinhole axis
    phi   : azimuthal angle of XS around pinhole axis
    indegrees : whether the input angles are in degrees, radians if not True

    Attributes
    ----------
    alpha, phi : stored in radians. degree-values can be retrieved as alpha_deg
    v_x        : emission vector toward origin
    """

    def __init__(self, mat='Cu', r=24.14, alpha=45, phi=0, indegrees=True):
        self.mat = mat
        self.r = r
        if indegrees:
            self.alpha_deg, self.phi_deg = alpha, phi
            self.alpha = radians(alpha)
            self.phi = radians(phi)
        else:
            self.alpha, self.phi = alpha, phi
            self.alpha_deg = np.degrees(alpha)
            self.phi_deg = np.degrees(phi)
            
        sin_alpha, cos_alpha = sin(self.alpha), cos(self.alpha)
        sin_phi = np.around(sin(self.phi), 8) # np.around to handle rounding error
        cos_phi = np.around(cos(self.phi), 8)

        self.v_x = r * np.array((sin_alpha * -cos_phi, # assumes XS position at -z
                                 sin_alpha * -sin_phi,
                                 cos_alpha))

        self.M = self.get_M()

    def get_M(self):
        if self.phi != 0:
            print("Non phi_x=0 XraySource Matrix not implemented")
            return np.matrix(np.eye(3))

        cosa, sina = cos(self.alpha), sin(self.alpha)
        return np.array(((cosa, 0, -sina), # transformation matrix
                         (0,    1, 0),
                         (sina, 0, cosa)))

    def transform_to_box_coords(self, twotheta, phi=0):
        """Return (beta, phi_d) for given (twotheta_n, phi)."""
        input_shape = np.shape(twotheta)
        twotheta = np.ravel(twotheta)
        phi = np.ravel(phi)
        cosq, sinq = np.cos(twotheta), np.sin(twotheta)
        cosp, sinp = np.cos(phi), np.sin(phi)
        
        # xs direct image coords, and transformed to box coords
        v_1 = np.squeeze(np.transpose(np.dstack((sinq * cosp, sinq * sinp, cosq)))) # words for 1d array of vectors
        v_2 = np.transpose(np.dot(self.M, v_1))
        
        beta = np.reshape(zenith(v_2), input_shape)
        phi_d = np.reshape(azimuth(v_2), input_shape)

        return beta, phi_d


class Pinhole():
    """Cylindrical aperture in an opaque material.
    
    The diagnostic coordinate system (DCS) uses the pinhole center as the
      origin, and the pinhole axis as the z-axis
    
    Parameters
    ----------
    mat    : pinhole substrate material (e.g. Ta, W, Pt)
    diam   : pinhole aperture diameter [mm]
    thick  : pinhole substrate thickness [mm]
    pos    : pinhole center position; optional, default (0,0,0)
    normal : pinhole normal unit vector; optional, default (0,0,1)
    
    Attributes
    ----------
    all __init__ parameters are saved as attributes
    d, h : aliases for diam and thick, respectively
    critical_angle : critical angle at which pinhole is effectively closed [rad]
    """
    def __init__(self, mat='Ta', diam=0.4, thick=0.1,
                 pos=(0,0,0), normal=(0,0,1), **kws):
        self.mat = mat
        self.diam = self.d = diam
        self.thick = self.h = thick
        self.pos = np.array(pos)
        self.normal = np.array(normal)

    @property
    def critical_angle(self):
        return np.arctan(self.diam / self.thick)


class Detector():
    """A detector single element or array of elements."""
    PXRDIP_LWH = (75, 50, 25) # [mm] length, width, and offset of pxrdip box
    
    def __init__(self, v_d, id=None):
        """v_d contains the 3-vector(s) of the detector elements in xyz."""
        self.id = id
        self.v_d = v_d
        self.r_d = np.sqrt(np.sum(self.v_d**2,axis=2)) # nominal beta of each pixel
        self.beta = zenith(self.v_d) # nominal beta of each pixel
        self.phi_d = azimuth(self.v_d) # azimuthal angle of each pixel

    @classmethod
    def pxrdip_from_id(cls, plate_id='D', centered_ph=True, shape=None):
        """Generate an element array corresponding to a given PXRDIP plate."""
        plate_id = plate_id.upper()[0]
        L, W, _ = cls.PXRDIP_LWH # width and length of PXRDIP image plates
        if plate_id == "B":
            L = W # square back plate

        if shape is None: # default to 1 element per mm
            shape = (int(W) + 1, int(L) + 1)
        N1, N2 = shape

        # p,q in image plate coordinate system (IPCS)
        p = np.linspace(0, W, N1).reshape(N1, 1) - W / 2
        q = np.linspace(0, L, N2).reshape(1, N2)

        # convert to diagnostic coordinate system (DCS)
        v_d = np.empty((N1, N2, 3))  # placeholder for detector pixel positions
        if plate_id == "B":
            q -= L / 2

            v_d[..., 0] = p
            v_d[..., 1] = q
            v_d[..., 2] = cls.PXRDIP_LWH[0]
        else:
            offset_sign = (-1, 1)[int(plate_id in ('R', 'U'))]
            ordering = (1, -1)[int(plate_id in ('L', 'U'))]
            i_offset = int(plate_id in ('L', 'R'))
            i_p = 1 - i_offset # v_d index corresponding to "p" direction

            v_d[..., i_offset] = offset_sign * W / 2
            v_d[..., i_p] = p[::ordering]
            v_d[..., 2] = q

        obj = cls(v_d, plate_id)
        obj.plate_id = plate_id
        obj.centered_ph = centered_ph
        return obj

    @classmethod
    def pxrdip_from_twothetaphi(cls, xs, twotheta_n=None, phi=None, centered_ph=True):
        """Generate element array for given twotheta,phi on PXRDIP box."""
        L, _, H = cls.PXRDIP_LWH # mm, length of PXRDIP box, height of pinhole vs D plate
        if twotheta_n is None and phi is None:
            qq = np.arange(10, xs.alpha_deg + 89, 1) # zenith every 1 degrees
            pp = np.arange(0, 180 + 1, 5) # azimuth every 5 degrees (calculate half since assuming symmetric so far)
            twotheta_n, phi = np.meshgrid(qq, pp)
        
        twotheta_n = np.radians(twotheta_n)
        phi = np.radians(phi)
        
        beta, phi_d = xs.transform_to_box_coords(twotheta_n, phi)
        sinb, cosb = sin(beta), cos(beta)
        
        # deduce detector element distances, r_d
        rmax = np.sqrt(L**2 + 2 * H**2)
        back_betamax = np.arccos(L / rmax)
        down_betamin = np.arccos(L / np.sqrt(L**2 + 1 * H**2))
        r_back = np.where(beta < back_betamax, L / cosb, rmax)
        r_drul = np.where(np.logical_and(down_betamin < beta, beta < np.pi/2),
                          H / (sinb * cos(radians((180/pi*phi_d+45)%90 - 45))), rmax)
        r_d = np.minimum(r_back, r_drul)

        x = r_d * sinb * cos(phi_d)
        y = r_d * sinb * sin(phi_d)
        z = r_d * cosb
        v_d = np.dstack((x, y, z))

        obj = cls(v_d, id="pxrdip")
        obj.xs = xs
        obj.twotheta_n = twotheta_n
        obj.phi = phi
        obj.centered_ph = centered_ph
        return obj

    @classmethod
    def from_point_params(cls, r_d=24, beta=45, phi_d=0, id=None):
        """Single-point detector given by spherical coordinates."""
        beta = radians(beta)
        phi_d = radians(phi_d)
        v_d = r_d * np.array((sin(beta)*cos(phi_d), sin(beta)*sin(phi_d), cos(beta)))
        return cls(v_d, id)
    
    @property
    def shape(self):
        return self.beta.shape

    def calc_twotheta_n(self, xs):
        """Return nominal twotheta for all elements for given XS and phi=0 vector."""
        return calc_twotheta_nominal(xs.alpha, self.beta, xs.phi, self.phi_d)

    def calc_phi(self, xs, v_phi0=(0,0,1)):
        """Return phi for all elements for given XS and phi=0 vector."""
        v_phi0 = np.array(v_phi0)
        v_x = xs.v_x
        v_d = self.v_d

        phi = azimuth(v_d, v0=v_x, v1=v_phi0)
        return phi


class ReportFileWriter():
    """Writes out the twotheta_pinhole correction report file with tables."""
    TWOTHETA_MIN = 10 # [deg] no useful data this close to direct image
    BETA_MIN = 10 # [deg] VISAR hole is 25 mm diam at 75 mm distance =~10 deg
    BETA_MAX = 88 # [deg] correction value very close to 90 deg is dubious
    DEFAULT_NP = 240 # default number of pinhole phi points for integration

    def __init__(self, params=None):
        p = params if isinstance(params, dict) else EP_BASELINE
        self.params = p
        self.xs = XraySource(p['mat_x'], p['r_x'], p['alpha'], p['phi_x'])
        self.ph = Pinhole(p['mat_p'], p['d_p'], p['h_p'])
        self.centered_ph = p['centered_ph']

        self.beta_min, self.beta_max = self.BETA_MIN, self.BETA_MAX
        self.Np = self.DEFAULT_NP
        self.mu_p = MU_P_LOOKUP.get((self.ph.mat, self.xs.mat), 1/3.0) # [um^-1] attenuation coefficient
        self.run_calc()

    @property
    def title(self):
        xs, ph = self.xs, self.ph
        return "{}{}_a{:.3g}_r{:.0f}_d{:.0f}_h{:.0f}".format(xs.mat, ph.mat, xs.alpha_deg, xs.r, 1000*ph.diam, 1000*ph.thick)
    
    def run_calc(self):
        if True: # standard 2theta and phi sample spacing
            d2theta, dphi = 2, 5
        else: # extra fine 2theta and phi sample spacing
            d2theta, dphi = 0.5, 1 

        # create detector, and generate range for twotheta and phi
        xs, ph = self.xs, self.ph
        qq = np.arange(self.TWOTHETA_MIN, xs.alpha_deg + self.beta_max, d2theta)
        pp = np.arange(0, 180 + 0.001, dphi)
        twotheta_n, phi = np.meshgrid(qq, pp)
        det = Detector.pxrdip_from_twothetaphi(xs, twotheta_n, phi, self.centered_ph)
        
        self.qq, self.pp = qq, pp
        self.twotheta_n, self.phi = twotheta_n, phi
        self.det = det

        qq_n = det.twotheta_n
        qq_p, V_p = calc_twotheta_pinhole(xs, ph, det, Np=self.Np)
        qq_JHE = calc_twotheta_JHE(xs, ph, det)
    
        qq_p[det.beta < np.radians(self.beta_min)] = np.nan
        qq_p[det.beta > np.radians(self.beta_max)] = np.nan
        dq_pn = 180 / pi * (qq_p - qq_n) # convert to degrees
        dq_pj = 180 / pi * (qq_p - qq_JHE) # convert to degrees

        self.qq_n, self.qq_p = qq_n, qq_p
        self.dq_pn, self.dq_pj = dq_pn, dq_pj
    
        if False: # switch on/off additional tables
            write_table(dq_pn.T, self.title + "_pn")
            write_table(dq_pj.T, self.title + "_pj")
            write_table(180/pi*qq_n[0,:][None,:], self.title + "_qq_n", fmt="%.0f")
            write_table(180/pi*det.phi[:,0][None,:], self.title + "_phi", fmt="%.0f")


    def write(self, filename=None, tables="standard"):
        """Write pxrdip twotheta pinhole correction tables to file"""
        if filename is None:
            name = "pxrdip-2theta-pinhole-correction_{}.txt".format(self.title)
            filename = osp.join(OUTPUT_PATH, name)
        if tables == "standard":
            self.tables = tables = (
                (self.dq_pn.T, "%6.3f", "twotheta_p - twotheta_n", "deg"),
                (self.dq_pj.T, "%6.3f", "twotheta_p - twotheta_JHE", "deg"),
            )
        ss = [HEADER]
        ss.append("[parameters]")
        ss.extend(self.param_output_list())

        ss.append("\n[{} rows, mapped to twotheta_n, deg]".format(len(self.qq)))
        ss.append(" ".join(["{:.1f}".format(q) for q in self.qq]))
        ss.append("\n[{} columns, mapped to phi, deg]".format(len(self.pp)))
        ss.append(" ".join(["{:.0f}".format(q) for q in self.pp]))

        for i, table in enumerate(tables):
            array, fmt, label, uom = table
            ss.append("\n[table{}, {}, {}]".format(i, label, uom))
            with io.StringIO() as sio:
                np.savetxt(sio, array, fmt=fmt)
                ss.append(sio.getvalue())
        
        with open(filename, "w") as f:
            f.write("\n".join(ss))

    def param_output_list(self):
        d = self.params
        ss = []
        ss.append("xsf: mat={mat_x}, r_x={r_x} mm, alpha={alpha} deg, phi_x={phi_x} deg".format(**d))
        ss.append("pinhole: mat={}, d_p={:.0f} um, h_p={:.0f} um".format(d['mat_p'], 1000*d['d_p'], 1000*d['h_p']))
        ss.append("attenuation coefficient: mu_p=1/({:.2f} um)".format(1/self.mu_p))
        ss.append("diagnostic: pxrdip, centered pinhole")
        ss.append("integration_resolution: Np={}".format(self.Np))
        ss.append("evaluation_script: {} version {}".format(osp.split(__file__)[1], __version__))
        ss.append("evaluation_timestamp: {}".format(datetime.datetime.today()))
        ss.append("row, column axes: twotheta_n, phi")

        for i, table in enumerate(self.tables):
            array, fmt, label, uom = table
            ss.append("table{}: {}".format(i, label))
        return ss

    def save_image(self, filename=None):
        if filename is None:
            name = "pxrdip-2theta-pinhole-correction_{}.png".format(self.title)
            filename = osp.join(OUTPUT_PATH, name)

        xs, ph, det = self.xs, self.ph, self.det
        qq_n = self.qq_n
        dq_pn, dq_pj = self.dq_pn, self.dq_pj

        extent = 180/pi*np.array((np.nanmin(qq_n), np.nanmax(qq_n), np.min(det.phi), np.max(det.phi)))
        extent2 = extent * np.array((1,1,-1,-1))
        kws = dict(extent=extent, aspect='auto', cmap='coolwarm')
        kws2 = dict(extent=extent2, aspect='auto', cmap='coolwarm')
        kws_contour = dict(colors='k', linestyles='--')
    
        def show_image(ax, value, label):
            vmax = max(np.nanmax(value), -np.nanmin(value))
            im = ax.imshow(value, vmin=-vmax, vmax=vmax, **kws2)
            im = ax.imshow(value, vmin=-vmax, vmax=vmax, **kws)
            plt.colorbar(im, ax=ax, label=label)
            ax.contour(det.beta, [ph.critical_angle], extent=extent, **kws_contour)
            ax.contour(det.beta, [ph.critical_angle], extent=extent2, **kws_contour)
    
        thetalabel, philabel = r'$2\theta_n$ [deg]', r'$\phi$ [deg]'
        fig = plt.figure(figsize=(5,6), dpi=DPI_SCREEN)
        ax1 = plt.subplot(211, xlabel=thetalabel, ylabel=philabel, title=" PXRDIP {}".format(self.title))
        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1, xlabel=thetalabel, ylabel=philabel)
        show_image(ax1, dq_pn, r'$2\theta_p$ - $2\theta_n$ [deg]')
        show_image(ax2, dq_pj, r'$2\theta_p$ - $2\theta_{JHE}$ [deg]')
    
        # formatting tweaks
        ax1.set_xticks(np.arange(0,151,15))
        ax1.set_xticks(np.arange(0,151,5), minor=True)
        ax1.set_yticks(np.arange(-180,181,90))
        ax1.set_yticks(np.arange(-180,181,45), minor=True)
        ax1.axis((self.TWOTHETA_MIN, xs.alpha_deg + self.beta_max, -180, 180))
        plt.tight_layout()
    
        if False: # optional: toggle printing additional info
            print("shape (Nphi, N2theta) and size of output array:", np.shape(dq_pj), np.size(dq_pj))
            print("max difference in twotheta direction: {:.4f} deg".format(np.nanmax(dq_pj[:,1:]-dq_pj[:,:-1])))
            print("max difference in phi direction: {:.4f} deg".format(np.nanmax(dq_pj[1:,:]-dq_pj[:-1,:])))
            print("max correction {:.4f} deg".format(np.nanmax(dq_pj[1:,:])))
            print("min correction {:.4f} deg".format(np.nanmin(dq_pj[1:,:])))

        fig.savefig(filename, dpi=200)


def write_table(A, filename=None, fmt="%6.3f"):
    """write table to file"""
    if filename is None:
        filename = osp.join(OUTPUT_PATH, "output.txt")
    else:
        filename = osp.join(OUTPUT_PATH, filename + ".txt")
    np.savetxt(filename, A, fmt=fmt)


def sandbox1():
    """Sandbox for exploring twotheta calculations."""
    # speed vs fidelity parameters
    calc_type = 1
    if calc_type in ("fast", 0):
        d2theta, dphi = 5, 15 # 2theta and phi sample spacing
        Np = 72 # number of azimuthal points around pinhole
    elif calc_type in ("hires", 2):
        d2theta, dphi = 0.5, 1 # 2theta and phi sample spacing
        Np = 240 # number of azimuthal points around pinhole
    else:
        d2theta, dphi = 2, 5 # 2theta and phi sample spacing
        Np = 240 # number of azimuthal points around pinhole

    twothetamin = 10 # [deg] no useful data this close to direct image
    betamin, betamax = 10, 88 # deg

    # xray source and pinhole
    if False:
#        p = OMEGA_BASELINE
        p = EP_BASELINE
    else:
#        p = dict(mat_x='Cu', r_x=24.24, alpha=22.5, phi_x=0,
#                 mat_p='Ta', d_p=0.3, h_p=0.075, centered_ph=True)
#        p = dict(mat_x='Cu', r_x=24.24, alpha=22.5, phi_x=0, z_s=0.1,
#                 mat_p='Ta', d_p=0.8, h_p=0.075, centered_ph=True) # Danae sodium, although actually with non-centered pinhole
        p = dict(mat_x='Cu', r_x=17, alpha=45, phi_x=0, z_s=0.1,
                 mat_p='W', d_p=0.5, h_p=0.075, centered_ph=True) # Michelle TATB

    xs = XraySource(p['mat_x'], p['r_x'], p['alpha'], p['phi_x'])
    ph = Pinhole(p['mat_p'], p['d_p'], p['h_p'])
    
    # create detector, and generate range for twotheta and phi
    qq = np.arange(twothetamin, xs.alpha_deg + betamax, d2theta)
    pp = np.arange(0, 180 + 1, dphi)
    twotheta_n, phi = np.meshgrid(qq, pp)
    det = Detector.pxrdip_from_twothetaphi(xs, twotheta_n, phi, p['centered_ph'])
    
    title = "{}{}_a{:.3g}_d{:.0f}".format(xs.mat, ph.mat, xs.alpha_deg, 1000*ph.diam)

    qq_n = det.twotheta_n
    qq_p, V_p = calc_twotheta_pinhole(xs, ph, det, Np=Np)
    qq_JHE = calc_twotheta_JHE(xs, ph, det)

    qq_p[det.beta<np.radians(betamin)] = np.nan
    qq_p[det.beta>np.radians(betamax)] = np.nan
    dq_pn = 180/pi*(qq_p - qq_n)
    dq_pj = 180/pi*(qq_p - qq_JHE)

    extent = 180/pi*np.array((np.nanmin(qq_n), np.nanmax(qq_n), np.min(det.phi), np.max(det.phi)))
    extent2 = extent * np.array((1,1,-1,-1))
    kws = dict(extent=extent, aspect='auto', cmap='coolwarm')
    kws2 = dict(extent=extent2, aspect='auto', cmap='coolwarm')
    kws_contour = dict(colors='k', linestyles='--')


    def show_image(ax, value, label):
        vmax = max(np.nanmax(value), -np.nanmin(value))
        im = ax.imshow(value, vmin=-vmax, vmax=vmax, **kws2)
        im = ax.imshow(value, vmin=-vmax, vmax=vmax, **kws)
        plt.colorbar(im, ax=ax, label=label)
        ax.contour(det.beta, [ph.critical_angle], extent=extent, **kws_contour)
        ax.contour(det.beta, [ph.critical_angle], extent=extent2, **kws_contour)

    plt.figure(figsize=(6,7), dpi=DPI_SCREEN)
    ax1 = plt.subplot(211, xlabel=r'$2\theta_n$', ylabel=r'$\phi$', title=title)
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1, xlabel=r'$2\theta_n$', ylabel=r'$\phi$')
    show_image(ax1, dq_pn, r'$2\theta_p$ - $2\theta_n$')
    show_image(ax2, dq_pj, r'$2\theta_p$ - $2\theta_{JHE}$')

    # formatting tweaks
#    [ax.minorticks_on() for ax in (ax1, ax2)]
    ax1.set_xticks(np.arange(0,151,15))
    ax1.set_xticks(np.arange(0,151,5), minor=True)
    ax1.set_yticks(np.arange(-180,181,90))
    ax1.set_yticks(np.arange(-180,181,45), minor=True)
    ax1.axis((twothetamin, xs.alpha_deg + betamax, -180, 180))
    plt.tight_layout()

    print("shape (Nphi, N2theta) and size of output array:", np.shape(dq_pj), np.size(dq_pj))
    print("max difference in twotheta direction: {:.4f} deg".format(np.nanmax(dq_pj[:,1:]-dq_pj[:,:-1])))
    print("max difference in phi direction: {:.4f} deg".format(np.nanmax(dq_pj[1:,:]-dq_pj[:-1,:])))

    if calc_type not in ("hires", 2):
        write_table(dq_pn.T, title + "_pn")
        write_table(dq_pj.T, title + "_pj")
        write_table(180/pi*qq_n[0,:][None,:], title + "_qq_n", fmt="%.0f")
        write_table(180/pi*det.phi[:,0][None,:], title + "_phi", fmt="%.0f")

    if True: # impact parameter calc
        z_s = p['z_s']
        b_center = z_s * np.tan(det.beta)
        b_max = ph.d/2 + (z_s - ph.h/2) * np.tan(det.beta)
        b_center[det.beta >= ph.critical_angle] = np.nan
        b_max[det.beta >= ph.critical_angle] = np.nan
        plt.figure(figsize=(6,7), dpi=DPI_SCREEN)
        ax1 = plt.subplot(211, xlabel=r'$2\theta_n$', ylabel=r'$\phi$', title=title)
        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1, xlabel=r'$2\theta_n$', ylabel=r'$\phi$')
        
        im = ax1.imshow(b_center, vmin=0, **kws2)
        im = ax1.imshow(b_center, vmin=0, **kws)
        plt.colorbar(im, ax=ax1, label="b center [mm]")
        im = ax2.imshow(b_max, vmin=ph.d/2, **kws2)
        im = ax2.imshow(b_max, vmin=ph.d/2, **kws)
        plt.colorbar(im, ax=ax2, label="b max [mm]")

        ax1.contour(det.beta, [ph.critical_angle], extent=extent, **kws_contour)
        ax1.contour(det.beta, [ph.critical_angle], extent=extent2, **kws_contour)
        ax1.set_xticks(np.arange(0,151,15))
        ax1.set_xticks(np.arange(0,151,5), minor=True)
        ax1.set_yticks(np.arange(-180,181,90))
        ax1.set_yticks(np.arange(-180,181,45), minor=True)
        ax1.axis((twothetamin, xs.alpha_deg + betamax, -180, 180))

        plt.tight_layout()
        
        if calc_type not in ("hires", 2):
            write_table(b_center.T, title + "_b-center")
        if calc_type not in ("hires", 2):
            write_table(b_max.T, title + "_b-max")

# ============================ MAIN ===========================================
if __name__ == "__main__":
    if True:
        p = OMEGA_BASELINE
    else:
        p = EP_BASELINE
    rfw = ReportFileWriter(p)
    rfw.write()
    rfw.save_image()

#    sandbox1()
    
    plt.show()
