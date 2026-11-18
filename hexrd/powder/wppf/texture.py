# standard imports
# ---------
import copy
from warnings import warn

import numba
import numpy as np

# 3rd party imports
# -----------------
from lmfit import Minimizer, Parameters
from matplotlib import pyplot as plt
from scipy.special import sph_harm_y

from hexrd.core import constants

# hexrd imports
# -------------
from hexrd.core.transforms.xfcapi import angles_to_gvec
from hexrd.phase_transition.orientation_relationship import OrientationRelationship
from hexrd.phase_transition.texture import (
    kernels,
    unimodal_odf,
)
from hexrd.powder.wppf.phase import Material_Rietveld
from hexrd.rotations import rotMatOfExpMap

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
eta_ref = constants.lab_x

SYMLIST = [
    'ci',
    'c2h',
    'd2h',
    'c4h',
    'd4h',
    's6',
    'd3d',
    'c6h',
    'd6h',
    'th',
    'oh',  # end of crystal symmtries
    'monoclinic',
    'orthorhombic',
    'triclinic',
    'axial',
]  # end of sample symmetries

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
    'th': {
        2: np.zeros((0, 3)),
        4: np.array(
            [
                [
                    0.4564354645876387,
                    0.0,
                    0.7637626158259737,
                    0.0,
                    0.4564354645876387,
                ]
            ]
        ),
        6: np.array(
            [
                [
                    -0.39528470752104744,
                    0.0,
                    0.5863019699779289,
                    0.0,
                    0.5863019699779287,
                    0.0,
                    -0.3952847075210474,
                ],
                [
                    0.0,
                    0.6614378277661476,
                    0.0,
                    -0.35355339059327384,
                    0.0,
                    0.6614378277661477,
                    0.0,
                ],
            ]
        ),
        8: np.array(
            [
                [
                    0.4114253678777394,
                    0.0,
                    0.27003086243366087,
                    0.0,
                    0.7180703308172529,
                    0.0,
                    0.270030862433661,
                    0.0,
                    0.4114253678777394,
                ]
            ]
        ),
        10: np.array(
            [
                [
                    -0.32179907341941766,
                    -0.3026705127825632,
                    0.08783983261480693,
                    -0.25429409464318103,
                    0.4478970205736254,
                    0.25236027118158605,
                    0.44789702057362524,
                    -0.25429409464318115,
                    0.08783983261480709,
                    -0.3026705127825629,
                    -0.3217990734194177,
                ],
                [
                    0.24992195050936286,
                    -0.3897180314346175,
                    -0.06821990525403415,
                    -0.3274286386826981,
                    -0.34785462810576917,
                    0.32493865092120033,
                    -0.34785462810576934,
                    -0.3274286386826982,
                    -0.06821990525403382,
                    -0.38971803143461725,
                    0.24992195050936322,
                ],
            ]
        ),
        12: np.array(
            [
                [
                    -0.10774660850851898,
                    0.35440527769341207,
                    0.042183172840390366,
                    -0.5617029300070009,
                    -0.11564076236589332,
                    0.14294418606853107,
                    -0.15291919171429025,
                    0.1429441860685311,
                    -0.1156407623658934,
                    -0.5617029300070007,
                    0.042183172840390734,
                    0.35440527769341246,
                    -0.10774660850851901,
                ],
                [
                    0.3411000537498401,
                    0.10028705753178294,
                    -0.2620437799555088,
                    -0.15894665684441117,
                    0.42787445167677257,
                    0.04044931809534165,
                    0.4357742152904641,
                    0.040449318095341594,
                    0.4278744516767727,
                    -0.15894665684441128,
                    -0.2620437799555087,
                    0.100287057531783,
                    0.3411000537498399,
                ],
                [
                    0.19715691408140235,
                    0.02017705548403037,
                    0.561512087862538,
                    -0.03197895713646226,
                    -0.09548336992239026,
                    0.00813812026783474,
                    0.5200389527975391,
                    0.00813812026783466,
                    -0.09548336992239025,
                    -0.0319789571364624,
                    0.5615120878625383,
                    0.02017705548403026,
                    0.19715691408140232,
                ],
            ]
        ),
        14: np.array(
            [
                [
                    -0.28955062552019833,
                    -0.2972304474122511,
                    0.010117717673205564,
                    -0.2447888749918319,
                    0.12465750846560059,
                    -0.22811657797300322,
                    0.38999831765926807,
                    0.3102101852656362,
                    0.3899983176592681,
                    -0.228116577973003,
                    0.1246575084656004,
                    -0.24478887499183172,
                    0.010117717673205809,
                    -0.297230447412251,
                    -0.2895506255201985,
                ],
                [
                    0.28772627934090717,
                    -0.2991150553540337,
                    -0.010053969858649543,
                    -0.2463409739167105,
                    -0.12387208985743604,
                    -0.22956296517271021,
                    -0.387541089533881,
                    0.31217709203398175,
                    -0.3875410895338812,
                    -0.22956296517271035,
                    -0.12387208985743622,
                    -0.2463409739167103,
                    -0.010053969858649592,
                    -0.29911505535403393,
                    0.28772627934090733,
                ],
            ]
        ),
        16: np.array(
            [
                [
                    0.16618905573198167,
                    -0.3654438510445056,
                    -0.059213072462354854,
                    0.17954391047429719,
                    -0.002851898776857607,
                    0.4610540410294765,
                    0.19021431714221818,
                    -0.17805079589901174,
                    0.2127310570059948,
                    -0.17805079589901207,
                    0.19021431714221823,
                    0.4610540410294766,
                    -0.002851898776857436,
                    0.1795439104742971,
                    -0.05921307246235514,
                    -0.36544385104450566,
                    0.16618905573198162,
                ],
                [
                    0.30796148329941286,
                    0.1641675465982823,
                    -0.26096265404956615,
                    -0.08065612050929621,
                    -0.09371337438188433,
                    -0.20711830435427173,
                    0.44600261056672585,
                    0.07998537189520905,
                    0.305305250454509,
                    0.07998537189520934,
                    0.4460026105667255,
                    -0.20711830435427164,
                    -0.09371337438188454,
                    -0.08065612050929624,
                    -0.26096265404956615,
                    0.16416754659828256,
                    0.3079614832994129,
                ],
                [
                    0.21028019819195082,
                    0.04839013590249786,
                    0.4381584188768272,
                    -0.02377425205947472,
                    0.29639275890473565,
                    -0.06105033000294009,
                    -0.07659622573459653,
                    0.02357654174909582,
                    0.5707783675995487,
                    0.02357654174909576,
                    -0.07659622573459626,
                    -0.06105033000294006,
                    0.29639275890473576,
                    -0.023774252059474876,
                    0.4381584188768272,
                    0.048390135902498174,
                    0.2102801981919509,
                ],
            ]
        ),
    },
    'oh': {
        2: np.zeros((0, 1)),
        4: np.array(
            [
                [
                    -0.4564354645876382,
                    -0.7637626158259734,
                    -0.4564354645876386,
                ]
            ]
        ),
        6: np.array(
            [
                [
                    0.6614378277661479,
                    -0.3535533905932739,
                    0.6614378277661476,
                ]
            ]
        ),
        8: np.array(
            [
                [
                    -0.4114253678777393,
                    -0.2700308624336607,
                    -0.7180703308172534,
                    -0.2700308624336608,
                    -0.4114253678777394,
                ]
            ]
        ),
        10: np.array(
            [
                [
                    -0.4934466367636259,
                    -0.41457809879442503,
                    0.4114253678777398,
                    -0.41457809879442514,
                    -0.49344663676362605,
                ]
            ]
        ),
        12: np.array(
            [
                [
                    -0.40844758180620133,
                    -0.04107687246151441,
                    -0.34173954767501197,
                    -0.6552821453699549,
                    -0.34173954767501197,
                    -0.04107687246151456,
                    -0.40844758180620117,
                ],
                [
                    0.0,
                    0.6197216133464934,
                    -0.2979605473965942,
                    0.23308639662726535,
                    -0.29796054739659417,
                    0.6197216133464934,
                    0.0,
                ],
            ]
        ),
        14: np.array(
            [
                [
                    -0.421682054643464,
                    -0.3472829807952012,
                    -0.32362992464387486,
                    0.4400964619641172,
                    -0.32362992464387463,
                    -0.34728298079520126,
                    -0.4216820546434642,
                ]
            ]
        ),
        16: np.array(
            [
                [
                    0.40826074902286213,
                    0.004724991415765744,
                    0.0808097864409005,
                    0.3744089876279584,
                    0.6108821877615941,
                    0.3744089876279583,
                    0.0808097864409004,
                    0.004724991415765709,
                    0.4082607490228618,
                ],
                [
                    0.0,
                    -0.5133889064323339,
                    -0.3001812382731627,
                    0.3174660719799944,
                    -0.3017891584619822,
                    0.3174660719799949,
                    -0.30018123827316284,
                    -0.5133889064323339,
                    0.0,
                ],
            ]
        ),
    },
}

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


def check_symmetry(sym, symtype='either'):
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


def check_degrees(ell, sym):
    if ell < 0:
        msg = 'the degree is negative'
        raise ValueError(msg)
    if np.mod(ell, 2) != 0:
        msg = 'only even degrees allowed due toinversion symmetry'
        raise ValueError(msg)
    if sym in ['td', 'oh']:
        if ell > 16:
            msg = 'maximum degree available is 16'
            raise ValueError(msg)
    return True


def calc_sym_sph_harm(deg, theta, phi, sym='oh'):
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
        msg = (
            f'unknown crystal/sample symmetry. '
            f'It should be one of {SYMLIST[0:11]} '
            f'for crystal symmetries and {SYMLIST[11:]} '
            f'for sample symmetries'
        )
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
    elif sym in ['th', 'c2h', 'd2h', 'orthorhombic', 'monoclinic']:
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

    n = int(deg / nfold)

    if sym in ['th', 'oh']:
        coeff = Blmn[sym][deg]

        num_sym = coeff.shape[0]

        Intensity = np.zeros((theta.shape[0], num_sym)).astype(complex)

        for ii in range(num_sym):
            for jj in range(-n, n + 1):
                Intensity[:, ii] += coeff[ii, jj + n] * sph_harm_y(
                    deg, nfold * jj, theta, phi
                )
        Intensity = np.real(Intensity)

    elif sym == 'axial':
        Intensity = np.zeros((theta.shape[0], 1))
        Intensity[:, 0] = np.real(sph_harm_y(deg, 0, theta, phi))

    elif sym in ['d2h', 'd3d', 'd4h', 'd6h', 'orthorhombic']:
        '''this is the dihedral symmetry with the
        following selection rules:
        1. l = 2l'
        2. m = [0, nfold*m'] for all allowed m's
        '''
        num_sym = n + 1
        Intensity = np.zeros((theta.shape[0], num_sym))
        for ii in range(n + 1):
            if ii == 0:
                Intensity[:, ii] = np.real(sph_harm_y(deg, nfold * ii, theta, phi))
            else:
                '''we need special attention for laue
                group d3d and d6h. for this cases the
                real/imaginary component flips for
                m = (2k+1)*nfold*m'
                '''
                Intensity[:, ii] = np.real(
                    (
                        sph_harm_y(deg, -nfold * ii, theta, phi)
                        + sph_harm_y(deg, nfold * ii, theta, phi)
                    )
                    / np.sqrt(2.0)
                )
                if (
                    sym
                    in [
                        'd3d',
                    ]
                    and np.mod(ii, 2) != 0
                ):
                    Intensity[:, ii] = np.imag(
                        (
                            sph_harm_y(deg, -nfold * ii, theta, phi)
                            + sph_harm_y(deg, nfold * ii, theta, phi)
                        )
                        / np.sqrt(2.0)
                    )
    elif sym in ['ci', 'c2h', 's6', 'c4h', 'c6h', 'monoclinic', 'triclinic']:
        '''this is the cyclic symmetry with the
        following selection rules:
        1. l = 2l'
        2. m = 2m' for all allowed m's
        '''
        num_sym = 2 * n + 1
        Intensity = np.zeros((theta.shape[0], num_sym))
        for ii in range(-n, n + 1):
            if ii < 0:
                Intensity[:, ii + n] = np.imag(sph_harm_y(deg, nfold * ii, theta, phi))
            else:
                Intensity[:, ii + n] = np.real(sph_harm_y(deg, nfold * ii, theta, phi))

    return Intensity


def get_num_sym_harm(ell, sym='oh'):
    # crystal symmetries
    if sym in ['th', 'oh']:
        return Blmn[sym][ell].shape[0]
    elif sym == 'ci':
        return 2 * ell + 1
    elif sym == 'c2h':
        return 2 * int(ell / 2) + 1
    elif sym == 'd2h':
        return int(ell / 2) + 1
    elif sym == 'c4h':
        return 2 * int(ell / 4) + 1
    elif sym == 'd4h':
        return int(ell / 4) + 1
    elif sym == 's6':
        return 2 * int(ell / 3) + 1
    elif sym == 'd3d':
        return int(ell / 3) + 1
    elif sym == 'c6h':
        return 2 * int(ell / 6) + 1
    elif sym == 'd6h':
        return int(ell / 6) + 1

    # sample symmetries
    elif sym == 'axial':
        return 1
    elif sym == 'monoclinic':
        return 2 * int(ell / 2) + 1
    elif sym == 'orthorhombic':
        return int(ell / 2) + 1
    elif sym == 'triclinic':
        return 2 * ell + 1


class MarchDollaseModel:
    """this class implements the well known March Dollase
    texture model for axisymmetric textures. the functional
    form used is the same as the Bragg-Brentano geometry which
    is also valid from Debye-Scherrer geometry as detailed in
    Howard and Kisi, J. Appl. Cryst. (2000). 33, 1434-1435.

    P_k = 1/m_k sum{i=1, m_k} (P^2cos^2alpha + sin^2 alpha/P)^-1.5

    this is a simple model with just one refinement parameter.
    However, we need to make sure that only one of the texture
    models is used inside the wppf module.
    """

    def __init__(
        self,
        material: Material_Rietveld = None,
        HKL: np.ndarray = np.array([1.0, 1.0, 1.0]),
        P_MD: float = 1.0,
    ) -> None:
        self.material = material
        self.HKL = HKL
        self.P_MD = P_MD

    def get_texture_factor(self) -> None:
        hkls = self.material.hkls
        self.texture_factors = np.empty(hkls.shape[0])

        for ii, g in enumerate(hkls):
            self.texture_factors[ii] = self.get_texture_factor_reflection(g)

    def get_texture_factor_reflection(self, g: np.ndarray) -> np.ndarray:
        """get the sum of symmetrically equivalent hkls"""
        gsym = self.material.CalcStar(g, 'r')
        pre = 1.0 / gsym.shape[0]
        texture_factor = 0.0
        for h in gsym:
            cos2_alpha, sin2_alpha = self.calc_cos_sin_alpha(h)
            texture_factor += (
                self.P_MD**2 * cos2_alpha + sin2_alpha / self.P_MD
            ) ** -1.5

        return pre * texture_factor

    def calc_cos_sin_alpha(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g_mag = self.material.CalcLength(g, 'r')
        gdotHKL = self.material.CalcDot(g, self.HKL, 'r')
        cos2_alpha = (gdotHKL / g_mag / self.HKL_length) ** 2
        sin2_alpha = 1 - cos2_alpha

        return cos2_alpha, sin2_alpha

    def calc_pf_rings(
        self,
        params: Parameters,
        eta_min: float = -np.pi,
        eta_max: float = np.pi,
        eta_step: float = np.radians(0.1),
        calc_type: str = 'spectrum_2d',
    ) -> dict:
        '''this function computes the intensity variation
        along the debye-scherrer rings for each hkl in the material.
        the intensity variation around the ring is computed between
        eta_min and eta_max. Default values are (-pi,pi) which gives
        the full ring. eta_step is the angular step size in azimuth.
        Default value is 0.1 degrees for eta_step
        '''
        pname = f'{self.material.name}_p_md'
        self.P_MD = params[pname].value
        # floor with a tolerance, to match bin_polar_view
        nspec = int(np.floor((eta_max - eta_min) / eta_step + 1e-8)) - 1

        if calc_type != 'spectrum_2d':
            msg = 'calc_type parameter set incorrectly'
            raise ValueError(msg)
        self.intensities_rings_2d = {}
        for ii, h in enumerate(self.material.hkls):
            hkey = tuple(h)
            self.intensities_rings_2d[hkey] = self.texture_factors[ii] * np.ones(
                [
                    nspec,
                ]
            )

    def calc_texture_factor(
        self,
        params: Parameters,
        eta_min: float = -np.pi,
        eta_max: float = np.pi,
        eta_step: float = np.radians(1),
        eta_mask: dict | None = None,
    ) -> np.ndarray:
        """Return texture factors, matching the HarmonicModel interface."""
        pname = f'{self.material.name}_p_md'
        if pname in params:
            self.P_MD = params[pname].value
        return self.texture_factors

    def texture_index(self, params: Parameters) -> float:
        """Return P_MD as the texture index, analogous to HarmonicModel.J."""
        pname = f'{self.material.name}_p_md'
        if pname in params:
            self.P_MD = params[pname].value
        return self.P_MD

    def get_parameters(
        self, params: Parameters | None = None, vary: bool = True
    ) -> Parameters:
        '''
        make lmfit parameter class for refinement
        '''
        if params is None:
            params = Parameters()

        pname = f'{self.material.name}_p_md'

        params.add(pname, value=self.P_MD, min=1e-3, vary=vary)

        return params

    @property
    def material(self) -> Material_Rietveld:
        return self._material

    @material.setter
    def material(self, mat: Material_Rietveld) -> None:
        if mat is not None and not isinstance(mat, Material_Rietveld):
            raise Exception(f'Invalid material: {mat}')

        self._material = mat

    @property
    def HKL(self) -> np.ndarray:
        return self._HKL

    @HKL.setter
    def HKL(self, hkl: list | np.ndarray) -> None:
        if isinstance(hkl, list):
            hkl = np.array(hkl)
            if hkl.shape != (3,):
                msg = 'HKL should be a list of length 3'
                raise ValueError(msg)
        elif isinstance(hkl, np.ndarray):
            if hkl.shape != (3,):
                msg = 'HKL should be an array with shape (3,)'
                raise ValueError(msg)

        else:
            msg = 'HKL must be a list or numpy.ndarray'
            raise TypeError(msg)
        self._HKL = hkl
        self._HKL_length = self.material.CalcLength(self._HKL, 'r')

        if hasattr(self, '_P_MD'):
            self.get_texture_factor()

    @property
    def HKL_length(self) -> float:
        return self._HKL_length

    @property
    def P_MD(self) -> float:
        return self._P_MD

    @P_MD.setter
    def P_MD(self, val: float):
        if isinstance(val, (float, int)):
            self._P_MD = float(val)
        else:
            msg = 'P_MD must be a float'
            raise TypeError(msg)

        if hasattr(self, '_HKL'):
            self.get_texture_factor()


class HarmonicModel:
    """this is the abstract class which is used by
    both the harmonic model as well as the pole figure
    class

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
    bvec: numpy.ndarray
            unit vector of xray beam in the lab frame. default value
            is -Z direction
    evec: numpy.ndarray
            direction which defines zero azimuth. default value is
            x direction
    sample_normal: numpy.ndarray
            inclination of sample frame about x direction. default
            value is 0, corresponding to no tilt.
    """

    def __init__(
        self,
        material=None,
        ssym='axial',
        ell_max=10,
        bvec=bvec_ref,
        evec=eta_ref,
        sample_rmat=np.eye(3),
        pfdata=None,
    ):
        self.material = material
        self.ssym = ssym
        self.csym = self.material.sg.laueGroup
        self.ell_max = ell_max
        self.bvec = bvec
        self.etavec = evec
        self.sample_rmat = sample_rmat
        self.pfdata = pfdata

    @property
    def csym(self):
        return self._csym

    @property
    def ssym(self):
        return self._ssym

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, mat):
        if not isinstance(mat, Material_Rietveld):
            raise Exception(f'Invalid material: {mat}')

        self._material = mat

    @ssym.setter
    def ssym(self, val):
        if check_symmetry(val, symtype='sample'):
            self._ssym = val
        else:
            msg = 'unknown sample symmetry'
            raise ValueError(msg)

    @csym.setter
    def csym(self, val):
        if check_symmetry(val, symtype='crystal'):
            self._csym = val
        else:
            msg = 'unknown sample symmetry'
            raise ValueError(msg)

    @property
    def ell_max(self):
        return self._ell_max

    @ell_max.setter
    def ell_max(self, val):
        if check_degrees(val, self.csym):
            self._ell_max = val

    @property
    def bvec(self):
        return self._bvec

    @property
    def etavec(self):
        return self._etavec

    @bvec.setter
    def bvec(self, bHat_l):
        self._bvec = bHat_l

    @etavec.setter
    def etavec(self, eHat_l):
        self._etavec = eHat_l

    @property
    def sample_normal(self):
        return np.dot(self.sample_rmat, np.array([0, 0, 1]))

    @property
    def sample_rmat(self):
        return self._sample_rmat

    @sample_rmat.setter
    def sample_rmat(self, val):
        if not isinstance(val, np.ndarray) or val.shape != (3, 3):
            msg = 'either sample_rmat is not an array, or the shape is not (3,3)'
            raise ValueError(msg)

        self._sample_rmat = val
        self._ref_frame_rmat = val

    @property
    def ref_frame_rmat(self):
        return self._sample_rmat

    def get_total_sym_harm(self, symtype):
        '''function to return the total symmetrized harmonics
        for a given degree and symmetry
        '''
        if symtype == 'crystal':
            sym = self.csym
        elif symtype == 'sample':
            sym = self.ssym
        num = 0
        for ell in range(2, self.ell_max + 1, 2):
            num += get_num_sym_harm(ell, sym=sym)

        return num

    def get_parameters(self, params=None, vary=True):
        '''
        make lmfit parameter class for refinement
        '''
        if params is None:
            params = Parameters()

        for name in self.parameter_names:
            params.add(name, value=0.0, vary=vary)

        return params

    @property
    def parameter_names(self):
        phase = self.material.name
        ret = []
        for ell in range(2, self.ell_max + 1, 2):
            nc = get_num_sym_harm(ell, sym=self.csym)
            ns = get_num_sym_harm(ell, sym=self.ssym)

            for ii in range(ns):
                for jj in range(nc):
                    ret.append(f'{phase}_c_{ell}_{ii}_{jj}')

        return ret

    def convert_hkls_to_cartesian(self, hkls):
        """
        this routine converts hkls in the crystallographic frame to
        the cartesian frame and normalizes them
        """
        hkls_c = np.atleast_2d(np.empty(hkls.shape))

        for ii, g in enumerate(hkls):
            v = self.material.TransSpace(g, "r", "c")
            v = v / np.linalg.norm(v)
            hkls_c[ii, :] = v
        return hkls_c

    def get_c_matrix(self, params, ell):
        nc = get_num_sym_harm(ell, sym=self.csym)
        ns = get_num_sym_harm(ell, sym=self.ssym)
        phase = self.material.name
        cmat = np.empty([ns, nc])
        for ii in range(ns):
            for jj in range(nc):
                pname = f'{phase}_c_{ell}_{ii}_{jj}'
                cmat[ii, jj] = params[pname].value
        return cmat

    def get_sph_s_matrix(self, h, ell, gridtype=None):
        if gridtype is None:
            ngrid = self.angs[h].shape[0]
            Ylm = self.sph_s[h]

        elif gridtype == 'new':
            ngrid = self.angs_new[h].shape[0]
            Ylm = self.sph_s_new[h]

        elif gridtype == 'rings':
            ngrid = self.rotated_angs_rings[h].shape[0]
            Ylm = self.sph_s_rings[h]

        elif gridtype == 'rings_spectrum_2d':
            ngrid = self.rotated_angs_rings_2d[h].shape[0]
            Ylm = self.sph_s_rings_2d[h]

        ns = get_num_sym_harm(ell, sym=self.ssym)

        smat = np.empty((ngrid, ns))
        for ii in range(ns):
            yname = f's_{ell}_{ii}'
            smat[:, ii] = Ylm[yname]

        return smat

    def get_sph_c_matrix(self, h, ell, gridtype=None):
        nc = get_num_sym_harm(ell, sym=self.csym)

        ngrid = 1
        if gridtype is None:
            Ylm = self.sph_c[h]
        elif gridtype == 'new':
            Ylm = self.sph_c_new[h]
        elif gridtype in ['rings', 'rings_spectrum_2d']:
            Ylm = self.sph_c_rings[h]

        cmat = np.empty((nc, ngrid))
        for ii in range(nc):
            yname = f'c_{ell}_{ii}'
            cmat[ii, :] = Ylm[yname]

        return cmat

    def precompute_spherical_harmonics(
        self, eta_min, eta_max, eta_step, calc_type='texture_factor'
    ):
        '''this function precomputes the spherical
        harmonic functions so we don't keep repeating
        the calculations
        '''

        # floor with a tolerance, to match bin_polar_view
        nspec = int(np.floor((eta_max - eta_min) / eta_step + 1e-8)) - 1
        eta_grid = eta_min + eta_step * np.arange(1, nspec + 1)
        self.eta_grid = eta_grid

        '''initialize all the dictionaries which will store the data
        '''
        if calc_type == 'texture_factor':
            self.angs_rings = {}
            self.rotated_angs_rings = {}
        elif calc_type == 'spectrum_2d':
            self.angs_rings_2d = {}
            self.rotated_angs_rings_2d = {}

        self.hkl_angles_rings = {}

        hkls = self.material.hkls
        hkls_c = self.convert_hkls_to_cartesian(hkls)
        tth = np.radians(self.material.getTTh(self.material.wavelength))

        for ii, (t, h, hc) in enumerate(zip(tth, hkls, hkls_c)):
            hkey = tuple(h)
            if calc_type == 'texture_factor':
                angs = np.vstack(
                    (
                        np.full(self.eta_grid.shape, t),
                        self.eta_grid,
                        np.zeros_like(self.eta_grid),
                    )
                ).T
            elif calc_type == 'spectrum_2d':
                angs = np.vstack(
                    (
                        np.full(eta_grid.shape, t),
                        eta_grid,
                        np.zeros_like(eta_grid),
                    )
                ).T

            pfgrid = angles_to_gvec(angs, beam_vec=self.bvec, eta_vec=self.etavec)

            # Next part runs in a numba function to accelerate it.
            if calc_type == 'texture_factor':
                (
                    self.angs_rings[hkey],
                    self.rotated_angs_rings[hkey],
                    self.hkl_angles_rings[hkey],
                ) = _calc_ang_rings(
                    pfgrid,
                    t,
                    hc,
                    self.ref_frame_rmat,
                    self.eta_grid,
                )
            elif calc_type == 'spectrum_2d':
                (
                    self.angs_rings_2d[hkey],
                    self.rotated_angs_rings_2d[hkey],
                    self.hkl_angles_rings[hkey],
                ) = _calc_ang_rings(
                    pfgrid,
                    t,
                    hc,
                    self.ref_frame_rmat,
                    eta_grid,
                )

            '''also pre-compute the spherical harmonics if they are
            not already precomputed
            '''
            if not hasattr(self, 'sph_c_rings'):
                self.sph_c_rings = {}

            if hkey not in self.sph_c_rings:
                self.sph_c_rings[hkey] = {}

                theta = np.array([self.hkl_angles_rings[hkey][0]])
                phi = np.array([self.hkl_angles_rings[hkey][1]])

                for ell in range(2, self.ell_max + 1, 2):
                    Ylm = calc_sym_sph_harm(ell, theta, phi, sym=self.csym)
                    for jj in range(Ylm.shape[1]):
                        kname = f'c_{ell}_{jj}'
                        self.sph_c_rings[hkey][kname] = Ylm[:, jj]

            if calc_type == 'texture_factor':
                if not hasattr(self, 'sph_s_rings'):
                    self.sph_s_rings = {}

                if hkey not in self.sph_s_rings:
                    self.sph_s_rings[hkey] = {}

                theta_samp = self.angs_rings[hkey][:, 0]
                phi_samp = self.angs_rings[hkey][:, 1]

            elif calc_type == 'spectrum_2d':
                if not hasattr(self, 'sph_s_rings_2d'):
                    self.sph_s_rings_2d = {}

                if hkey not in self.sph_s_rings_2d:
                    self.sph_s_rings_2d[hkey] = {}

                theta_samp = self.angs_rings_2d[hkey][:, 0]
                phi_samp = self.angs_rings_2d[hkey][:, 1]

            for ell in range(2, self.ell_max + 1, 2):
                Ylm = calc_sym_sph_harm(ell, theta_samp, phi_samp, sym=self.ssym)
                if calc_type == 'texture_factor':
                    for jj in range(Ylm.shape[1]):
                        kname = f's_{ell}_{jj}'
                        self.sph_s_rings[hkey][kname] = Ylm[:, jj]
                elif calc_type == 'spectrum_2d':
                    for jj in range(Ylm.shape[1]):
                        kname = f's_{ell}_{jj}'
                        self.sph_s_rings_2d[hkey][kname] = Ylm[:, jj]

    def calc_pf_rings(
        self,
        params,
        eta_min=-np.pi,
        eta_max=np.pi,
        eta_step=np.radians(0.1),
        calc_type='texture_factor',
    ):
        '''this function computes the intensity variation
        along the debye-scherrer rings for each hkl in the material.
        the intensity variation around the ring is computed between
        eta_min and eta_max. Default values are (-pi,pi) which gives
        the full ring. eta_step is the angular step size in azimuth.
        Default value is 0.1 degrees for eta_step
        '''

        # floor with a tolerance, to match bin_polar_view
        nspec = int(np.floor((eta_max - eta_min) / eta_step + 1e-8)) - 1
        eta_grid = eta_min + eta_step * np.arange(1, nspec + 1)

        if not calc_type in ['texture_factor', 'spectrum_2d']:
            msg = 'unknown type of grid for precomputingspherical harmonics'
            raise ValueError(msg)

        if calc_type == 'texture_factor':
            gtype = 'rings'
            self.intensities_rings = {}
        elif calc_type == 'spectrum_2d':
            gtype = 'rings_spectrum_2d'
            self.intensities_rings_2d = {}

        self.precompute_spherical_harmonics(
            eta_min=eta_min,
            eta_max=eta_max,
            eta_step=eta_step,
            calc_type=calc_type,
        )

        for h in self.material.hkls:
            hkey = tuple(h)
            '''perform the series sum with given coefficients
            '''
            term = np.ones_like(eta_grid)

            for ell in range(2, self.ell_max + 1, 2):
                cmat = self.get_c_matrix(params, ell)
                sph_s_mat = self.get_sph_s_matrix(hkey, ell, gridtype=gtype)
                sph_c_mat = self.get_sph_c_matrix(hkey, ell, gridtype=gtype)
                pre = 4 * np.pi / (2 * ell + 1)
                t = pre * np.dot(sph_s_mat, np.dot(cmat, sph_c_mat))
                term += t.flatten()
            if calc_type == 'texture_factor':
                self.intensities_rings[hkey] = term
            elif calc_type == 'spectrum_2d':
                self.intensities_rings_2d[hkey] = term

    def calc_texture_factor(
        self,
        params,
        eta_min=-np.pi,
        eta_max=np.pi,
        eta_step=np.radians(1),
        eta_mask=None,
    ):
        self.calc_pf_rings(
            params,
            eta_min=eta_min,
            eta_max=eta_max,
            eta_step=eta_step,
            calc_type='texture_factor',
        )
        tf = np.zeros(
            [
                len(self.intensities_rings),
            ]
        )
        for ii, (k, v) in enumerate(self.intensities_rings.items()):
            if eta_mask is None:
                tf[ii] = np.mean(v)
            else:
                if k in eta_mask:
                    mask = eta_mask[k]
                else:
                    mask = np.zeros_like(v, dtype=bool)

                if np.all(mask):
                    # The ring is not observed at all. Use the full ring,
                    # rather than a NaN that would poison the spectrum.
                    mask = np.zeros_like(v, dtype=bool)

                tf[ii] = np.mean(v[~mask])
        return tf

    def J(self, params):
        '''this is the texture index for the harmonic
        model. A value of 1 is a random texture and any
        value > 1 is preferred orientation.
        '''
        J = 1.0
        phase = self.material.name
        for ell in range(2, self.ell_max + 1, 2):
            nc = get_num_sym_harm(ell, sym=self.csym)
            ns = get_num_sym_harm(ell, sym=self.ssym)
            for ii in range(ns):
                for jj in range(nc):
                    pname = f'{phase}_c_{ell}_{ii}_{jj}'
                    pre = 1 / (2 * ell + 1)
                    J += pre * (params[pname].value ** 2)
        return J

    def texture_index(self, params: Parameters) -> float:
        """Return the texture index, matching the MarchDollaseModel interface."""
        return self.J(params)

    """
    the following functions deal with everything related to pole figures.
    pole figures can be initialized in a number of ways. the most
    basic being the (x,y,z, intensities) array. There are
    other formats which will be slowly added to this class. a list of
    hkl and a material/material_rietveld class is also supplied along
    with the (angle, intensity) info to get a class holding all the
    information.
    @DATE 10/06/2021 SS changes input from angles to unit vectors
                        and added routines to compute the angles

    Extra Parameters
    ----------------

    hkls: numpy.ndarray
        reciprocal lattice vectors for which pfdata is available
        size is nx3
    pfdata: dict
        dictionary of pole figure (x,y,z,intensity) array. each key
        is the corresponding hkl

    """

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
            data = np.hstack((angs, intensities))
            np.savetxt(fname, data, delimiter="\t")

    def stereographic_radius(self, new=False):
        sr = {}
        angs = self.angs
        if new:
            angs = self.angs_new
        for h, angs in angs.items():
            t = angs[:, 0]
            cth = np.cos(t)
            sth = np.sin(t)
            sr[h] = sth / (1 + np.abs(cth))
        return sr

    def plot_pf(
        self,
        filled=False,
        grid=False,
        cmap='jet',
        colorbar=True,
        colorbar_label='m.r.d.',
        show=True,
        recalculated=False,
    ):
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
        nrows = int(n / 3) + 1
        if nrows == 1:
            ncols = np.min((3, n))
        else:
            ncols = 3

        self.fig, self.ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            subplot_kw={'projection': 'polar'},
            figsize=(12, 4 * nrows),
        )
        self.ax = np.atleast_2d(self.ax)

        [ax.set_axis_off() for ax in self.ax.flatten()]
        [ax.set_yticklabels([]) for ax in self.ax.flatten()]
        [ax.set_xticklabels([]) for ax in self.ax.flatten()]
        [ax.grid(False) for ax in self.ax.flatten()]

        for ii, h in enumerate(self.pfdata):
            nr = int(ii / 3)
            nc = int(np.mod(ii, 3))
            rho = self.angs[h][:, 1]
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
                pf = self.ax[nr][nc].tricontourf(rho, r, I, levels=20, cmap=cmap)
            else:
                pf = self.ax[nr][nc].tricontour(rho, r, I, levels=20, cmap=cmap)
            self.ax[nr][nc].set_yticklabels([])
            self.ax[nr][nc].grid(grid)
            hkl_str = ', '.join(str(int(x)) for x in h)
            self.ax[nr][nc].set_title(f'({hkl_str})')
            plt.colorbar(pf, label=colorbar_label)

        if show:
            self.fig.show()

    @property
    def new_pf_plots_visible(self):
        if not hasattr(self, 'fig_new'):
            return False

        return self.fig_new.canvas.isVisible()

    def plot_new_pf(
        self,
        filled=False,
        grid=False,
        cmap='jet',
        colorbar=True,
        colorbar_label='m.r.d.',
        show=True,
    ):
        '''first get the layout of the subplot'''
        n = self.num_pfs_new
        nrows = int(np.ceil(n / 3))
        if nrows == 1:
            ncols = np.min((3, n))
        else:
            ncols = 3

        self.fig_new, self.ax_new = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            subplot_kw={'projection': 'polar'},
            figsize=(12, 4 * nrows),
            tight_layout=True,
        )
        self.ax_new = np.atleast_2d(self.ax_new)

        title = f'Pole Figures for {self.material.name}'
        self.fig_new.canvas.manager.set_window_title(title)

        [ax.set_axis_off() for ax in self.ax_new.flatten()]
        [ax.set_yticklabels([]) for ax in self.ax_new.flatten()]
        [ax.set_xticklabels([]) for ax in self.ax_new.flatten()]
        [ax.grid(False) for ax in self.ax_new.flatten()]

        self.new_pf_filled = filled
        self.new_pf_cmap = cmap
        self.new_pf_grid = grid
        self.new_pf_colorbar_label = colorbar_label

        self.update_new_pf_plots()

        if show:
            self.fig_new.show()

    def update_new_pf_plot_data(self, params):
        self.calc_new_pfdata(params, self.hkls_new, pfgrid=self.pfgrid_new)
        self.update_new_pf_plots()
        self.fig_new.canvas.draw_idle()

    def update_new_pf_plots(self):
        filled = self.new_pf_filled
        cmap = self.new_pf_cmap
        grid = self.new_pf_grid
        colorbar_label = self.new_pf_colorbar_label
        for ii, h in enumerate(self.angs_new):
            nr = int(ii / 3)
            nc = int(np.mod(ii, 3))
            rho = self.angs_new[h][:, 1]
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

            ax = self.ax_new[nr][nc]
            # Remove any old artists
            if hasattr(ax, '_plt_colorbar'):
                ax._plt_colorbar.remove()

            if hasattr(ax, '_plt_pf'):
                ax._plt_pf.remove()

            func = ax.tricontourf if filled else ax.tricontour
            pf = func(rho, r, I, levels=20, cmap=cmap)
            ax._plt_pf = pf

            ax.set_yticklabels([])
            ax.grid(grid)
            hkl_str = ', '.join(str(int(x)) for x in h)
            ax.set_title(f'({hkl_str})')
            ax._plt_colorbar = self.fig_new.colorbar(pf, label=colorbar_label)

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

            for ell in range(2, self.ell_max + 1, 2):
                pre = 4 * np.pi / (2 * ell + 1)
                cmat = self.get_c_matrix(params, ell)
                sph_s_mat = self.get_sph_s_matrix(h, ell)
                sph_c_mat = self.get_sph_c_matrix(h, ell)
                t = pre * np.dot(sph_s_mat, np.dot(cmat, sph_c_mat))
                term += np.squeeze(t)

            calculated = np.concatenate((calculated, term))

        residual = np.sqrt(self.weights * (measured - calculated) ** 2)
        if np.all(np.isnan(residual)):
            return 1e10
        return residual[~np.isnan(residual)]

    def recalculated_pf(self, params, new=False):
        if not new:
            self.intensities_recalc = {}

            for h, v in self.intensities.items():
                term = np.ones_like(v)

                for ell in range(2, self.ell_max + 1, 2):
                    pre = 4 * np.pi / (2 * ell + 1)
                    cmat = self.get_c_matrix(params, ell)
                    sph_s_mat = self.get_sph_s_matrix(h, ell)
                    sph_c_mat = self.get_sph_c_matrix(h, ell)
                    t = pre * np.dot(sph_s_mat, np.dot(cmat, sph_c_mat))
                    term += np.squeeze(t)

                self.intensities_recalc[h] = term
        else:
            self.intensities_new = {}

            for h, v in self.angs_new.items():
                term = np.ones_like(v[:, 0])

                for ell in range(2, self.ell_max + 1, 2):
                    pre = 4 * np.pi / (2 * ell + 1)
                    cmat = self.get_c_matrix(params, ell)
                    sph_s_mat = self.get_sph_s_matrix(h, ell, gridtype='new')
                    sph_c_mat = self.get_sph_c_matrix(h, ell, gridtype='new')
                    t = pre * np.dot(sph_s_mat, np.dot(cmat, sph_c_mat))
                    term += np.squeeze(t)

                self.intensities_new[h] = term

    def calc_new_pole_figure(self, params, hkls=None, pfgrid=None, plot=False):
        '''calculate pole figure for new poles
        given by hkls. the data will be computed
        on the same grid as the input grid. this
        is done so that the spherical harmonics
        don't have to be recomputed.

        by default, we will pick the pf grid 5 degrees
        in theta and 10 degrees in azimuth

        hkls has the shape nx3
        '''
        if hkls is None:
            hkls = np.array(list(self.pfdata))

        hkls = np.atleast_2d(hkls)

        self.hkls_new = hkls
        self.pfgrid_new = pfgrid
        self.num_pfs_new = hkls.shape[0]
        self.hkls_c_new = self.convert_hkls_to_cartesian(hkls)
        self.calc_new_pfdata(params, hkls, pfgrid=pfgrid)

        if plot:
            self.plot_new_pf(
                filled=True,
                grid=False,
                cmap='jet',
                colorbar=True,
                colorbar_label='m.r.d.',
                show=True,
            )

    def calc_new_pfdata(self, params, hkls, pfgrid=None):
        '''this routine computes the new pfdata which will
        will be used in computing pole figures for a new hkl
        pole
        '''

        # @TODO make the naming convention same as calc_pf_rings
        # currently the rotated angs are used for calculating the
        # values of the spherical harmonics, which is not the case
        # with the other function

        self.intensities_new = {}
        self.angs_new = {}
        self.rotated_angs_new = {}
        self.hkl_angles_new = {}

        '''also pre-compute the spherical harmonics
        '''
        self.sph_c_new = {}
        self.sph_s_new = {}

        '''get the densest grid and associated data
        if the user does not specify a grid of points

        otherwise use the grid data provided by the
        user to generate the angs and the values of the
        spherical harmonics

        '''
        if pfgrid is None:
            t = np.radians(np.linspace(0, 90, 19))
            e = np.radians(np.linspace(-180, 180, 73))
            tt, ee = np.meshgrid(t, e)

            tt = tt.flatten()
            ee = ee.flatten()

            ctt = np.cos(tt)
            stt = np.sin(tt)

            cee = np.cos(ee)
            see = np.sin(ee)

            v = np.vstack((cee * stt, see * stt, ctt)).T

            vr = np.dot(self.ref_frame_rmat, v.T).T

            tr = np.arccos(vr[:, 2])
            rhor = np.arctan2(vr[:, 1], vr[:, 0])

            angs = np.vstack((tt, ee)).T

            rotated_angs = np.vstack((tr, rhor)).T

        else:
            norm = np.linalg.norm(pfgrid, axis=1)
            v = pfgrid / np.tile(norm, [3, 1]).T

            # sanitize v[:, 2] for arccos operation
            mask = np.abs(v[:, 2]) > 1.0
            v[mask, 2] = np.sign(v[mask, 2])

            t = np.arccos(v[:, 2])
            rho = np.arctan2(v[:, 1], v[:, 0])

            vr = np.dot(self.ref_frame_rmat, v[:, 0:3].T).T

            # sanitize vr[:, 2] for arccos operation
            mask = np.abs(vr[:, 2]) > 1.0
            vr[mask, 2] = np.sign(vr[mask, 2])

            tr = np.arccos(vr[:, 2])
            rhor = np.arctan2(vr[:, 1], vr[:, 0])

            angs = np.vstack((t, rho)).T
            rotated_angs = np.vstack((tr, rhor)).T

        hkls_c = self.convert_hkls_to_cartesian(hkls)

        for ii, h in enumerate(hkls):
            hkey = tuple(h)
            self.angs_new[hkey] = angs.copy()
            self.rotated_angs_new[hkey] = rotated_angs.copy()
            self.stereo_radius_new = self.stereographic_radius(new=True)
            self.hkl_angles_new[hkey] = np.array(
                [
                    np.arccos(hkls_c[ii, 2]),
                    np.arctan2(hkls_c[ii, 1], hkls_c[ii, 0]),
                ]
            )

            v = self.hkl_angles_new[hkey]
            '''needs special attention for monoclininc case
                since the 2-fold axis is aligned with b*
            '''
            theta = np.array([v[0]])
            phi = np.array([v[1]])
            '''is the requested data on a new grid?
            '''
            self.sph_c_new[hkey] = {}

            for ell in range(2, self.ell_max + 1, 2):
                Ylm = calc_sym_sph_harm(ell, theta, phi, sym=self.csym)
                for jj in range(Ylm.shape[1]):
                    kname = f'c_{ell}_{jj}'
                    self.sph_c_new[hkey][kname] = Ylm[:, jj]

            self.sph_s_new[hkey] = {}
            theta_samp = self.rotated_angs_new[hkey][:, 0]
            phi_samp = self.rotated_angs_new[hkey][:, 1]

            for ell in range(2, self.ell_max + 1, 2):
                Ylm = calc_sym_sph_harm(ell, theta_samp, phi_samp, sym=self.ssym)
                for jj in range(Ylm.shape[1]):
                    kname = f's_{ell}_{jj}'
                    self.sph_s_new[hkey][kname] = Ylm[:, jj]

        self.recalculated_pf(params, new=True)

    def calculate_harmonic_coefficients(self, params, hkls=None):
        '''precompute the spherical harmonics for
        the given hkls and sample directions
        '''
        if hkls is None:
            # If no hkls are specified, use all of them
            hkls = np.array(list(self.pfdata))

        # Make a copy of the parameters to modify. We want to disable `vary`
        # for all parameters other than the texture parameters, because
        # varying other parameters will make no sense and may produce bad
        # results.
        harmonic_params = Parameters()
        for name in self.parameter_names:
            harmonic_params[name] = copy.deepcopy(params[name])

        if not any(param.vary for param in harmonic_params.values()):
            # No parameters are marked as vary. Return early.
            return None

        self.sph_c = {}
        self.sph_s = {}

        hkls_c = self.convert_hkls_to_cartesian(hkls)
        hkl_angles = {}
        for ii, hkl in enumerate(hkls):
            k = tuple(hkl)
            hkl_angles[k] = np.array(
                [
                    np.arccos(hkls_c[ii, 2]),
                    np.arctan2(hkls_c[ii, 1], hkls_c[ii, 0]),
                ]
            )

        for ii, (h, v) in enumerate(hkl_angles.items()):
            '''needs special attention for monoclininc case
            since the 2-fold axis is aligned with b*
            '''
            theta = np.array([v[0]])
            phi = np.array([v[1]])
            self.sph_c[h] = {}
            self.sph_s[h] = {}
            for ell in range(2, self.ell_max + 1, 2):
                Ylm = calc_sym_sph_harm(ell, theta, phi, sym=self.csym)
                for jj in range(Ylm.shape[1]):
                    kname = f'c_{ell}_{jj}'
                    self.sph_c[h][kname] = Ylm[:, jj]

            '''use the rotated angles to account for the
            misalignment between sample normal and beam
            propagation vector
            '''
            theta = self.rotated_angs[h][:, 0]
            phi = self.rotated_angs[h][:, 1]
            for ell in range(2, self.ell_max + 1, 2):
                Ylm = calc_sym_sph_harm(ell, theta, phi, sym=self.ssym)
                for jj in range(Ylm.shape[1]):
                    kname = f's_{ell}_{jj}'
                    self.sph_s[h][kname] = Ylm[:, jj]

        fdict = {
            'ftol': 1e-6,
            'xtol': 1e-6,
            'gtol': 1e-6,
            'verbose': 2,
            'max_nfev': 20000,
            'method': 'trf',
            'jac': '3-point',
        }

        fitter = Minimizer(self.calc_residual, harmonic_params)
        results = fitter.least_squares(**fdict)

        # Update parameters with the new values
        for param_name, param in results.params.items():
            if not param.vary:
                continue

            params[param_name].value = param.value
            params[param_name].stderr = param.stderr

        return results

    @property
    def num_pfs(self):
        """number of pole figures (read only)"""
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
        if not val:
            return

        for ii, (k, v) in enumerate(val.items()):
            norm = np.linalg.norm(v[:, 0:3], axis=1)
            v[:, 0:3] = v[:, 0:3] / np.tile(norm, [3, 1]).T

            # sanitize v[:, 2] for arccos operation
            mask = np.abs(v[:, 2]) > 1.0
            v[mask, 2] = np.sign(v[mask, 2])

            t = np.arccos(v[:, 2])
            rho = np.arctan2(v[:, 1], v[:, 0])

            vr = np.dot(self.ref_frame_rmat, v[:, 0:3].T).T

            # sanitize vr[:, 2] for arccos operation
            mask = np.abs(vr[:, 2]) > 1.0
            vr[mask, 2] = np.sign(vr[mask, 2])

            tr = np.arccos(vr[:, 2])
            rhor = np.arctan2(vr[:, 1], vr[:, 0])

            self._gvecs[k] = v[:, 0:3]
            self._pfdata[k] = v
            self._intensities[k] = v[:, 3]
            self._weights = np.concatenate((self._weights, v[:, 3]))
            self._angs[k] = np.vstack((t, rho)).T
            self._rotated_angs[k] = np.vstack((tr, rhor)).T
            self._stereo_radius = self.stereographic_radius()
        self._weights = 1 / self._weights
        self._weights = np.nan_to_num(self._weights)
        mask = np.abs(self._weights) > 1e2
        self._weights[mask] = 0.0

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
    def densest_grid(self):
        sz = 0
        dg = ''
        for h, v in self.pfdata.items():
            if v.shape[0] > sz:
                dg = h
                sz = v.shape[0]
        return dg


class PhaseTransitionTextureModel:
    """
    this is a special case of dynamic compression of
    single crystals. in this case, the texture is more
    or less known if the phase transition OR is known.
    is such cases, we don't need to refine the texture.
    we can simply provide the phase transition OR and the
    starting orientation of the parent phase to get a good
    starting point for the expected texture.

    In this case, we will compute the ODF based on the
    radial distribution functions. the refinable parameters
    will only be the starting orientation of the parent
    crystal(s), the FWHM of the De la Vallé Poussin kernel,
    and the relative variant fractions of the possible
    orientation variants

    this is quite a non-standard technique and I have not seen
    this used in the literature, so we are breaking new ground.

    one important point to note is that there is no sample symmetry
    here because that symmetry is statistical in nature and we
    are working with single/few crystal orientations. By default
    the sample symmetry will be "ci" or just the inversion symmetry.

    we will retain the option to specify the sample symmetry in case
    of phase transitions with textured starting samples.

    Parameters
    ----------
    OR: orientation_relationship class
        helper class for defining orientation relationship
        between parent and daughter phases. has information
        about the materials
    params: lmfit.Parameters
        Parameter class for the refinement
    bvec: numpy.ndarray
        unit vector of xray beam in the lab frame. default value
        is -Z direction
    evec: numpy.ndarray
        direction which defines zero azimuth. default value is
        x direction
    sample_symmetry: str
        sample symmetry for the starting sample. default case for
        this model is "ci" because the sample is single crystal but
        will maintain flexibility to specify this for textured
        starting samples

    """

    def __init__(
        self,
        OR: OrientationRelationship,
        params: Parameters,
        bvec: np.ndarray = bvec_ref,
        evec: np.ndarray = eta_ref,
        sample_symmetry: str = "ci",
    ):

        self.OR = OR
        self.beam_vector = bvec
        self.eta_vector = evec
        self.sample_symmetry = sample_symmetry

        self.parent_kernel_width = float(
            params[f"{OR.material_parent.name}_kernel_width"].value
        )
        self.daughter_kernel_width = float(
            params[f"{OR.material_daughter.name}_kernel_width"].value
        )
        # small misorientation of the parent from the assumed orientation
        # (zero if the parameters are not given)
        self.parent_misorientation = np.zeros(3)
        self.parent_misorientation = self.get_misorientation_angles_from_parameters(
            params
        )

        self.init_kernels()
        self.compute_odfs()

        # one view per phase. each view exposes the HarmonicModel
        # interface for a single phase so the views can be passed to
        # the Rietveld class as its texture_model dictionary
        self._views = {
            'parent': PhaseTransitionTextureView(self, 'parent'),
            'daughter': PhaseTransitionTextureView(self, 'daughter'),
        }

    # the parent misorientation is a rotation vector (exponential map)
    # in the sample frame, in degrees: rotation by |v| about v/|v|. unlike
    # Euler angles, it is regular at the identity, so all three components
    # can be refined starting from zero misorientation (with zxz Euler
    # angles at zero the first and last angle are the same rotation and
    # there is no rotation about y, which makes the fit singular).
    MISORIENTATION_ROOT = "parent_misorientation_"
    MISORIENTATION_COMPONENTS = ('x', 'y', 'z')

    @property
    def misorientation_parameter_names(self) -> list:
        '''names of the three parent misorientation parameters'''
        root = self.MISORIENTATION_ROOT
        return [f"{root}{d}" for d in self.MISORIENTATION_COMPONENTS]

    def get_misorientation_angles_from_parameters(self, params):
        '''rotation vector (degrees, sample frame) of the parent
        misorientation from params. components missing from params keep
        their current value (zero when the model is created)
        '''
        current = self.parent_misorientation
        return np.array(
            [
                float(params[name].value) if name in params else current[ii]
                for ii, name in enumerate(self.misorientation_parameter_names)
            ]
        )

    @property
    def parent_misorientation(self) -> np.ndarray:
        '''rotation vector of the parent misorientation in degrees'''
        return self._parent_misorientation

    @parent_misorientation.setter
    def parent_misorientation(self, val):
        val = np.array(val, dtype=float)
        if val.shape != (3,):
            msg = 'parent_misorientation must have 3 components'
            raise ValueError(msg)
        self._parent_misorientation = val

    @property
    def parent_misorientation_rmat(self) -> np.ndarray:
        '''rotation matrix of the parent misorientation. it multiplies the
        parent and the daughter orientations from the left, i.e. it is a
        rotation in the sample frame'''
        return rotMatOfExpMap(np.radians(self.parent_misorientation))

    @property
    def texture_models(self) -> dict:
        '''per-phase models with the same interface as HarmonicModel,
        keyed by material name. intended to be passed directly to the
        Rietveld class, i.e.

            Rietveld(..., texture_model=model.texture_models)

        the views share the kernels and the ODFs of this model, so a
        kernel width updated through one of them is seen by all.
        '''
        return {view.material.name: view for view in self._views.values()}

    @property
    def parameter_names(self) -> list:
        '''names of the refinable parameters of both phases'''
        names = []
        for view in self._views.values():
            names.extend(view.parameter_names)
        return names

    def get_parameters(self, params=None, vary=True):
        '''add the refinable parameters of both phases to params
        (a new lmfit.Parameters if None) and return it
        '''
        if params is None:
            params = Parameters()

        for view in self._views.values():
            view.get_parameters(params, vary=vary)

        return params

    def _material(self, key):
        '''parent or daughter material'''
        if key == 'parent':
            return self.OR.material_parent
        elif key == 'daughter':
            return self.OR.material_daughter

        msg = f'unknown phase "{key}", must be "parent" or "daughter"'
        raise ValueError(msg)

    def _phase_ring_pole_densities(self, key, eta_grid):
        '''pole density (m.r.d.) of every hkl of one phase ("parent" or
        "daughter") on its debye-scherrer ring sampled at eta_grid.
        same computation as the inner loop of calc_pf_rings, but for a
        single phase and without touching the kernel widths.

        Returns
        -------
        intensities: dict
            {tuple(hkl): array over eta_grid}, NaN for hkls that are not
            accessible at the wavelength
        gvecs: dict
            {tuple(hkl): (n_eta, 3) unit scattering vectors}
        '''
        mat = self._material(key)
        odf = self.odfs[key]

        intensities = {}
        gvecs = {}

        tth = np.radians(mat.getTTh(mat.wavelength))
        for t, h in zip(tth, mat.hkls):
            hkey = tuple(h)

            if np.isnan(t):
                # reflection not accessible at this wavelength
                intensities[hkey] = np.full(eta_grid.shape, np.nan)
                gvecs[hkey] = np.full((eta_grid.shape[0], 3), np.nan)
                continue

            angs = np.vstack(
                (
                    np.full(eta_grid.shape, t),
                    eta_grid,
                    np.zeros_like(eta_grid),
                )
            ).T

            # unit scattering vectors on the ring in the sample frame
            pfgrid = angles_to_gvec(
                angs,
                beam_vec=self.beam_vector,
                eta_vec=self.eta_vector,
            )

            hc = mat.TransSpace(h, 'r', 'c')
            intensities[hkey] = np.atleast_1d(
                odf.pole_density(hc, pfgrid, antipodal=True)
            )
            gvecs[hkey] = pfgrid

        return intensities, gvecs

    def init_kernels(self):
        self.parent_kernel = kernels.DeLaValleePoussinKernel(
            np.radians(self.parent_kernel_width),
            crystal_symmetry=self.OR.material_parent.laueGroup,
            sample_symmetry=self.sample_symmetry,
        )

        self.daughter_kernel = kernels.DeLaValleePoussinKernel(
            np.radians(self.daughter_kernel_width),
            crystal_symmetry=self.OR.material_daughter.laueGroup,
            sample_symmetry=self.sample_symmetry,
        )

    def compute_odfs(self):
        self.odfs = dict.fromkeys(["parent", "daughter"])
        prmat = self.parent_misorientation_rmat
        rmat_parent = [prmat @ r for r in self.OR.rmat_parent]
        self.odfs["parent"] = unimodal_odf.UnimodalODF(
            rmat_parent,
            self.parent_kernel,
        )

        rmat_variants = [prmat @ r for r in self.OR.rmat_variants_rotated]
        self.odfs["daughter"] = unimodal_odf.UnimodalODF(
            rmat_variants,
            self.daughter_kernel,
        )

    def calc_pf_rings(
        self,
        params,
        eta_min=-np.pi,
        eta_max=np.pi,
        eta_step=np.radians(0.1),
        calc_type='texture_factor',
    ):
        '''this function computes the intensity variation
        along the debye-scherrer rings for each hkl in the material.
        the intensity variation around the ring is computed between
        eta_min and eta_max. Default values are (-pi,pi) which gives
        the full ring. eta_step is the angular step size in azimuth.
        Default value is 0.1 degrees for eta_step

        the interface is the same as HarmonicModel.calc_pf_rings. the
        difference is in how the pole density is evaluated: instead of
        summing a truncated spherical harmonic series, the pole figure
        of each (kernel based) ODF is evaluated in closed form through
        the radon transform of the kernel at the scattering vectors
        on the ring.

        since this model describes both the parent and the daughter
        phase, the results are stored as nested dictionaries keyed
        first by the material name and then by the hkl tuple, i.e.

            self.intensities_rings[material.name][tuple(hkl)]

        for calc_type='texture_factor' and self.intensities_rings_2d
        with the same layout for calc_type='spectrum_2d'. values are
        in m.r.d. (multiples of random distribution). the unit
        scattering vectors on each ring (sample frame, which coincides
        with the lab frame here) are stored in self.gvecs_rings or
        self.gvecs_rings_2d with the same layout.

        the kernel widths are read from params using the keys
        "{material.name}_kernel_width" (in degrees) for the parent
        and daughter phases. if a value differs from the one used to
        build the current ODFs, the kernels and the ODFs are rebuilt
        so the pole figures always reflect params. if a key is not
        present in params, the current kernel width is retained.
        '''

        nspec = int((eta_max - eta_min) / eta_step) - 1
        eta_grid = eta_min + eta_step * np.arange(1, nspec + 1)
        self.eta_grid = eta_grid

        if not calc_type in ['texture_factor', 'spectrum_2d']:
            msg = 'unknown type of grid for computing pole figures on rings'
            raise ValueError(msg)

        if calc_type == 'texture_factor':
            self.intensities_rings = {}
            self.gvecs_rings = {}
            intensities = self.intensities_rings
            gvecs = self.gvecs_rings
        elif calc_type == 'spectrum_2d':
            self.intensities_rings_2d = {}
            self.gvecs_rings_2d = {}
            intensities = self.intensities_rings_2d
            gvecs = self.gvecs_rings_2d

        phases = {
            'parent': self.OR.material_parent,
            'daughter': self.OR.material_daughter,
        }

        '''make sure the ODFs are consistent with the kernel widths and
        the parent misorientation in params. only rebuilds if something
        changed since building the kernels is not free
        '''
        self._sync_kernel_widths(params)

        '''loop over both the phases and compute the pole
        density of every hkl on its debye-scherrer ring
        '''
        for key, mat in phases.items():
            odf = self.odfs[key]
            intensities[mat.name] = {}
            gvecs[mat.name] = {}

            hkls = mat.hkls
            tth = np.radians(mat.getTTh(mat.wavelength))

            for t, h in zip(tth, hkls):
                hkey = tuple(h)

                if np.isnan(t):
                    # reflection not accessible at this wavelength
                    intensities[mat.name][hkey] = np.full(eta_grid.shape, np.nan)
                    gvecs[mat.name][hkey] = np.full((eta_grid.shape[0], 3), np.nan)
                    continue

                angs = np.vstack(
                    (
                        np.full(eta_grid.shape, t),
                        eta_grid,
                        np.zeros_like(eta_grid),
                    )
                ).T

                # unit scattering vectors on the ring in the
                # sample frame (omega = chi = 0)
                pfgrid = angles_to_gvec(
                    angs,
                    beam_vec=self.beam_vector,
                    eta_vec=self.eta_vector,
                )

                # pole in the cartesian crystal frame. the
                # normalization is done inside pole_density
                hc = mat.TransSpace(h, 'r', 'c')

                intensities[mat.name][hkey] = np.atleast_1d(
                    odf.pole_density(hc, pfgrid, antipodal=True)
                )
                gvecs[mat.name][hkey] = pfgrid

    def calc_texture_factor(
        self,
        params,
        eta_min=-np.pi,
        eta_max=np.pi,
        eta_step=np.radians(1),
        eta_mask=None,
    ):
        '''compute the texture factor of every hkl of the parent and the
        daughter phase. same interface as HarmonicModel.calc_texture_factor:
        the texture factor of a reflection is the mean of the pole density
        (in m.r.d.) around its debye-scherrer ring between eta_min and
        eta_max, optionally excluding the masked azimuths.

        since this model describes two phases, the return value and the
        mask are keyed by material name

        Parameters
        ----------
        params: lmfit.Parameters
            parameters holding "{material.name}_kernel_width" (degrees)
            for the parent and daughter phases
        eta_min, eta_max, eta_step: float
            azimuthal range and step in radians
        eta_mask: dict, optional
            nested dictionary eta_mask[material.name][tuple(hkl)] of
            boolean arrays the size of the azimuthal grid. True means the
            azimuth is excluded from the average. missing phases or hkls
            are not masked, and a fully masked ring falls back to the full
            ring (same as HarmonicModel) rather than returning NaN

        Returns
        -------
        dict
            texture factors keyed by material name. each value is an array
            with one entry per hkl, in the order of material.hkls. hkls
            not accessible at the wavelength give NaN
        '''
        self.calc_pf_rings(
            params,
            eta_min=eta_min,
            eta_max=eta_max,
            eta_step=eta_step,
            calc_type='texture_factor',
        )

        tf = {}
        for name, rings in self.intensities_rings.items():
            phase_mask = None
            if eta_mask is not None:
                phase_mask = eta_mask.get(name)

            tf[name] = np.zeros(
                [
                    len(rings),
                ]
            )
            for ii, (k, v) in enumerate(rings.items()):
                if phase_mask is None:
                    tf[name][ii] = np.mean(v)
                else:
                    if k in phase_mask:
                        mask = phase_mask[k]
                    else:
                        mask = np.zeros_like(v, dtype=bool)

                    if np.all(mask):
                        # The ring is not observed at all. Use the full ring,
                        # rather than a NaN that would poison the spectrum.
                        mask = np.zeros_like(v, dtype=bool)

                    tf[name][ii] = np.mean(v[~mask])
        return tf

    def J(self, params):
        '''this is the texture index J = <f^2> of the parent and the
        daughter ODFs, the same quantity HarmonicModel.J returns. A value
        of 1 is a random texture and any value > 1 is preferred
        orientation.

        the ODFs are weighted sums of symmetrized de la Vallee Poussin
        kernels, f(g) = sum_i w_i K_sym(g, g_i), with K_sym the average of
        K over the symmetry images s g_i c. J is therefore a double sum of
        overlap integrals of two kernels,

            J = 1/(|S||C|) sum_ij w_i w_j sum_(s,c) P(w(g_i, s g_j c)),

        where P(w) = <K(g, a) K(g, b)> only depends on the misorientation
        angle w between a and b. P is reduced to a 1d periodic integral
        (see _kernel_overlap) and evaluated by quadrature, so J is exact
        to floating point and deterministic (no Monte Carlo noise, which
        matters when J is used during a refinement).

        Parameters
        ----------
        params: lmfit.Parameters
            parameters holding "{material.name}_kernel_width" (degrees).
            the kernels and ODFs are rebuilt if the widths changed

        Returns
        -------
        dict
            texture index keyed by material name
        '''
        self._sync_kernel_widths(params)

        phases = {
            'parent': self.OR.material_parent,
            'daughter': self.OR.material_daughter,
        }
        return {
            mat.name: self._odf_texture_index(self.odfs[key])
            for key, mat in phases.items()
        }

    def texture_index(self, params: Parameters) -> dict:
        """Return the texture index, matching the HarmonicModel interface."""
        return self.J(params)

    def _sync_kernel_widths(self, params):
        '''update the kernel widths and the parent misorientation from
        params and rebuild the kernels and ODFs if anything changed. keys
        that are not in params leave the current value untouched. used
        by calc_pf_rings, J and the per-phase views
        '''
        phases = {
            'parent': self.OR.material_parent,
            'daughter': self.OR.material_daughter,
        }
        rebuild_kernels = False
        for key, mat in phases.items():
            pname = f'{mat.name}_kernel_width'
            attr = f'{key}_kernel_width'
            current = float(getattr(self, attr))
            if pname in params:
                width = float(params[pname].value)
            else:
                width = current
            # exact comparison: the finite difference steps of a fit are
            # ~1e-6 relative, which np.isclose would treat as unchanged
            if width != current:
                rebuild_kernels = True
            setattr(self, attr, width)

        mis = self.get_misorientation_angles_from_parameters(params)
        rebuild_odfs = not np.array_equal(mis, self.parent_misorientation)
        self.parent_misorientation = mis

        if rebuild_kernels:
            self.init_kernels()
        if rebuild_kernels or rebuild_odfs:
            self.compute_odfs()

    @staticmethod
    def _odf_texture_index(odf):
        '''exact texture index of a UnimodalODF built on a de la Vallee
        Poussin kernel (see J for the formula)
        '''
        kernel = odf.kernel
        crystal_ops = kernel.crystal_symmetry_operations
        sample_ops = kernel.sample_symmetry_operations
        g = odf.modal_orientations
        w = odf.weights

        # the J formula needs proper rotations. an improper modal
        # orientation (det = -1, e.g. from the variant calculation) is
        # replaced by -g, which is equivalent for a centrosymmetric (Laue)
        # crystal symmetry and gives identical pole figures
        g = g * np.sign(np.linalg.det(g))[:, None, None]
        # symmetry images s g_j c, shape (n_sample, n_modes, n_crystal, 3, 3)
        images = np.einsum('sab,nbc,kcd->snkad', sample_ops, g, crystal_ops)

        # trace(g_i^T s g_j c) -> cos(w/2) = sqrt((1 + trace) / 4)
        trace = np.einsum('iba,snkba->isnk', g, images)
        cos_half = np.sqrt(np.clip(0.25 * (1.0 + trace), 0.0, 1.0))

        overlap = PhaseTransitionTextureModel._kernel_overlap(
            kernel, cos_half.ravel()
        ).reshape(cos_half.shape)

        J = np.einsum('i,j,isjk->', w, w, overlap)
        return float(J / (sample_ops.shape[0] * crystal_ops.shape[0]))

    @staticmethod
    def _kernel_overlap(kernel, cos_half):
        '''overlap integral P = <K(g, a) K(g, b)> over SO(3) (normalized
        Haar measure) of two de la Vallee Poussin kernels with the same
        shape parameter kappa, centered at a and b. cos_half is
        cos(w/2) = |q_a . q_b| for the misorientation angle w between a
        and b.

        with K(g, a) = C |q . q_a|^(2 kappa) and q uniform on S^3, the
        projection (q . e1, q . e2) is uniformly distributed on the unit
        disk. doing the radial integral analytically leaves

            P = C^2 / (pi (4 kappa + 2))
                * int_0^2pi |(cos(w/2) + cos(psi)) / 2|^(2 kappa) dpsi

        the integrand is smooth and periodic, so the trapezoidal rule
        converges very fast. the grid resolves the peak at psi = 0, whose
        width is ~ 1/sqrt(2 kappa). for w = 0 this reduces to the
        analytic single mode texture index of UnimodalODF.
        '''
        kappa = kernel.kappa
        norm_c = kernel.norm_constant

        width = 1.0 / np.sqrt(2.0 * kappa)
        npts = int(max(2048, np.ceil(16 * 2 * np.pi / width)))
        cos_psi = np.cos(np.linspace(0.0, 2 * np.pi, npts, endpoint=False))

        cos_half = np.atleast_1d(np.asarray(cos_half, dtype=float))
        integral = np.empty(cos_half.shape)

        # chunk to keep the (n_values, npts) work array small
        chunk = max(1, int(2e6 // npts))
        for start in range(0, cos_half.size, chunk):
            cb = cos_half[start : start + chunk, None]
            x = 0.5 * np.abs(cb + cos_psi[None, :])
            with np.errstate(divide='ignore'):
                vals = np.exp(2.0 * kappa * np.log(x))
            integral[start : start + chunk] = 2 * np.pi * vals.mean(axis=1)

        return norm_c**2 / (np.pi * (4.0 * kappa + 2.0)) * integral


class PhaseTransitionTextureView:
    """
    single phase view of a PhaseTransitionTextureModel with the same
    interface as HarmonicModel, so it can be used as a drop-in
    replacement in the Rietveld class.

    the Rietveld class expects one texture model per phase, while a
    PhaseTransitionTextureModel describes both the parent and the
    daughter phase. a view selects one of the two. it does not hold any
    texture state of its own: the kernels, the ODFs and the kernel widths
    live on the parent model and are shared by both views. the only
    per-view state is the pole figure data (pfdata) used by
    calculate_harmonic_coefficients and the ring intensities from the
    last call to calc_pf_rings.

    the views are created by the PhaseTransitionTextureModel and are
    normally obtained through PhaseTransitionTextureModel.texture_models

        rietveld = Rietveld(..., texture_model=model.texture_models)

    the refinable parameter of a phase is its kernel half width,
    "{material.name}_kernel_width", in degrees. the parent view also owns
    the parent misorientation, "parent_misorientation_x/y/z" (rotation
    vector in degrees, sample frame), which rotates both phases.

    Parameters
    ----------
    model: PhaseTransitionTextureModel
        model holding the kernels and the ODFs
    phase: str
        "parent" or "daughter"
    pfdata: dict, optional
        pole figure data, see the pfdata property
    """

    def __init__(self, model, phase, pfdata=None):
        if phase not in ('parent', 'daughter'):
            msg = f'unknown phase "{phase}", must be "parent" or "daughter"'
            raise ValueError(msg)

        self._model = model
        self._phase = phase
        self.pfdata = pfdata

    def __repr__(self):
        return (
            f'{type(self).__name__}(phase={self.phase!r}, '
            f'material={self.material.name!r}, '
            f'kernel_width={self.kernel_width})'
        )

    @property
    def model(self):
        '''the PhaseTransitionTextureModel this view belongs to'''
        return self._model

    @property
    def phase(self):
        '''"parent" or "daughter" (read only)'''
        return self._phase

    @property
    def material(self):
        '''Material_Rietveld of this phase (read only)'''
        return self._model._material(self._phase)

    @property
    def odf(self):
        '''ODF of this phase (read only)'''
        return self._model.odfs[self._phase]

    @property
    def kernel_width(self):
        '''current kernel half width in degrees (read only, set through
        params)'''
        return float(getattr(self._model, f'{self._phase}_kernel_width'))

    @property
    def parameter_name(self):
        '''name of the kernel width parameter of this phase'''
        return f'{self.material.name}_kernel_width'

    @property
    def parameter_names(self):
        '''refinable parameters of this view: the kernel width and, for
        the parent view, the three components of the parent
        misorientation. the misorientation also rotates the daughter
        orientations, but it is owned (added to params and refined) by
        the parent view only, so it is not refined twice by
        Rietveld.RefineTexture'''
        names = [self.parameter_name]
        if self._phase == 'parent':
            names.extend(self._model.misorientation_parameter_names)
        return names

    def get_parameters(self, params=None, vary=True):
        '''
        make lmfit parameter class for refinement. the kernel half width
        is in degrees and starts at the current value of the model. the
        parent view also adds the parent misorientation (rotation vector
        in degrees, sample frame), starting at the current value
        '''
        if params is None:
            params = Parameters()

        params.add(
            self.parameter_name,
            value=self.kernel_width,
            min=0.1,
            max=90.0,
            vary=vary,
        )

        if self._phase == 'parent':
            names = self._model.misorientation_parameter_names
            values = self._model.parent_misorientation
            for name, value in zip(names, values):
                params.add(name, value=float(value), vary=vary)

        return params

    def calc_pf_rings(
        self,
        params,
        eta_min=-np.pi,
        eta_max=np.pi,
        eta_step=np.radians(0.1),
        calc_type='texture_factor',
    ):
        '''this function computes the intensity variation
        along the debye-scherrer rings for each hkl in the material.
        the intensity variation around the ring is computed between
        eta_min and eta_max. Default values are (-pi,pi) which gives
        the full ring. eta_step is the angular step size in azimuth.
        Default value is 0.1 degrees for eta_step

        same interface and output layout as HarmonicModel.calc_pf_rings:
        the results are stored in self.intensities_rings[tuple(hkl)] for
        calc_type='texture_factor' and self.intensities_rings_2d for
        calc_type='spectrum_2d', in m.r.d. hkls that are not accessible at
        the wavelength give NaN. the kernel widths are updated from params
        first (see PhaseTransitionTextureModel.calc_pf_rings).
        '''
        if not calc_type in ['texture_factor', 'spectrum_2d']:
            msg = 'unknown type of grid for computing pole figures on rings'
            raise ValueError(msg)

        nspec = int((eta_max - eta_min) / eta_step) - 1
        eta_grid = eta_min + eta_step * np.arange(1, nspec + 1)
        self.eta_grid = eta_grid

        self._model._sync_kernel_widths(params)
        intensities, gvecs = self._model._phase_ring_pole_densities(
            self._phase, eta_grid
        )

        if calc_type == 'texture_factor':
            self.intensities_rings = intensities
            self.gvecs_rings = gvecs
        elif calc_type == 'spectrum_2d':
            self.intensities_rings_2d = intensities
            self.gvecs_rings_2d = gvecs

    def calc_texture_factor(
        self,
        params,
        eta_min=-np.pi,
        eta_max=np.pi,
        eta_step=np.radians(1),
        eta_mask=None,
    ):
        '''texture factor of every hkl of this phase, i.e. the mean pole
        density around its debye-scherrer ring. same interface and return
        value as HarmonicModel.calc_texture_factor: an array in the order
        of material.hkls. eta_mask is a dictionary {tuple(hkl): boolean
        array} where True excludes the azimuth from the average.
        '''
        self.calc_pf_rings(
            params,
            eta_min=eta_min,
            eta_max=eta_max,
            eta_step=eta_step,
            calc_type='texture_factor',
        )
        tf = np.zeros(
            [
                len(self.intensities_rings),
            ]
        )
        for ii, (k, v) in enumerate(self.intensities_rings.items()):
            if eta_mask is None:
                tf[ii] = np.mean(v)
            else:
                if k in eta_mask:
                    mask = eta_mask[k]
                else:
                    mask = np.zeros_like(v, dtype=bool)

                if np.all(mask):
                    # The ring is not observed at all. Use the full ring,
                    # rather than a NaN that would poison the spectrum.
                    mask = np.zeros_like(v, dtype=bool)

                tf[ii] = np.mean(v[~mask])
        return tf

    def J(self, params):
        '''texture index J = <f^2> of the ODF of this phase. A value of 1
        is a random texture and any value > 1 is preferred orientation.
        see PhaseTransitionTextureModel.J for the method
        '''
        self._model._sync_kernel_widths(params)
        return self._model._odf_texture_index(self.odf)

    def texture_index(self, params: Parameters) -> float:
        """Return the texture index, matching the HarmonicModel interface."""
        return self.J(params)

    def calc_residual(self, params):
        '''weighted difference between the measured pole figure
        intensities (pfdata) and the pole densities of the ODF, with the
        kernel width taken from params. same weighting as
        HarmonicModel.calc_residual
        '''
        self._model._sync_kernel_widths(params)

        hkls = getattr(self, '_fit_hkls', None)
        if hkls is None:
            hkls = list(self.intensities)

        measured = np.empty(0)
        calculated = np.empty(0)
        weights = np.empty(0)
        for h in hkls:
            v = self.intensities[h]
            hc = self.material.TransSpace(np.asarray(h, dtype=float), 'r', 'c')
            term = np.atleast_1d(
                self.odf.pole_density(hc, self.gvecs[h], antipodal=True)
            )
            measured = np.concatenate((measured, v))
            calculated = np.concatenate((calculated, term))
            weights = np.concatenate((weights, self._weights_by_hkl[h]))

        residual = np.sqrt(weights * (measured - calculated) ** 2)
        if np.all(np.isnan(residual)):
            return 1e10
        return residual[~np.isnan(residual)]

    def recalculated_pf(self, params):
        '''pole densities of the ODF at the pfdata points, stored in
        self.intensities_recalc[tuple(hkl)]
        '''
        self._model._sync_kernel_widths(params)
        self.intensities_recalc = {}
        for h, g in self.gvecs.items():
            hc = self.material.TransSpace(np.asarray(h, dtype=float), 'r', 'c')
            self.intensities_recalc[h] = np.atleast_1d(
                self.odf.pole_density(hc, g, antipodal=True)
            )

    def calculate_harmonic_coefficients(self, params, hkls=None):
        '''fit the texture parameters of this view to the pole figure data
        (pfdata). the name and interface are the same as
        HarmonicModel.calculate_harmonic_coefficients so the Rietveld
        class can refine this model through RefineTexture.

        the parameters are those in parameter_names that are present in
        params: the kernel width of this phase and, for the parent view,
        the parent misorientation. only the ones marked as vary are
        refined, against the pfdata of this view. the refined values and
        their standard errors are written back to params. returns the
        lmfit result, or None if no parameter is marked as vary.
        '''
        if hkls is None:
            # If no hkls are specified, use all of them
            hkls = list(self.pfdata)

        fit_params = Parameters()
        for name in self.parameter_names:
            if name in params:
                fit_params[name] = copy.deepcopy(params[name])

        if not any(param.vary for param in fit_params.values()):
            # No parameters are marked as vary. Return early.
            return None

        self._fit_hkls = [tuple(h) for h in hkls]

        fdict = {
            'ftol': 1e-6,
            'xtol': 1e-6,
            'gtol': 1e-6,
            'verbose': 0,
            'max_nfev': 20000,
            'method': 'trf',
            'jac': '3-point',
        }

        try:
            fitter = Minimizer(self.calc_residual, fit_params)
            results = fitter.least_squares(**fdict)

            # Update parameters with the new values
            for param_name, param in results.params.items():
                if not param.vary:
                    continue

                params[param_name].value = param.value
                params[param_name].stderr = param.stderr
        finally:
            # the fit changes the kernel widths of the shared model. leave
            # it consistent with params (refined values on success, the
            # starting values otherwise)
            self._fit_hkls = None
            self._model._sync_kernel_widths(params)

        return results

    @property
    def num_pfs(self):
        """number of pole figures (read only)"""
        return len(self.pfdata)

    """
    the pfdata property returns the pole figure data in the form of a
    dictionary with keys as the hkl tuples and the value as an (n, 4)
    array of (x, y, z, intensity), where (x, y, z) is the scattering
    vector in the lab frame (same as HarmonicModel). this is what
    Rietveld.compute_texture_data sets.
    """

    @property
    def pfdata(self):
        return self._pfdata

    @pfdata.setter
    def pfdata(self, val):
        self._pfdata = {}
        self._intensities = {}
        self._weights = np.empty(0)
        self._weights_by_hkl = {}
        self._gvecs = {}
        self._angs = {}
        if not val:
            return

        for k, v in val.items():
            k = tuple(k)
            v = np.array(v, dtype=float)
            norm = np.linalg.norm(v[:, 0:3], axis=1)
            v[:, 0:3] = v[:, 0:3] / np.tile(norm, [3, 1]).T

            # sanitize v[:, 2] for arccos operation
            mask = np.abs(v[:, 2]) > 1.0
            v[mask, 2] = np.sign(v[mask, 2])

            t = np.arccos(v[:, 2])
            rho = np.arctan2(v[:, 1], v[:, 0])

            # same weights as HarmonicModel: 1/intensity, with
            # non-finite and very large weights set to zero
            with np.errstate(divide='ignore', invalid='ignore'):
                w = 1 / v[:, 3]
            w = np.nan_to_num(w)
            w[np.abs(w) > 1e2] = 0.0

            self._gvecs[k] = v[:, 0:3]
            self._pfdata[k] = v
            self._intensities[k] = v[:, 3]
            self._weights_by_hkl[k] = w
            self._weights = np.concatenate((self._weights, w))
            self._angs[k] = np.vstack((t, rho)).T

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
    def weights(self):
        return self._weights


class InversePoleFigures:
    """
    this class deals with everything related to inverse pole figures.

    """

    def __init__(self, sample_dir, sampling="equiangular", resolution=5.0):
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

    def initialize_crystal_dir(self, samplingtype, resolution=5.0):
        """
        this function prepares the unit vectors
        of the stereogram
        """
        if samplingtype.lower() == "equiangular":
            angs = []
            for tth in range(0, 91, resolution):
                for eta in range(0, 360, resolution):
                    angs.append([np.radians(tth), np.radians(eta)])
                    if tth == 0:
                        break
            angs = np.array(angs)
            self.crystal_dir = np.zeros([angs.shape[0], 3])
            for i, a in enumerate(angs):
                t, r = a
                st = np.sin(t)
                ct = np.cos(t)
                sr = np.sin(r)
                cr = np.cos(r)
                self.crystal_dir[i, :] = np.array([st * cr, st * sr, ct])

        if samplingtype.lower() == "fem":
            msg = "sampling type FEM not implemented yet."
            raise ValueError(msg)

    @property
    def sample_dir(self):
        """sample direction for IPF"""
        return self._sample_dir

    @sample_dir.setter
    def sample_dir(self, val):
        # interpret sample_dir input
        # sample_dir size = nx3
        if isinstance(val, str):
            if val.upper() == "RD":
                self._sample_dir = np.atleast_2d([1.0, 0.0, 0.0])
                self._sample_dir_name = "RD"
            elif val.upper() == "TD":
                self._sample_dir = np.atleast_2d([0.0, 1.0, 0.0])
                self._sample_dir_name = "TD"
            elif val.upper() == "ND":
                self._sample_dir = np.atleast_2d([0.0, 0.0, 1.0])
                self._sample_dir_name = "ND"
            else:
                msg = "unknown direction."
                raise ValueError(msg)
        elif isinstance(val, np.array):
            v = np.atleast_2d(val)
            if v.shape[1] != 3:
                msg = (
                    f"incorrect shape for sample_dir input.\n"
                    f"expected nx3, got {val.shape[0]}x{val.shape[1]}"
                )
                raise ValueError(msg)
            self._sample_dir = v
            self._sample_dir_name = "array"

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        if val < 1.0:
            msg = (
                "the resolution appears to be very fine.\n"
                "Are you sure the value is in degrees?"
            )
            warn(msg)
        self._resolution = val

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, val):
        if val.lower() == "equiangular":
            self.initialize_crystal_dir("equiangular", resolution=self.resolution)

        elif val.lower() == "fem":
            self.initialize_crystal_dir("FEM")

    @property
    def angs(self):
        polar = np.arccos(self.crystal_dir[:, 2])
        az = np.arctan2(self.crystal_dir[:, 1], self.crystal_dir[:, 0])
        return np.degrees(np.vstack((polar, az)).T)


# These are here only for backward-compatibility
harmonic_model = HarmonicModel
inverse_polar_figures = InversePoleFigures


@numba.njit(nogil=True, cache=True)
def _calc_ang_rings(pfgrid, t, hc, ref_frame_rmat, eta_grid):
    norm = np.sqrt(np.sum(pfgrid**2, axis=1))
    v = pfgrid / norm.repeat(3).reshape((3, -1)).T
    v = np.dot(ref_frame_rmat, v.T).T

    # sanitize v[:, 2] for arccos operation
    mask = np.abs(v[:, 2]) > 1.0
    v[mask, 2] = np.sign(v[mask, 2])

    tt = np.arccos(v[:, 2])
    rho = np.arctan2(v[:, 1], v[:, 0])

    ang_rings = np.vstack((tt, rho)).T

    # in the beam frame
    rotated_ang_rings = np.vstack(
        (
            np.pi / 2 - t / 2 * np.ones_like(eta_grid),
            eta_grid,
        )
    ).T

    hkl_ang_rings = np.array([np.arccos(hc[2]), np.arctan2(hc[1], hc[0])])

    return ang_rings, rotated_ang_rings, hkl_ang_rings
