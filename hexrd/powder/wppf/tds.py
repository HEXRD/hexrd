import numpy as np
import warnings
from hexrd.powder.wppf.phase import (
    Phases_Rietveld,
    Material_Rietveld,
)
from hexrd.wppf.xtal import _calcxrayformfactor, _calcanomalousformfactor
from hexrd.constants import (
    ptableinverse,
    fNT,
    frel,
    scatfac,
    cPlanck,
    cBoltzmann,
    cAvogadro,
    ATOM_WEIGHTS_DICT,
)
from scipy.ndimage import gaussian_filter1d

TDS_MODEL_TYPES = ['warren', 'experimental']


def check_model_type(model):
    if not model in TDS_MODEL_TYPES:
        msg = f'unknown TDS model type'
        raise ValueError(msg)


class TDS_material:
    '''
    >> @AUTHOR:     Saransh Singh,
                    Lawrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       01/09/2026 SS 1.0 original
    >> @DETAILS:    thermal diffuse scattering class can be used
                    to fit the broad, interpeak diffuse signal
                    in the Rietveld refinement. The primary purpose is to
                    extract temperature and also account for the diffuse signal
                    when extracting incipient melt phase fraction. Currently,
                    the models supported are experimentally extracted diffuse
                    signal which can be scaled and shifted, or a theoretical
                    model detailed in Warren's X-Ray diffraction textbook. The
                    theoretical model is only given for FCC and BCC crystals, so
                    only those space groups will be supported for now. Further
                    development is needed to get the correct models for other
                    symmetries.


    Attributes
    ----------
    model_type : str
        allowed model types are "warren" for the model described in
        Chapter 11, B.E. Warren, X-Ray Diffraction, Dover publication (1969).
        Additionally, the TDS signal can also specified as a text file which
        can be imported. This is the "experimental" model type.
        The experimental data can be shifted and scaled

    material : hexrd.wppf.Material_Rietveld
        rietveld material class

    wavelength : float
        wavelength of x-rays in Angstrom

    theta_D: float
        Debye temperature for this phase. If the model type is
        "warren" then this is a user input and should not be refined.
        Otherwise, this is not required

    model_data: numpy.ndarray
        if the "experimental" model type is used, then model_data
        is a numpy array containing the 2theta-intensity of the
        measured TDS signal. the signal in model_data will
        be shifted and scaled to get the best fit with the observed
        data.

    scale: if model is "experimental", then this quantifies the
        scale factor. otherwise, not present

    shift: if model is "experimental", then this quantifies the
        shift in 2theta. otherwise, not present

    smoothing: if model is "experimental", then this specifies how
        much (if any) gaussian smoothing to apply to the lineout
    '''

    def __init__(
        self,
        model_type="warren",
        material=None,
        wavelength=None,
        theta_D=None,
        tth=None,
        model_data=None,
        scale=None,
        shift=None,
        smoothing=None,
    ):
        check_model_type(model_type)
        self.model_type = model_type

        if self.model_type == 'warren':
            if material is None:
                msg = f'material has to be specified for warren model'
                raise ValueError(msg)
            if not isinstance(material, Material_Rietveld):
                msg = f'specify material as Material_Rietveld class'
                raise ValueError(msg)
            if not material.sgnum in [225, 229]:
                msg = f'only FCC and BCC crystals are supported at the moment.'
                raise ValueError(msg)
            self.mat = material

            if wavelength is None:
                msg = f'wavelength has to be specified for warren model'
                raise ValueError(msg)
            self._wavelength = wavelength

            if theta_D is not None:
                self._that_D = theta_D
            else:
                self._theta_D = np.nan

            if tth is None:
                msg = f'tth has to be specified for warren model'
                raise ValueError(msg)
            self.tth = tth

        if self.model_type == 'experimental':
            if model_data is None or not isinstance(model_type, np.ndarray):
                msg = f'model_data needs to be nx2 numpy array'
                raise ValueError(msg)
            self.model_data = model_data

            scalec = scale
            if scale is None:
                scalec = 1.0
            self.scale = scalec

            shiftc = shift
            if shift is None:
                shiftc = 0.0
            self.shift = shiftc

        self.smoothing = smoothing

    def WarrenFunctionalForm(self, x, xhkl):
        xx = np.abs(x - xhkl)
        xx = self.agm / xx
        mask = xx > 1.0
        y = np.log(xx)
        y[~mask] = 0.0
        return y

    def formfactor(self, q):
        s = (q / 4 / np.pi) ** 2

        formfact = {}
        f_anomalous_data = self.mat.f_anomalous_data
        f_anomalous_data_sizes = self.mat.f_anomalous_data_sizes

        scattering_factors = np.zeros([self.mat.atom_ntype, 11])
        for i, a in enumerate(self.mat.atom_type):
            k = ptableinverse[a]
            scattering_factors[i, :] = scatfac[k]
            k = ptableinverse[a]
            fNT_k = np.array([fNT[k]])
            frel_k = np.array([frel[k]])

        formfact = {}
        for i, a in enumerate(self.mat.atom_type):
            k = ptableinverse[a]
            formfact[k] = np.zeros_like(s)
            for ii, ss in enumerate(s):
                formfact[k][ii] = (
                    np.squeeze(
                        np.abs(
                            _calcxrayformfactor(
                                self.mat.wavelength,
                                ss,
                                np.array([a]),
                                scattering_factors,
                                fNT_k,
                                frel_k,
                                f_anomalous_data,
                                f_anomalous_data_sizes,
                            )
                        )
                    )
                    ** 2
                )
        return formfact

    def get_TDS_contribution_hkl(self, g, j, q):
        """texture factor for random powder crystal"""
        glen = self.mat.CalcLength(g, "r")
        # the factor of 10 cancels out here
        xhkl = glen * self.mat.lparms[0]
        # factor of 10 in lparms for nm --> A
        x = 10 * self.mat.lparms[0] * q / np.pi / 2
        if np.isclose(glen, 0):
            C = np.zeros_like(x)
            C = self.agm**2 / 3 / x**2
            # mask = np.abs(x) < self.agm
            # C[mask] = self.agm**2 / 3 / x[mask] ** 2
        else:
            pre = self.agm**2 * (j / xhkl) / (6 * x)
            C = pre * self.WarrenFunctionalForm(x, xhkl)
        return C

    def calcTDS(self):
        thr = np.radians(self.tth * 0.5)

        q = self.q
        s = q / 4 / np.pi

        sth = np.sin(thr)
        cth = np.cos(thr)
        c2th = np.cos(2 * thr)

        formfact = self.formfactor(q)

        M = self.Mdict
        exp2M = dict.fromkeys(M.keys())

        for k in M.keys():
            mass = ATOM_WEIGHTS_DICT[k]
            exp2M[k] = np.exp(-2 * M[k])

        # qb = self.get_qb()

        hkl = self.mat.hkls
        multiplicity = self.mat.multiplicity

        C = np.zeros_like(q)
        # C = self.get_TDS_contribution_hkl(np.array([0, 0, 0]), 1, q)
        for g, j in zip(hkl, multiplicity):
            C += self.get_TDS_contribution_hkl(g, j, q)

        thermal_diffuse = np.zeros_like(self.tth)
        for k, v in formfact.items():
            thermal_diffuse = v * (
                (1 - exp2M[k]) + exp2M[k] * (2 * M[k] + M[k] ** 2) * (C - 1)
            )

        return thermal_diffuse

    @property
    def Mdict(self):
        self._Mdict = {}
        if not self.mat.aniU:
            for U, a in zip(self.mat.U, self.mat.atom_type):
                key = ptableinverse[a]
                self._Mdict[key] = 8.0 * np.pi**2 * U
        else:
            msg = (
                f'at least one atom has an anisotropic debye-waller factor.'
                f' That model is in development'
            )
            raise ValueError(msg)
        return self._Mdict

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def theta_D(self):
        return self._theta_D

    @property
    def tth(self):
        return self._tth

    @tth.setter
    def tth(self, ttharray):
        self._tth = ttharray
        thr = np.radians(ttharray * 0.5)
        self._q = 4 * np.pi * np.sin(thr) / self.wavelength

    @property
    def agm(self):
        # value for FCC crystal
        self._agm = (3 / np.pi) ** (1.0 / 3.0)
        if self.mat.sgnum == 229:
            # handle the BCC case
            self._agm = (3 / np.pi / 2) ** (1.0 / 3.0)
        return self._agm

    @property
    def tds_lineout(self):
        lo = self.calcTDS()
        if self.smoothing is not None:
            lo = gaussian_filter1d(lo, self.smoothing)
        return lo

    @property
    def q(self):
        return self._q


class TDS:
    '''
    >> @AUTHOR:     Saransh Singh,
                    Lawrence Livermore National Lab,
                    saransh1@llnl.gov
    >> @DATE:       01/09/2026 SS 1.0 original
    >> @DETAILS:    this is just a wrapper class for the TDS_amterials
                    to loop over all materials in the phase and wavelength
                    and compute the overall signal
    '''

    def __init__(
        self,
        model_type="warren",
        phases=None,
        theta_D_dict=None,
        tth=None,
        model_data=None,
        scale=None,
        shift=None,
        cagliotti_uvw=None,
    ):
        self.model_type = model_type
        self.theta_D = theta_D_dict
        self.tth = tth
        self.cagliotti = cagliotti_uvw
        self.model_data = model_data
        self.scale = scale
        self.shift = shift
        self.phases = phases

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, phases):
        if not isinstance(phases, Phases_Rietveld):
            msg = f'input phase as Phases_Rietveld class'
            raise ValueError(msg)
        self._phases = phases
        self.TDSmodels = {}
        kwargs = {
            "model_type": "warren",
            "tth": self.tth,
            "model_data": self.model_data,
            "scale": self.scale,
            "shift": self.shift,
            "smoothing": self.smoothing,
        }
        for pname in self.phases:
            self.TDSmodels[pname] = {}
            for wavn, lam in self.phases.wavelength.items():
                matr = self.phases[pname][wavn]
                kwargs = {
                    **kwargs,
                    "material": matr,
                    "wavelength": 10 * lam[0].getVal("nm"),
                    "theta_D": self.theta_D[pname],
                }
                self.TDSmodels[pname][wavn] = TDS_material(**kwargs)

    @property
    def tds_lineout(self):
        lineout = np.zeros_like(self.tth)
        for p in self.phases:
            for l in self.phases.wavelength:
                weight = self.phases.wavelength[l][1]
                lineout += weight * self.TDSmodels[p][l].tds_lineout

        return lineout

    @property
    def cagliotti(self):
        return self._cagliotti

    @cagliotti.setter
    def cagliotti(self, uvw):
        '''this number defines the smoothing
        term for the tds model. the nstrumental
        broadening will affect tds just like the
        elastic scattering signal. we use the
        averageinstrumental broadening across the
        field of view as the first approximation
        '''
        self._cagliotti = uvw
        th = np.radians(0.5 * self.tth)
        tanth = np.tan(th)
        sigsqr = uvw[0] * tanth**2 + uvw[1] * tanth + uvw[2]
        sigsqr = np.mean(sigsqr)
        if sigsqr <= 0.0:
            sigsqr = 1.0e-12

        self.instrumental_broadening = np.sqrt(sigsqr) * 1e-2

        tth_step = self.tth[1] - self.tth[0]
        if not np.isclose(tth_step, 0.0):
            self.smoothing = int(np.rint(self.instrumental_broadening / tth_step))
        else:
            self.smoothing = 1
