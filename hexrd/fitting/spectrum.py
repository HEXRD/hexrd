import numpy as np
from numpy.polynomial import chebyshev

from lmfit import Model, Parameters

from hexrd.constants import fwhm_to_sigma
from hexrd.imageutil import snip1d

from .utils import (_calc_alpha, _calc_beta,
                    _mixing_factor_pv,
                    _gaussian_pink_beam,
                    _lorentzian_pink_beam,
                    _parameter_arg_constructor,
                    _extract_parameters_by_name,
                    _set_bound_constraints,
                    _set_refinement_by_name,
                    _set_width_mixing_bounds,
                    _set_equality_constraints,
                    _set_peak_center_bounds)

# =============================================================================
# PARAMETERS
# =============================================================================

_function_dict_1d = {
    'gaussian': ['amp', 'cen', 'fwhm'],
    'lorentzian': ['amp', 'cen', 'fwhm'],
    'pvoigt': ['amp', 'cen', 'fwhm', 'mixing'],
    'split_pvoigt': ['amp', 'cen', 'fwhm_l', 'fwhm_h', 'mixing_l', 'mixing_h'],
    'pink_beam_dcs': ['amp', 'cen',
                      'alpha0', 'alpha1',
                      'beta0', 'beta1',
                      'fwhm_g', 'fwhm_l'],
    'constant': ['c0'],
    'linear': ['c0', 'c1'],
    'quadratic': ['c0', 'c1', 'c2'],
    'cubic': ['c0', 'c1', 'c2', 'c3'],
    'quartic': ['c0', 'c1', 'c2', 'c3', 'c4'],
    'quintic': ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
}

num_func_params = dict.fromkeys(_function_dict_1d)
for key, val in _function_dict_1d.items():
    num_func_params[key] = len(val)

pk_prefix_tmpl = "pk%d_"

alpha0_DFLT, alpha1_DFLT, beta0_DFLT, beta1_DFLT = np.r_[
    14.45, 0.0, 3.0162, -7.9411
]

param_hints_DFLT = (True, None, None, None, None)

fwhm_min = 1e-3
fwhm_DFLT = 0.1
mixing_DFLT = 0.5
pk_sep_min = 0.01


# =============================================================================
# SIMPLE FUNCTION DEFS
# =============================================================================


def constant_bkg(x, c0):
    # return c0
    return c0


def linear_bkg(x, c0, c1):
    # return c0 + c1*x
    cheb_cls = chebyshev.Chebyshev(
        [c0, c1], domain=(min(x), max(x))
    )
    return cheb_cls(x)


def quadratic_bkg(x, c0, c1, c2):
    # return c0 + c1*x + c2*x**2
    cheb_cls = chebyshev.Chebyshev(
        [c0, c1, c2], domain=(min(x), max(x))
    )
    return cheb_cls(x)


def cubic_bkg(x, c0, c1, c2, c3):
    # return c0 + c1*x + c2*x**2 + c3*x**3
    cheb_cls = chebyshev.Chebyshev(
        [c0, c1, c2, c3], domain=(min(x), max(x))
    )
    return cheb_cls(x)


def quartic_bkg(x, c0, c1, c2, c3, c4):
    # return c0 + c1*x + c2*x**2 + c3*x**3
    cheb_cls = chebyshev.Chebyshev(
        [c0, c1, c2, c3, c4], domain=(min(x), max(x))
    )
    return cheb_cls(x)


def quintic_bkg(x, c0, c1, c2, c3, c4, c5):
    # return c0 + c1*x + c2*x**2 + c3*x**3
    cheb_cls = chebyshev.Chebyshev(
        [c0, c1, c2, c3, c4, c5], domain=(min(x), max(x))
    )
    return cheb_cls(x)


def chebyshev_bkg(x, *args):
    cheb_cls = chebyshev.Chebyshev(args, domain=(min(x), max(x)))
    return cheb_cls(x)


def gaussian_1d(x, amp, cen, fwhm):
    return amp * np.exp(-(x - cen)**2 / (2*(fwhm_to_sigma*fwhm)**2))


def lorentzian_1d(x, amp, cen, fwhm):
    return amp * (0.5*fwhm)**2 / ((x - cen)**2 + (0.5*fwhm)**2)


def pvoigt_1d(x, amp, cen, fwhm, mixing):
    return mixing*gaussian_1d(x, amp, cen, fwhm) \
        + (1 - mixing)*lorentzian_1d(x, amp, cen, fwhm)


def split_pvoigt_1d(x, amp, cen, fwhm_l, fwhm_h, mixing_l, mixing_h):
    idx_l = x <= cen
    idx_h = x > cen
    return np.concatenate(
        [pvoigt_1d(x[idx_l], amp, cen, fwhm_l, mixing_l),
         pvoigt_1d(x[idx_h], amp, cen, fwhm_h, mixing_h)]
    )


def pink_beam_dcs(x, amp, cen, alpha0, alpha1, beta0, beta1, fwhm_g, fwhm_l):
    """
    @author Saransh Singh, Lawrence Livermore National Lab
    @date 10/18/2021 SS 1.0 original
    @details pink beam profile for DCS data for calibration.
    more details can be found in
    Von Dreele et. al., J. Appl. Cryst. (2021). 54, 3–6
    """
    alpha = _calc_alpha((alpha0, alpha1), cen)
    beta = _calc_beta((beta0, beta1), cen)

    arg1 = np.array([alpha, beta, fwhm_g], dtype=np.float64)
    arg2 = np.array([alpha, beta, fwhm_l], dtype=np.float64)

    p_g = np.hstack([[amp, cen], arg1]).astype(np.float64, order='C')
    p_l = np.hstack([[amp, cen], arg2]).astype(np.float64, order='C')

    eta, fwhm = _mixing_factor_pv(fwhm_g, fwhm_l)

    G = _gaussian_pink_beam(p_g, x)
    L = _lorentzian_pink_beam(p_l, x)

    return eta*L + (1. - eta)*G


def _amplitude_guess(x, x0, y, fwhm):
    pt_l = np.argmin(np.abs(x - (x0 - 0.5*fwhm)))
    pt_h = np.argmin(np.abs(x - (x0 + 0.5*fwhm)))
    return np.max(y[pt_l:pt_h + 1])


def _initial_guess(peak_positions, x, f,
                   pktype='pvoigt', bgtype='linear',
                   fwhm_guess=None, min_ampl=0.):
    """
    Generate function-specific estimate for multi-peak parameters.

    Parameters
    ----------
    peak_positions : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    pktype : TYPE, optional
        DESCRIPTION. The default is 'pvoigt'.
    bgtype : TYPE, optional
        DESCRIPTION. The default is 'linear'.
    fwhm_guess : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    p0 : TYPE
        DESCRIPTION.
    """
    npts = len(x)
    assert len(f) == npts, "ordinate and data must be same length!"

    num_pks = len(peak_positions)

    if fwhm_guess is None:
        fwhm_guess = (np.max(x) - np.min(x))/(20.*num_pks)
    fwhm_guess = np.atleast_1d(fwhm_guess)
    if(len(fwhm_guess) < 2):
        fwhm_guess = fwhm_guess*np.ones(num_pks)

    # estimate background with snip1d
    # !!! using a window size based on abcissa
    bkg = snip1d(np.atleast_2d(f),
                 w=int(np.floor(len(f)/num_pks/2.))).flatten()

    bkg_mod = chebyshev.Chebyshev(
        [0., 0.], domain=(min(x), max(x))
    )
    fit_bkg = bkg_mod.fit(x, bkg, 1)
    coeff = fit_bkg.coef

    # make lin bkg subtracted spectrum
    fsubtr = f - fit_bkg(x)

    # number of parmaters from reference dict
    nparams_pk = num_func_params[pktype]
    nparams_bg = num_func_params[bgtype]

    pkparams = np.zeros((num_pks, nparams_pk))

    # case processing
    # !!! used to use (f[pt] - min_val) for ampl
    if pktype == 'gaussian' or pktype == 'lorentzian':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, peak_positions[ii], fsubtr, fwhm_guess[ii]
            )
            pkparams[ii, :] = [
                max(amp_guess, min_ampl),
                peak_positions[ii],
                fwhm_guess[ii]
            ]
    elif pktype == 'pvoigt':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, peak_positions[ii], fsubtr, fwhm_guess[ii]
            )
            pkparams[ii, :] = [
                max(amp_guess, min_ampl),
                peak_positions[ii],
                fwhm_guess[ii],
                0.5
            ]
    elif pktype == 'split_pvoigt':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, peak_positions[ii], fsubtr, fwhm_guess[ii]
            )
            pkparams[ii, :] = [
                max(amp_guess, min_ampl),
                peak_positions[ii],
                fwhm_guess[ii],
                fwhm_guess[ii],
                0.5,
                0.5
            ]
    elif pktype == 'pink_beam_dcs':
        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            amp_guess = _amplitude_guess(
                x, peak_positions[ii], fsubtr, fwhm_guess[ii]
            )
            pkparams[ii, :] = [
                max(amp_guess, min_ampl),
                peak_positions[ii],
                alpha0_DFLT,
                alpha1_DFLT,
                beta0_DFLT,
                beta1_DFLT,
                fwhm_guess[ii],
                fwhm_guess[ii],
            ]

    if bgtype == 'constant':
        bgparams = np.average(bkg)
    else:
        bgparams = np.hstack([coeff, np.zeros(nparams_bg - 2)])

    return np.hstack([pkparams.flatten(), bgparams])

# =============================================================================
# MODELS
# =============================================================================


def _build_composite_model(npeaks=1, pktype='gaussian', bgtype='linear'):
    if pktype == 'gaussian':
        pkfunc = gaussian_1d
    elif pktype == 'lorentzian':
        pkfunc = lorentzian_1d
    elif pktype == 'pvoigt':
        pkfunc = pvoigt_1d
    elif pktype == 'split_pvoigt':
        pkfunc = split_pvoigt_1d
    elif pktype == 'pink_beam_dcs':
        pkfunc = pink_beam_dcs

    spectrum_model = Model(pkfunc, prefix=pk_prefix_tmpl % 0)
    for i in range(1, npeaks):
        spectrum_model += Model(pkfunc, prefix=pk_prefix_tmpl % i)

    if bgtype == 'constant':
        spectrum_model += Model(constant_bkg)
    elif bgtype == 'linear':
        spectrum_model += Model(linear_bkg)
    elif bgtype == 'quadratic':
        spectrum_model += Model(quadratic_bkg)
    elif bgtype == 'cubic':
        spectrum_model += Model(cubic_bkg)
    elif bgtype == 'quartic':
        spectrum_model += Model(quartic_bkg)
    elif bgtype == 'quintic':
        spectrum_model += Model(quintic_bkg)
    return spectrum_model


# =============================================================================
# CLASSES
# =============================================================================


class SpectrumModel(object):
    def __init__(self, data, peak_centers,
                 pktype='pvoigt', bgtype='linear',
                 fwhm_init=None, min_ampl=1e-4, min_pk_sep=pk_sep_min):
        """
        Instantiates spectrum model.

        Parameters
        ----------
        data : array_like
            The (n, 2) array representing the spectrum as (2θ, intensity).
            The units of tth are assumed to be in degrees.
        peak_centers : array_like
            The (n, ) vector of ideal peak 2θ positions in degrees.
        pktype : TYPE, optional
            DESCRIPTION. The default is 'pvoigt'.
        bgtype : TYPE, optional
            DESCRIPTION. The default is 'linear'.

        Returns
        -------
        None.

        Notes
        -----
        - data abscissa and peak centers are in DEGREES.

        """
        # peak and background spec
        assert pktype in _function_dict_1d.keys(), \
            "peak type '%s' not recognized" % pktype
        assert bgtype in _function_dict_1d.keys(), \
            "background type '%s' not recognized" % bgtype
        self._pktype = pktype
        self._bgtype = bgtype

        master_keys_pks = _function_dict_1d[pktype]
        master_keys_bkg = _function_dict_1d[bgtype]

        # spectrum data
        data = np.atleast_2d(data)
        assert data.shape[1] == 2, \
            "data must be [[tth_0, int_0], ..., [tth_N, int_N]"
        assert len(data > 10), \
            "check your input spectrum; you provided fewer than 10 points."
        self._data = data

        xdata, ydata = data.T
        window_range = (np.min(xdata), np.max(xdata))
        ymax = np.max(ydata)

        self._tth0 = peak_centers
        num_peaks = len(peak_centers)

        if fwhm_init is None:
            fwhm_init = np.diff(window_range)/(20.*num_peaks)

        self._min_pk_sep = min_pk_sep

        # model
        spectrum_model = _build_composite_model(
            num_peaks, pktype=pktype, bgtype=bgtype
        )
        self._model = spectrum_model

        p0 = _initial_guess(
            self._tth0, xdata, ydata,
            pktype=self._pktype, bgtype=self._bgtype,
            fwhm_guess=fwhm_init, min_ampl=min_ampl
        )
        psplit = num_func_params[bgtype]
        p0_pks = np.reshape(p0[:-psplit], (num_peaks, num_func_params[pktype]))
        p0_bkg = p0[-psplit:]

        # peaks
        initial_params_pks = Parameters()
        for i, pi in enumerate(p0_pks):
            new_keys = [pk_prefix_tmpl % i + k for k in master_keys_pks]
            initial_params_pks.add_many(
                *_parameter_arg_constructor(
                    dict(zip(new_keys, pi)), param_hints_DFLT
                )
            )

        # set some special constraints
        _set_width_mixing_bounds(
            initial_params_pks,
            min_w=fwhm_min,
            max_w=0.9*float(np.diff(window_range))
        )
        _set_bound_constraints(
            initial_params_pks, 'amp', min_val=min_ampl, max_val=1.5*ymax
        )
        _set_peak_center_bounds(
            initial_params_pks, window_range, min_sep=min_pk_sep
        )
        if pktype == 'pink_beam_dcs':
            # set bounds on fwhm params and mixing (where applicable)
            # !!! important for making pseudo-Voigt behave!
            _set_refinement_by_name(initial_params_pks, 'alpha', vary=False)
            _set_refinement_by_name(initial_params_pks, 'beta', vary=False)
            _set_equality_constraints(
                initial_params_pks,
                zip(_extract_parameters_by_name(initial_params_pks, 'fwhm_g'),
                    _extract_parameters_by_name(initial_params_pks, 'fwhm_l'))
            )
        elif pktype == 'split_pvoigt':
            mparams = _extract_parameters_by_name(
                initial_params_pks, 'mixing_l'
            )
            for mp in mparams[1:]:
                _set_equality_constraints(
                    initial_params_pks, ((mp, mparams[0]), )
                )
            mparams = _extract_parameters_by_name(
                initial_params_pks, 'mixing_h'
            )
            for mp in mparams[1:]:
                _set_equality_constraints(
                    initial_params_pks, ((mp, mparams[0]), )
                )

        # background
        initial_params_bkg = Parameters()
        initial_params_bkg.add_many(
            *_parameter_arg_constructor(
                dict(zip(master_keys_bkg, p0_bkg)),
                param_hints_DFLT
            )
        )

        self._peak_params = initial_params_pks
        self._background_params = initial_params_bkg

    @property
    def pktype(self):
        return self._pktype

    @property
    def bgtype(self):
        return self._bgtype

    @property
    def min_pk_sep(self):
        return self._min_pk_sep

    @property
    def data(self):
        return self._data

    @property
    def tth0(self):
        return self._tth0

    @property
    def num_peaks(self):
        return len(self.tth0)

    @property
    def model(self):
        return self._model

    @property
    def peak_params(self):
        return self._peak_params

    @property
    def background_params(self):
        return self._background_params

    @property
    def params(self):
        return self._peak_params + self._background_params

    def fit(self):
        xdata, ydata = self.data.T
        window_range = (np.min(xdata), np.max(xdata))
        if self.pktype == 'pink_beam_dcs':
            for pname, param in self.peak_params.items():
                if 'alpha' in pname or 'beta' in pname or 'fwhm' in pname:
                    param.vary = False

            res0 = self.model.fit(ydata, params=self.params, x=xdata)
            if res0.success:
                new_p = res0.params
                _set_refinement_by_name(new_p, 'alpha0', vary=True)
                _set_refinement_by_name(new_p, 'alpha1', vary=False)
                _set_refinement_by_name(new_p, 'beta', vary=True)
                _set_equality_constraints(new_p, 'alpha')
                _set_equality_constraints(new_p, 'beta')
                _set_bound_constraints(
                    new_p, 'alpha', min_val=-10, max_val=30
                )
                _set_bound_constraints(
                    new_p, 'beta', min_val=-10, max_val=30
                )
                _set_width_mixing_bounds(
                    new_p,
                    min_w=fwhm_min,
                    max_w=0.9*float(np.diff(window_range))
                )
                # !!! not sure on this, but it seems
                #     to give more stable results with many peaks
                _set_equality_constraints(
                    new_p,
                    zip(_extract_parameters_by_name(new_p, 'fwhm_g'),
                        _extract_parameters_by_name(new_p, 'fwhm_l'))
                )
                try:
                    _set_peak_center_bounds(new_p, window_range,
                                            min_sep=self.min_pk_sep)
                except(RuntimeError):
                    return res0

                # refit
                res1 = self.model.fit(ydata, params=new_p, x=xdata)
            else:
                return res0
        else:
            res1 = self.model.fit(ydata, params=self.params, x=xdata)

        return res1
