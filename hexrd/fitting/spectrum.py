import numpy as np

from lmfit import Model, Parameters, report_fit

from hexrd.constants import fwhm_to_sigma

from utils import (_calc_alpha, _calc_beta,
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
    'linear': ['c0', 'c1'],
    'quadratic': ['c0', 'c1', 'c2'],
    'cubic': ['c0', 'c1', 'c2', 'c3']
}

npp = dict.fromkeys(_function_dict_1d)
for key, val in _function_dict_1d.items():
    npp[key] = len(val)

pk_prefix_tmpl = "pk%d_"

alpha0_DFLT, alpha1_DFLT, beta0_DFLT, beta1_DFLT = np.r_[
    14.45, 0.0, 3.0162, -7.9411
]

param_hints_DFLT = (True, None, None, None, None)


# =============================================================================
# SIMPLE FUNCTION DEFS
# =============================================================================


def linear_bkg(x, c0, c1):
    return c0 + c1*x


def quadratic_bkg(x, c0, c1, c2):
    return c0 + c1*x + c2*x**2


def cubic_bkg(x, c0, c1, c2, c3):
    return c0 + c1*x + c2*x**2 + c3*x**3


def gaussian_1d(x, amp, cen, fwhm):
    return amp * np.exp(-(x - cen)**2 / (2*fwhm_to_sigma*fwhm)**2)


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

    if bgtype == 'linear':
        spectrum_model += Model(linear_bkg)
    elif bgtype == 'quadratic':
        spectrum_model += Model(quadratic_bkg)
    elif bgtype == 'cubic':
        spectrum_model += Model(cubic_bkg)

    return spectrum_model

# =============================================================================
# %% TESTING
# =============================================================================
import h5py
from hexrd.fitting import fitpeak
from hexrd import material
from hexrd import valunits
from matplotlib import pyplot as plt

# fit real snippet
# pv = h5py.File('./test/DCS_Ceria_pv.h5', 'r')
pv = h5py.File('./test/GE_1ID_Ceria_pv.h5', 'r')
pimg = np.array(pv['intensities'])
etas = np.array(pv['eta_coordinates'])
tths = np.array(pv['tth_coordinates'])
# lineout = np.vstack(
#     [tths[0, :],
#       np.nanmean(pimg.reshape(20, 36, 1027), axis=0)[7]]

# ).T  # rebin to 10deg
lineout = np.array(pv['azimuthal_integration'])

# for DCS CeO2
# idx = np.logical_and(lineout[:, 0] >= 7., lineout[:, 0] <= 14)
# tth0 = np.r_[9.67, 11.17]
# idx = np.logical_and(lineout[:, 0] >= 14.48, lineout[:, 0] <= 21.073)
# tth0 = np.r_[15.83, 18.58, 19.42]
# idx = np.logical_and(lineout[:, 0] >= 21.073, lineout[:, 0] <= 30.964)
# tth0 = np.r_[22.46, 24.50, 25.15, 27.59, 29.30]
# idx = np.logical_and(lineout[:, 0] >= 7., lineout[:, 0] <= 27.06)
# tth0 = np.r_[9.67, 11.17, 15.83, 18.58, 19.42, 22.46, 24.50, 25.15]
#
# for GE CeO2
dmin = valunits.valWUnit('dmin', 'length', 0.55, 'angstrom')
kev = valunits.valWUnit('kev', 'energy', 80.725, 'keV')
mat_dict = material.load_materials_hdf5('./test/materials.h5',
                                        dmin=dmin, kev=kev)
ridx = 0
matl = mat_dict['CeO2']
pd = matl.planeData
pd.tThWidth = np.radians(0.35)
tthi, tthr = pd.getMergedRanges(cullDupl=True)
tth0 = np.degrees(pd.getTTh()[tthi[ridx]]) \
    + 0.01*np.random.randn(len(tthi[ridx]))
idx = np.logical_and(lineout[:, 0] >= np.degrees(tthr[ridx][0]),
                     lineout[:, 0] <= np.degrees(tthr[ridx][1]))

snippet = lineout[idx, :]
xdata = snippet[:, 0]
ydata = snippet[:, 1]
window_range = (np.min(xdata), np.max(xdata))

# parameters
pktype = 'split_pvoigt'
bgtype = 'linear'
psplit = 2

npks = len(tth0)

p0 = fitpeak.estimate_mpk_parms_1d(
    tth0, xdata, ydata, pktype=pktype, bgtype=bgtype
)[0]

master_keys_pks = _function_dict_1d[pktype]
master_keys_bkg = _function_dict_1d[bgtype]

# peaks
initial_params_pks = Parameters()
for i, pi in enumerate(p0[:-psplit].reshape(npks, npp[pktype])):
    new_keys = [pk_prefix_tmpl % i + k for k in master_keys_pks]
    initial_params_pks.add_many(
        *_parameter_arg_constructor(
            dict(zip(new_keys, pi)), param_hints_DFLT
        )
    )

if pktype == 'pink_beam_dcs':
    # set bounds on fwhm params and mixing (where applicable)
    # !!! important for making pseudo-Voigt behave!
    _set_refinement_by_name(initial_params_pks, 'alpha', vary=False)
    _set_refinement_by_name(initial_params_pks, 'beta', vary=False)
    _set_width_mixing_bounds(initial_params_pks, min_w=0.01, max_w=np.inf)
    _set_equality_constraints(
        initial_params_pks,
        zip(_extract_parameters_by_name(initial_params_pks, 'fwhm_g'),
            _extract_parameters_by_name(initial_params_pks, 'fwhm_l'))
    )
    pass

_set_width_mixing_bounds(initial_params_pks, min_w=0.01, max_w=np.inf)
_set_bound_constraints(initial_params_pks, 'amp',
                       min_val=0., max_val=np.max(ydata))
_set_peak_center_bounds(initial_params_pks, window_range, min_sep=0.01)
print('\nINITITAL PARAMETERS\n------------------\n')
initial_params_pks.pretty_print()

# background
initial_params_bkg = Parameters()
initial_params_bkg.add_many(
    *_parameter_arg_constructor(
        dict(zip(master_keys_bkg, p0[-psplit:])), param_hints_DFLT
    )
)

# model
spectrum_model = _build_composite_model(
    npks, pktype=pktype, bgtype=bgtype)

fig, ax = plt.subplots()
f = spectrum_model.eval(params=initial_params_pks+initial_params_bkg, x=xdata)
ax.plot(xdata, ydata, 'r+', label='measured')
ax.plot(xdata, f, 'g:', label='initial')

# %%

if pktype == 'pink_beam_dcs':
    for pname, param in initial_params_pks.items():
        if 'alpha' in pname or 'beta' in pname or 'fwhm' in pname:
            param.vary = False

    res0 = spectrum_model.fit(
        ydata, params=initial_params_pks + initial_params_bkg, x=xdata
    )
    ax.plot(xdata, res0.best_fit, 'g-', lw=1, label='first fit')

    new_p = Parameters()
    new_p.add_many(
        *_parameter_arg_constructor(res0.best_values, param_hints_DFLT)
    )
    _set_equality_constraints(new_p, 'alpha0')
    _set_equality_constraints(new_p, 'beta0')
    _set_refinement_by_name(new_p, 'alpha1', vary=False)
    _set_refinement_by_name(new_p, 'beta1', vary=False)
    _set_width_mixing_bounds(new_p, min_w=0.01, max_w=np.inf)
    # _set_equality_constraint(
    #     new_p,
    #     zip(_extract_parameters_by_name(new_p, 'fwhm_g'),
    #         _extract_parameters_by_name(new_p, 'fwhm_l'))
    # )
    _set_peak_center_bounds(new_p, window_range, min_sep=0.01)
    print('\nRE-FIT PARAMETERS\n------------------\n')
    new_p.pretty_print()

    res1 = spectrum_model.fit(ydata, params=new_p, x=xdata)
    ax.plot(xdata, res1.best_fit, 'c-', lw=1, label='second fit')
else:
    '''
    mpn = 'fwhm'
    for ip in range(1, npks):
        pkey = pk_prefix_tmpl % ip + mpn
        initial_params_pks[pkey].expr = pk_prefix_tmpl % 0 + mpn
    '''
    res1 = spectrum_model.fit(
        ydata, params=initial_params_pks + initial_params_bkg, x=xdata
    )
    ax.plot(xdata, res1.best_fit, 'y-', lw=1, label='fit')

# print report
print('\nPOST-FIT REPORT\n---------------\n')
print('Final sum of squared residuals: %1.3e\n' % np.sum(res1.residual**2))
report_fit(res1.params)

ax.set_xlabel(r'$2\theta$ [degrees]')
ax.set_ylabel(r'Intensity [arb.]')
ax.legend(loc='upper right')
fig.suptitle(r'multi-peak fitting for %s + %s background' % (pktype, bgtype))

# %%
class SpectrumModel(object):
    def __init__(self, *models, **param_dicts):
        raise NotImplementedError
