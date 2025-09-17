from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from hexrd.instrument import HEDMInstrument
from hexrd.material import Material
from hexrd.projections.polar import PolarView
from hexrd.transforms.xfcapi import anglesToGVec
from hexrd.valunits import valWUnit
from hexrd.wppf import Rietveld
from hexrd.wppf.phase import Material_Rietveld
from hexrd.wppf.texture import HarmonicModel, PoleFigures
from hexrd.wppf.WPPF import extract_intensities
from hexrd.wppf.wppfsupport import _generate_default_parameters_LeBail


@pytest.fixture
def texture_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'tests/wppf/texture'


@pytest.fixture
def texture_instrument(texture_examples_path: Path) -> HEDMInstrument:
    filepath = texture_examples_path / 'config.hexrd'
    with h5py.File(filepath) as f:
        return HEDMInstrument(f)


@pytest.fixture
def texture_img_dict(
    texture_examples_path: Path,
    texture_instrument: HEDMInstrument,
) -> dict[str: np.ndarray]:
    instr = texture_instrument
    img_dict = dict.fromkeys(instr.detectors.keys())
    for k in img_dict:
        path = texture_examples_path / f'{k}.tiff'
        img_dict[k] = np.array(Image.open(path))
    return img_dict


def _angstrom(x):
    return valWUnit('wavelength', 'length', x, 'angstrom')


def _kev(x):
    return valWUnit('xray_energy', 'energy', x, 'keV')


def get_mean(ii,
             pv,
             eta_step,
             azimuthal_interval,
             integration_range):

    start = int(((ii+1)*azimuthal_interval-integration_range)/eta_step)
    stop = int(((ii+1)*azimuthal_interval+integration_range)/eta_step)

    return np.squeeze(np.nanmean(pv[start:stop, :], axis=0))


def bin_polar_view(polar_obj,
                   pv,
                   azimuthal_interval,
                   integration_range):
    '''bin the polar view image into a coarser
    grid by integration around +/- "integration_range"
    every "azimuthal_interval" degree
    '''
    eta_mi = np.degrees(polar_obj.eta_min)
    eta_ma = np.degrees(polar_obj.eta_max)

    nspec = int((eta_ma - eta_mi)/azimuthal_interval)-1

    pv_binned = np.zeros((nspec, pv.shape[1]))

    eta_step = polar_obj.eta_pixel_size

    for ii in np.arange(nspec):
        pv_binned[ii, :] = get_mean(ii,
                                    pv,
                                    eta_step,
                                    azimuthal_interval,
                                    integration_range)

    return pv_binned


def get_lineout(pv):
    return np.nanmean(pv, axis=0)


def test_wppf_texture(texture_instrument, texture_img_dict, test_data_dir):

    # get the basic hexrd objects
    # 1. instrument
    # 2. image dictionary
    # 3. material
    # 4. polar view class

    instr = texture_instrument
    img_dict = texture_img_dict

    sample_normal = np.array([
        np.sin(np.radians(22.5)),
        0,
        np.cos(np.radians(22.5)),
    ])

    mat = Material(dmin=_angstrom(0.5), kev=_kev(instr.beam_energy))
    exc = np.zeros_like(mat.planeData.exclusions).astype(bool)
    mat.planeData.exclusions = exc

    matr = Material_Rietveld(material_obj=mat)

    polar_obj = PolarView(
        (2, 40),
        instr,
        eta_min=-100.,
        eta_max=100.,
        pixel_size=(0.05, 0.1),
        cache_coordinate_map=True,
    )

    pv = polar_obj.warp_image(img_dict,
                              pad_with_nans=True,
                              do_interpolation=True)

    ttharray = np.degrees(polar_obj.angular_grid[1][0, :])

    lo_full = get_lineout(pv)

    '''now we'll subdivide the generate polar view image
    using coarser binning. as a start, we will subdivide
    in steps of 5 degrees. The integrated intensity will
    be given by a sum of +/- 1 degree around each 5 degree
    azimuthal angle
    '''
    # define constants
    azimuthal_interval = 5
    integration_range = 1
    pv_binned = bin_polar_view(
        polar_obj,
        pv,
        azimuthal_interval,
        integration_range,
    )

    '''generate instance of rietveld class with a texture model
    attached to it
    '''

    # pf = get_pole_figures(instr, mat)

    kwargs = {
        'material': matr,
        'ssym': 'axial',
        'ell_max': 16,
        'bvec': instr.beam_vector,
        'evec': instr.eta_vector,
        'sample_normal': sample_normal,
    }
    hm = HarmonicModel(**kwargs)

    expt_spec = np.vstack((ttharray[~lo_full.mask],
                           lo_full.data[~lo_full.mask])).T

    kwargs = {
        'expt_spectrum': expt_spec,
        'wavelength': {'XFEL': [_angstrom(instr.beam_wavelength), 1.0]},
        'bkgmethod': {'chebyshev': 1},
        'peakshape': "pvtch",
        'phases': [mat],
        'texture_model': {'Ni': hm}
    }
    R = Rietveld(**kwargs)

    R.params['scale'].vary = True
    R.params['bkg_0'].vary = True
    R.params['bkg_1'].vary = True

    R.params['Ni_X'].value = 0
    R.params['Ni_Y'].value = 0

    R.Refine()

    assert R.Rwp < 1

    R.params['scale'].vary = False
    R.params['bkg_0'].vary = False
    R.params['bkg_1'].vary = False

    R.params['U'].vary = True
    R.params['V'].vary = True
    R.params['W'].vary = True

    R.Refine()

    assert R.Rwp < 0.95

    R.params['scale'].vary = True

    R.Refine()

    assert R.Rwp < 0.4

    R.params_vary_off()

    R.params['scale'].vary = True
    R.texture_parameters_vary(True)

    R.Refine()

    assert R.Rwp < 0.35

    params = _generate_default_parameters_LeBail(mat, 1, {"chebyshev": 1})

    for p in params:
        params[p].value = R.params[p].value

    params["U"].vary = True
    params["V"].vary = True
    params["W"].vary = True
    params["Ni_a"].vary = True

    Ic = R.compute_intensities()

    mask = np.isnan(pv_binned)

    results = extract_intensities(**{
        'polar_view': np.ma.masked_array(pv_binned, mask=mask),
        'tth_array': ttharray,
        'params': params,
        'phases': [mat],
        'wavelength': {"XFEL": [_angstrom(instr.beam_wavelength), 1.]},
        'bkgmethod': {"chebyshev": 1},
        'intensity_init': Ic,
        'termination_condition': {"rwp_perct_change": 0.05, "max_iter": 10},
        'peakshape': "pvtch",
    })

    # we have to divide by the computed instensites
    # to get the texture contribution
    pfdata = {}
    hkl_master = []
    Ic = R.compute_intensities()

    for ii in range(pv_binned.shape[0]):
        eta = np.radians(-100 + (ii+1)*azimuthal_interval)
        t = results[2][ii]['Ni']['XFEL']
        hkl = results[1][ii]['Ni']['XFEL']
        I = results[0][ii]['Ni']['XFEL']  # noqa
        Icomp = Ic['Ni']['XFEL']

        nn = np.min((t.shape[0], Icomp.shape[0]))
        for jj in np.arange(nn):
            if jj < 3:
                angs = np.atleast_2d([np.radians(t[jj]), eta, 0])
                v = anglesToGVec(
                    angs,
                    bHat_l=instr.beam_vector,
                    eHat_l=instr.eta_vector,
                )
                h = hkl[jj, :]
                hstr = str(h).strip('[').strip(']').replace(" ", "")
                data = np.hstack((v, np.atleast_2d(I[jj]/Icomp[jj])))
                if hstr not in pfdata:
                    pfdata[hstr] = data
                else:
                    pfdata[hstr] = np.vstack((pfdata[hstr], data))

    for k in pfdata:
        h = np.array([int(k[0]), int(k[1]), int(k[2])])
        hkl_master.append(h)

    hkl_master = np.array(hkl_master)

    pf = PoleFigures(R.phases['Ni']['XFEL'],
                     hkl_master,
                     pfdata,
                     ell_max=16,
                     bvec=instr.beam_vector,
                     sample_normal=sample_normal,
                     ssym='axial')

    pf.calculate_harmonic_coefficients()

    pf.calc_new_pole_figure(np.atleast_2d(hkl_master[0:3, :]))

    # R.spectrum_expt.plot(0, '-k', lw=2.5)
    # R.spectrum_sim.plot(1, '--r', lw=1.75)

    # Load refs and verify they match
    ref_path = test_data_dir / 'test_wppf_texture_expected.npy'
    ref_data = np.load(ref_path, allow_pickle=True).item()
    ref_map = {
        'angs': 'angs_new',
        'stereo_radius': 'stereo_radius_new',
        'intensities': 'intensities_new',
    }
    for ref_name, pf_name in ref_map.items():
        d1 = ref_data[ref_name]
        d2 = getattr(pf, pf_name)

        assert sorted(list(d1)) == sorted(list(d2))
        for key in d1:
            assert np.allclose(d1[key], d2[key], rtol=1e-3)
