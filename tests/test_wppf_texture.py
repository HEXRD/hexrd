from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from hexrd.instrument import HEDMInstrument
from hexrd.material import Material
from hexrd.projections.polar import bin_polar_view, PolarView
from hexrd.valunits import valWUnit
from hexrd.wppf import Rietveld
from hexrd.wppf.phase import Material_Rietveld
from hexrd.wppf.texture import HarmonicModel
from hexrd.wppf.WPPF import extract_intensities
from hexrd.wppf.wppfsupport import _generate_default_parameters_LeBail
from hexrd.rotations import quatOfAngleAxis, rotMatOfQuat

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

    q = quatOfAngleAxis(np.array(
        [np.radians(22.5)]), 
        np.atleast_2d(np.array([0,1,0])).T)

    sample_rmat = rotMatOfQuat(q)


    mat = Material(dmin=_angstrom(0.5), kev=_kev(instr.beam_energy))
    exc = np.zeros_like(mat.planeData.exclusions).astype(bool)
    mat.planeData.exclusions = exc

    matr = Material_Rietveld(material_obj=mat)

    polar_obj = PolarView(
        (2, 40),
        instr,
        eta_min=-180.,
        eta_max=180.,
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
        'sample_rmat': sample_rmat,
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
        'texture_model': {'Ni': hm},
        'eta_min': -180,
        'eta_max': 180,
        'eta_step': 5,
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

    R.compute_texture_data(
        pv_binned,
        bvec=instr.beam_vector,
        evec=instr.eta_vector,
        azimuthal_interval=azimuthal_interval,
    )

    R.params_vary_off()
    R.texture_parameters_vary(True)

    # The pole figure data will have been set on the harmonic model
    # Now we can refine the texture
    R.RefineTexture()
    hm.calc_new_pole_figure(R.params)

    # R.spectrum_expt.plot(0, '-k', lw=2.5)
    # R.spectrum_sim.plot(1, '--r', lw=1.75)

    # Compute the 2D spectrum
    R.computespectrum_2D()

    # Load refs and verify they match
    ref_path = test_data_dir / 'test_wppf_texture_expected.npy'
    ref_data = np.load(ref_path, allow_pickle=True).item()
    ref_map = {
        'angs': 'angs_new',
        'stereo_radius': 'stereo_radius_new',
        'intensities': 'intensities_new',
    }

    ref_on_rietveld_obj = [
        'simulated_2d',
    ]

    for ref_name, hm_name in ref_map.items():
        if hm_name in ref_on_rietveld_obj:
            # This one is on the Rietveld object
            continue

        d1 = ref_data[ref_name]
        d2 = getattr(hm, hm_name)

        assert sorted(list(d1)) == sorted(list(d2))
        for key in d1:
            assert np.allclose(d1[key], d2[key], rtol=1e-3)

    for name in ref_on_rietveld_obj:
        assert np.allclose(ref_data[name], getattr(R, name), rtol=1e-2)
