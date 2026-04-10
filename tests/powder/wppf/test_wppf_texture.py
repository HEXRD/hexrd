from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from hexrd.instrument import HEDMInstrument
from hexrd.material import Material
from hexrd.projections.polar import bin_polar_view, PolarView
from hexrd.valunits import _angstrom, _kev
from hexrd.wppf import Rietveld
from hexrd.wppf.phase import Material_Rietveld
from hexrd.wppf.texture import HarmonicModel, MarchDollaseModel
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
) -> dict[str : np.ndarray]:
    instr = texture_instrument
    img_dict = dict.fromkeys(instr.detectors.keys())
    for k in img_dict:
        path = texture_examples_path / f'{k}.tiff'
        img_dict[k] = np.array(Image.open(path))
    return img_dict


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

    q = quatOfAngleAxis(
        np.array([np.radians(22.5)]), np.atleast_2d(np.array([0, 1, 0])).T
    )

    sample_rmat = rotMatOfQuat(q)

    mat = Material(dmin=_angstrom(0.5), kev=_kev(instr.beam_energy))
    exc = np.zeros_like(mat.planeData.exclusions).astype(bool)
    mat.planeData.exclusions = exc

    matr = Material_Rietveld(material_obj=mat)

    polar_obj = PolarView(
        (2, 40),
        instr,
        eta_min=-180.0,
        eta_max=180.0,
        pixel_size=(0.05, 0.1),
        cache_coordinate_map=True,
    )

    pv = polar_obj.warp_image(img_dict, pad_with_nans=True, do_interpolation=True)

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

    expt_spec = np.vstack((ttharray[~lo_full.mask], lo_full.data[~lo_full.mask])).T

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

    comparison_dict = {}
    for ref_name, hm_name in ref_map.items():
        if hm_name in ref_on_rietveld_obj:
            # This one is on the Rietveld object
            continue

        comparison_dict[ref_name] = getattr(hm, hm_name)

    for name in ref_on_rietveld_obj:
        # Wrap this in a dict so we can do the same comparison later
        comparison_dict[name] = {'result': getattr(R, name)}

    # When the test data needs to be updated, save this out:
    # np.save(ref_path, comparison_dict)

    # Now do the comparison
    for root_key in ref_data:
        d1 = ref_data[root_key]
        d2 = comparison_dict[root_key]
        assert sorted(list(d1)) == sorted(list(d2))
        for key in d1:
            assert np.allclose(d1[key], d2[key], equal_nan=True, rtol=1e-2)


def test_wppf_march_dollase_texture(texture_instrument, texture_img_dict):
    instr = texture_instrument
    img_dict = texture_img_dict

    polar_obj = PolarView(
        (2, 40),
        instr,
        eta_min=-180.0,
        eta_max=180.0,
        pixel_size=(0.05, 0.1),
        cache_coordinate_map=True,
    )

    pv = polar_obj.warp_image(img_dict, pad_with_nans=True, do_interpolation=True)

    ttharray = np.degrees(polar_obj.angular_grid[1][0, :])
    lo_full = get_lineout(pv)

    expt_spec = np.vstack((ttharray[~lo_full.mask], lo_full.data[~lo_full.mask])).T

    mat = Material(dmin=_angstrom(0.5), kev=_kev(instr.beam_energy))
    exc = np.zeros_like(mat.planeData.exclusions).astype(bool)
    mat.planeData.exclusions = exc

    matr = Material_Rietveld(material_obj=mat)
    HKL = np.array([1, 1, 1])
    P_MD = 1.0

    march = MarchDollaseModel(material=matr, HKL=HKL, P_MD=P_MD)

    kwargs = {
        'expt_spectrum': expt_spec,
        'wavelength': {'XFEL': [_angstrom(instr.beam_wavelength), 1.0]},
        'bkgmethod': {'chebyshev': 1},
        'peakshape': "pvtch",
        'phases': [mat],
        'texture_model': {'Ni': march},
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

    R.params['scale'].vary = False
    R.params['bkg_0'].vary = False
    R.params['bkg_1'].vary = False
    R.params['U'].vary = True
    R.params['V'].vary = True
    R.params['W'].vary = True
    R.Refine()

    R.params['scale'].vary = True
    R.Refine()

    assert R.Rwp < 0.20

    rwp_before_pmd = R.Rwp

    for p in R.params:
        R.params[p].vary = False
    R.params['scale'].vary = True
    R.params["Ni_p_md"].vary = True

    R.Refine()

    # P_MD refinement should not make Rwp worse
    assert R.Rwp <= rwp_before_pmd + 1e-4

    # Test texture_index (polymorphic interface)
    ti = R.texture_index
    assert 'Ni' in ti
    assert np.isclose(ti['Ni'], R.texture_model['Ni'].P_MD)

    # Test calc_pf_rings
    march = R.texture_model['Ni']
    march.calc_pf_rings(R.params)
    assert hasattr(march, 'intensities_rings_2d')
    assert len(march.intensities_rings_2d) > 0
    for hkey, vals in march.intensities_rings_2d.items():
        assert len(hkey) == 3
        assert vals.ndim == 1
        assert np.all(np.isfinite(vals))
        # Each ring should be constant (no azimuthal variation in MD model)
        assert np.all(vals == vals[0])


def test_march_dollase_texture_factors(texture_instrument, texture_img_dict):
    """Test that texture factors and calc_pf_rings produce correct values
    for a known P_MD, using recorded reference values."""
    instr = texture_instrument
    img_dict = texture_img_dict

    mat = Material(dmin=_angstrom(0.5), kev=_kev(instr.beam_energy))
    exc = np.zeros_like(mat.planeData.exclusions).astype(bool)
    mat.planeData.exclusions = exc
    matr = Material_Rietveld(material_obj=mat)

    # P_MD=1.0 should give all texture factors = 1.0 (random texture)
    md_random = MarchDollaseModel(material=matr, HKL=np.array([1, 1, 1]), P_MD=1.0)
    assert np.allclose(md_random.texture_factors, 1.0)

    # P_MD=1.5 should give known non-trivial texture factors
    md = MarchDollaseModel(material=matr, HKL=np.array([1, 1, 1]), P_MD=1.5)
    expected_texture_factors = np.array([
        1.0437682967, 0.7660393300, 1.1397846812, 0.9461665933,
        1.0437682967, 0.7660393300, 1.1080127852, 0.9800541809,
        1.0294009260, 1.0437682967, 0.8465996891, 1.1397846812,
        1.0383456645, 1.0812194483, 0.7660393300, 0.8771545998,
        1.0466261309, 0.9461665933, 1.0437682967, 1.1270512730,
        0.8094535100, 1.0698998879,
    ])
    assert np.allclose(md.texture_factors, expected_texture_factors, rtol=1e-6)

    # Build Rietveld to get params for calc_pf_rings
    polar_obj = PolarView(
        (2, 40), instr,
        eta_min=-180.0, eta_max=180.0,
        pixel_size=(0.05, 0.1),
        cache_coordinate_map=True,
    )
    pv = polar_obj.warp_image(img_dict, pad_with_nans=True, do_interpolation=True)
    ttharray = np.degrees(polar_obj.angular_grid[1][0, :])
    lo_full = get_lineout(pv)
    expt_spec = np.vstack((ttharray[~lo_full.mask], lo_full.data[~lo_full.mask])).T

    kwargs = {
        'expt_spectrum': expt_spec,
        'wavelength': {'XFEL': [_angstrom(instr.beam_wavelength), 1.0]},
        'bkgmethod': {'chebyshev': 1},
        'peakshape': "pvtch",
        'phases': [mat],
        'texture_model': {'Ni': md},
        'eta_min': -180,
        'eta_max': 180,
        'eta_step': 5,
    }
    R = Rietveld(**kwargs)
    R.params['Ni_p_md'].value = 1.5

    # calc_pf_rings should produce constant rings matching texture_factors
    md.calc_pf_rings(R.params)
    for ii, hkey in enumerate(md.material.hkls):
        ring_vals = md.intensities_rings_2d[tuple(hkey)]
        assert np.allclose(ring_vals, md.texture_factors[ii])

    # texture_index should return P_MD
    assert np.isclose(md.texture_index(R.params), 1.5)

    # calc_texture_factor should return the texture_factors array
    tf = md.calc_texture_factor(R.params)
    assert np.allclose(tf, expected_texture_factors, rtol=1e-6)


def test_march_dollase_validation():
    """Test property validation on MarchDollaseModel."""
    from hexrd.wppf.phase import Material_Rietveld
    from hexrd.material import Material
    from hexrd.valunits import _angstrom, _kev

    mat = Material(dmin=_angstrom(0.5), kev=_kev(50.0))
    matr = Material_Rietveld(material_obj=mat)

    # Construct via list HKL (exercises the branch we fixed)
    md = MarchDollaseModel(material=matr, HKL=[1, 1, 1], P_MD=1.0)
    assert md.HKL.shape == (3,)

    # HKL setter: wrong shape list
    with pytest.raises(ValueError):
        md.HKL = [1, 1]

    # HKL setter: wrong shape array
    with pytest.raises(ValueError):
        md.HKL = np.array([1, 1])

    # HKL setter: wrong type
    with pytest.raises(TypeError):
        md.HKL = "111"

    # P_MD setter: wrong type
    with pytest.raises(TypeError):
        md.P_MD = "bad"

    # P_MD setter: int is accepted and stored as float
    md.P_MD = 2
    assert isinstance(md.P_MD, float)
    assert md.P_MD == 2.0

    # material setter: wrong type
    with pytest.raises(Exception):
        md.material = "not a material"
