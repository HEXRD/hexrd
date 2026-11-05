from pathlib import Path

import h5py
import lmfit
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
            assert np.allclose(d1[key], d2[key], equal_nan=True, rtol=2e-2)


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

    # P_MD should refine away from random and improve the fit
    assert not np.isclose(R.params['Ni_p_md'].value, 1.0)
    assert R.Rwp < rwp_before_pmd - 0.01

    # Harmonic texture refinement skips March-Dollase models
    R.RefineTexture()

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
    assert R.params['Ni_p_md'].value == 1.5  # taken from P_MD

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

    # With the first peaks outside the tth range, each factor must still
    # go to its own hkl
    first_peaks = np.sort(R.tth['Ni']['XFEL'])[:2]
    kwargs['expt_spectrum'] = expt_spec[expt_spec[:, 0] > first_peaks[-1] + 0.1]
    R = Rietveld(**kwargs)
    hkls = R.hkls['Ni']['XFEL']
    assert len(hkls) < len(md.material.hkls)
    expected = [
        md.texture_factors[np.all(md.material.hkls == h, axis=1)][0] for h in hkls
    ]
    assert np.allclose(R.phase_texture_factors('Ni', 'XFEL'), expected)

    # A texture model whose hkls differ from the phase's is rejected
    md.material.hkls = md.material.hkls[::-1]
    with pytest.raises(ValueError):
        R.phase_texture_factors('Ni', 'XFEL')


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


def test_texture_factor_fully_masked_ring(texture_instrument: HEDMInstrument) -> None:
    mat = Material(dmin=_angstrom(0.5), kev=_kev(texture_instrument.beam_energy))
    hm = HarmonicModel(
        material=Material_Rietveld(material_obj=mat),
        ssym='axial',
        ell_max=4,
        bvec=texture_instrument.beam_vector,
        evec=texture_instrument.eta_vector,
    )
    params = lmfit.Parameters()
    hm.get_parameters(params, vary=False)
    expected = hm.calc_texture_factor(params, eta_step=np.radians(5))

    # A ring that is masked at every eta falls back to the full ring
    hkl = next(iter(hm.intensities_rings))
    eta_mask = {hkl: np.ones_like(hm.intensities_rings[hkl], dtype=bool)}
    tf = hm.calc_texture_factor(params, eta_step=np.radians(5), eta_mask=eta_mask)
    np.testing.assert_allclose(tf, expected)


def test_harmonic_model_fits_superposed_overlap():
    # 333 and 511 of fcc Ni sit at the same 2theta; the data of the pair is
    # sum_k c_k P_k / sum_k c_k with c = m |F|^2 (8 : 24, equal |F|^2)
    import contextlib
    import io

    from hexrd.valunits import _kev

    ni = Material(kev=_kev(100.0), dmin=_angstrom(0.6))
    ni.planeData.exclusions = None
    mat = Material_Rietveld(material_obj=ni)
    hkls = np.array([[1, 1, 1], [2, 0, 0], [3, 3, 3], [5, 1, 1]])

    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(400, 3))
    dirs[:, 2] = np.abs(dirs[:, 2])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    truth = HarmonicModel(material=mat, ssym='triclinic', ell_max=6)
    true = truth.get_parameters()
    for name in true:
        true[name].value = 0.3 * rng.normal()
    truth.calc_new_pole_figure(true, hkls=hkls, pfgrid=dirs)
    P = {tuple(h): truth.intensities_new[tuple(h)] for h in hkls}

    c = np.array([8.0, 24.0])
    pair = (c[0] * P[(3, 3, 3)] + c[1] * P[(5, 1, 1)]) / c.sum()

    def fit(superposition):
        model = HarmonicModel(material=mat, ssym='triclinic', ell_max=6)
        model.superposition = superposition
        model.pfdata = {
            (1, 1, 1): np.column_stack((dirs, P[(1, 1, 1)])),
            (2, 0, 0): np.column_stack((dirs, P[(2, 0, 0)])),
            (3, 3, 3): np.column_stack((dirs, pair)),
        }
        params = model.get_parameters()
        with contextlib.redirect_stdout(io.StringIO()):
            model.calculate_harmonic_coefficients(params)
        return np.array([params[name].value for name in true])

    expected = np.array([true[name].value for name in true])
    superposed = fit({(3, 3, 3): ([(3, 3, 3), (5, 1, 1)], c)})
    np.testing.assert_allclose(superposed, expected, atol=1e-6)

    # Read as a pure 333 pole figure, the same data give the wrong texture
    assert np.max(np.abs(fit({}) - expected)) > 1e-2


# ---------------------------------------------------------------------------
# DeLaValleePoussinModel
# ---------------------------------------------------------------------------


def _ni_rietveld_material(dmin=0.6):
    from hexrd.valunits import _kev

    ni = Material(kev=_kev(100.0), dmin=_angstrom(dmin))
    ni.planeData.exclusions = None
    return Material_Rietveld(material_obj=ni)


@pytest.mark.parametrize('ssym', ['triclinic', 'orthorhombic', 'axial'])
def test_kernel_model_operator_matches_hexrd_pole_density(ssym):
    import hexrd.phase_transition.texture as hx
    from hexrd.wppf.texture import DeLaValleePoussinModel

    model = DeLaValleePoussinModel(
        material=_ni_rietveld_material(), ssym=ssym, halfwidth=np.radians(10)
    )
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(100, 3))
    cols = rng.choice(len(model.orientations), 4, replace=False)
    for hkl in [(1, 1, 1), (3, 1, 1)]:
        A = model.pole_figure_operator(hkl, dirs, cols)
        hc = model._crystal_direction(hkl)
        ref = np.column_stack(
            [
                hx.pole_density(hx.UnimodalODF(model.orientations[c], model.kernel), hc, dirs)
                for c in cols
            ]
        )
        np.testing.assert_allclose(A, ref, rtol=1e-9, atol=1e-9 * ref.max())


def test_kernel_model_recovers_texture_from_single_rings():
    import hexrd.phase_transition.texture as hx
    from scipy.spatial.transform import Rotation
    from hexrd.transforms.xfcapi import angles_to_gvec
    from hexrd.wppf.texture import DeLaValleePoussinModel, bvec_ref, eta_ref

    mat = _ni_rietveld_material()
    sample_rmat = rotMatOfQuat(
        quatOfAngleAxis(np.array([np.radians(22.5)]), np.atleast_2d([0, 1, 0]).T)
    ).reshape(3, 3)

    # truth: two 8 degree components on 20 % random
    centres = Rotation.from_euler('ZXZ', [[35, 45, 10], [80, 20, 60]], degrees=True)
    truth = hx.CompositeODF(
        [
            hx.UnimodalODF(
                centres.as_matrix(),
                hx.DeLaValleePoussinKernel(np.radians(8), crystal_symmetry='oh'),
                weights=[0.6, 0.4],
            )
        ],
        coefficients=[0.8],
        constant=0.2,
    )
    tth = dict(zip(map(tuple, mat.hkls), np.radians(mat.getTTh(mat.wavelength))))
    eta = np.radians(np.arange(-180, 180, 5) + 2.5)

    def ring(hkl):
        angs = np.vstack((np.full(eta.size, tth[hkl]), eta, np.zeros(eta.size))).T
        g = angles_to_gvec(angs, beam_vec=bvec_ref, eta_vec=eta_ref)
        return g, (sample_rmat @ g.T).T

    def model_for(scale):
        model = DeLaValleePoussinModel(
            material=mat, ssym='triclinic', halfwidth=np.radians(7.5), sample_rmat=sample_rmat
        )
        pfdata = {}
        for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]:
            g, y = ring(hkl)
            pfdata[hkl] = np.column_stack(
                (g, scale * hx.pole_density(truth, model._crystal_direction(hkl), y))
            )
        model.pfdata = pfdata
        model.fit_texture()
        return model

    model = model_for(1.0)
    assert model.fit_result['relative_residual'] < 0.05

    # one lmfit parameter stands for all coefficients: its bounds bound
    # every coefficient, and the fit reports their mean standard error
    params = model.get_parameters()
    name = model.coefficients_parameter_name
    assert (params[name].min, params[name].max) == (0, np.inf)
    model.fit_texture(params)
    assert model.fit_result['bounds'] == (0.0, np.inf)
    mean = model.fit_result['mean_stderr']
    assert mean is None or (mean > 0 and params[name].stderr == mean)
    assert params[name].value == pytest.approx(model.kernel_weights[model.active].mean())
    params[name].set(max=0.004)
    model.fit_texture(params)
    assert model.fit_result['method'] == 'bvls'
    assert model.kernel_weights.max() * model.data_scale <= 0.004 + 1e-9
    params[name].set(min=-1.0)
    with pytest.raises(ValueError, match='>= 0'):
        model.fit_texture(params)
    model = model_for(1.0)
    for hkl in [(2, 2, 2), (3, 3, 1), (4, 2, 0)]:   # rings that were not fitted
        _, y = ring(hkl)
        true = hx.pole_density(truth, model._crystal_direction(hkl), y)
        assert np.linalg.norm(model.pole_density(hkl, y) - true) < 0.15 * np.linalg.norm(true)

    # the ODF is normalised to 1 MRD: an error in the data scale is absorbed
    scaled = model_for(1.07)
    assert np.isclose(scaled.data_scale, 1.07 * model.data_scale)
    np.testing.assert_allclose(
        scaled.kernel_weights, model.kernel_weights, rtol=1e-6, atol=1e-12
    )

    # the fitted ODF survives new or cleared pole figure data
    active = model.active
    model.pfdata = {}
    np.testing.assert_array_equal(model.active, active)
    model.pfdata = scaled.pfdata
    np.testing.assert_array_equal(model.active, active)
    tf = scaled.calc_texture_factor(None, eta_step=np.radians(5))
    assert tf.shape == (len(mat.hkls),) and np.all(tf > 0)

    # pole figures on a new grid, and plots of the ring data (which
    # tricontour alone cannot triangulate)
    import matplotlib

    matplotlib.use('Agg')
    model.calc_new_pole_figure(None, hkls=np.array([[1, 1, 1], [2, 0, 0]]))
    assert set(model.intensities_new) == {(1, 1, 1), (2, 0, 0)}
    assert all(np.all(v >= 0) for v in model.intensities_new.values())
    model.plot_new_pf(show=False)
    assert model.plot_pf(show=False) is model.fig
    fig = model.plot_pf(show=False, recalculated=True, filled=True)
    assert len(fig.axes) >= 2 * len(model.pfdata)   # panels and colorbars
    matplotlib.pyplot.close('all')
    assert not model.new_pf_plots_visible


def test_wppf_kernel_texture(texture_instrument, texture_img_dict):
    from hexrd.wppf.texture import DeLaValleePoussinModel

    instr = texture_instrument
    q = quatOfAngleAxis(np.array([np.radians(22.5)]), np.atleast_2d(np.array([0, 1, 0])).T)
    mat = Material(dmin=_angstrom(0.5), kev=_kev(instr.beam_energy))
    mat.planeData.exclusions = np.zeros_like(mat.planeData.exclusions).astype(bool)
    model = DeLaValleePoussinModel(
        material=Material_Rietveld(material_obj=mat),
        ssym='axial',
        halfwidth=np.radians(10),
        bvec=instr.beam_vector,
        evec=instr.eta_vector,
        sample_rmat=rotMatOfQuat(q),
    )
    polar_obj = PolarView(
        (2, 40), instr, eta_min=-180.0, eta_max=180.0, pixel_size=(0.05, 0.1),
        cache_coordinate_map=True,
    )
    pv = polar_obj.warp_image(texture_img_dict, pad_with_nans=True, do_interpolation=True)
    ttharray = np.degrees(polar_obj.angular_grid[1][0, :])
    lo = get_lineout(pv)
    R = Rietveld(
        expt_spectrum=np.vstack((ttharray[~lo.mask], lo.data[~lo.mask])).T,
        wavelength={'XFEL': [_angstrom(instr.beam_wavelength), 1.0]},
        bkgmethod={'chebyshev': 1},
        peakshape='pvtch',
        phases=[mat],
        texture_model={'Ni': model},
        eta_min=-180,
        eta_max=180,
        eta_step=5,
    )
    # before any fit the model is the random texture
    assert R.texture_index['Ni'] == 1.0

    for names in (['scale', 'bkg_0', 'bkg_1'], ['U', 'V', 'W'], ['scale']):
        R.params_vary_off()
        for name in names:
            R.params[name].vary = True
        if 'bkg_0' in names:
            R.params['Ni_X'].value = 0
            R.params['Ni_Y'].value = 0
        R.Refine()
    rwp_random = R.Rwp

    R.compute_texture_data(
        bin_polar_view(polar_obj, pv, 5, 1),
        bvec=instr.beam_vector,
        evec=instr.eta_vector,
        azimuthal_interval=5,
    )
    R.params_vary_off()
    R.RefineTexture()   # coefficients not marked as vary: nothing is fitted
    assert len(model.active) == 0

    R.texture_parameters_vary(True)
    coefficients = R.params['Ni_kernel_coefficients']
    assert coefficients.vary and (coefficients.min, coefficients.max) == (0, np.inf)
    R.RefineTexture()

    assert 0 < len(model.active) < len(model.orientations)
    assert coefficients.value == pytest.approx(model.kernel_weights[model.active].mean())
    assert coefficients.stderr is None or coefficients.stderr > 0
    assert R.Rwp < rwp_random - 0.02
    assert R.texture_index['Ni'] > 1.5
    R.computespectrum_2D()
    assert R.simulated_2d.ndim == 2
