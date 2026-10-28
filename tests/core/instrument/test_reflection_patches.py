import numpy as np
import pytest

from hexrd.core.gridutil import cellConnectivity, make_tolerance_grid
from hexrd.core.instrument import HEDMInstrument
from hexrd.hedm.xrdutil.utils import _reflection_patch_grid, make_reflection_patches


@pytest.mark.parametrize('npdiv', [1, 2, 4])
def test_reflection_grid_keeps_tolerance_grid_rounding(npdiv):
    for pix, tth_tol, eta_tol in [
        (0.02, 0.125, 1.0),
        (0.06, 0.2, 3.0),
        (1.0, 0.075, 0.5),
    ]:
        ntth, tth_edges = make_tolerance_grid(pix, tth_tol, npdiv)
        neta, eta_edges = make_tolerance_grid(3 * pix, eta_tol, npdiv)
        tth, eta = np.meshgrid(tth_edges, eta_edges)
        expected = np.radians(np.c_[tth.ravel(), eta.ravel(), np.zeros(tth.size)])
        offsets, conn = _reflection_patch_grid(ntth, neta, tth_tol, eta_tol)
        np.testing.assert_array_equal(offsets, expected)
        np.testing.assert_array_equal(conn, cellConnectivity(neta, ntth, origin='ll'))


@pytest.mark.parametrize('npdiv', [1, 2, 4])
@pytest.mark.parametrize('distorted', [False, True])
def test_sample_only_patches_keep_full_output_sample_points(npdiv, distorted):
    instr = HEDMInstrument()
    panel = next(iter(instr.detectors.values()))
    panel.saturation_level = 65535
    config = panel.config_dict(
        instr.chi,
        instr.tvec,
        beam_energy=instr.beam_energy,
        beam_vector=instr.beam_vector,
        style='hdf5',
    )
    if distorted:
        config['detector']['distortion'] = {
            'function_name': 'GE_41RT',
            'parameters': [0.01, -0.005, 0.001, 2.0, 2.0, 2.0],
        }
    rng = np.random.default_rng(42)
    angles = np.radians(rng.uniform([3.0, -180.0], [15.0, 180.0], (40, 2)))
    pixels = np.radians(rng.uniform([0.015, 0.04], [0.08, 0.15], (40, 2)))
    args = (config, angles, pixels)
    kwargs = dict(
        omega=rng.uniform(-np.pi, np.pi, 40),
        npdiv=npdiv,
        tvec_c=np.array([0.1, -0.2, 0.3]),
    )
    full = list(make_reflection_patches(*args, **kwargs))
    minimal = list(make_reflection_patches(*args, **kwargs, full_output=False))
    batches = list(
        make_reflection_patches(*args, **kwargs, full_output=False, _packed=True)
    )
    for start, edges, shapes, offsets, rows, cols in batches:
        for i, shape in enumerate(shapes):
            expected = minimal[start + i]
            np.testing.assert_array_equal(
                rows[offsets[i] : offsets[i + 1]].reshape(shape), expected[5][0]
            )
            np.testing.assert_array_equal(
                cols[offsets[i] : offsets[i + 1]].reshape(shape), expected[5][1]
            )
            tth, eta = expected[0]
            np.testing.assert_array_equal(
                edges[i],
                [tth[0, 0], eta[0, 0], tth[0, 1] - tth[0, 0], eta[1, 0] - eta[0, 0]],
            )
    for expected, actual in zip(full, minimal):
        assert actual[1:4] == (None, None, None)
        for index in (0, 4, 5):
            np.testing.assert_array_equal(actual[index], expected[index])
    # A caller may change the connectivity it receives without changing
    # subsequent patches that use the cached grid.
    full[0][2][:] = -1
    again = list(make_reflection_patches(*args, **kwargs))
    assert np.all(again[0][2] >= 0)


def test_reduced_projection_uses_exact_fallback_at_pixel_boundaries(monkeypatch):
    from hexrd.core.extensions import transforms as native
    from hexrd.core.transforms import xfcapi
    from hexrd.hedm.xrdutil import utils

    instr = HEDMInstrument()
    panel = next(iter(instr.detectors.values()))
    panel.saturation_level = 65535
    cfg = panel.config_dict(
        instr.chi,
        instr.tvec,
        beam_energy=instr.beam_energy,
        beam_vector=instr.beam_vector,
        style='hdf5',
    )
    angles = np.radians(np.array([[5.0, 13.0, 30.0]]))
    rc = xfcapi.make_rmat_of_expmap(np.array([0.2, -0.3, 0.4]))
    tc = np.array([0.1, -0.2, 0.3])
    shape = np.array([[10, 4]])
    grid = utils._reflection_patch_offsets(4, 10, 0.2, 1.0)
    points, _, _ = native._spotPatchGrid(angles, shape, [grid])
    transform = cfg['detector']['transform']
    rd = xfcapi.make_rmat_of_expmap(np.array(transform['tilt']))
    td = np.array(transform['translation'])
    xy, _, _ = utils._project_on_detector_plane(
        points, rd, rc, instr.chi, td, tc, instr.tvec, beamVec=instr.beam_vector
    )
    # With a flat default detector, translate its x-origin onto the first
    # sampling point. Its pixel choice now depends on transform roundoff.
    transform['translation'][0] += xy[0, 0]
    args = (cfg, angles[:, :2], np.radians([[0.1, 0.2]]))
    kwargs = dict(
        omega=angles[:, 2],
        tth_tol=0.2,
        eta_tol=1.0,
        rmat_c=rc,
        tvec_c=tc,
        npdiv=2,
        full_output=False,
    )
    expected = list(make_reflection_patches(*args, **kwargs))[0]
    original = utils._project_on_detector_plane
    calls = []

    def track(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(utils, '_project_on_detector_plane', track)
    _, _, shapes, offsets, rows, cols = list(
        make_reflection_patches(*args, **kwargs, _packed=True)
    )[0]
    assert calls, 'boundary-sensitive samples must use the original projection'
    np.testing.assert_array_equal(rows.reshape(shapes[0]), expected[5][0])
    np.testing.assert_array_equal(cols.reshape(shapes[0]), expected[5][1])


@pytest.mark.parametrize('tilted', [False, True])
@pytest.mark.parametrize(
    'beam', [[0.0, 0.0, -1.0], [-6e-17, -6e-17, -1.0], [0.1, -0.2, -1.0]]
)
def test_fused_pixel_grid_matches_reference_for_beam_and_detector_geometry(
    tilted, beam
):
    instr = HEDMInstrument()
    instr.chi = 0.17
    panel = next(iter(instr.detectors.values()))
    panel.saturation_level = 65535
    if tilted:
        panel.tilt = [0.1, -0.03, 0.02]
    config = panel.config_dict(
        instr.chi,
        instr.tvec,
        beam_energy=instr.beam_energy,
        beam_vector=np.array(beam),
        style='hdf5',
    )
    rng = np.random.default_rng(102)
    angles = rng.uniform([0.02, -np.pi], [0.4, np.pi], (20, 2))
    pixels = rng.uniform([0.001, 0.003], [0.004, 0.009], (20, 2))
    kwargs = dict(
        omega=rng.uniform(-np.pi, np.pi, 20),
        npdiv=1,
        tvec_c=np.array([0.1, -0.2, 0.3]),
        tth_tol=0.2,
        eta_tol=1.0,
        full_output=False,
    )
    expected = list(make_reflection_patches(config, angles, pixels, **kwargs))
    for start, edges, shapes, offsets, rows, cols in make_reflection_patches(
        config,
        angles,
        pixels,
        **kwargs,
        _packed=True,
    ):
        for index, shape in enumerate(shapes):
            slc = slice(offsets[index], offsets[index + 1])
            np.testing.assert_array_equal(
                rows[slc].reshape(shape), expected[start + index][5][0]
            )
            np.testing.assert_array_equal(
                cols[slc].reshape(shape), expected[start + index][5][1]
            )
