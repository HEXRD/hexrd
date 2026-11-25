import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from hexrd.core.fitting.calibration.powder import PowderCalibrator

# --- Fixtures ---


@pytest.fixture
def mocks():
    """Consolidated mock environment."""
    instr = MagicMock()
    det = MagicMock(
        rows=200,
        cols=200,
        pixel_area=0.01,
        rmat=np.eye(3),
        tvec=np.zeros(3),
        distortion=None,
    )
    det.cart_to_angles.side_effect = lambda xy, **kwargs: (
        np.column_stack([xy[:, 0] * 0.1, xy[:, 1] * 0.1]),
        None,
    )
    det.angles_to_cart.side_effect = lambda angs, **kwargs: angs[:, :2] * 10.0

    instr.detectors = {'det1': det}
    instr.tvec, instr.beam_energy, instr.beam_wavelength = (
        np.zeros(3),
        50.0,
        0.5,
    )
    instr.xrs_beam_energy.return_value = 50.0

    mat = MagicMock()
    mat.planeData.getPlaneSpacings.return_value = np.array([1.0])
    mat.planeData.hkls.T = np.array([[1, 1, 1]])
    mat.planeData.getMergedRanges.return_value = ([[0]], [])
    mat.planeData.latVecOps = {'B': np.eye(3)}
    mat.planeData.tThWidth = np.radians(0.5)

    return {'instr': instr, 'mat': mat}


@pytest.fixture
def pc(mocks):
    mocks['instr'].extract_line_positions.return_value = {
        'det1': [
            [
                (
                    (np.array([1.0]), np.array([0.5])),
                    np.array([100.0]),
                    [np.array([0.1])],
                )
            ]
        ]
    }
    return PowderCalibrator(
        mocks['instr'],
        mocks['mat'],
        {'det1': np.zeros((10, 10))},
        tth_tol=0.5,
        eta_tol=5.0,
    )


# --- Initialization & Properties ---


def test_init_and_properties(mocks, pc):
    with pytest.raises(AssertionError):
        PowderCalibrator(mocks['instr'], mocks['mat'], {'bad_key': []})

    cal_picks = {'det1': {'1 1 1': [[10.0, 10.0]]}}
    pc_picks = PowderCalibrator(
        mocks['instr'], mocks['mat'], {'det1': []}, calibration_picks=cal_picks
    )
    np.testing.assert_array_equal(
        pc_picks.data_dict['det1'][0][:, :2], [[10.0, 10.0]]
    )

    pc.tth_tol = 1.0
    assert np.isclose(pc.plane_data.tThWidth, np.radians(0.5))
    assert pc.spectrum_kwargs['min_ampl'] == 0.0
    assert pc.plane_data == mocks['mat'].planeData

    mock_dist = MagicMock()
    pc.tth_distortion = {'det1': mock_dist}
    assert pc.tth_distortion['det1'].panel == mocks['instr'].detectors['det1']
    assert pc.tth_distortion['det1'] is not mock_dist


# --- LMFit Parameter Logic ---


@patch(
    'hexrd.core.fitting.calibration.powder.create_material_params',
    return_value=[('Mat_a', 1.0, True)],
)
@patch('hexrd.core.fitting.calibration.powder.update_material_from_params')
def test_lmfit_params(mock_update, mock_create, pc):
    assert len(pc.create_lmfit_params([])) == 1
    assert len(pc.create_lmfit_params([('Mat_a', 1.0)])) == 0
    pc.param_names = ['Mat_a']
    pc.update_from_lmfit_params({})
    mock_update.assert_called()


# --- Autopick Points ---


@patch('hexrd.core.fitting.calibration.powder.mutil.findDuplicateVectors')
def test_autopick_scenarios(mock_dupe, pc, mocks):
    mock_dupe.return_value = (None, np.array([0]))
    rhs = pc.autopick_points()
    assert np.isclose(rhs['det1'][0][0, 2], np.radians(0.1))

    pd = mocks['mat'].planeData
    pd.getPlaneSpacings.return_value, pd.hkls.T = np.array(
        [1.0, 1.0]
    ), np.array([[1, 1, 1], [2, 0, 0]])
    pd.getMergedRanges.return_value = ([[0, 1]], [])
    mock_dupe.return_value = (None, np.array([0]))

    res_deg = pc.autopick_points()
    np.testing.assert_array_equal(res_deg['det1'][0][0, 3:6], [1, 1, 1])

    mock_dupe.return_value = (None, np.array([0, 1]))
    multi_peak = [
        (
            (np.array([1.0, 1.1]), np.array([0.5])),
            np.array([100]),
            [np.array([0.1, 0.2])],
        )
    ]
    mocks['instr'].extract_line_positions.return_value = {'det1': [multi_peak]}

    res_multi = pc.autopick_points()
    assert len(res_multi['det1'][0]) == 2


@patch('hexrd.core.fitting.calibration.powder.switch_xray_source')
def test_autopick_empty(mock_switch, pc, mocks):
    mocks['instr'].extract_line_positions.return_value = {
        'det1': [[((np.array([1]), np.array([1])), np.array([]), [0.1])]]
    }
    assert pc.autopick_points()['det1'][0].size == 0

    mocks['instr'].extract_line_positions.return_value = {
        'det1': [[((np.array([1]), np.array([1])), np.array([100]), [None])]]
    }
    assert pc.autopick_points()['det1'][0].size == 0


# --- Calibration Picks & Evaluation ---


def test_calibration_picks_access(pc):
    pc.data_dict = {'det1': [np.array([[10, 10, 0.1, 1, 1, 1, 1.0, 0.5]])]}

    picks = pc.calibration_picks
    assert picks['det1']['1 1 1'] == [[10.0, 10.0]]

    picks['det1']['2 0 0'] = []
    pc.calibration_picks = picks
    assert len(pc.data_dict['det1']) == 1
    np.testing.assert_array_equal(pc.data_dict['det1'][0][:, 3:6], [[1, 1, 1]])


def test_evaluation(pc):
    pc.data_dict = {'det1': [np.array([[10.0, 10.0, 0.5, 1, 1, 1, 1.0, 1.0]])]}

    assert isinstance(pc.residual()[0], float)
    assert pc.model().size == 2

    mock_dist = MagicMock()
    mock_dist.apply.return_value = np.array([[0.1, 0.0]])
    pc.tth_distortion = {'det1': mock_dist}
    pc.residual()
    pc.tth_distortion['det1'].apply.assert_called()

    with pytest.raises(RuntimeError):
        pc._evaluate(output='bad')
    pc.data_dict = {'det1': []}
    assert pc.residual().size == 0


# --- Fixed pink-beam asymmetry forwarding ---


def test_fixed_pink_asymmetry_forwarded_to_spectrum_kwargs(mocks: dict) -> None:
    # The asymmetry dict is stored and passed through to the SpectrumModel
    # via spectrum_kwargs, so it reaches the per-ring fit.
    fixed = {'alpha0': 11.0, 'alpha1': 0.4, 'beta0': 2.1, 'beta1': -5.2}
    pc = PowderCalibrator(
        mocks['instr'],
        mocks['mat'],
        {'det1': np.zeros((10, 10))},
        tth_tol=0.5,
        eta_tol=5.0,
        pktype='pink_beam_dcs',
        fixed_pink_asymmetry=fixed,
    )
    assert pc.fixed_pink_asymmetry == fixed
    assert pc.spectrum_kwargs['fixed_pink_asymmetry'] == fixed


def test_fixed_pink_asymmetry_defaults_to_none(pc: PowderCalibrator) -> None:
    # Default behavior is unchanged: no asymmetry is forwarded.
    assert pc.fixed_pink_asymmetry is None
    assert pc.spectrum_kwargs['fixed_pink_asymmetry'] is None


# --- Fit report ---


class ConstantTThShift:
    """A tth distortion that shifts every point by the same amount"""

    def __init__(self, shift: float):
        self.shift = shift
        self.panel = None

    def apply(self, xy: np.ndarray, return_nominal: bool = True) -> np.ndarray:
        return np.column_stack([np.full(len(xy), self.shift), np.zeros(len(xy))])


def test_fit_report() -> None:
    from hexrd.core.instrument import HEDMInstrument
    from hexrd.core.material import Material

    instr = HEDMInstrument()
    det_key, panel = next(iter(instr.detectors.items()))
    material = Material()
    bmat = material.planeData.latVecOps['B']

    # Pick the Ni (113) and (200) rings, out of 2-theta order
    etas = np.radians(np.linspace(-170, 170, 20))
    rng = np.random.default_rng(0)
    picks = {}
    picked_tths = []
    ring_tths = []
    for hkl in [(1, 1, 3), (2, 0, 0)]:
        dsp = 1 / np.linalg.norm(bmat @ hkl)
        ring_tth = 2 * np.arcsin(instr.beam_wavelength / (2 * dsp))
        tth = ring_tth + rng.normal(0, np.radians(0.01), len(etas))
        xy = panel.angles_to_cart(np.column_stack([tth, etas]))
        picks['%d %d %d' % hkl] = xy.tolist()
        picked_tths.insert(0, tth)
        ring_tths.insert(0, ring_tth)

    # Invalid points should be dropped
    picks['2 0 0'].append([np.nan, np.nan])

    shift = np.radians(0.2)
    calibrator = PowderCalibrator(
        instr,
        material,
        {det_key: None},
        tth_distortion={det_key: ConstantTThShift(shift)},
        calibration_picks={det_key: picks},
    )

    rings = calibrator.fit_report()
    assert [r.hkl.tolist() for r in rings] == [[2, 0, 0], [1, 1, 3]]
    for ring, tth, ring_tth in zip(rings, picked_tths, ring_tths):
        assert np.allclose(ring.tth, np.degrees(tth - shift), atol=1e-6)
        assert np.allclose(ring.eta, np.degrees(etas), atol=1e-6)
        assert np.all(ring.detectors == det_key)
        assert np.isclose(ring.material_tth, np.degrees(ring_tth))
        assert ring.index is None and ring.fit_tth is None

    # The report should describe exactly what the calibration minimizes
    report_residual = np.hstack([r.tth - r.material_tth for r in rings])
    residual = np.degrees(calibrator.residual())
    residual = residual[np.isfinite(residual)]
    assert np.allclose(np.sort(report_residual), np.sort(residual))
