"""
Eiger subpanel stability tests.

Verifies that the relative tilts/translations between Eiger subpanels
remain stable across calibrations. The core idea: start from a reference
instrument (eig_composite.yml) whose relative subpanel geometry we trust,
adjust the global position (beam energy, chi, mean center) to roughly
match a new dataset, then:

  1. Auto-pick powder points using a well-calibrated instrument
  2. Group-calibrate (rigid body shift of the whole detector)
  3. Individually calibrate each subpanel
  4. Verify relative tilts/translations barely changed in step 3

If the reference subpanel geometry is correct, individual calibration
should be a near-no-op.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pytest
import yaml

from hexrd import imageseries
from hexrd.core.fitting.calibration import (
    InstrumentCalibrator,
    PowderCalibrator,
)
from hexrd.core.fitting.calibration.relative_constraints import (
    RelativeConstraintsType,
)
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.core.instrument.hedm_instrument import HEDMInstrument
from hexrd.core.material.material import load_materials_hdf5, Material


EULER_CONVENTION = ('xyz', True)  # extrinsic XYZ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_relative_translations(
    instr: HEDMInstrument,
    reference_det: str,
) -> dict[str, np.ndarray]:
    ref_tvec = instr.detectors[reference_det].tvec
    return {
        key: panel.tvec - ref_tvec
        for key, panel in instr.detectors.items()
    }


def compute_relative_rmats(
    instr: HEDMInstrument,
    reference_det: str,
) -> dict[str, np.ndarray]:
    ref_rmat = instr.detectors[reference_det].rmat
    return {
        key: panel.rmat @ ref_rmat.T
        for key, panel in instr.detectors.items()
    }


def max_relative_translation_change(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    reference_det: str,
) -> float:
    worst = 0.0
    for key in before:
        if key == reference_det:
            continue
        diff = np.linalg.norm(after[key] - before[key])
        worst = max(worst, diff)
    return worst


def max_relative_rmat_change(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    reference_det: str,
) -> float:
    worst = 0.0
    for key in before:
        if key == reference_det:
            continue
        diff = np.linalg.norm(after[key] - before[key])
        worst = max(worst, diff)
    return worst


def build_img_dict(
    instr: HEDMInstrument,
    monolithic_image: np.ndarray,
) -> dict[str, np.ndarray]:
    ims = imageseries.open(None, format='array', data=monolithic_image)
    img_dict = {}
    for det_key, panel in instr.detectors.items():
        pims = ProcessedImageSeries(ims, oplist=[('rectangle', panel.roi)])
        img_dict[det_key] = pims[0]
    return img_dict


def make_powder_calibrator(
    instr: HEDMInstrument,
    material: Material,
    img_dict: dict[str, np.ndarray],
    picks: dict | None = None,
    tth_tol: float = 0.25,
    eta_tol: float = 0.25,
) -> PowderCalibrator:
    kwargs = dict(
        instr=instr,
        material=material,
        img_dict=img_dict,
        tth_tol=tth_tol,
        eta_tol=eta_tol,
        pktype='gaussian',
    )
    if picks is not None:
        kwargs['calibration_picks'] = picks
    return PowderCalibrator(**kwargs)


def make_instrument_calibrator(
    powder_cal: PowderCalibrator,
    relative_constraints_type: RelativeConstraintsType = (
        RelativeConstraintsType.none
    ),
) -> InstrumentCalibrator:
    calibrator = InstrumentCalibrator(
        powder_cal,
        engineering_constraints=None,
        euler_convention=EULER_CONVENTION,
    )

    # Don't refine material lattice parameters
    for name in powder_cal.param_names:
        calibrator.params[name].vary = False

    if relative_constraints_type != RelativeConstraintsType.none:
        calibrator.relative_constraints_type = relative_constraints_type

    return calibrator


def set_group_params_vary(
    calibrator: InstrumentCalibrator,
    instr: HEDMInstrument,
    vary_tvec: bool = True,
    vary_tilt: bool = True,
):
    """Enable/disable refinement for group-level parameters."""
    for group in instr.detector_groups:
        prefix = group.replace('-', '_')
        for suffix in ('_tvec_x', '_tvec_y', '_tvec_z'):
            name = f'{prefix}{suffix}'
            if name in calibrator.params:
                calibrator.params[name].vary = vary_tvec
        for suffix in ('_euler_x', '_euler_y', '_euler_z'):
            name = f'{prefix}{suffix}'
            if name in calibrator.params:
                calibrator.params[name].vary = vary_tilt

    if 'beam_energy' in calibrator.params:
        calibrator.params['beam_energy'].vary = vary_tvec


def set_individual_params_vary(
    calibrator: InstrumentCalibrator,
    instr: HEDMInstrument,
    vary_tvec: bool = True,
    vary_tilt: bool = True,
):
    """Enable/disable refinement for individual detector parameters."""
    for det_key in instr.detectors:
        for suffix in ('_tvec_x', '_tvec_y', '_tvec_z'):
            name = f'{det_key}{suffix}'
            if name in calibrator.params:
                calibrator.params[name].vary = vary_tvec
        for suffix in ('_euler_x', '_euler_y', '_euler_z'):
            name = f'{det_key}{suffix}'
            if name in calibrator.params:
                calibrator.params[name].vary = vary_tilt


def apply_global_shift(
    instr: HEDMInstrument,
    tvec_delta: np.ndarray,
):
    """Shift all panels by the same translation delta."""
    for panel in instr.detectors.values():
        panel.tvec = panel.tvec + tvec_delta


def adjust_global_position(
    instr: HEDMInstrument,
    target_instr: HEDMInstrument,
):
    """Adjust instr's global position to roughly match target_instr.

    Copies beam energy and chi from target, then shifts all panels so
    the mean group center matches.  Preserves relative subpanel geometry.
    """
    # Match beam energy and chi
    instr.beam_energy = target_instr.beam_energy
    instr.chi = target_instr.chi

    # Shift all panels so the mean center matches
    group = instr.detector_groups[0]
    src_center = instr.mean_group_center(group)
    tgt_center = target_instr.mean_group_center(group)
    apply_global_shift(instr, tgt_center - src_center)


def print_calibration_summary(
    instr: HEDMInstrument,
    reference_det: str,
    label: str,
    relative_tvecs_before: dict | None = None,
    relative_rmats_before: dict | None = None,
):
    """Print a summary of current relative geometry (useful for debugging)."""
    rel_tvecs = compute_relative_translations(instr, reference_det)
    rel_rmats = compute_relative_rmats(instr, reference_det)

    print(f'\n=== {label} ===')
    if relative_tvecs_before is not None:
        max_tvec = max_relative_translation_change(
            relative_tvecs_before, rel_tvecs, reference_det,
        )
        max_rmat = max_relative_rmat_change(
            relative_rmats_before, rel_rmats, reference_det,
        )
        print(f'  Max relative tvec change: {max_tvec:.6f} mm')
        print(f'  Max relative rmat change: {max_rmat:.6e}')

    return rel_tvecs, rel_rmats


def print_per_panel_breakdown(
    instr: HEDMInstrument,
    reference_det: str,
    before_tvecs: dict[str, np.ndarray],
    after_tvecs: dict[str, np.ndarray],
    before_rmats: dict[str, np.ndarray],
    after_rmats: dict[str, np.ndarray],
    tvec_tol: float,
    rmat_tol: float,
    picks: dict | None = None,
):
    print(f'\nPer-panel breakdown:')
    for det_key in sorted(instr.detectors):
        if det_key == reference_det:
            continue
        tvec_diff = np.linalg.norm(
            after_tvecs[det_key] - before_tvecs[det_key]
        )
        rmat_diff = np.linalg.norm(
            after_rmats[det_key] - before_rmats[det_key]
        )
        flag = ''
        if tvec_diff > tvec_tol or rmat_diff > rmat_tol:
            flag = ' ***'

        n_picks = 0
        n_hkls = 0
        if picks and det_key in picks:
            det_picks = picks[det_key]
            n_hkls = len(det_picks)
            n_picks = sum(len(v) for v in det_picks.values())

        print(
            f'  {det_key}: '
            f'tvec={tvec_diff:.4f} mm, rmat={rmat_diff:.2e}, '
            f'picks={n_picks} ({n_hkls} hkls){flag}'
        )


def picks_to_monolithic_pixels(
    picks: dict,
    instr: HEDMInstrument,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cartesian picks to monolithic image pixel coordinates."""
    all_rows = []
    all_cols = []
    for det_key, det_picks in picks.items():
        panel = instr.detectors[det_key]
        roi = panel.roi
        # roi is ((row_start, row_end), (col_start, col_end))
        roi_row_start = roi[0][0]
        roi_col_start = roi[1][0]
        for hkl_str, coords in det_picks.items():
            coords = np.asarray(coords, dtype=float)
            if coords.ndim != 2 or coords.shape[0] == 0:
                continue
            # Cartesian (x, y) in mm -> panel pixel (i, j)
            panel_j = coords[:, 0] / panel.pixel_size_col + panel.cols / 2
            panel_i = -coords[:, 1] / panel.pixel_size_row + panel.rows / 2
            # Panel pixel -> monolithic pixel
            all_rows.append(roi_row_start + panel_i)
            all_cols.append(roi_col_start + panel_j)

    return np.concatenate(all_rows), np.concatenate(all_cols)


def save_picks_image(
    monolithic_image: np.ndarray,
    instr: HEDMInstrument,
    picks: dict,
    path: str | Path,
    title: str = 'Picks overlay',
):
    """Save the monolithic image with picked points overlaid."""
    rows, cols = picks_to_monolithic_pixels(picks, instr)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    vmin = max(1, np.percentile(monolithic_image, 5))
    vmax = np.percentile(monolithic_image, 99.5)
    ax.imshow(
        monolithic_image,
        cmap='gray',
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation='nearest',
    )
    ax.scatter(cols, rows, s=3, c='lime', alpha=0.7, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('column (pixels)')
    ax.set_ylabel('row (pixels)')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'\nSaved picks image to {path}')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def eiger_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'eiger'


@pytest.fixture
def ceria_examples_path(eiger_examples_path: Path) -> Path:
    return eiger_examples_path / 'first_ceria'


@pytest.fixture
def reference_instrument(
    eiger_examples_path: Path,
) -> HEDMInstrument:
    """The reference Eiger composite with trusted relative subpanel geometry."""
    path = eiger_examples_path / 'chess_eiger_composite_standard.yml'
    with open(path, 'r') as rf:
        config = yaml.safe_load(rf)
    return HEDMInstrument(config)


@pytest.fixture
def ceria_calibrated_instrument(
    ceria_examples_path: Path,
) -> HEDMInstrument:
    """Already-calibrated instrument for ceria (used to get global position
    and for auto-picking)."""
    instr_path = ceria_examples_path / 'eiger_ceria_calibrated_composite.hexrd'
    with h5py.File(instr_path, 'r') as rf:
        return HEDMInstrument(rf)


@pytest.fixture
def ceria_image(ceria_examples_path: Path) -> np.ndarray:
    data_path = ceria_examples_path / 'ff_000_data_000001.h5'
    with h5py.File(data_path, 'r') as rf:
        return rf['/entry/data/data'][0]


@pytest.fixture
def ceria_material(ceria_examples_path: Path) -> Material:
    path = ceria_examples_path / 'ceria.h5'
    materials = load_materials_hdf5(path)
    return materials['CeO2']


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEigerSubpanelStabilityPowder:
    """Verify Eiger subpanel relative geometry is stable for powder data."""

    REFERENCE_DET = 'eiger_0_0'

    # Tolerances: how much individual subpanel calibration can move panels
    # relative to the group-calibrated result.
    # tvec tolerance in mm, rmat tolerance is Frobenius norm of rmat diff.
    # These are intentionally generous starting points; tighten as the
    # reference calibration improves.
    MAX_RELATIVE_TVEC_CHANGE_MM = 2.0
    MAX_RELATIVE_RMAT_CHANGE = 0.05

    def test_subpanel_stability_ceria(
        self,
        reference_instrument: HEDMInstrument,
        ceria_calibrated_instrument: HEDMInstrument,
        ceria_material: Material,
        ceria_image: np.ndarray,
    ):
        """
        Starting from eig_composite.yml (reference subpanel geometry):
        1. Adjust global position to roughly match ceria dataset
        2. Auto-pick using the calibrated instrument (known good positions)
        3. Group-calibrate the reference instrument (rigid body)
        4. Individually calibrate each subpanel
        5. Verify relative geometry barely changed in step 4
        """
        instr = reference_instrument
        calibrated_instr = ceria_calibrated_instrument
        material = ceria_material
        material.planeData.exclusions = None

        # --- Step 1: Adjust global position ---
        adjust_global_position(instr, calibrated_instr)

        img_dict = build_img_dict(instr, ceria_image)

        # --- Step 2: Auto-pick using the calibrated instrument ---
        # We auto-pick with the calibrated instrument because its positions
        # are already correct, giving us reliable picks. These picks (in
        # cartesian detector coords) are then used to calibrate the
        # reference instrument.
        cal_img_dict = build_img_dict(calibrated_instr, ceria_image)
        cal_powder = make_powder_calibrator(
            calibrated_instr, material, cal_img_dict,
        )
        cal_powder.autopick_points(fit_tth_tol=1.0, int_cutoff=1e-4)
        picks = cal_powder.calibration_picks

        picks_per_det = {}
        for det_key, det_picks in picks.items():
            picks_per_det[det_key] = sum(
                len(v) for v in det_picks.values()
            )

        total_picks = sum(picks_per_det.values())
        print(f'\nAuto-picked {total_picks} total points')
        assert total_picks > 2000, (
            f'Too few total picks ({total_picks}), expected >2000'
        )
        for det_key, n in picks_per_det.items():
            assert n > 10, (
                f'{det_key} has only {n} picks, expected >10'
            )

        # Record the reference relative geometry before any calibration
        orig_rel_tvecs = compute_relative_translations(
            instr, self.REFERENCE_DET,
        )
        orig_rel_rmats = compute_relative_rmats(instr, self.REFERENCE_DET)

        # --- Step 3: Group-calibrate (rigid body) ---
        powder_cal_group = make_powder_calibrator(
            instr, material, img_dict, picks=picks,
        )
        group_calibrator = make_instrument_calibrator(
            powder_cal_group,
            relative_constraints_type=RelativeConstraintsType.group,
        )
        set_group_params_vary(group_calibrator, instr)

        group_calibrator.run_calibration({})

        post_group_rel_tvecs = compute_relative_translations(
            instr, self.REFERENCE_DET,
        )
        post_group_rel_rmats = compute_relative_rmats(
            instr, self.REFERENCE_DET,
        )

        print_calibration_summary(
            instr, self.REFERENCE_DET, 'After group calibration',
            orig_rel_tvecs, orig_rel_rmats,
        )

        # --- Step 4: Individual subpanel calibration ---
        powder_cal_indiv = make_powder_calibrator(
            instr, material, img_dict, picks=picks,
        )
        indiv_calibrator = make_instrument_calibrator(powder_cal_indiv)
        set_individual_params_vary(indiv_calibrator, instr)

        indiv_calibrator.run_calibration({})

        # --- Step 5: Verify relative geometry barely changed ---
        post_indiv_rel_tvecs = compute_relative_translations(
            instr, self.REFERENCE_DET,
        )
        post_indiv_rel_rmats = compute_relative_rmats(
            instr, self.REFERENCE_DET,
        )

        print_calibration_summary(
            instr, self.REFERENCE_DET,
            'After individual calibration',
            post_group_rel_tvecs, post_group_rel_rmats,
        )

        max_tvec_change = max_relative_translation_change(
            post_group_rel_tvecs,
            post_indiv_rel_tvecs,
            self.REFERENCE_DET,
        )
        max_rmat_change = max_relative_rmat_change(
            post_group_rel_rmats,
            post_indiv_rel_rmats,
            self.REFERENCE_DET,
        )

        print(f'\n--- Results ---')
        print(
            f'Max relative tvec change after individual calibration: '
            f'{max_tvec_change:.6f} mm '
            f'(tolerance: {self.MAX_RELATIVE_TVEC_CHANGE_MM} mm)'
        )
        print(
            f'Max relative rmat change after individual calibration: '
            f'{max_rmat_change:.6e} '
            f'(tolerance: {self.MAX_RELATIVE_RMAT_CHANGE})'
        )

        print_per_panel_breakdown(
            instr, self.REFERENCE_DET,
            post_group_rel_tvecs, post_indiv_rel_tvecs,
            post_group_rel_rmats, post_indiv_rel_rmats,
            self.MAX_RELATIVE_TVEC_CHANGE_MM,
            self.MAX_RELATIVE_RMAT_CHANGE,
            picks=picks,
        )

        save_picks_image(
            ceria_image, instr, picks,
            path='/tmp/eiger_ceria_picks.png',
            title='Ceria picks after individual calibration',
        )

        assert max_tvec_change < self.MAX_RELATIVE_TVEC_CHANGE_MM, (
            f'Subpanel relative translation changed by {max_tvec_change:.4f} '
            f'mm after individual calibration (tolerance: '
            f'{self.MAX_RELATIVE_TVEC_CHANGE_MM} mm)'
        )
        assert max_rmat_change < self.MAX_RELATIVE_RMAT_CHANGE, (
            f'Subpanel relative rmat changed by {max_rmat_change:.2e} '
            f'after individual calibration (tolerance: '
            f'{self.MAX_RELATIVE_RMAT_CHANGE})'
        )


class TestEigerSubpanelStabilityHEDM:
    """Placeholder for HEDM dataset stability tests.

    To add a new HEDM dataset:
    1. Add fixtures for the instrument, image data, material, and grain params
    2. Write a test method following the pattern in the powder class:
       - Load instrument and data
       - Pick spots (or use pre-existing picks)
       - Group-calibrate (rigid body)
       - Individual-calibrate
       - Verify relative geometry stability
    """
    pass
