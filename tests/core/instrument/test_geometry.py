"""Tests for hexrd.core.instrument.geometry.

The decomposition splits the geometry change between two calibrations of one
instrument into a single whole-detector "monolith" move plus small per-panel
residuals. We exercise it on the 32-subpanel Eiger composite instrument that
ships in the example repo, in two complementary ways:

* On a SYNTHETIC final instrument built from the real Eiger geometry by planting
  a *known* monolith move (plus tiny residuals, and -- for the robust tests --
  a couple of deliberately-moved panels). Because the ground truth is known, we
  can assert the monolith and residuals are recovered.

* On the REAL uncalibrated -> calibrated Eiger pair, where we check the
  structural invariants that must hold for any input (exact reconstruction,
  mean-zero residuals, the monolith absorbing the common motion).
"""

import copy
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.instrument.geometry import (
    apply_monolith_move,
    decompose_instrument_motion,
    reconstruct_final,
)

PlantedTruth = tuple[HEDMInstrument, np.ndarray, np.ndarray]
PlantedWithOutliers = tuple[HEDMInstrument, np.ndarray, np.ndarray, str, str]


# ======================================================================
# Fixtures: the 32-subpanel Eiger composite instruments
# ======================================================================
@pytest.fixture
def eiger_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'eiger' / 'first_ceria'


@pytest.fixture
def eiger_uncalibrated(eiger_examples_path: Path) -> HEDMInstrument:
    path = eiger_examples_path / 'eiger_ceria_uncalibrated_composite.hexrd'
    with h5py.File(path, 'r') as rf:
        return HEDMInstrument(rf)


@pytest.fixture
def eiger_calibrated(eiger_examples_path: Path) -> HEDMInstrument:
    path = eiger_examples_path / 'eiger_ceria_calibrated_composite.hexrd'
    with h5py.File(path, 'r') as rf:
        return HEDMInstrument(rf)


# ======================================================================
# Helpers
# ======================================================================
def _expmap(rmat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rmat).as_rotvec()


def _rmat(tilt: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(tilt).as_matrix()


def _misorientation_deg(tilt_a: np.ndarray, tilt_b: np.ndarray) -> float:
    q = Rotation.from_rotvec(tilt_a).inv() * Rotation.from_rotvec(tilt_b)
    return float(np.degrees(q.magnitude()))


def plant_motion(base: HEDMInstrument, monolith_tilt: np.ndarray,
                 monolith_trans: np.ndarray, res_tilts: np.ndarray,
                 res_disps: np.ndarray) -> HEDMInstrument:
    """Build a 'final' instrument from ``base`` by applying a KNOWN motion.

    The monolith rotation turns about the (uniform) panel centroid and then the
    centroid is translated by ``monolith_trans``; each panel additionally gets a
    small residual rotation about its own center and a residual center shift.
    This is exactly the forward model the decomposition inverts, so a clean
    planted move should be recovered.
    """
    out = copy.deepcopy(base)
    keys = list(out.detectors)
    tA = np.array([np.asarray(base.detectors[k].tvec, float) for k in keys])
    cen = tA.mean(axis=0)
    QM = _rmat(np.asarray(monolith_tilt, float))

    for i, k in enumerate(keys):
        det = out.detectors[k]
        RA = base.detectors[k].rmat
        R_int = QM @ RA
        t_int = QM @ (tA[i] - cen) + cen + np.asarray(monolith_trans, float)
        QS = _rmat(np.asarray(res_tilts[i], float))
        det.tilt = _expmap(QS @ R_int)
        det.tvec = t_int + np.asarray(res_disps[i], float)
    return out


@pytest.fixture
def planted_truth(
    eiger_uncalibrated: HEDMInstrument,
) -> PlantedTruth:
    """A synthetic clean move: known monolith + tiny mean-zero residuals."""
    rng = np.random.default_rng(0)
    n = len(eiger_uncalibrated.detectors)

    monolith_tilt = np.array([0.004, -0.003, 0.0015])   # ~0.3 deg
    monolith_trans = np.array([0.30, -0.15, 0.80])      # mm

    res_tilts = rng.normal(0, 2e-4, (n, 3))             # ~0.01 deg jitter
    res_disps = rng.normal(0, 0.01, (n, 3))             # ~10 micron jitter
    # Make the residual translations *exactly* common-mode free so the planted
    # monolith translation is recovered without a centroid bias.
    res_disps -= res_disps.mean(axis=0)

    final = plant_motion(eiger_uncalibrated, monolith_tilt, monolith_trans,
                         res_tilts, res_disps)
    return final, monolith_tilt, monolith_trans


# ======================================================================
# Synthetic ground-truth recovery
# ======================================================================
def test_recovers_planted_monolith(
    eiger_uncalibrated: HEDMInstrument, planted_truth: PlantedTruth,
) -> None:
    final, monolith_tilt, monolith_trans = planted_truth

    d = decompose_instrument_motion(eiger_uncalibrated, final)

    # Monolith move recovered (tiny residual jitter sets the floor).
    assert _misorientation_deg(d.monolith.tilt, monolith_tilt) < 1e-2
    np.testing.assert_allclose(d.monolith.translation, monolith_trans, atol=2e-3)

    # Residuals stay at the planted jitter scale, nowhere near the monolith.
    assert d.max_residual_angle < 0.05      # deg  (planted ~0.01 deg)
    assert d.max_residual_distance < 0.05   # mm   (planted ~0.01 mm)
    assert d.monolith.angle > 0.25
    assert d.monolith.distance > 0.7


def test_reconstruct_recovers_planted_final(
    eiger_uncalibrated: HEDMInstrument, planted_truth: PlantedTruth,
) -> None:
    final, _, _ = planted_truth
    d = decompose_instrument_motion(eiger_uncalibrated, final)

    rec = reconstruct_final(eiger_uncalibrated, d)
    for k in final.detectors:
        assert _misorientation_deg(
            rec.detectors[k].tilt, final.detectors[k].tilt) < 1e-9
        np.testing.assert_allclose(
            rec.detectors[k].tvec, final.detectors[k].tvec, atol=1e-9)


# ======================================================================
# Robust variant: isolate deliberately-moved panels
# ======================================================================
@pytest.fixture
def planted_with_outliers(
    eiger_uncalibrated: HEDMInstrument,
) -> PlantedWithOutliers:
    rng = np.random.default_rng(1)
    keys = list(eiger_uncalibrated.detectors)
    n = len(keys)

    monolith_tilt = np.array([0.004, -0.003, 0.0015])
    monolith_trans = np.array([0.30, -0.15, 0.80])

    res_tilts = rng.normal(0, 2e-4, (n, 3))
    res_disps = rng.normal(0, 0.01, (n, 3))

    # Two genuine movers: a big extra tilt and a big extra shift.
    tilt_mover, disp_mover = keys[7], keys[20]
    res_tilts[7] += np.array([0.02, 0.0, 0.0])          # ~1.1 deg
    res_disps[20] += np.array([1.0, -0.8, 0.0])         # ~1.28 mm

    final = plant_motion(eiger_uncalibrated, monolith_tilt, monolith_trans,
                         res_tilts, res_disps)
    return final, monolith_tilt, monolith_trans, tilt_mover, disp_mover


def test_robust_recovers_monolith_despite_outliers(
    eiger_uncalibrated: HEDMInstrument,
    planted_with_outliers: PlantedWithOutliers,
) -> None:
    final, monolith_tilt, monolith_trans, _, _ = planted_with_outliers

    plain = decompose_instrument_motion(eiger_uncalibrated, final)
    robust = decompose_instrument_motion(eiger_uncalibrated, final, robust=True)

    # The robust monolith is closer to the planted truth than the plain one,
    # which the two movers have biased -- in both rotation and translation.
    plain_err = _misorientation_deg(plain.monolith.tilt, monolith_tilt)
    robust_err = _misorientation_deg(robust.monolith.tilt, monolith_tilt)
    assert robust_err < plain_err
    assert robust_err < 1e-2

    # (Down-weighting the movers leaves a little of the monolith rotation-swing
    # in the weighted centroid, so robust translation lands within ~0.02 mm
    # rather than exactly -- still well inside the plain estimate's bias.)
    plain_trans_err = np.linalg.norm(plain.monolith.translation - monolith_trans)
    robust_trans_err = np.linalg.norm(robust.monolith.translation - monolith_trans)
    assert robust_trans_err < plain_trans_err
    np.testing.assert_allclose(
        robust.monolith.translation, monolith_trans, atol=2e-2)


def test_robust_flags_the_movers(
    eiger_uncalibrated: HEDMInstrument,
    planted_with_outliers: PlantedWithOutliers,
) -> None:
    final, _, _, tilt_mover, disp_mover = planted_with_outliers
    d = decompose_instrument_motion(eiger_uncalibrated, final, robust=True)

    # The tilt mover has the largest residual angle; the shift mover the largest
    # residual distance -- each well clear of the jitter floor.
    angles = d.residual_angles
    dists = d.residual_distances
    assert max(angles, key=angles.get) == tilt_mover
    assert max(dists, key=dists.get) == disp_mover
    assert angles[tilt_mover] > 0.5      # deg
    assert dists[disp_mover] > 0.5       # mm

    # Movers are down-weighted; the quiet panels keep ~full weight.
    assert d.residuals[tilt_mover].weight < 0.5
    assert d.residuals[disp_mover].weight < 0.5
    quiet = [k for k in final.detectors if k not in (tilt_mover, disp_mover)]
    assert np.median([d.residuals[k].weight for k in quiet]) > 0.9


# ======================================================================
# Structural invariants on the REAL uncalibrated -> calibrated pair
# ======================================================================
def test_real_pair_reconstructs_exactly(
    eiger_uncalibrated: HEDMInstrument, eiger_calibrated: HEDMInstrument,
) -> None:
    d = decompose_instrument_motion(eiger_uncalibrated, eiger_calibrated)
    rec = reconstruct_final(eiger_uncalibrated, d)

    for k in eiger_calibrated.detectors:
        assert _misorientation_deg(
            rec.detectors[k].tilt, eiger_calibrated.detectors[k].tilt) < 1e-9
        np.testing.assert_allclose(
            rec.detectors[k].tvec, eiger_calibrated.detectors[k].tvec, atol=1e-9)


def test_real_pair_residual_translations_are_mean_zero(
    eiger_uncalibrated: HEDMInstrument, eiger_calibrated: HEDMInstrument,
) -> None:
    d = decompose_instrument_motion(eiger_uncalibrated, eiger_calibrated)

    # The monolith absorbed the centroid displacement, so residual center shifts
    # average to zero and residual rotations average to identity.
    mean_disp = np.mean(
        [d.residuals[k].displacement for k in d.detector_keys], axis=0)
    np.testing.assert_allclose(mean_disp, 0.0, atol=1e-9)

    mean_res_rot = Rotation.from_rotvec(
        [d.residuals[k].tilt for k in d.detector_keys]).mean().magnitude()
    assert np.degrees(mean_res_rot) < 1e-6


def test_real_pair_monolith_absorbs_common_motion(
    eiger_uncalibrated: HEDMInstrument, eiger_calibrated: HEDMInstrument,
) -> None:
    d = decompose_instrument_motion(eiger_uncalibrated, eiger_calibrated)

    raw = np.mean([
        np.linalg.norm(np.asarray(eiger_calibrated.detectors[k].tvec)
                       - np.asarray(eiger_uncalibrated.detectors[k].tvec))
        for k in eiger_uncalibrated.detectors
    ])
    residual = np.mean(list(d.residual_distances.values()))
    # Pulling out the single common move shrinks the typical per-panel motion.
    assert residual < raw
    assert d.monolith.distance > 1.0


# ======================================================================
# apply_monolith_move semantics
# ======================================================================
def test_apply_monolith_move_copy_by_default(
    eiger_uncalibrated: HEDMInstrument, planted_truth: PlantedTruth,
) -> None:
    final, _, _ = planted_truth
    d = decompose_instrument_motion(eiger_uncalibrated, final)

    before = {k: (np.asarray(p.tilt).copy(), np.asarray(p.tvec).copy())
              for k, p in eiger_uncalibrated.detectors.items()}
    moved = apply_monolith_move(eiger_uncalibrated, d)

    # Original untouched...
    for k, (tilt0, tvec0) in before.items():
        np.testing.assert_array_equal(eiger_uncalibrated.detectors[k].tilt, tilt0)
        np.testing.assert_array_equal(eiger_uncalibrated.detectors[k].tvec, tvec0)
    # ...and a different object was returned.
    assert moved is not eiger_uncalibrated

    # After the monolith move, the only thing left between the moved instrument
    # and ``final`` is each panel's residual: tvec difference == residual disp.
    for k in final.detectors:
        np.testing.assert_allclose(
            np.asarray(final.detectors[k].tvec)
            - np.asarray(moved.detectors[k].tvec),
            d.residuals[k].displacement, atol=1e-9)


def test_apply_monolith_move_inplace(
    eiger_uncalibrated: HEDMInstrument, planted_truth: PlantedTruth,
) -> None:
    final, _, _ = planted_truth
    d = decompose_instrument_motion(eiger_uncalibrated, final)

    target = copy.deepcopy(eiger_uncalibrated)
    returned = apply_monolith_move(target, d, inplace=True)
    assert returned is target
    # It actually moved.
    k0 = d.detector_keys[0]
    assert not np.allclose(target.detectors[k0].tvec,
                           eiger_uncalibrated.detectors[k0].tvec)


# ======================================================================
# Validation / error handling
# ======================================================================
def test_mismatched_detectors_raises(
    eiger_uncalibrated: HEDMInstrument,
) -> None:
    other = copy.deepcopy(eiger_uncalibrated)
    # Drop a detector so the key sets differ.
    dropped = list(other.detectors)[0]
    del other.detectors[dropped]

    with pytest.raises(ValueError, match='different detectors'):
        decompose_instrument_motion(eiger_uncalibrated, other)


def test_detector_keys_subset(
    eiger_uncalibrated: HEDMInstrument, planted_truth: PlantedTruth,
) -> None:
    final, _, _ = planted_truth
    subset = list(eiger_uncalibrated.detectors)[:4]

    d = decompose_instrument_motion(
        eiger_uncalibrated, final, detector_keys=subset)
    assert d.detector_keys == subset
    assert set(d.residuals) == set(subset)


def test_unknown_detector_key_raises(
    eiger_uncalibrated: HEDMInstrument, planted_truth: PlantedTruth,
) -> None:
    final, _, _ = planted_truth
    with pytest.raises(ValueError, match='not present in both'):
        decompose_instrument_motion(
            eiger_uncalibrated, final, detector_keys=['does_not_exist'])
