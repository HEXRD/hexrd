"""Decompose the geometry change between two calibrations of one instrument.

When a multi-panel detector (e.g. a 32-subpanel Eiger, where every subpanel is
its own hexrd ``Detector``) is re-calibrated, the panels tend to move *together*:
most of the apparent per-panel motion is really a single rigid motion of the
whole detector assembly. This module re-expresses the change between an
``initial`` and a ``final`` instrument as

    step 1 : one MONOLITH rigid motion applied to every panel at once
             (a single tilt + translation for the whole detector), plus
    step 2 : small RESIDUAL per-panel motions left over after step 1.

The residuals are the quantity of interest -- "how much did each subpanel
*actually* move on its own" -- which should be tiny if the relative geometry
between panels is stable. They are reported per panel as a tilt (deg) and a
center displacement (mm).

METHOD
------
  * monolith ROTATION    = the chordal (L2/Frobenius) mean on SO(3) of the
                           per-panel rotation changes Q_i = R_i^final
                           (R_i^initial)^T, via scipy's ``Rotation.mean()``.
  * monolith TRANSLATION = the displacement of the panel centroid.

Averaging is done on rotation matrices, not on the exp-map ``tilt`` vectors,
which wrap and are singular near pi. The monolith rotation is applied as a rigid
rotation about the initial centroid, so panel centers swing with it; residual
center shifts are measured *after* that swing (otherwise an outer panel shows a
large fake translation ~ lever_arm * tilt).

POSE CONVENTION (matches hexrd)
-------------------------------
Each panel carries ``tilt`` (exp-map rotation vector, radians) and ``tvec``
(lab-frame translation, mm), with ``rmat = expmap(tilt)`` and
``x_lab = rmat @ x_local + tvec``. hexrd's exp map equals scipy's ``rotvec``, so
we use ``scipy.spatial.transform.Rotation`` here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from hexrd.core.instrument.hedm_instrument import HEDMInstrument


# ======================================================================
# Pose <-> rotation-matrix helpers (scipy rotvec == hexrd exp map)
# ======================================================================
def _rmats_from_tilts(tilts: NDArray) -> NDArray:
    """Exp-map tilt(s) -> rotation matrix(es); single (3,) or stacked (N, 3)."""
    return Rotation.from_rotvec(np.asarray(tilts, float)).as_matrix()


def _expmaps_from_rmats(rmats: NDArray) -> NDArray:
    """Rotation matrix(es) -> exp-map tilt(s); single (3, 3) or stacked (N, 3, 3)."""
    return Rotation.from_matrix(rmats).as_rotvec()


def _angle_deg(v: NDArray) -> float:
    """Magnitude of a rotation vector, in degrees."""
    return float(np.degrees(np.linalg.norm(v)))


def _norm(v: NDArray) -> float:
    """Euclidean magnitude of a vector, as a plain float."""
    return float(np.linalg.norm(v))


def _mean_rotation(rmats: NDArray, weights: Optional[NDArray] = None) -> NDArray:
    """The chordal (L2/Frobenius) mean of a stack of rotation matrices.

    Delegates to scipy's ``Rotation.mean``, which solves it via the largest
    eigenvector of the (weighted) quaternion outer-product sum. That can never
    return a reflection, so no explicit ``det`` guard is needed.
    """
    return Rotation.from_matrix(rmats).mean(weights=weights).as_matrix()


# ======================================================================
# Result containers
# ======================================================================
@dataclass
class MonolithMove:
    """A single rigid motion applied to the whole detector assembly.

    Attributes
    ----------
    tilt : (3,) exp-map vector of the monolith rotation (radians).
    rmat : (3, 3) rotation matrix form of ``tilt``.
    translation : (3,) centroid displacement, mm. The interpretable "how far did
        the detector move" number; pivot-independent.
    pivot : (3,) point the monolith rotation turns about (the initial, weighted,
        panel centroid), mm.
    offset : (3,) lab-origin offset used to apply the move exactly:
        ``tvec_new = rmat @ tvec_old + offset``.
    angle : magnitude of ``tilt``, degrees.
    distance : magnitude of ``translation``, mm.
    """

    tilt: NDArray
    rmat: NDArray
    translation: NDArray
    pivot: NDArray
    offset: NDArray

    @property
    def angle(self) -> float:
        return _angle_deg(self.tilt)

    @property
    def distance(self) -> float:
        return _norm(self.translation)


@dataclass
class ResidualMove:
    """The leftover per-panel motion after the monolith move is removed.

    Attributes
    ----------
    tilt : (3,) residual rotation as an exp-map vector (radians), applied about
        the panel's own center.
    displacement : (3,) residual shift of the panel center, mm.
    angle : magnitude of ``tilt``, degrees.
    distance : magnitude of ``displacement``, mm.
    weight : consensus weight this panel carried (1.0 unless ``robust``
        down-weighted it).
    """

    tilt: NDArray
    displacement: NDArray
    weight: float = 1.0

    @property
    def angle(self) -> float:
        return _angle_deg(self.tilt)

    @property
    def distance(self) -> float:
        return _norm(self.displacement)


@dataclass
class InstrumentMotionDecomposition:
    """Full initial -> final decomposition: one monolith move + residuals."""

    monolith: MonolithMove
    residuals: dict[str, ResidualMove]
    detector_keys: list[str] = field(default_factory=list)

    @property
    def residual_angles(self) -> dict[str, float]:
        """Per-detector residual tilt magnitude, degrees."""
        return {k: r.angle for k, r in self.residuals.items()}

    @property
    def residual_distances(self) -> dict[str, float]:
        """Per-detector residual center-shift magnitude, mm."""
        return {k: r.distance for k, r in self.residuals.items()}

    @property
    def max_residual_angle(self) -> float:
        return max((r.angle for r in self.residuals.values()), default=0.0)

    @property
    def max_residual_distance(self) -> float:
        return max((r.distance for r in self.residuals.values()), default=0.0)


# ======================================================================
# Core decomposition (array level)
# ======================================================================
def _decompose_arrays(
    tiltsA: NDArray,
    tvecsA: NDArray,
    tiltsB: NDArray,
    tvecsB: NDArray,
    weights: Optional[NDArray] = None,
) -> dict[str, NDArray]:
    """Split the A -> B motion of N panels into a monolith move + residuals.

    ``tilts*`` are (N, 3) exp-map vectors, ``tvecs*`` are (N, 3) lab-frame
    translations in mm. ``weights`` is an optional (N,) non-negative array applied
    consistently to both the rotation mean and the centroid. Returns a dict of
    arrays.
    """
    RA = _rmats_from_tilts(tiltsA)
    RB = _rmats_from_tilts(tiltsB)
    tA = np.asarray(tvecsA, float)
    tB = np.asarray(tvecsB, float)
    n = len(tA)

    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, float)
        w = w / w.sum()

    # Per-panel A->B orientation change in the lab frame.
    Q = RB @ RA.transpose(0, 2, 1)

    # Monolith rotation = (weighted) consensus of the per-panel changes.
    QM = _mean_rotation(Q, weights=w)

    # Monolith translation = (weighted) centroid displacement, pivoting about
    # the state-A centroid.
    cenA = w @ tA
    cenB = w @ tB
    offset = cenB - QM @ cenA            # tvec_new = QM @ tvec_old + offset
    trans_M = cenB - cenA                # pivot-independent reported translation

    # Intermediate poses after only the monolith step.
    R_int = QM @ RA
    t_int = (QM @ tA.T).T + offset

    # Residual per-panel motion taking intermediate -> B.
    QS = RB @ R_int.transpose(0, 2, 1)
    res_tilt = _expmaps_from_rmats(QS)
    res_dvec = tB - t_int

    return dict(
        QM=QM, offset=offset, tilt_M=_expmaps_from_rmats(QM),
        trans_M=trans_M, pivot=cenA,
        res_tilt=res_tilt, res_dvec=res_dvec,
        weights=w * n,  # report relative weights (1.0 == nominal)
    )


def _robust_decompose_arrays(
    tiltsA: NDArray,
    tvecsA: NDArray,
    tiltsB: NDArray,
    tvecsB: NDArray,
    n_iter: int = 5,
    huber_angle: Optional[float] = None,
    huber_distance: Optional[float] = None,
) -> dict[str, NDArray]:
    """IRLS/Huber version of :func:`_decompose_arrays`.

    A single genuinely-shifted or stuck module would otherwise bias the consensus
    and leak its motion into every panel's residual. A few iteratively-reweighted
    passes down-weight big movers when forming the monolith, so they stand out as
    large residuals instead.
    """
    w = None
    d = None
    for _ in range(n_iter):
        d = _decompose_arrays(tiltsA, tvecsA, tiltsB, tvecsB, weights=w)
        res_angle = np.linalg.norm(d['res_tilt'], axis=1)
        res_disp = np.linalg.norm(d['res_dvec'], axis=1)
        ka = huber_angle if huber_angle else max(np.median(res_angle), 1e-9)
        kt = huber_distance if huber_distance else max(np.median(res_disp), 1e-9)
        r = np.hypot(res_angle / ka, res_disp / kt)
        w = np.where(r <= 1.0, 1.0, 1.0 / r)
    return d


# ======================================================================
# Public API
# ======================================================================
def _shared_detector_keys(
    initial: 'HEDMInstrument',
    final: 'HEDMInstrument',
    detector_keys: Optional[list[str]] = None,
) -> list[str]:
    """Validate and return the ordered list of detector keys to compare."""
    keys_a = list(initial.detectors)
    keys_b = set(final.detectors)
    if detector_keys is None:
        missing = [k for k in keys_a if k not in keys_b]
        extra = [k for k in final.detectors if k not in set(keys_a)]
        if missing or extra:
            raise ValueError(
                'initial and final instruments have different detectors '
                f'(only in initial: {missing}; only in final: {extra}). '
                'Pass detector_keys=... to compare a common subset.'
            )
        return keys_a

    keys = list(detector_keys)
    for k in keys:
        if k not in initial.detectors or k not in keys_b:
            raise ValueError(f'detector {k!r} is not present in both instruments')
    return keys


def _stack(instr: 'HEDMInstrument', keys: list[str], attr: str) -> NDArray:
    """Stack one float attribute (``tilt``/``tvec``) over ``keys`` into (N, 3)."""
    return np.array(
        [np.asarray(getattr(instr.detectors[k], attr), float) for k in keys]
    )


def decompose_instrument_motion(
    initial: 'HEDMInstrument',
    final: 'HEDMInstrument',
    *,
    detector_keys: Optional[list[str]] = None,
    robust: bool = False,
    weights: Optional[NDArray] = None,
    n_iter: int = 5,
    huber_angle: Optional[float] = None,
    huber_distance: Optional[float] = None,
) -> InstrumentMotionDecomposition:
    """Decompose the geometry change between two calibrations of one instrument.

    Reads each shared detector's ``tilt``/``tvec`` from ``initial`` and ``final``
    and splits the change into one whole-detector monolith move plus small
    per-panel residuals (see the module docstring for the method).

    Parameters
    ----------
    initial, final : HEDMInstrument
        Two calibrations of the SAME detector. Their detectors must share keys
        (or pass ``detector_keys`` to compare a common subset).
    detector_keys : list[str], optional
        Restrict the comparison to these detectors (default: all shared keys).
    robust : bool
        If True, down-weight outlier panels (genuinely moved/stuck modules) when
        forming the monolith so they do not contaminate everyone's residual.
    weights : (N,) array, optional
        Fixed per-panel weights for the (non-robust) consensus. Ignored when
        ``robust`` is True. Order follows ``detector_keys`` / instrument order.
    n_iter, huber_angle, huber_distance :
        IRLS controls used only when ``robust`` is True. The Huber knees default
        to the median residual of each channel (radians / mm).

    Returns
    -------
    InstrumentMotionDecomposition
        ``.monolith`` (a :class:`MonolithMove`) and ``.residuals`` (a dict of
        :class:`ResidualMove` keyed by detector id).
    """
    keys = _shared_detector_keys(initial, final, detector_keys)

    tiltsA = _stack(initial, keys, 'tilt')
    tvecsA = _stack(initial, keys, 'tvec')
    tiltsB = _stack(final, keys, 'tilt')
    tvecsB = _stack(final, keys, 'tvec')

    if robust:
        d = _robust_decompose_arrays(
            tiltsA, tvecsA, tiltsB, tvecsB,
            n_iter=n_iter, huber_angle=huber_angle, huber_distance=huber_distance,
        )
    else:
        d = _decompose_arrays(tiltsA, tvecsA, tiltsB, tvecsB, weights=weights)

    monolith = MonolithMove(
        tilt=d['tilt_M'], rmat=d['QM'], translation=d['trans_M'],
        pivot=d['pivot'], offset=d['offset'],
    )
    residuals = {
        k: ResidualMove(
            tilt=d['res_tilt'][i], displacement=d['res_dvec'][i],
            weight=float(d['weights'][i]),
        )
        for i, k in enumerate(keys)
    }
    return InstrumentMotionDecomposition(
        monolith=monolith, residuals=residuals, detector_keys=keys,
    )


def apply_monolith_move(
    instr: 'HEDMInstrument',
    decomposition: InstrumentMotionDecomposition,
    *,
    inplace: bool = False,
) -> 'HEDMInstrument':
    """Apply ONLY the monolith move to ``instr`` (the common rigid motion).

    Each affected panel becomes ``rmat_new = QM @ rmat_old`` and
    ``tvec_new = QM @ tvec_old + offset``. Applied to the ``initial`` instrument,
    this lands every panel at its post-monolith intermediate pose; what is left
    between that and ``final`` is exactly the per-panel residual.

    Returns a deep copy by default; pass ``inplace=True`` to mutate ``instr``.
    """
    out = instr if inplace else copy.deepcopy(instr)
    QM = decomposition.monolith.rmat
    offset = decomposition.monolith.offset
    for k in decomposition.detector_keys:
        det = out.detectors[k]
        det.tvec = QM @ np.asarray(det.tvec, float) + offset
        det.tilt = _expmaps_from_rmats(QM @ det.rmat)
    return out


def reconstruct_final(
    initial: 'HEDMInstrument',
    decomposition: InstrumentMotionDecomposition,
    *,
    inplace: bool = False,
) -> 'HEDMInstrument':
    """Apply the monolith move and then the per-panel residuals to ``initial``.

    Recovers the ``final`` instrument the decomposition was built from, to
    machine precision. Useful for round-trip checks and for rebuilding a final
    instrument from a stored decomposition. The residual rotation is applied
    about each panel's own center, so the center only moves by the residual
    displacement.

    Returns a deep copy by default; pass ``inplace=True`` to mutate ``initial``.
    """
    out = apply_monolith_move(initial, decomposition, inplace=inplace)
    for k in decomposition.detector_keys:
        det = out.detectors[k]
        res = decomposition.residuals[k]
        QS = _rmats_from_tilts(res.tilt)
        det.tilt = _expmaps_from_rmats(QS @ det.rmat)
        det.tvec = np.asarray(det.tvec, float) + res.displacement
    return out
