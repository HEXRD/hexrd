from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingReport:
    """The measured angles of one Debye-Scherrer ring, in degrees

    `tth` includes the tth distortion correction, just as it does in the
    calibration residual.
    """

    xray_source: str
    tth: np.ndarray
    eta: np.ndarray
    # The detector key for each point
    detectors: np.ndarray
    # The hkl of the ring and its 2-theta. For powder calibration, this is
    # what the residual is measured from. For structureless calibration,
    # this is the nearest hkl, if plane data was provided.
    hkl: np.ndarray | None = None
    material_tth: float | None = None
    # For structureless calibration only: the ring index, which matches the
    # `DS_ring_{index}` parameter, and the fitted mean 2-theta of the ring
    index: int | None = None
    fit_tth: float | None = None


def make_ring_report(
    xray_source: str,
    tth: np.ndarray,
    eta: np.ndarray,
    detectors: np.ndarray,
    **kwargs,
) -> RingReport | None:
    """Make a ring report from angles in radians

    Points that are not finite are dropped, as they are in the calibration
    residuals. If no points are left, this returns None.
    """
    valid = np.isfinite(tth) & np.isfinite(eta)
    if not valid.any():
        return None

    return RingReport(
        xray_source=xray_source,
        tth=np.degrees(tth[valid]),
        eta=np.degrees(eta[valid]),
        detectors=detectors[valid],
        **kwargs,
    )
