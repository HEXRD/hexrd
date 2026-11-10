"""Validation and omega-frame mapping for frame-dependent beam energy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hexrd.core.imageseries.omega import OmegaImageSeries


def validate_beam_energy_series(
    beam_energies: ArrayLike,
    omega_image_series: OmegaImageSeries,
) -> NDArray[np.float64]:
    """Return a validated energy array with one positive keV value per frame."""
    frame_count = len(omega_image_series)
    energies = np.asarray(beam_energies, dtype=float)
    if energies.shape != (frame_count,):
        raise ValueError(
            'beam_energies must be a 1-D array with one value per omega frame; '
            f'expected ({frame_count},), got {energies.shape}'
        )
    if not np.all(np.isfinite(energies)) or np.any(energies <= 0):
        raise ValueError('beam_energies must contain only finite positive values')
    # OmegaImageSeries maps modulo 360 degrees and chooses the first wedge.
    # Reject scans where that would silently select the wrong acquisition.
    bounds = np.array(
        [[w['ostart'], w['ostop']] for w in omega_image_series.omegawedges.wedges]
    )
    widths = bounds[:, 1] - bounds[:, 0]
    tolerance = OmegaImageSeries.DFLT_TOL
    if not np.all(np.isfinite(bounds)) or np.any(
        (widths <= 0) | (widths > 360 + tolerance)
    ):
        raise ValueError(
            'frame energies require finite, increasing omega wedges of at most '
            '360 degrees; trim overscan frames before fitting or prediction'
        )
    starts = bounds[:, 0] % 360
    intervals = np.column_stack([starts, starts + widths])
    # Include the next turn to detect overlaps that cross zero degrees.
    intervals = np.vstack([intervals, intervals + 360])
    intervals = intervals[np.argsort(intervals[:, 0])]
    if np.any(intervals[1:, 0] < intervals[:-1, 1] - tolerance):
        raise ValueError(
            'frame energies require omega wedges that do not overlap modulo '
            '360 degrees; trim overlapping frames before fitting or prediction'
        )
    return energies


def load_beam_energy_file(
    filename: str | Path, omega_image_series: OmegaImageSeries
) -> NDArray[np.float64]:
    """Read one absolute energy in keV per frame from a single-column text file."""
    try:
        energies = np.loadtxt(filename, dtype=float, ndmin=2)
        if energies.shape[1] != 1:
            raise ValueError('expected one energy column')
        return validate_beam_energy_series(energies[:, 0], omega_image_series)
    except ValueError as exc:
        raise ValueError(f'{filename}: invalid frame energies: {exc}') from exc


def omegas_to_frames(
    omegas_deg: ArrayLike,
    omega_image_series: OmegaImageSeries,
) -> NDArray[np.int64]:
    """Map finite omega values to frames, vectorizing normal wedge metadata."""
    omegas_deg = np.asarray(omegas_deg, dtype=float)
    frames = np.full(omegas_deg.shape, -1, dtype=np.int64)
    wedges = omega_image_series.omegawedges.wedges

    frame_offset = 0
    finite = np.isfinite(omegas_deg)
    for wedge in wedges:
        start = float(wedge['ostart'])
        stop = float(wedge['ostop'])
        steps = int(wedge['nsteps'])
        delta = (stop - start) / steps
        wrapped = start + np.mod(np.where(finite, omegas_deg, start) - start, 360.0)
        selected = finite & (frames < 0) & (wrapped >= start) & (wrapped < stop)
        local_frames = np.floor((wrapped[selected] - start) / delta).astype(int)
        frames[selected] = frame_offset + np.minimum(local_frames, steps - 1)
        frame_offset += steps
    return frames
