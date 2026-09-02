"""The HEDM experiment: shared experiment inputs plus the parameters of the
HEDM analyses, as typed, frozen config sections.

:class:`HedmExperiment` extends :class:`hexrd.core.experiment.Experiment`
with the ``find_orientations`` section of the config.
"""
import enum
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from hexrd.core.experiment import Experiment


class SeedSearchMethod(enum.StrEnum):
    LABEL = 'label'
    BLOB_LOG = 'blob_log'
    BLOB_DOG = 'blob_dog'


class ClusteringAlgorithm(enum.StrEnum):
    DBSCAN = 'dbscan'
    ORT_DBSCAN = 'ort-dbscan'
    SPH_DBSCAN = 'sph-dbscan'
    FCLUSTERDATA = 'fclusterdata'


def _hkl_array(value: Any, key: str) -> Optional[np.ndarray]:
    """Canonicalize an hkl selection: an (n, 3) array of [h, k, l] triples,
    an (n,) array of master hkl IDs, or None for all rings.

    Both spellings appear in existing configs; anything else (single ints,
    'all', floats) is rejected here so downstream code only sees these
    three shapes.
    """
    if value is None:
        return None
    hkls = np.asarray(value)
    if not np.issubdtype(hkls.dtype, np.integer) or hkls.ndim not in (1, 2) \
            or (hkls.ndim == 2 and hkls.shape[1] != 3):
        raise ValueError(
            f'{key} must be a list of [h, k, l] integer triples or of '
            f'integer hkl IDs (or null for all rings), got {value!r}')
    return hkls


@dataclass(frozen=True)
class OrientationMaps:
    threshold: float | None
    active_hkls: Optional[np.ndarray]   # (n, 3) hkl vectors or (n,) master
                                        # hkl IDs; None -> all rings
    eta_step: float                     # degrees
    file: Optional[str]                 # maps cache; None -> default path
    filter_maps: bool                   # subtract each eta column's median
    filter_fwhm: Optional[float]        # additionally LoG-filter, this FWHM in pixels

    @classmethod
    def from_dict(cls, d: dict) -> 'OrientationMaps':
        filter_maps = d.get('filter_maps', False)
        if not isinstance(filter_maps, bool):
            raise ValueError(
                'filter_maps must be a boolean; give the LoG width (pixels) '
                'separately as filter_fwhm')
        filter_fwhm = d.get('filter_fwhm')
        if filter_fwhm is not None and not filter_maps:
            raise ValueError('filter_fwhm requires filter_maps: true')
        # a null threshold means "no thresholding", same as 0 for count data
        return cls(float(d.get('threshold') or 0),
                   _hkl_array(d.get('active_hkls'), 'active_hkls'),
                   float(d.get('eta_step', 0.25)), d.get('file'),
                   filter_maps,
                   None if filter_fwhm is None else float(filter_fwhm))


@dataclass(frozen=True)
class SeedSearch:
    hkl_seeds: np.ndarray               # indices into the active rings
    fiber_step: float                   # radians
    method: SeedSearchMethod
    method_kwargs: dict[str, float]

    @classmethod
    def from_dict(cls, d: dict, omega_tolerance_deg: float) -> 'SeedSearch':
        # `method` is a one-entry mapping: {name: {kwargs...}}
        method_dict = d.get('method') or {'label': {}}
        (name, kwargs), = method_dict.items()
        seeds = np.asarray(d.get('hkl_seeds', []), dtype=np.intp)
        if seeds.ndim != 1:
            raise ValueError(f'hkl_seeds must be a list of ring indices, '
                             f'got {d.get("hkl_seeds")!r}')
        return cls(seeds, np.radians(d.get('fiber_step', omega_tolerance_deg)),
                   SeedSearchMethod(name), kwargs or {})

    @property
    def fiber_ndiv(self) -> int:
        return int(round(2 * np.pi / self.fiber_step))


@dataclass(frozen=True)
class Omega:
    tolerance: float           # radians

    @classmethod
    def from_dict(cls, d: dict) -> 'Omega':
        return cls(np.radians(d.get('tolerance', 0.5)))


@dataclass(frozen=True)
class Eta:
    tolerance: float           # radians
    mask: Optional[float]      # radians

    @classmethod
    def from_dict(cls, d: dict) -> 'Eta':
        mask = d.get('mask', 5)
        return cls(np.radians(d.get('tolerance', 0.5)),
                   np.radians(mask) if mask is not None else None)

    @property
    def range(self) -> np.ndarray:
        """Valid eta spans, masking the region near the rotation axis.

        A null mask means no masking: the full circle is valid.
        """
        if self.mask is None:
            return np.array([[-np.pi, np.pi]])
        return np.array([[-np.pi / 2 + self.mask,     np.pi / 2 - self.mask],
                         [np.pi / 2 + self.mask, 3 * np.pi / 2 - self.mask]])


@dataclass(frozen=True)
class Clustering:
    radius: float
    completeness: float
    algorithm: ClusteringAlgorithm

    @classmethod
    def from_dict(cls, d: dict) -> 'Clustering':
        missing = {'radius', 'completeness'} - d.keys()
        if missing:
            raise ValueError(f'clustering config requires {sorted(missing)}')
        return cls(float(d['radius']), float(d['completeness']),
                   ClusteringAlgorithm(d.get('algorithm', 'dbscan')))


@dataclass(frozen=True)
class FindOrientations:
    orientation_maps: OrientationMaps
    seed_search: SeedSearch
    omega: Omega
    eta: Eta
    clustering: Clustering
    threshold: float
    use_quaternion_grid: Optional[str]  # .npy of trial quats; replaces seed search

    @classmethod
    def from_dict(cls, d: dict) -> 'FindOrientations':
        omega = Omega.from_dict(d.get('omega', {}))
        return cls(OrientationMaps.from_dict(d.get('orientation_maps', {})),
                   SeedSearch.from_dict(d.get('seed_search', {}),
                                        np.degrees(omega.tolerance)),
                   omega,
                   Eta.from_dict(d.get('eta', {})),
                   Clustering.from_dict(d.get('clustering', {})),
                   float(d.get('threshold', 1)),
                   d.get('use_quaternion_grid'))


def _pair(value):
    return [value, value] if isinstance(value, (int, float)) else value


@dataclass(frozen=True)
class FitGrainsTolerance:
    eta: list
    omega: list
    tth: list

    @classmethod
    def from_dict(cls, d: dict) -> 'FitGrainsTolerance':
        return cls(_pair(d.get('eta')), _pair(d.get('omega')),
                   _pair(d.get('tth')))


@dataclass(frozen=True)
class FitGrains:
    do_fit: bool
    estimate: str | None
    npdiv: int
    threshold: float
    tolerance: FitGrainsTolerance
    refit: list | None
    tth_max: bool | float
    reset_exclusions: bool
    exclusion_parameters: dict

    @classmethod
    def from_dict(cls, d: dict, path) -> 'FitGrains':
        estimate = d.get('estimate')
        if estimate is not None and not os.path.isabs(estimate):
            estimate = path(estimate)
        names = ('dmin', 'dmax', 'tthmin', 'tthmax', 'sfacmin', 'sfacmax',
                 'pintmin', 'pintmax')
        exclusions = {name: d.get(name) for name in names}
        exclusions['sfacmin'] = d.get('sfacmin', d.get('min_sfac_ratio'))
        return cls(d.get('do_fit', True), estimate, d.get('npdiv', 2),
                   d.get('threshold'),
                   FitGrainsTolerance.from_dict(d.get('tolerance', {})),
                   _pair(d.get('refit')), d.get('tth_max', True),
                   d.get('reset_exclusions', True), exclusions)


class HedmExperiment(Experiment):
    """An :class:`Experiment` plus the HEDM find-orientations parameters."""

    def __init__(self, filename: str, study: int | None = None):
        super().__init__(filename, study)
        self.find_orientations = FindOrientations.from_dict(
            self.config['find_orientations'])
        self.fit_grains = FitGrains.from_dict(
            self.config.get('fit_grains', {}), self._path)

    @property
    def eta_ome_maps_file(self) -> str:
        """Where the eta-omega maps cache lives (orientation_maps: file, or default)."""
        configured = self.find_orientations.orientation_maps.file
        if configured:
            return configured if os.path.isabs(configured) else self._path(configured)
        actmat = self.active_material.active.strip().replace(' ', '-')
        return os.path.join(self.analysis_dir, f'eta-ome-maps-{actmat}.npz')

    @property
    def quaternion_grid_file(self) -> str | None:
        """Absolute path of the trial-quaternion grid, if configured."""
        grid = self.find_orientations.use_quaternion_grid
        if grid is None:
            return None
        return grid if os.path.isabs(grid) else self._path(grid)
