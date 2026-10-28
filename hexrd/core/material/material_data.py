"""Material bridge exposing crystallographic arrays to workflows."""
from __future__ import annotations

from dataclasses import asdict, field
from pathlib import Path

import numpy as np
from pydantic.dataclasses import dataclass

from hexrd.core import material as _material
from hexrd.core import valunits
from hexrd.core.utils.npz import fingerprint, load_array_cache, save_array_cache


def _plane_data_fingerprint(definitions, settings):
    """Invalidate snapshots on material, selection, implementation, or data edits."""
    core = Path(__file__).resolve().parents[1]
    from hexrd.core.extensions import transforms

    sources = [Path(definitions), Path(__file__), core / 'constants.py',
               core / 'rotations.py', Path(transforms.__file__),
               core / 'matrixutil.py', core / 'utils/decorators.py',
               core / 'valunits.py', core.parent / 'powder/wppf/xtal.py',
               core / 'resources/Anomalous.h5']
    sources.extend(sorted((core / 'material').glob('*.py')))
    return fingerprint(settings, files=sources, packages=('numpy', 'scipy', 'numba'))


def load_plane_data_cached(definitions, cache_file, settings, load_material):
    """Reuse numeric crystallography without constructing a full Material.

    This is a derived-data cache, never a pickle. Both contents and the code
    producing them are fingerprinted; writes are atomic, and unavailable or
    damaged caches fall back to the normal material loader.
    """
    key = _plane_data_fingerprint(definitions, settings)
    data = load_array_cache(cache_file, key)
    if data is not None:
        try:
            sizes = data.pop('symm_sizes')
            data['symm_hkls'] = np.split(
                data['symm_hkls'], np.cumsum(sizes)[:-1], axis=1) if sizes.size else []
            data['laue_group'] = str(data['laue_group'])
            data['wavelength'] = float(data['wavelength'])
            data['strain_magnitude'] = float(data['strain_magnitude'])
            data['point_group_number'] = int(data['point_group_number'])
            return PlaneData(**data)
        except (KeyError, ValueError, TypeError):
            pass

    plane_data = load_material().plane_data
    arrays = asdict(plane_data)
    arrays.update(
        symm_sizes=np.array([a.shape[1] for a in plane_data.symm_hkls]),
        symm_hkls=(np.hstack(plane_data.symm_hkls)
                   if plane_data.symm_hkls else np.empty((3, 0))))
    save_array_cache(cache_file, key, **arrays)
    return plane_data


@dataclass(frozen=True, slots=True, config={'arbitrary_types_allowed': True})
class PlaneData:
    """Everything the diffraction workflows need to know about a crystal.

    hkls            : (n, 3) every reference reflection, excluded ones included
    exclusions      : (n,) True where a reflection is excluded (dmin cut,
                      structure-factor cut, ...)
    unexcluded_hkls : (m, 3) the reflections in play, i.e. hkls[~exclusions]
    two_thetas      : (n,) Bragg angle per reference reflection, radians
    symm_hkls       : per unexcluded reflection, its (3, k) array of
                      symmetry-equivalent hkls
    laue_group      : Schoenflies symbol of the Laue group (e.g. 'd3d')
    q_sym           : (4, q) quaternion symmetry group of the lattice
    B               : (3, 3) reciprocal-lattice B matrix
    wavelength      : beam wavelength, angstrom
    master_hkl_ids  : legacy identifiers of the unexcluded reflections
    lattice_parameters, strain_magnitude : legacy metadata for eta-omega exports
    point_group_number : crystallographic point group (1-32); 0 if unspecified
    """

    hkls: np.ndarray
    exclusions: np.ndarray
    unexcluded_hkls: np.ndarray
    two_thetas: np.ndarray
    symm_hkls: list[np.ndarray]
    laue_group: str
    q_sym: np.ndarray
    B: np.ndarray
    wavelength: float
    master_hkl_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    lattice_parameters: np.ndarray = field(default_factory=lambda: np.empty(0))
    strain_magnitude: float | None = 0.0
    point_group_number: int = 0

    def __post_init__(self):
        if not self.master_hkl_ids.size:
            object.__setattr__(self, 'master_hkl_ids', np.flatnonzero(~self.exclusions))


class Material:
    """One crystal material from a materials HDF5 file.

    ``definitions`` is an open h5py file/group (or a path) holding the named
    material.  ``dmin`` (angstrom) bounds the generated reflections;
    ``sfacmin`` excludes reflections whose structure factor falls below that
    proportion of the maximum; ``beam_energy`` (keV) overrides the beam
    energy stored with the material.
    """

    def __init__(
        self,
        name: str,
        definitions,
        dmin: float | None = None,
        sfacmin: float | None = None,
        beam_energy: float | None = None,
        reset_exclusions: bool = True,
    ):
        self.name = name
        self.dmin = dmin
        self.sfacmin = sfacmin
        self.reset_exclusions = reset_exclusions
        kwargs = {}
        if reset_exclusions and dmin is not None:
            kwargs['dmin'] = valunits.valWUnit('dmin', 'length', dmin, 'angstrom')
        if beam_energy is not None:
            kwargs['kev'] = valunits.valWUnit('kev', 'energy', beam_energy, 'keV')
        self._material = _material.Material(
            name, material_file=definitions, **kwargs
        )
        self.plane_data = self._make_plane_data()

    def _make_plane_data(self) -> PlaneData:
        pd = self._material.planeData
        if self.reset_exclusions:
            pd.exclude(dmin=self.dmin, sfacmin=self.sfacmin)
        hkls = np.vstack([d['hkl'] for d in pd.hklDataList])
        return PlaneData(
            hkls=hkls,
            exclusions=np.asarray(pd.exclusions, dtype=bool),
            unexcluded_hkls=np.asarray(pd.hkls.T),
            two_thetas=np.asarray(pd.getTTh(allHKLs=True)),
            symm_hkls=list(pd.getSymHKLs()),
            laue_group=pd.laueGroup,
            q_sym=np.ascontiguousarray(pd.q_sym),
            B=np.asarray(pd.latVecOps['B']),
            wavelength=float(pd.wavelength),
            master_hkl_ids=np.asarray(pd.getHKLID(pd.hkls, master=True), dtype=int),
            lattice_parameters=np.asarray(pd.getParams()[0]),
            strain_magnitude=pd.getParams()[3],
            point_group_number=self._material.pgnum,
        )


def load_materials(definitions, **kwargs) -> dict[str, Material]:
    """Every material defined in a materials HDF5 file (path or open handle),
    keyed by name.  Keyword arguments are passed to :class:`Material`."""
    import h5py

    if isinstance(definitions, h5py.Group):
        names = list(definitions)
    else:
        with h5py.File(definitions, 'r') as f:
            names = list(f)
    return {name: Material(name, definitions, **kwargs) for name in names}
