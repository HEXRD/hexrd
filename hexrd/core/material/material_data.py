"""A small, typed material interface for the analysis workflows.

:class:`Material` here is a thin bridge: it exposes the minimal, immutable
crystallographic surface the workflows actually consume -- a
:class:`PlaneData` of plain arrays -- while the numbers are produced by
hexrd's full crystallography (:mod:`hexrd.core.material`), so every lattice
type hexrd supports works unchanged.

The intent is that workflow code never reaches into the big legacy classes:
everything it may touch is a field on this PlaneData.  (Replacing the
underlying implementation with the verified exact-integer space-group port
is future work; the interface here is what would stay stable.)
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from hexrd.core import material as _material
from hexrd.core import valunits


@dataclass(frozen=True, slots=True)
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
    ):
        self.name = name
        self.dmin = dmin
        self.sfacmin = sfacmin
        kwargs = {}
        if dmin is not None:
            kwargs['dmin'] = valunits.valWUnit('dmin', 'length', dmin, 'angstrom')
        if beam_energy is not None:
            kwargs['kev'] = valunits.valWUnit('kev', 'energy', beam_energy, 'keV')
        self._material = _material.Material(
            name, material_file=definitions, **kwargs
        )
        self.plane_data = self._make_plane_data()

    def _make_plane_data(self) -> PlaneData:
        pd = self._material.planeData
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
        )


def load_materials(definitions, **kwargs) -> dict[str, Material]:
    """Every material defined in a materials HDF5 file (path or open handle),
    keyed by name.  Keyword arguments are passed to :class:`Material`."""
    if isinstance(definitions, h5py.Group):
        names = list(definitions)
    else:
        with h5py.File(definitions, 'r') as f:
            names = list(f)
    return {name: Material(name, definitions, **kwargs) for name in names}
