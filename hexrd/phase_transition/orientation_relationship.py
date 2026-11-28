import numpy as np

from hexrd.phase_transition.variants import getPhaseTransformationVariants, plot_OR
from hexrd.powder.wppf.phase import Material_Rietveld


class OrientationRelationship:
    """
    this is just a helper class to hold information about
    the orientation relationship between a parent and a
    daughter phase. the information that is contained here
    is simply the material information about the parent
    and daughter crystals, the parallel planes and the
    parallel directions of the two phases.

    Parameters
    -----------

    material_parent: Material_Rietveld
        material of the parent phase
    material_daughter: Material_Rietveld
        material of the daughter phase
    parallel_planes: dict
        {"parent": hkl, "daughter": hkl} of the parallel planes
    parallel_directions: dict
        {"parent": uvw, "daughter": uvw} of the parallel directions
    rmat_parent: list
        orientations of the parent crystal(s). default is [np.eye(3)]
    verbose: bool
        log the number of variants

    """

    def __init__(
        self,
        material_parent: Material_Rietveld,
        material_daughter: Material_Rietveld,
        parallel_planes: dict[str, np.ndarray],
        parallel_directions: dict[str, np.ndarray],
        rmat_parent: list[np.ndarray] | None = None,
        verbose: bool = False,
    ) -> None:
        self._rmat_variants = None
        self.verbose = verbose
        self.material_parent = material_parent
        self.material_daughter = material_daughter
        self.parallel_planes = parallel_planes
        self.parallel_directions = parallel_directions
        self.rmat_parent = rmat_parent

        # compute the variants now, so bad input fails here
        self.compute_variant_rmats()

    def compute_variant_rmats(self) -> None:
        self._rmat_variants, _ = getPhaseTransformationVariants(
            self.material_parent,
            self.material_daughter,
            [self.plane_parent, self.plane_daughter],
            [self.direction_parent, self.direction_daughter],
            rmat_parent=[np.eye(3)],
            apply_parent_rotation_to_variants=False,
            plot=False,
            verbose=self.verbose,
        )

    def plot_OR(self) -> None:
        plot_OR(
            self.rmat_parent,
            self.rmat_variants_rotated,
            self.material_parent,
            self.material_daughter,
        )

    @property
    def material_parent(self) -> Material_Rietveld:
        return self._material_parent

    @material_parent.setter
    def material_parent(self, mat: Material_Rietveld) -> None:
        if not isinstance(mat, Material_Rietveld):
            msg = "incorrect material class of parent phase"
            raise TypeError(msg)
        self._material_parent = mat
        # the variants depend on the materials, the planes and the
        # directions, so they are recomputed when one of them changes
        self._rmat_variants = None

    @property
    def material_daughter(self) -> Material_Rietveld:
        return self._material_daughter

    @material_daughter.setter
    def material_daughter(self, mat: Material_Rietveld) -> None:
        if not isinstance(mat, Material_Rietveld):
            msg = "incorrect material class of daughter phase"
            raise TypeError(msg)
        self._material_daughter = mat
        self._rmat_variants = None

    @property
    def parallel_planes(self) -> dict[str, np.ndarray]:
        return self._parallel_planes

    @parallel_planes.setter
    def parallel_planes(self, pp: dict[str, np.ndarray]) -> None:
        _validate_parallel_pair(pp, "parallel planes", "plane")
        self._parallel_planes = pp
        self._rmat_variants = None

    @property
    def parallel_directions(self) -> dict[str, np.ndarray]:
        return self._parallel_directions

    @parallel_directions.setter
    def parallel_directions(self, pd: dict[str, np.ndarray]) -> None:
        _validate_parallel_pair(pd, "parallel directions", "direction")
        self._parallel_directions = pd
        self._rmat_variants = None

    @property
    def plane_parent(self) -> np.ndarray:
        return self.parallel_planes["parent"]

    @property
    def plane_daughter(self) -> np.ndarray:
        return self.parallel_planes["daughter"]

    @property
    def direction_parent(self) -> np.ndarray:
        return self.parallel_directions["parent"]

    @property
    def direction_daughter(self) -> np.ndarray:
        return self.parallel_directions["daughter"]

    @property
    def rmat_parent(self) -> list[np.ndarray]:
        return self._rmat_parent

    @rmat_parent.setter
    def rmat_parent(self, rmat_p: list[np.ndarray] | None) -> None:
        if rmat_p is None:
            rmat_p = [np.eye(3)]
        if not isinstance(rmat_p, list):
            msg = "rmat_parent not list type"
            raise TypeError(msg)
        self._rmat_parent = rmat_p

    @property
    def rmat_variants(self) -> np.ndarray:
        '''variants for a parent in the identity orientation, (n, 3, 3)'''
        if self._rmat_variants is None:
            self.compute_variant_rmats()
        return self._rmat_variants

    @property
    def rmat_variants_rotated(self) -> list[np.ndarray]:
        '''variants of every parent orientation in rmat_parent'''
        return [
            parent @ variant
            for parent in self.rmat_parent
            for variant in self.rmat_variants
        ]

    @property
    def num_variants(self) -> int:
        return len(self.rmat_variants)


def _validate_parallel_pair(pair: dict[str, np.ndarray], name: str, item: str) -> None:
    '''check a {"parent": vector, "daughter": vector} dictionary'''
    if len(pair) != 2:
        msg = f"incorrect shape of list defining {name}"
        raise ValueError(msg)

    if not ("parent" in pair and "daughter" in pair):
        msg = f"incorrect keys in {name}"
        raise ValueError(msg)

    if not all(p.shape == (3,) for p in pair.values()):
        msg = f"{item} should be array of size (3,)"
        raise ValueError(msg)
