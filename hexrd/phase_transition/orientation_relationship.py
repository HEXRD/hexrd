# standard imports
# ---------
import logging

# 3rd party imports
# -----------------
import numpy as np

# hexrd imports
# -------------
from hexrd.core.material import Material
from hexrd.powder.wppf.phase import Material_Rietveld

logger = logging.getLogger(__name__)


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

    material_parent: float
        minimum azimuthal angle
    eta_max: float
        maximum azimuthal angle
    eta_step: float
        step size in azimuth. this determines the
        dimensions in the azimuthal direction

    Returns
    --------
    simulated_2d: np.ndarray
        simulated 2D diffraction pattern

    """

    def __init__(
        self,
        material_parent: Material | Material_Rietveld,
        material_daughter: Material | Material_Rietveld,
        parallel_planes: dict[str, np.ndarray],
        parallel_directions: dict[str, np.ndarray],
    ):

        if len(parallel_directions) != 2:
            msg = "incorrect shape of list defining parallel directions"
            logger.info(msg)

        shape_check = [p.shape == (3,) for p in parallel_directions]
        if not all(shape_check):
            msg = "directions should be array of size (3,)"
            logger.info(msg)

        self.material_parent = material_parent
        self.material_daughter = material_daughter
        self.parallel_planes = parallel_planes
        self.parallel_directions = parallel_directions

    @property
    def material_parent(self):
        return self._material_parent

    @property
    def material_daughter(self):
        return self._material_daughter

    @property
    def direction_parent(self):
        return self._direction_parent

    @property
    def direction_daughter(self):
        return self._direction_daughter

    @property
    def plane_parent(self):
        return self._plane_parent

    @property
    def plane_daughter(self):
        return self._plane_daughter

    @property
    def parallel_planes(self):
        return self._parallel_planes

    @property
    def parallel_directions(self):
        return self._parallel_directions

    @property
    def plane_parent_c(self):
        return self._plane_parent_c

    @property
    def plane_daughter_c(self):
        return self._plane_daughter_c

    @property
    def direction_parent_c(self):
        return self._direction_parent_c

    @property
    def direction_daughter_c(self):
        return self._direction_daughter_c

    @property
    def rmat_variants(self):
        return self._rmat_variants

    @property
    def num_variants(self):
        return self._num_variants

    @material_parent.setter
    def material_parent(self, mat):
        if not isinstance(mat, (Material, Material_Rietveld)):
            msg = "incorrect material class of parent phase"
            logger.info(msg)
        self._material_parent = mat

    @material_daughter.setter
    def material_daughter(self, mat):
        if not isinstance(mat, (Material, Material_Rietveld)):
            msg = "incorrect material class of parent phase"
            logger.info(msg)
        self._material_daughter = mat

    @parallel_planes.setter
    def parallel_planes(self, pp):
        if len(pp) != 2:
            msg = "incorrect shape of list defining parallel planes"
            logger.info(msg)

        if not ("parent" in pp | "daughter" in pp):
            msg = "incorrect keys in parallel planes"
            logger.info(msg)

        shape_check = [p.shape == (3,) for p in pp.values()]
        if not all(shape_check):
            msg = "plane should be array of size (3,)"
            logger.info(msg)

        self._parallel_planes = pp
        self._plane_parent = pp["parent"]
        self._plane_daughter = pp["daughter"]

        if all([hasattr(self, "material_parent"), hasattr(self, "material_daughter")]):
            mat_p = self.material_parent
            mat_d = self.material_daughter
            p1, p2 = self.plane_parent, self.plane_daughter

            p1 = mat_p.unitcell.TransSpace(p1, "r", "c")
            p1 = mat_p.unitcell.NormVec(p1, "c")
            self._plane_parent_c = p1

            p2 = mat_d.unitcell.TransSpace(p2, "r", "c")
            p2 = mat_d.unitcell.NormVec(p2, "c")
            self._plane_daughter_c = p2

        else:
            msg = (
                "materials for parent and daughter phases not set. "
                "skipping computation of the cartesian represantation"
                "of planes."
            )
            logger.info(msg)

    @parallel_directions.setter
    def parallel_directions(self, pd):
        if len(pd) != 2:
            msg = "incorrect shape of list defining parallel directions"
            logger.info(msg)

        if not ("parent" in pd | "daughter" in pd):
            msg = "incorrect keys in parallel directions"
            logger.info(msg)

        shape_check = [p.shape == (3,) for p in pd.values()]
        if not all(shape_check):
            msg = "direction should be array of size (3,)"
            logger.info(msg)

        self._parallel_directions = pd
        self._direction_parent = pd["parent"]
        self._direction_daughter = pd["daughter"]

        if all([hasattr(self, "material_parent"), hasattr(self, "material_daughter")]):
            mat_p = self.material_parent
            mat_d = self.material_daughter
            d1, d2 = self.direction_parent, self.direction_daughter

            d1 = mat_p.unitcell.TransSpace(d1, "d", "c")
            d1 = mat_p.unitcell.NormVec(d1, "c")
            self._direction_parent_c = d1

            d2 = mat_d.unitcell.TransSpace(d2, "d", "c")
            d2 = mat_d.unitcell.NormVec(d2, "c")
            self._direction_daughter_c = d2
        else:
            msg = (
                "materials for parent and daughter phases not set. "
                "skipping computation of the cartesian represantation"
                "of directions."
            )
            logger.info(msg)
