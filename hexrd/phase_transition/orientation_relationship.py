# standard imports
# ---------
import logging

import matplotlib

# 3rd party imports
# -----------------
import numpy as np

# hexrd imports
# -------------
from hexrd.core.material import Material
from hexrd.core.material.spacegroup import get_symmetry_directions
from hexrd.phase_transition.variants import getPhaseTransformationVariants
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
        rmat_parent: list[np.ndarray] | None = None,
        verbose: bool = False,
    ):

        self.material_parent = material_parent
        self.material_daughter = material_daughter
        self.parallel_planes = parallel_planes
        self.parallel_directions = parallel_directions
        self.verbose = verbose
        self.compute_variant_rmats()
        self.rmat_parent = rmat_parent

    def compute_variant_rmats(self):
        self._rmat_variants, _ = getPhaseTransformationVariants(
            self.material_parent,
            self.material_daughter,
            [pp for pp in self.parallel_planes.values()],
            [pd for pd in self.parallel_directions.values()],
            rmat_parent=[np.eye(3)],
            apply_parent_rotation_to_variants=False,
            plot=False,
            verbose=self.verbose,
        )

    def plot_OR_mat(
        self,
        parent_or_daughter: str,
        title: str,
        fig: matplotlib.figure.Figure = None,
        ax: matplotlib.axes.Axes = None,
    ):
        font = {
            "family": "serif",
            "weight": "bold",
            "size": 12,
        }

        if parent_or_daughter.lower() == "parent":
            mat = self.material_parent
            rmat = self.rmat_parent
        elif parent_or_daughter.lower() == "daughter":
            mat = self.material_daughter
            rmat = self.rmat_variants_rotated

        Us = get_symmetry_directions(mat)
        markers = ["ok", "^g", "sb"]

        leg = []
        for U in Us:
            leg.append(rf"$[{U[0]},{U[1]},{U[2]}]$")

        for m, u in zip(markers, Us):
            uvw = mat.CalcStar(u, "d")
            dip = []
            strike = []

            for v in uvw:
                vv = mat.TransSpace(v, "d", "c")
                vec = mat.NormVec(vv, "c")
                for r in rmat:
                    xyz = np.dot(r, vec)
                    if xyz[2] < 0.0:
                        xyz = -xyz
                    dip.append(np.degrees(np.arccos(xyz[2])))
                    strike.append(180.0 + np.degrees(np.arctan2(xyz[1], xyz[0])))

            ax.pole(strike, dip, m, markersize=12, markeredgecolor="red")
        ax.tick_params(axis="both", size=24)
        ax.grid(True, which="both", ls=":", lw=1)
        ax.set_longitude_grid(10)
        # ax.set_latitude_grid(10)
        ax.set_azimuth_ticks([])
        ax.legend(leg, bbox_to_anchor=(1.15, 1.15), loc="upper right", prop=font)
        ax.set_title(title, **font)

    def plot_OR(self):
        try:
            import mplstereonet
        except ImportError:
            raise RuntimeError('mplstereonet must be installed to use plot_OR')

        fig, ax = mplstereonet.subplots(
            ncols=2, figsize=(10, 5), projection='equal_angle_stereonet'
        )

        self.plot_OR_mat("parent", fig=fig, ax=ax[0], title="Parent")
        self.plot_OR_mat("daughter", fig=fig, ax=ax[1], title="Variants")
        # cosmetic
        fig.subplots_adjust(hspace=0, wspace=0.05, left=0.01, bottom=0.1, right=0.99)
        fig.show()

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
    def rmat_parent(self):
        return self._rmat_parent

    @property
    def rmat_variants_rotated(self):
        return self._rmat_variants_rotated

    @property
    def verbose(self):
        return self._verbose

    @property
    def num_variants(self):
        return len(self._rmat_variants)

    @material_parent.setter
    def material_parent(self, mat):
        if not isinstance(mat, Material_Rietveld):
            msg = "incorrect material class of parent phase"
            raise TypeError(msg)
        self._material_parent = mat

    @material_daughter.setter
    def material_daughter(self, mat):
        if not isinstance(mat, Material_Rietveld):
            msg = "incorrect material class of parent phase"
            raise TypeError(msg)
        self._material_daughter = mat

    @parallel_planes.setter
    def parallel_planes(self, pp):
        if len(pp) != 2:
            msg = "incorrect shape of list defining parallel planes"
            raise ValueError(msg)

        if not ("parent" in pp or "daughter" in pp):
            msg = "incorrect keys in parallel planes"
            raise ValueError(msg)

        shape_check = [p.shape == (3,) for p in pp.values()]
        if not all(shape_check):
            msg = "plane should be array of size (3,)"
            raise ValueError(msg)

        self._parallel_planes = pp
        self._plane_parent = pp["parent"]
        self._plane_daughter = pp["daughter"]

        if all([hasattr(self, "material_parent"), hasattr(self, "material_daughter")]):
            mat_p = self.material_parent
            mat_d = self.material_daughter
            p1, p2 = self.plane_parent, self.plane_daughter

            p1 = mat_p.TransSpace(p1, "r", "c")
            p1 = mat_p.NormVec(p1, "c")
            self._plane_parent_c = p1

            p2 = mat_d.TransSpace(p2, "r", "c")
            p2 = mat_d.NormVec(p2, "c")
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
            raise ValueError(msg)

        if not ("parent" in pd or "daughter" in pd):
            msg = "incorrect keys in parallel directions"
            raise ValueError(msg)

        shape_check = [p.shape == (3,) for p in pd.values()]
        if not all(shape_check):
            msg = "direction should be array of size (3,)"
            raise ValueError(msg)

        self._parallel_directions = pd
        self._direction_parent = pd["parent"]
        self._direction_daughter = pd["daughter"]

        if all([hasattr(self, "material_parent"), hasattr(self, "material_daughter")]):
            mat_p = self.material_parent
            mat_d = self.material_daughter
            d1, d2 = self.direction_parent, self.direction_daughter

            d1 = mat_p.TransSpace(d1, "d", "c")
            d1 = mat_p.NormVec(d1, "c")
            self._direction_parent_c = d1

            d2 = mat_d.TransSpace(d2, "d", "c")
            d2 = mat_d.NormVec(d2, "c")
            self._direction_daughter_c = d2
        else:
            msg = (
                "materials for parent and daughter phases not set. "
                "skipping computation of the cartesian represantation"
                "of directions."
            )
            logger.info(msg)

    @rmat_parent.setter
    def rmat_parent(self, rmat_p):
        if rmat_p is None:
            self._rmat_parent = [np.eye(3)]
            return
        if not isinstance(rmat_p, list):
            msg = "rmat_parent not list type"
            raise TypeError(msg)
        self._rmat_parent = rmat_p

        if hasattr(self, "_rmat_variants"):
            self._rmat_variants_rotated = [
                parent @ variant
                for parent in self.rmat_parent
                for variant in self.rmat_variants
            ]

    @verbose.setter
    def verbose(self, v):
        if not isinstance(v, bool):
            msg = "verbose should be type bool"
            raise TypeError(msg)
        self._verbose = v

    @rmat_variants.setter
    def rmat_variants(self, rmat_v):
        if not isinstance(rmat_v, list):
            msg = "rmat_variants should be list type"
            raise TypeError(msg)
        self._rmat_variants = rmat_v
