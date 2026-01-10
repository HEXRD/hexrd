import logging

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from hexrd.core.material import Material
from hexrd.core.material.spacegroup import get_symmetry_directions

logger = logging.getLogger(__name__)

"""
define some helper functions
"""


def removeinversion(ksym: np.ndarray) -> np.ndarray:
    """remove all hkl, -hkl pairs from list of symmetrically"""
    klist = []
    for i in range(ksym.shape[0]):
        k = ksym[i, :]
        kk = list(k)
        nkk = list(-k)
        if not klist:
            klist.append(kk if np.sum(k) > np.sum(-k) else nkk)
        else:
            if kk not in klist and nkk not in klist:
                klist.append(kk)

    return np.array(klist)


def expected_num_variants(
    mat1: Material, mat2: Material, r1: np.ndarray, r2: np.ndarray
) -> int:
    """
    calculate expected number of orientational
    variants using the formula given in
    C. Cayron, J. Appl. Cryst. (2007). 40, 1179–1182
    page 2
    N_alpha = |G_beta|/|H_beta|
    """
    T = np.dot(r1, r2.T)
    sym1 = mat1.unitcell.SYM_PG_c
    sym2 = mat2.unitcell.SYM_PG_c
    ctr = 0
    for s2 in sym2:
        ss = np.dot(T, np.dot(s2, T.T))
        for s1 in sym1:
            diff = np.sum(np.abs(ss - s1))
            if diff < 1e-6:
                ctr += 1
    return int(len(sym1) / ctr)


def get_rmats(p1: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """get rotation matrix"""
    z = p1
    x = d1
    y = np.cross(z, x)
    return np.vstack((x, y, z)).T


def isnew(mat: np.ndarray, rmat: list[np.ndarray], sym: np.ndarray) -> bool:
    """check if rotation matrix generated is a new one"""
    for r in rmat:
        for s in sym:
            rr = np.dot(s, r)
            diff = np.sum(np.abs(rr - mat))
            if diff < 1e-6:
                return False

    return True


def prepare_data(
    mat1: Material,
    mat2: Material,
    parallel_planes: list[np.ndarray],
    parallel_directions: list[np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    prepare the planes and directions by:
        1. converting to cartesian space
        2. normalizing magnitude"""
    p1, p2 = parallel_planes
    d1, d2 = parallel_directions

    p1 = mat1.unitcell.TransSpace(p1, "r", "c")
    p1 = mat1.unitcell.NormVec(p1, "c")
    d1 = mat1.unitcell.TransSpace(d1, "d", "c")
    d1 = mat1.unitcell.NormVec(d1, "c")

    p2 = mat2.unitcell.TransSpace(p2, "r", "c")
    p2 = mat2.unitcell.NormVec(p2, "c")
    d2 = mat2.unitcell.TransSpace(d2, "d", "c")
    d2 = mat2.unitcell.NormVec(d2, "c")

    return (p1, p2), (d1, d2)


# main function
# get the variants
def getOR(r1: np.ndarray, r2: np.ndarray, mat1: Material, mat2: Material) -> np.ndarray:
    """
    r1 ---> mat1
    r2 ---> mat2
    """
    rmat = []
    sym1 = mat1.unitcell.SYM_PG_c
    sym2 = mat2.unitcell.SYM_PG_c
    for sa in sym1:
        ma = np.dot(sa, r1)
        # for sb in sym2:
        #     mb = np.dot(sb, r2)
        mb = r2
        m = np.dot(mb, ma.T)
        if isnew(m, rmat, sym2):
            rmat.append(m)

    rmat_t = [r.T for r in rmat]
    return np.array(rmat_t)


def plot_OR_mat(rmat: np.ndarray, mat: Material, fig: Figure, ax: Axes, title: str):
    font = {
        "family": "serif",
        "weight": "bold",
        "size": 12,
    }

    Gs = get_symmetry_directions(mat)
    markers = ["ok", "^g", "sb"]

    leg = []
    for G in Gs:
        leg.append(rf"$[{G[0]},{G[1]},{G[2]}]$")

    for m, g in zip(markers, Gs):
        hkl = mat.unitcell.CalcStar(g, "d")
        dip = []
        strike = []

        for v in hkl:
            vv = mat.unitcell.TransSpace(v, "d", "c")
            vec = mat.unitcell.NormVec(vv, "c")
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


def plot_OR(
    rmat_parent: np.ndarray,
    rmat_variants: np.ndarray,
    mat1: Material,
    mat2: Material,
    fig: Figure | None = None,
    ax: Axes | None = None,
):
    try:
        import mplstereonet
    except ImportError:
        raise RuntimeError('mplstereonet must be installed to use plot_OR')

    if fig is None or ax is None:
        fig, ax = mplstereonet.subplots(
            ncols=2, figsize=(10, 5), projection='equal_angle_stereonet'
        )

    plot_OR_mat(rmat_parent, mat1, fig, ax[0], title="Parent")
    plot_OR_mat(rmat_variants, mat2, fig, ax[1], title="Variants")
    # cosmetic
    fig.subplots_adjust(hspace=0, wspace=0.05, left=0.01, bottom=0.1, right=0.99)
    fig.show()


def getPhaseTransformationVariants(
    mat1: Material,
    mat2: Material,
    parallel_planes: np.ndarray,
    parallel_directions: np.ndarray,
    rmat_parent: np.ndarray | None = None,
    plot: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """
    main function to get the phase transformation variants between two materials
    give a set of parallel planes and parallel directions

    Parameters
    ----------
    mat1 : hexrd.material.Material
        Material class for parent phase
    mat2 : hexrd.material.Material
        Material class for child phase
    parallel_planes : np.ndarray
        list of numpy arrays with parallel planes, length=2
    parallel_planes : np.ndarray
        list of numpy arrays with parallel directions, length=2
    rmat_parent : np.ndarray
        ???, default is np.array([np.eye(3)])
    plot : boolean
        plot the data in stereographic projection if true
    """
    if rmat_parent is None:
        rmat_parent = np.array([np.eye(3)])

    (p1, p2), (d1, d2) = prepare_data(mat1, mat2, parallel_planes, parallel_directions)

    r1 = get_rmats(p1, d1)
    r2 = get_rmats(p2, d2)

    rmat_variants = getOR(r1, r2, mat1, mat2)

    num_var = expected_num_variants(mat1, mat2, r1, r2)

    if verbose:
        logger.info(f'Expected # of orientational variants = {num_var}')
        logger.info(
            f'number of orientation variants in phase transition = {len(rmat_variants)}'
        )

    if plot:
        plot_OR(rmat_parent, rmat_variants, mat1, mat2, fig=None, ax=None)

    return rmat_variants, num_var
