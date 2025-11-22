import numpy as np
from hexrd.rotations import misorientation
from hexrd.material import Material
from matplotlib import pyplot as plt
from hexrd.valunits import _angstrom, _kev
import mplstereonet
from numba import njit
from hexrd.material.spacegroup import get_symmetry_directions

"""
define some helper functions
"""

"""
    remove all hkl, -hkl pairs from list of symmetrically
equivalent hkls"""


def removeinversion(ksym):
    klist = []
    for i in range(ksym.shape[0]):
        k = ksym[i, :]
        kk = list(k)
        nkk = list(-k)
        if klist == []:
            if np.sum(k) > np.sum(-k):
                klist.append(kk)
            else:
                klist.append(nkk)

        else:
            if (kk in klist) or (nkk in klist):
                pass
            else:
                klist.append(kk)
    klist = np.array(klist)
    return klist


"""
    get expected number of phase transformation variants
    from group theoretic calculations
"""


def expected_num_variants(mat1, mat2, R1, R2):
    """
    calculate expected number of orientational
    variants using the formula given in
    C. Cayron, J. Appl. Cryst. (2007). 40, 1179–1182
    page 2
    N_alpha = |G_beta|/|H_beta|
    """
    T = np.dot(R1, R2.T)
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


"""
    get rotation matrix
"""


def get_rmats(p1, d1, mat, toprint=False):
    sym = mat.unitcell.SYM_PG_c
    z = p1
    x = d1
    y = np.cross(z, x)
    return np.vstack((x, y, z)).T


"""
    check if rotation matrix generated is a new one
"""


def isnew(mat, rmat, sym):
    res = True
    for r in rmat:
        for s in sym:
            rr = np.dot(s, r)
            diff = np.sum(np.abs(rr - mat))
            if diff < 1e-6:
                res = False
                break
    return res


def prepare_data(mat1, mat2, parallel_planes, parallel_directions):
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
def getOR(R1, R2, mat1, mat2):
    """
    R1 ---> mat1
    R2 ---> mat2
    """
    rmat = []
    sym1 = mat1.unitcell.SYM_PG_c
    sym2 = mat2.unitcell.SYM_PG_c
    for sa in sym1:
        ma = np.dot(sa, R1)
        # for sb in sym2:
        #     mb = np.dot(sb, R2)
        mb = R2
        m = np.dot(mb, ma.T)
        if isnew(m, rmat, sym2):
            rmat.append(m)

    rmat_t = [r.T for r in rmat]
    return rmat_t


def plot_OR_mat(rmat, mat, fig, ax, title):
    font = {
        "family": "serif",
        "weight": "bold",
        "size": 20,
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

        ax.pole(strike, dip, m, markersize=16, markeredgecolor="red")
    ax.set_azimuth_ticks([])
    ax.tick_params(axis="both", which="major", size=24)
    ax.grid()
    ax.legend(leg, bbox_to_anchor=(1.15, 1.15), loc="upper right", prop=font)
    ax.set_title(title, **font)


def plot_OR(rmat_parent, rmat_variants, mat1, mat2, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = mplstereonet.subplots(ncols=2, figsize=(16, 8))

    plot_OR_mat(rmat_parent, mat1, fig, ax[0], title="Parent")
    plot_OR_mat(rmat_variants, mat2, fig, ax[1], title="Variants")
    fig.show()


def getPhaseTransformationVariants(
    mat1,
    mat2,
    parallel_planes,
    parallel_directions,
    rmat_parent=[np.eye(3)],
    plot=False,
):
    """
    main function to get the phase transformation variants between two materials
    give a set of parallel planes and parallel directions

    Parameters
    ----------
    mat1 : hexrd.material.Material
        Material class for parent phase
    mat2 : hexrd.material.Material
        Material class for child phase
    parallel_planes : list/tuple
        list of numpy arrays with parallel planes, length=2
    parallel_planes : list/tuple
        list of numpy arrays with parallel directions, length=2
    plot : boolean
        plot the data in stereographic projection if true


    """
    (p1, p2), (d1, d2) = prepare_data(
        mat1, mat2, parallel_planes, parallel_directions
    )

    R1 = get_rmats(p1, d1, mat1)
    R2 = get_rmats(p2, d2, mat2)

    rmat_variants = getOR(R1, R2, mat1, mat2)

    num_var = expected_num_variants(mat1, mat2, R1, R2)

    print("Expected # of orientational variants = ", num_var)
    print(
        "number of orientation variants in phase transition = ",
        len(rmat_variants),
        "\n",
    )

    if plot:
        plot_OR(rmat_parent, rmat_variants, mat1, mat2, fig=None, ax=None)
