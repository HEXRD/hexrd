from pytest import fixture
from hexrd.core.material import Material, unitcell
from hexrd.core.material.spacegroup import Allowed_HKLs
from hexrd.core.material.unitcell import _hkls_within_dmin
import numpy as np


@fixture
def cell() -> unitcell.unitcell:
    return Material().unitcell


def get_space():
    return np.random.choice(['d', 'r', 'c'])


def test_calc_dot(cell: unitcell.unitcell):
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        v2 = np.random.rand(3) * 10 - 5
        space = get_space()
        v1c = cell.TransSpace(v1, space, 'c')
        v2c = cell.TransSpace(v2, space, 'c')
        assert np.allclose(np.dot(v1c, v2c), cell.CalcDot(v1, v2, space))


def test_calc_length(cell: unitcell.unitcell):
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        space = get_space()
        v1c = cell.TransSpace(v1, space, 'c')
        assert np.allclose(np.linalg.norm(v1c), cell.CalcLength(v1, space))


def test_calc_angle(cell: unitcell.unitcell):
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        v2 = np.random.rand(3) * 10 - 5
        space = get_space()
        v1c = cell.TransSpace(v1, space, 'c')
        v2c = cell.TransSpace(v2, space, 'c')
        norms = np.linalg.norm(v1c) * np.linalg.norm(v2c)
        assert np.allclose(
            np.arccos(np.dot(v1c, v2c) / norms), cell.CalcAngle(v1, v2, space)
        )


def test_calc_cross(cell: unitcell.unitcell):
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        v2 = np.random.rand(3) * 10 - 5
        inspace = get_space()
        outspace = get_space()
        v1c = cell.TransSpace(v1, inspace, 'c')
        v2c = cell.TransSpace(v2, inspace, 'c')
        expected = cell.TransSpace(np.cross(v1c, v2c), 'c', outspace)
        assert np.allclose(
            expected, cell.CalcCross(v1, v2, inspace, outspace, True)
        )


def test_norm_vec(cell: unitcell.unitcell):
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        space = get_space()
        norm_v1 = cell.NormVec(v1, space)
        assert np.allclose(1, cell.CalcLength(norm_v1, space))
        # Make sure we don't change the direction
        assert np.allclose(
            v1 / np.linalg.norm(v1), norm_v1 / np.linalg.norm(norm_v1)
        )


def test_trans_space(cell: unitcell.unitcell):
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        inspace = get_space()
        outspace = get_space()
        v1_out = cell.TransSpace(v1, inspace, outspace)
        assert np.allclose(v1, cell.TransSpace(v1_out, outspace, inspace))


def test_calc_star(cell: unitcell.unitcell):
    v = [1.0, 2.0, 3.0]
    vsym = cell.CalcStar(v, 'd', False)
    assert vsym.shape[0] == 48

    vsym = cell.CalcStar(v, 'r', False)
    assert vsym.shape[0] == 48

    vsym = cell.CalcStar(v, 'd', True)
    assert vsym.shape[0] == 48

    vsym = cell.CalcStar(v, 'r', True)
    assert vsym.shape[0] == 48

    v = [1.0, 1.0, 3.0]
    vsym = cell.CalcStar(v, 'd', False)
    assert vsym.shape[0] == 24

    vsym = cell.CalcStar(v, 'r', False)
    assert vsym.shape[0] == 24

    vsym = cell.CalcStar(v, 'd', True)
    assert vsym.shape[0] == 24

    vsym = cell.CalcStar(v, 'r', True)
    assert vsym.shape[0] == 24

    v = [1.0, 1.0, 0.0]
    vsym = cell.CalcStar(v, 'd', False)
    assert vsym.shape[0] == 12

    vsym = cell.CalcStar(v, 'r', False)
    assert vsym.shape[0] == 12

    vsym = cell.CalcStar(v, 'd', True)
    assert vsym.shape[0] == 12

    vsym = cell.CalcStar(v, 'r', True)
    assert vsym.shape[0] == 12

    v = [1.0, 1.0, 1.0]
    vsym = cell.CalcStar(v, 'd', False)
    assert vsym.shape[0] == 8

    vsym = cell.CalcStar(v, 'r', False)
    assert vsym.shape[0] == 8

    vsym = cell.CalcStar(v, 'd', True)
    assert vsym.shape[0] == 8

    vsym = cell.CalcStar(v, 'r', True)
    assert vsym.shape[0] == 8

    v = [1.0, 0.0, 0.0]
    vsym = cell.CalcStar(v, 'd', False)
    assert vsym.shape[0] == 6

    vsym = cell.CalcStar(v, 'r', False)
    assert vsym.shape[0] == 6

    vsym = cell.CalcStar(v, 'd', True)
    assert vsym.shape[0] == 6

    vsym = cell.CalcStar(v, 'r', True)
    assert vsym.shape[0] == 6


def test_hkls_within_dmin_matches_reference(cell: unitcell.unitcell):
    candidates = np.array(
        [
            [h, k, ell]
            for h in np.arange(cell.ih, -cell.ih - 1, -1)
            for k in np.arange(cell.ik, -cell.ik - 1, -1)
            for ell in np.arange(cell.il, -1, -1)
        ]
    )
    candidates = Allowed_HKLs(cell.sgnum, candidates)
    expected = np.array(
        [
            hkl
            for hkl in candidates
            if np.any(hkl) and 1.0 / cell.CalcLength(hkl, 'r') >= cell.dmin
        ],
        dtype=np.int32,
    )

    result = _hkls_within_dmin(
        cell.ih, cell.ik, cell.il, cell.rmt, cell.dmin, cell.sgnum
    )
    np.testing.assert_array_equal(result, expected)


def test_choose_symmetric_matches_reference(cell: unitcell.unitcell):
    hkls = _hkls_within_dmin(
        cell.ih, cell.ik, cell.il, cell.rmt, cell.dmin, cell.sgnum
    )
    mask = np.ones(len(hkls), dtype=bool)
    for i, hkl in enumerate(hkls):
        if not mask[i]:
            continue

        equivalents = cell.CalcStar(hkl, 'r', applyLaue=True).astype(int)
        for equivalent in equivalents[1:]:
            matches = np.where(np.all(equivalent == hkls, axis=1))
            mask[matches] = False

    expected = []
    for hkl in hkls[mask]:
        equivalents = cell.CalcStar(hkl, 'r', applyLaue=True)
        expected.append(equivalents[np.argmax(np.sum(equivalents, axis=1))])

    result = cell.ChooseSymmetric(hkls, InversionSymmetry=True)
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.int32))
