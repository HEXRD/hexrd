from pytest import fixture
from hexrd.core.material import Material, unitcell
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
    """
    Just ensuring that the outspace doesn't matter
    """
    np.random.seed(0)
    for _ in range(100):
        v1 = np.random.rand(3) * 10 - 5
        space = np.random.choice(['d', 'r'])
        v1c = cell.TransSpace(v1, space, 'c')
        assert np.allclose(
            cell.CalcStar(v1, space, False),
            cell.TransSpace(cell.CalcStar(v1c, 'c', False), 'c', space),
        )
        assert np.allclose(
            cell.CalcStar(v1, space, True),
            cell.TransSpace(cell.CalcStar(v1c, 'c', True), 'c', space),
        )
