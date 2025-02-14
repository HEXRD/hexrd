"""Testing GrainData class"""

from pathlib import Path

import pytest
import numpy as np

from hexrd.hedm.cli.fit_grains import GrainData


@pytest.fixture
def exp90():
    return (np.pi / 2) * np.identity(3)


@pytest.fixture
def quats90():
    c45, s45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
    return [[c45, s45, 0, 0], [c45, 0, s45, 0], [c45, 0, 0, s45]]


@pytest.fixture
def rmats90():
    return np.array(
        [
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        ]
    )


@pytest.fixture
def graindata_0(exp90):
    args = dict(
        id=np.array([0, 1, 2]),
        completeness=np.array([0, 0.5, 1.0]),
        chisq=np.array([2.1, 1.2, 0.1]),
        expmap=exp90,
        centroid=np.identity(3),
        inv_Vs=np.hstack((np.ones((3, 3)), np.zeros((3, 3)))),
        ln_Vs=np.zeros((3, 6)),
    )
    return GrainData(**args)


class TestLoadSave:

    def test_load_save(self, tmp_path, graindata_0):

        gdata = graindata_0
        save_path = tmp_path / "save.npz"
        gdata.save(save_path)
        gdata_cmp = GrainData.load(save_path)

        assert np.allclose(gdata.centroid, gdata_cmp.centroid)
        assert np.allclose(gdata.completeness, gdata_cmp.completeness)
        assert np.allclose(gdata.chisq, gdata_cmp.chisq)
        assert np.allclose(gdata.inv_Vs, gdata_cmp.inv_Vs)
        assert np.allclose(gdata.ln_Vs, gdata_cmp.ln_Vs)

    def test_grains_out(self, tmp_path, graindata_0):

        gdata = graindata_0
        save_path = tmp_path / "save.out"
        gdata.write_grains_out(save_path)
        gdata_cmp = GrainData.from_grains_out(save_path)

        assert np.allclose(gdata.centroid, gdata_cmp.centroid)
        assert np.allclose(gdata.completeness, gdata_cmp.completeness)
        assert np.allclose(gdata.chisq, gdata_cmp.chisq)
        assert np.allclose(gdata.inv_Vs, gdata_cmp.inv_Vs)
        assert np.allclose(gdata.ln_Vs, gdata_cmp.ln_Vs)


class TestRotations:

    def test_rotation_matrices(self, graindata_0, rmats90):
        assert np.allclose(graindata_0.rotation_matrices, rmats90)

    def test_quaternions(self, graindata_0, quats90):
        assert np.allclose(graindata_0.quaternions, quats90)


class TestSelect:

    def test_completeness(self, graindata_0):

        gd_sel = graindata_0.select(min_completeness=0.75)
        assert np.all(gd_sel.id == [2])

    def test_chisq(self, graindata_0):

        gd_sel = graindata_0.select(max_chisq=1.5)
        assert np.all(gd_sel.id == [1, 2])
