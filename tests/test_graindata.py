"""Testing GrainData class"""

from pathlib import Path

import pytest
import numpy as np

from hexrd.hedm.cli.fit_grains import GrainData


save_file = "save.npz"

exp90 = (np.pi / 2) * np.identity(3)
rmats90 = np.array(
    [
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    ]
)


@pytest.fixture
def graindata_0():
    args = dict(
        id=[0, 1, 2],
        completeness=[0, 0.5, 1.0],
        chisq=[2.1, 1.2, 0.1],
        expmap=exp90,
        centroid=np.identity(3),
        inv_Vs=np.zeros(6),
        ln_Vs=np.zeros(6),
    )
    return GrainData(**args)


def test_load_save(tmp_path, graindata_0):

    gdata = graindata_0
    save_path = tmp_path / save_file
    gdata.save(save_path)
    gdata_cmp = GrainData.load(save_path)

    assert np.allclose(gdata.centroid, gdata_cmp.centroid)
    assert np.allclose(gdata.completeness, gdata_cmp.completeness)
    assert np.allclose(gdata.chisq, gdata_cmp.chisq)
    assert np.allclose(gdata.inv_Vs, gdata_cmp.inv_Vs)
    assert np.allclose(gdata.ln_Vs, gdata_cmp.ln_Vs)


def test_rotation_matrices(graindata_0):

    gdata = graindata_0
    assert np.allclose(gdata.rotation_matrices, rmats90)
