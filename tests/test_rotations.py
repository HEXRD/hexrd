"""Test rotations module"""
import numpy as np
import pytest

from hexrd.material import symmetry
from hexrd import rotations


def test_misorientations():
    qsym = symmetry.quatOfLaueGroup('oh')
    q1 = qsym[:, 3:4]
    q2 = qsym[:, (6, 7, 18)]
    ang, mis = rotations.misorientation(q1, q2, (qsym,))
    print("ang: ", ang, "\nmis: ", mis)
    assert np.allclose(ang, 0.0)
    assert np.allclose(mis[0, :], 1.)
    assert np.allclose(mis[1:, :], 0.)
