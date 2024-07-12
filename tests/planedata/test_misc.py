import numpy as np

from hexrd.material.crystallography import PlaneData
from hexrd.rotations import quatOfLaueGroup
from hexrd.valunits import valWUnit


def test_misc():
    """
    Test some setters and getters to make sure they update properly
    """
    pd = PlaneData(
        np.array([[1, 2, 1], [2, 1, 3], [3, 2, 1]]),
        np.array([1, 1, 1, 1]),
        'C2h',
        0.5,
        0.01,
    )

    assert pd.laueGroup == 'C2h'
    assert pd.getLatticeType() == 'monoclinic'
    # Replace with pd.qSym when that is merged
    assert np.all(pd.getQSym() == quatOfLaueGroup('C2h'))
    assert np.all(pd.lparms == np.array([1, 1, 1, 1]))

    # Some getter and setter stuff
    pd.lparms = np.array([1, 1, 1, valWUnit('astr', 'angle', 0.1, 'radians')])

    assert np.abs(pd.lparms[3] - 5.72958) < 1e-4

    pd.strainMag = 0.025
    assert pd.strainMag == 0.025

    pd.laueGroup = 'D4h'
    assert pd.laueGroup == 'D4h'
    assert pd.getLatticeType() == 'tetragonal'
    # Replace with pd.qSym when that is merged
    # Also, this is a bug right now
    # assert pd.getQSym().shape == quatOfLaueGroup('D4h').shape and np.all(
    #     pd.getQSym() == quatOfLaueGroup('D4h')
    # )
