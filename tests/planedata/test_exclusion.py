import numpy as np

from hexrd.material.crystallography import PlaneData


def test_exclusion():
    pd = PlaneData(
        np.array([[1, 2, 1], [2, 1, 3], [3, 2, 1]]),
        np.array([1, 1, 1, 1]),
        'C2h',
        0.5,
        0.01,
    )
    assert pd.nHKLs == 3
    assert pd.getNhklRef() == 3

    # Exclude with a mask
    pd.exclusions = [True, False, True]
    assert pd.nHKLs == 1
    assert pd.getNhklRef() == 3

    # Exclude indices
    pd.exclusions = [2]
    assert pd.nHKLs == 2
    assert pd.getNhklRef() == 3

    try:
        # Exclude a range
        pd.exclusions = [[0,2]]
        assert pd.nHKLs == 1
        assert pd.getNhklRef() == 3
    except NotImplementedError:
        pass
