import numpy as np
import pytest

# TODO: Check that this test is still sensible after PlaneData change.
from hexrd.core.material.crystallography import PlaneData


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

    # Exclude a range
    pd.exclusions = [[0, 2]]
    assert pd.nHKLs == 1
    assert pd.getNhklRef() == 3


def test_exclusions_survive_lparms_change():
    pd = PlaneData(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([2.0, 3.0, 4.0, 90.0, 90.0, 90.0]),
        'D2h',
        0.5,
        0.01,
    )

    # Select only the second hkl in sorted order
    excl = np.ones(pd.getNhklRef(), dtype=bool)
    excl[1] = False
    pd.exclusions = excl
    selected_hkl = pd.getHKLs(allHKLs=True)[1].tolist()

    # Change lparms enough to re-sort
    pd.lparms = [2.0, 3.0, 1.5, 90.0, 90.0, 90.0]

    # The same hkl should still be selected
    all_hkls = pd.getHKLs(allHKLs=True)
    new_excl = pd.exclusions
    new_selected = [all_hkls[i].tolist() for i, e in enumerate(new_excl) if not e]
    assert new_selected == [selected_hkl]


def test_exclusions_with_unsorted_hkls() -> None:
    # Input order 220, 111, 200 sorts to 111, 200, 220.
    hkls = np.array([[2, 2, 0], [1, 1, 1], [2, 0, 0]]).T
    pd = PlaneData(hkls, [4.0], 'Oh', 80.0, 0.01)

    # All three forms exclude 111, the first reflection in sorted order.
    for exclusions in ([True, False, False], [0], [[0, 1]]):
        pd.exclusions = np.array(exclusions)
        assert pd.exclusions.tolist() == [True, False, False]
        assert pd.getHKLs().tolist() == [[2, 0, 0], [2, 2, 0]]
        # Master IDs refer to the input order, not the sorted order.
        assert pd.getHKLID((2, 0, 0), master=True) == 2
        assert pd.getHKLID((2, 2, 0), master=True) == 0
        assert pd.getHKLs(2).tolist() == [[2, 0, 0]]
        with pytest.raises(RuntimeError, match='excluded'):
            pd.getHKLs(1)
