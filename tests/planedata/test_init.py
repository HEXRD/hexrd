import numpy as np
import pytest

# TODO: Check that this test is still sensible after PlaneData change.
from hexrd.core.material.crystallography import PlaneData


def test_init_with_data_and_from_copy():
    hkls = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lparms = np.array([1, 1, 1, 1])
    laueGroup = 'C2h'
    wavelength = 0.5  # Angstroms
    strainMag = 0.01

    pd = PlaneData(hkls, lparms, laueGroup, wavelength, strainMag)
    pd2 = PlaneData(hkls, pd)
    pd3 = PlaneData(None, pd)
    pd4 = PlaneData(
        None,
        pd,
        doTThSort=False,
        exclusions=[False, False, False]
    )
    pd5 = PlaneData(
        None,
        pd,
        tThMax=6.0,
        tThWidth=3.0
    )
    assert pd2.hkls.shape == pd3.hkls.shape and np.all(pd2.hkls == pd3.hkls)
    assert pd2.hkls.shape == pd4.hkls.shape and np.all(pd2.hkls == pd4.hkls)


def test_init_with_invalid_params():
    # One extra parameter but not a PlaneData object
    with pytest.raises((NotImplementedError, RuntimeError)):
        PlaneData(np.array([[1, 0, 0], [0, 1, 0]]), 1)
    # Invalid number of parameters
    with pytest.raises((NotImplementedError, RuntimeError)):
        PlaneData(
            np.array([[1, 0, 0], [0, 1, 0]]), np.array([1, 1, 1, 1, 1]), 3
        )
    # Invalid number of unnamed parameters
    with pytest.raises((NotImplementedError, RuntimeError)):
        PlaneData(
            np.array([[1, 0, 0], [0, 1, 0]]),
            np.array([1, 1, 1, 1]),
            'C2h',
            0.5,
            doTThSort=True,
        )
    # Invalid named parameter
    with pytest.raises((NotImplementedError, RuntimeError)):
        PlaneData(
            np.array([[1, 0, 0], [0, 1, 0]]),
            np.array([1, 1, 1, 1]),
            'C2h',
            0.5,
            0.01,
            invalid_name=True,
        )
