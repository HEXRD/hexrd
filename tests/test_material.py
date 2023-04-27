import numpy as np
import h5py
import pytest

from hexrd.material import Material, load_materials_hdf5

# Tolerance for comparing floats
FLOAT_TOL = 1.e-8

# Use consistent units to simplify testing
DEFAULT_LENGTH_UNIT = 'angstrom'
DEFAULT_ANGLE_UNIT = 'degrees'


@pytest.fixture
def default_material():
    return Material()


@pytest.fixture
def test_materials_file(example_repo_path):
    return example_repo_path / 'NIST_ruby/single_GE/include/materials.h5'


def normalize_unit(v):
    if hasattr(v, 'unit'):
        # Assume it's a val unit
        if v.unit in ('radians', 'degrees'):
            # Assume it's an angle
            return v.getVal(DEFAULT_ANGLE_UNIT)
        else:
            # Assume it's a length
            return v.getVal(DEFAULT_LENGTH_UNIT)

    return v


def are_close(v1, v2, tol=FLOAT_TOL):
    return abs(normalize_unit(v1) - normalize_unit(v2)) < tol


def lparms_are_close(lparms, indices, tol=FLOAT_TOL):
    first = lparms[indices[0]]
    return all(are_close(first, lparms[i], tol) for i in indices[1:])


def test_sgnum_setter(default_material):
    # Just in case we change the default...
    default_material.sgnum = 225

    # Should have a == b == c and alpha == beta == gamma == 90
    lparms = default_material.latticeParameters
    assert lparms_are_close(lparms, [0, 1, 2])
    assert lparms_are_close(lparms, [3, 4, 5])
    assert are_close(lparms[5], 90)

    default_material.sgnum = 165
    lparms = default_material.latticeParameters

    # Gamma should be 120, the other angles should be 90
    assert not lparms_are_close(lparms, [3, 4, 5])
    assert lparms_are_close(lparms, [3, 4])
    assert are_close(lparms[3], 90)
    assert are_close(lparms[5], 120)


def test_load_materials(test_materials_file):
    materials = load_materials_hdf5(test_materials_file)

    with h5py.File(test_materials_file, 'r') as f:
        # Check that it loaded all of the materials
        mat_names = list(f.keys())
        assert all(x in materials for x in mat_names)

        # Check that the values for ruby match
        ruby = materials['ruby']
        params = f['ruby']['LatticeParameters'][()]

        # Convert to angstroms...
        for i in range(3):
            params[i] *= 10

        for i in range(6):
            assert are_close(params[i], ruby.latticeParameters[i])


class TestExclusions:

    def test_d(self, default_material):
        dmin, dmax = 1.0, 1.5
        pd = default_material.planeData
        pd.exclude(dmin=dmin, dmax=dmax)
        d = np.array(pd.getPlaneSpacings())
        assert (d.min() >= dmin) and (d.max() <= dmax)
