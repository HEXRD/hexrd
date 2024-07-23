import numpy as np

import pytest

from hexrd.material.crystallography import ltypeOfLaueGroup
from hexrd.material.material import Material
from hexrd.rotations import rotMatOfQuat


@pytest.fixture
def materials(test_data_dir):
    material_names = [
        "Ag(TeMo)6",
        "Al2SiO5",
        "C",
        "Cs",
        "AlCuO2",
        "Mg",
        "Si",
        "U",
    ]
    mats = {}
    for mat_name in material_names:
        # Load {test_data_dir}/materials/{mat_name}.cif
        mat = Material(
            mat_name, str(test_data_dir) + "/materials/" + mat_name + ".cif"
        )
        mats[mat_name] = mat.planeData
    return mats


def assertEqualNumpyArrays(a, b):
    a, b = np.array(a), np.array(b)
    # Make sure shape is the same and values are close
    assert a.shape == b.shape, f'Shape mismatch: {a.shape} vs {b.shape}'
    assert np.allclose(a, b), f'Numpy arrays not close: {a} vs {b}'


def assertAllArraysEqual(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assertEqualNumpyArrays(flatten(x), flatten(y))


# From Nick Craig-Wood at
# https://stackoverflow.com/questions/10632111/how-to-flatten-a-hetrogenous-list-of-list-into-a-single-list-in-python
def flatten(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result


def test_plane_data_with_data(test_data_dir, materials):
    data = np.load(test_data_dir / 'plane_data_test.npy', allow_pickle=True)
    for obj in data:
        pd = materials[obj['mat_name']]
        assertEqualNumpyArrays(pd.hkls, obj['hkls'])
        assertEqualNumpyArrays(pd.lparms, obj['lparms'])
        assert pd.laueGroup == obj['laueGroup']
        assert np.isclose(pd.strainMag, obj['strainMag'])
        assert pd.getLatticeType() == ltypeOfLaueGroup(obj['laueGroup'])
        assertEqualNumpyArrays(pd.hedm_intensity, obj['hedm_intensity'])
        assertEqualNumpyArrays(pd.powder_intensity, obj['powder_intensity'])
        # With the identity symmetry, zero rotation may have some sign issues,
        # but the rotation matrix should be pretty much the exact same
        assertEqualNumpyArrays(rotMatOfQuat(pd.getQSym()),
                               rotMatOfQuat(obj['q_sym']))
        assert pd.nHKLs == obj['nHKLs']
        assert pd.getNhklRef() == obj['nhklRef']
        assertEqualNumpyArrays(pd.getMultiplicity(), obj['multiplicity'])
        assertEqualNumpyArrays(pd.getTTh(), obj['tth'])
        assertAllArraysEqual(pd.getTThRanges(), obj['tth_ranges'])
        assertAllArraysEqual(
            pd.getMergedRanges(False), obj['unculled_merged_ranges']
        )
        assertAllArraysEqual(
            pd.getMergedRanges(True), obj['culled_merged_ranges']
        )


# def test_with_data_maker(test_data_dir, materials):
#     data = []
#     for mat_name, pd in materials.items():
#         obj = {"mat_name": mat_name}
#         obj['hkls'] = pd.hkls
#         obj['lparms'] = pd.lparms
#         obj['laueGroup'] = pd.laueGroup
#         obj['wavelength'] = pd.wavelength
#         obj['strainMag'] = pd.strainMag
#         obj['hedm_intensity'] = pd.hedm_intensity
#         obj['powder_intensity'] = pd.powder_intensity
#         obj['tth'] = pd.getTTh()
#         obj['tth_ranges'] = pd.getTThRanges()
#         obj['culled_merged_ranges'] = pd.getMergedRanges(True)
#         obj['unculled_merged_ranges'] = pd.getMergedRanges(False)
#         obj['multiplicity'] = pd.getMultiplicity()
#         obj['q_sym'] = pd.getQSym()
#         obj['nHKLs'] = pd.nHKLs
#         obj['nhklRef'] = pd.getNhklRef()
#         data.append(obj)
#     np.save(test_data_dir / 'plane_data_test.npy', data)
