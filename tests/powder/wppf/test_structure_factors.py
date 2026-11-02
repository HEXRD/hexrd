import numpy as np
import pytest

from hexrd.core.material import Material
from hexrd.core.valunits import _kev, _nm
from hexrd.powder.wppf.phase import Material_Rietveld


@pytest.fixture
def tetragonal_material():
    material = Material(dmin=_nm(0.2), kev=_kev(80.0))
    material.sgnum = 75  # P4: four rotations about the tetragonal axis.
    material.latticeParameters = [5, 5, 6, 90, 90, 90]
    material._set_atomdata(
        [28],
        [[0.1, 0.2, 0.3, 0.7]],
        [[0.020, 0.005, 0.010, 0.003, 0.002, 0.001]],
        ['0'],
    )
    return material


def _calculate_raw(target, hkls):
    if isinstance(target, Material_Rietveld):
        target.hkls = hkls
        target.multiplicity = target.getMultiplicity(hkls)
        weighted, raw = target.CalcXRSF(target.wavelength, 0.7)
        np.testing.assert_allclose(weighted, 0.7 * target.multiplicity * raw)
        return raw
    return target.CalcXRSF(hkls)


def _attenuation_ratio(target, hkls):
    # Divide out the common form factor and occupancy using zero displacement.
    saved_u = target.U.copy()
    try:
        target.U[:] = 0.0
        reference = _calculate_raw(target, hkls)
    finally:
        target.U[:] = saved_u
    return _calculate_raw(target, hkls) / reference


def _p4_attenuation_ratio(target, hkls):
    """Independent Cartesian Gaussian-displacement reference for a P4 site."""
    lengths_angstrom = 10.0 * np.asarray(target.lparms[:3])
    displaced = np.zeros(len(hkls), dtype=complex)
    undisplaced = np.zeros(len(hkls), dtype=complex)

    for u, atom in zip(target.U, target.atom_pos):
        xx, yy, zz, xy, xz, yz = u
        x, y, z, occupancy = atom
        # Write the four rotated tensors explicitly, including cross-term signs.
        tensors = np.array(
            [
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
                [[yy, -xy, -yz], [-xy, xx, xz], [-yz, xz, zz]],
                [[xx, xy, -xz], [xy, yy, -yz], [-xz, -yz, zz]],
                [[yy, -xy, yz], [-xy, xx, -xz], [yz, -xz, zz]],
            ]
        )
        positions = np.array([[x, y, z], [-y, x, z], [-x, -y, z], [y, -x, z]])
        if x == y == 0:
            # A site on the fourfold axis contributes one atom.
            tensors = tensors[:1]
            positions = positions[:1]

        for index, hkl in enumerate(hkls):
            q = 2.0 * np.pi * hkl / lengths_angstrom
            phases = np.exp(-2j * np.pi * (positions @ hkl))
            amplitudes = np.array([np.exp(-0.5 * q @ U @ q) for U in tensors])
            displaced[index] += occupancy * np.sum(amplitudes * phases)
            undisplaced[index] += occupancy * np.sum(phases)

    return abs(displaced / undisplaced) ** 2


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
@pytest.mark.parametrize('change', [None, 'U', 'lattice', 'position'])
def test_p4_anisotropic_structure_factors(tetragonal_material, caller, change):
    target = tetragonal_material.unitcell
    if caller == 'rietveld':
        target = Material_Rietveld(material_obj=tetragonal_material)

    hkls = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 3]], dtype=float)
    _calculate_raw(target, hkls)

    if change == 'U':
        target.U[0, 0] *= 1.5
        target.U[0, 3] *= 0.5
    elif change == 'lattice':
        if caller == 'rietveld':
            target.lparms[:2] *= 1.1
            target._calcrmt()
        else:
            tetragonal_material.latticeParameters = [5.5, 5.5, 6, 90, 90, 90]
            target = tetragonal_material.unitcell
    elif change == 'position':
        target.atom_pos[0, :3] = [0.13, 0.27, 0.31]
        target.CalcPositions()

    np.testing.assert_allclose(
        _attenuation_ratio(target, hkls),
        _p4_attenuation_ratio(target, hkls),
        rtol=1e-12,
    )
    raw = _calculate_raw(target, hkls)
    # Equivalent reflections must have the same intensity in this P4 crystal.
    np.testing.assert_allclose(raw[0], raw[1], rtol=1e-12)


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
def test_p4_special_position(tetragonal_material, caller):
    tetragonal_material._set_atomdata(
        [28], [[0, 0, 0.3, 0.7]], [[0.020, 0.020, 0.010, 0, 0, 0]], ['0']
    )
    target = tetragonal_material.unitcell
    if caller == 'rietveld':
        target = Material_Rietveld(material_obj=tetragonal_material)

    # All four rotations map to the same atom: include it exactly once.
    np.testing.assert_array_equal(target.numat, [1])
    assert target.orbit_rotations[0].shape == (1, 3, 3)
    hkls = np.array([[1, 0, 0], [2, 1, 3]], dtype=float)
    np.testing.assert_allclose(
        _attenuation_ratio(target, hkls),
        _p4_attenuation_ratio(target, hkls),
        rtol=1e-12,
    )


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
@pytest.mark.parametrize('U', [0.0, 0.005, 0.020])
@pytest.mark.parametrize(
    'sgnum,lattice',
    [
        pytest.param(16, [4, 5, 6, 90, 90, 90], id='orthorhombic'),
        pytest.param(75, [5, 5, 6, 90, 90, 90], id='tetragonal'),
        pytest.param(195, [5, 5, 5, 90, 90, 90], id='cubic'),
    ],
)
def test_equal_diagonal_anisotropic_matches_isotropic(
    tetragonal_material, caller, U, sgnum, lattice
):
    # Equal diagonal U entries with zero cross terms describe isotropic
    # displacement in orthogonal crystal axes, including unequal cell lengths.
    tetragonal_material.sgnum = sgnum
    tetragonal_material.latticeParameters = lattice
    tetragonal_material._set_atomdata([28], [[0.137, 0.271, 0.319, 0.7]], [U], ['0'])
    target = tetragonal_material.unitcell
    if caller == 'rietveld':
        target = Material_Rietveld(material_obj=tetragonal_material)

    hkls = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [2, 1, 3], [-1, 2, 3]],
        dtype=float,
    )
    assert not target.aniU
    reciprocal_lengths_squared = np.einsum('ni,ij,nj->n', hkls, target.rmt, hkls)
    expected = np.exp(-4.0 * np.pi**2 * U * reciprocal_lengths_squared * 1e-2)
    np.testing.assert_allclose(
        _attenuation_ratio(target, hkls), expected, rtol=1e-13, atol=0.0
    )

    isotropic = _calculate_raw(target, hkls)
    target.U = np.array([[U, U, U, 0, 0, 0]])
    if caller == 'rietveld':
        target.aniU = True
    assert target.aniU
    np.testing.assert_allclose(
        _calculate_raw(target, hkls), isotropic, rtol=1e-13, atol=0.0
    )


def test_calc_orbit_returns_aligned_rotations(tetragonal_material):
    cell = tetragonal_material.unitcell
    position = cell.atom_pos[0, :3]
    positions, count = cell.CalcOrbit(position)
    expanded, expanded_count, rotations = cell.CalcOrbit(
        position, return_rotations=True
    )
    assert count == expanded_count == 4
    np.testing.assert_array_equal(positions, expanded)
    np.testing.assert_allclose((rotations @ position) % 1.0, positions, atol=1e-14)


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
def test_multiple_sites_with_different_orbit_sizes(tetragonal_material, caller):
    tetragonal_material._set_atomdata(
        [28, 28],
        [[0.1, 0.2, 0.3, 0.7], [0, 0, 0.125, 0.4]],
        [
            [0.020, 0.005, 0.010, 0.003, 0.002, 0.001],
            [0.015, 0.015, 0.025, 0, 0, 0],
        ],
        ['0', '0'],
    )
    target = tetragonal_material.unitcell
    if caller == 'rietveld':
        target = Material_Rietveld(material_obj=tetragonal_material)

    np.testing.assert_array_equal(target.numat, [4, 1])
    hkls = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 3]], dtype=float)
    np.testing.assert_allclose(
        _attenuation_ratio(target, hkls),
        _p4_attenuation_ratio(target, hkls),
        rtol=1e-12,
    )


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
def test_hexagonal_anisotropic_structure_factors(tetragonal_material, caller):
    material = tetragonal_material
    material.sgnum = 168  # P6: rotations are not orthogonal in fractional axes.
    material.latticeParameters = [5, 5, 6, 90, 90, 120]
    cell = material.unitcell
    cartesian_u = np.array(
        [[0.020, 0.003, 0.002], [0.003, 0.005, 0.001], [0.002, 0.001, 0.010]]
    )
    reciprocal_directions = cell.rsm / np.linalg.norm(cell.rsm, axis=0)
    cif_u = reciprocal_directions.T @ cartesian_u @ reciprocal_directions
    material._set_atomdata(
        [28],
        [[0.13, 0.27, 0.31, 0.7]],
        [cif_u[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]],
        ['0'],
    )
    cell = material.unitcell
    target = cell if caller == 'unitcell' else Material_Rietveld(material_obj=material)
    hkls = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 3]], dtype=float)

    expected = []
    for hkl in hkls:
        q = 2.0 * np.pi * (cell.rsm @ hkl) * 0.1  # Inverse angstroms.
        displaced = 0j
        undisplaced = 0j
        for operation in cell.SYM_SG:
            cartesian_rotation = cell.dsm @ operation[:3, :3] @ np.linalg.inv(cell.dsm)
            rotated_u = cartesian_rotation @ cartesian_u @ cartesian_rotation.T
            position = operation[:3, :3] @ cell.atom_pos[0, :3] + operation[:3, 3]
            phase = np.exp(-2j * np.pi * (hkl @ position))
            displaced += np.exp(-0.5 * q @ rotated_u @ q) * phase
            undisplaced += phase
        expected.append(abs(displaced / undisplaced) ** 2)

    np.testing.assert_allclose(_attenuation_ratio(target, hkls), expected, rtol=1e-12)
