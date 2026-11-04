import numpy as np
import pytest

from hexrd.core.material import Material
from hexrd.core.material.unitcell import unitcell
from hexrd.core.valunits import _kev, _nm
from hexrd.powder.wppf.phase import Material_Rietveld

U_GENERAL = [0.020, 0.005, 0.010, 0.003, 0.002, 0.001]


@pytest.fixture
def tetragonal_material() -> Material:
    material = Material(dmin=_nm(0.2), kev=_kev(80.0))
    material.sgnum = 75  # P4: four rotations about the tetragonal axis.
    material.latticeParameters = [5, 5, 6, 90, 90, 90]
    material._set_atomdata([28], [[0.1, 0.2, 0.3, 0.7]], [U_GENERAL], ['0'])
    return material


def _calculate_raw(
    target: unitcell | Material_Rietveld, hkls: np.ndarray
) -> np.ndarray:
    if isinstance(target, Material_Rietveld):
        target.hkls = hkls
        target.multiplicity = target.getMultiplicity(hkls)
        weighted, raw = target.CalcXRSF(target.wavelength, 0.7)
        np.testing.assert_allclose(weighted, 0.7 * target.multiplicity * raw)
        return raw
    return target.CalcXRSF(hkls)


def _attenuation_ratio(
    target: unitcell | Material_Rietveld, hkls: np.ndarray
) -> np.ndarray:
    # Divide out the common form factor and occupancy using zero displacement.
    saved_u = target.U.copy()
    try:
        target.U[:] = 0.0
        reference = _calculate_raw(target, hkls)
    finally:
        target.U[:] = saved_u
    return _calculate_raw(target, hkls) / reference


def _full_group_ratio(
    target: unitcell | Material_Rietveld, hkls: np.ndarray
) -> np.ndarray:
    """Independent Cartesian reference summed over every space-group op.

    U is first averaged over the ops that leave the site in place, which
    imposes the site symmetry. Those ops reach the same atom, so each
    site's sums are also divided by their count.
    """
    direct = np.linalg.cholesky(np.linalg.inv(target.rmt)).T  # a, b, c
    reciprocal = np.linalg.inv(direct).T  # a*, b*, c* in inverse nm
    to_cif = reciprocal / np.linalg.norm(reciprocal, axis=0)
    from_cif = np.linalg.inv(to_cif)
    q = 0.2 * np.pi * hkls @ reciprocal.T  # Inverse angstroms.
    cartesian_rotations = [
        direct @ op[:3, :3] @ np.linalg.inv(direct) for op in target.SYM_SG
    ]
    displaced = np.zeros(len(hkls), dtype=complex)
    undisplaced = np.zeros(len(hkls), dtype=complex)

    for u, atom in zip(target.U, target.atom_pos):
        xx, yy, zz, xy, xz, yz = u
        cif_u = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        cartesian_u = from_cif.T @ cif_u @ from_cif
        site, occupancy = atom[:3], atom[3]
        stabilizer = []
        for operation, cartesian_rotation in zip(target.SYM_SG, cartesian_rotations):
            shift = operation[:3, :3] @ site + operation[:3, 3] - site
            if np.allclose(shift, np.round(shift), atol=1e-6):
                stabilizer.append(cartesian_rotation)
        cartesian_u = np.mean([s @ cartesian_u @ s.T for s in stabilizer], axis=0)

        site_displaced = np.zeros(len(hkls), dtype=complex)
        site_undisplaced = np.zeros(len(hkls), dtype=complex)
        for operation, cartesian_rotation in zip(target.SYM_SG, cartesian_rotations):
            position = operation[:3, :3] @ site + operation[:3, 3]
            rotated_u = cartesian_rotation @ cartesian_u @ cartesian_rotation.T
            phases = np.exp(-2j * np.pi * (hkls @ position))
            debye_waller = np.exp(-0.5 * np.einsum('ni,ij,nj->n', q, rotated_u, q))
            site_displaced += debye_waller * phases
            site_undisplaced += phases
        displaced += occupancy * site_displaced / len(stabilizer)
        undisplaced += occupancy * site_undisplaced / len(stabilizer)

    return abs(displaced / undisplaced) ** 2


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
@pytest.mark.parametrize('change', [None, 'U', 'lattice', 'position'])
def test_p4_anisotropic_structure_factors(
    tetragonal_material: Material, caller: str, change: str | None
) -> None:
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
        _full_group_ratio(target, hkls),
        rtol=1e-12,
    )
    raw = _calculate_raw(target, hkls)
    # Equivalent reflections must have the same intensity in this P4 crystal.
    np.testing.assert_allclose(raw[0], raw[1], rtol=1e-12)


@pytest.mark.parametrize('caller', ['unitcell', 'rietveld'])
@pytest.mark.parametrize(
    'sgnum,lattice,atoms,U,numat',
    [
        # The second P4 site is on the fourfold axis, so its orbit is 1 atom.
        pytest.param(
            75,
            [5, 5, 6, 90, 90, 90],
            [[0.1, 0.2, 0.3, 0.7], [0, 0, 0.125, 0.4]],
            [U_GENERAL, [0.015, 0.015, 0.025, 0, 0, 0]],
            [4, 1],
            id='P4-two-sites',
        ),
        # P6 rotations are not orthogonal in fractional axes.
        pytest.param(
            168,
            [5, 5, 6, 90, 90, 120],
            [[0.13, 0.27, 0.31, 0.7]],
            [U_GENERAL],
            [6],
            id='P6',
        ),
        # U12 and U23 break the mirror at y = 1/4, so they must average out.
        pytest.param(
            62,
            [5, 6, 7, 90, 90, 90],
            [[0.1, 0.25, 0.3, 0.7]],
            [U_GENERAL],
            [4],
            id='Pnma-mirror-site',
        ),
    ],
)
def test_anisotropic_matches_full_group_sum(
    tetragonal_material: Material,
    caller: str,
    sgnum: int,
    lattice: list[float],
    atoms: list[list[float]],
    U: list[list[float]],
    numat: list[int],
) -> None:
    tetragonal_material.sgnum = sgnum
    tetragonal_material.latticeParameters = lattice
    n = len(atoms)
    tetragonal_material._set_atomdata([28] * n, atoms, U, ['0'] * n)
    target = tetragonal_material.unitcell
    if caller == 'rietveld':
        target = Material_Rietveld(material_obj=tetragonal_material)

    np.testing.assert_array_equal(target.numat, numat)
    hkls = np.array([[1, 1, 1], [2, 1, 1], [2, 1, 3]], dtype=float)
    np.testing.assert_allclose(
        _attenuation_ratio(target, hkls),
        _full_group_ratio(target, hkls),
        rtol=1e-12,
    )
    # Every reflection equivalent to (2, 1, 3) must have the same intensity.
    equivalents = target.CalcStar(np.array([2, 1, 3]), 'r', applyLaue=True)
    raw = _calculate_raw(target, equivalents)
    np.testing.assert_allclose(raw, raw[0], rtol=1e-12)


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
    tetragonal_material: Material,
    caller: str,
    U: float,
    sgnum: int,
    lattice: list[float],
) -> None:
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


def test_calc_orbit_returns_aligned_rotations(tetragonal_material: Material) -> None:
    cell = tetragonal_material.unitcell
    position = cell.atom_pos[0, :3]
    positions, count = cell.CalcOrbit(position)
    expanded, expanded_count, rotations = cell.CalcOrbit(
        position, return_rotations=True
    )
    assert count == expanded_count == 4
    np.testing.assert_array_equal(positions, expanded)
    for rotation, expected in zip(rotations, positions):
        # P4 is symmorphic, so the rotations alone give the positions.
        np.testing.assert_allclose((rotation @ position) % 1.0, [expected])

    # A coordinate of 1 is the same atom as 0, so it is counted once, and the
    # identity still lands on the original site.
    positions, count, rotations = cell.CalcOrbit(
        np.array([1.0, 0.2, 0.3]), return_rotations=True
    )
    assert count == 4
    np.testing.assert_array_equal(rotations[0], [np.eye(3)])
