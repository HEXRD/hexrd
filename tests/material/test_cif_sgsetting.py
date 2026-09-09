from pathlib import Path

import pytest

from hexrd.material.material import Material


# Diamond, space group 227, which has two origin choices.  The atom sits
# at the origin, so the two settings give genuinely different structures.
CIF_TEMPLATE = """data_test
_cell_length_a 3.567
_cell_length_b 3.567
_cell_length_c 3.567
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
{space_group}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.0 0.0 0.0 1.0
"""


def write_cif(tmp_path: Path, space_group: str) -> Path:
    path = tmp_path / 'test.cif'
    path.write_text(CIF_TEMPLATE.format(space_group=space_group))
    return path


def load(tmp_path: Path, space_group: str) -> Material:
    path = write_cif(tmp_path, space_group)
    return Material(name='test', material_file=str(path))


@pytest.mark.parametrize(
    'space_group, expected',
    [
        # explicit coordinate system code
        ('_space_group_IT_number 227\n_space_group_IT_coordinate_system_code 1', 0),
        ('_space_group_IT_number 227\n_space_group_IT_coordinate_system_code 2', 1),
        # setting given as a suffix on the Hermann-Mauguin symbol
        ("_symmetry_space_group_name_H-M 'F d -3 m :2'", 1),
        # no setting given: fall back to the default
        ('_space_group_IT_number 227', Material.DFLT_SGSETTING),
        # an empty code is the same as not giving one
        (
            "_space_group_IT_number 227\n_space_group_IT_coordinate_system_code ''",
            Material.DFLT_SGSETTING,
        ),
    ],
)
def test_sgsetting_read_from_cif(
    tmp_path: Path, space_group: str, expected: int
) -> None:
    assert load(tmp_path, space_group).sgsetting == expected


def test_settings_give_different_structure_factors(tmp_path: Path) -> None:
    # The whole point of reading the setting: the two origin choices
    # describe different structures.
    first = load(
        tmp_path,
        '_space_group_IT_number 227\n_space_group_IT_coordinate_system_code 1',
    )
    second = load(
        tmp_path,
        '_space_group_IT_number 227\n_space_group_IT_coordinate_system_code 2',
    )
    assert first.planeData.structFact.tolist() != (second.planeData.structFact.tolist())


def test_unsupported_setting_is_rejected(tmp_path: Path) -> None:
    # hexrd only implements the standard descriptions.
    with pytest.raises(RuntimeError, match='does not support'):
        load(
            tmp_path,
            '_space_group_IT_number 14\n_space_group_IT_coordinate_system_code -b1',
        )


def test_standard_monoclinic_setting_is_accepted(tmp_path: Path) -> None:
    # Unique axis b, cell choice 1 is what hexrd already assumes.
    material = load(
        tmp_path,
        '_space_group_IT_number 14\n_space_group_IT_coordinate_system_code b1',
    )
    assert material.sgnum == 14
