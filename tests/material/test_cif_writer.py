from pathlib import Path

import numpy as np

from hexrd.material.material import Material


def assert_materials_equal(
    m1: Material, m2: Material, rtol: float = 1e-8, atol: float = 1e-8
) -> None:
    assert np.allclose(m1.lparms, m2.lparms, rtol=rtol, atol=atol)
    assert int(m1.sgnum) == int(m2.sgnum)
    assert np.array_equal(np.asarray(m1.atomtype), np.asarray(m2.atomtype))
    assert list(m1.charge) == list(m2.charge)
    assert np.allclose(m1.atominfo, m2.atominfo, rtol=rtol, atol=atol)
    assert np.allclose(
        np.asarray(m1.U), np.asarray(m2.U), rtol=rtol, atol=atol
    )


def test_cif_examples_export(tmp_path: Path) -> None:
    examples_dir = Path(__file__).resolve().parents[1] / 'data' / 'materials'
    sample_files = [
        examples_dir / 'C.cif',
        examples_dir / 'Mg.cif',
        examples_dir / 'Si.cif',
    ]

    for cif_path in sample_files:
        assert cif_path.exists(), f"missing example cif: {cif_path}"

        m1 = Material(name=cif_path.stem, material_file=str(cif_path))
        out_path = tmp_path / f"{cif_path.stem}_out.cif"
        m1.write_cif(out_path)

        assert out_path.exists()

        m2 = Material(name=cif_path.stem, material_file=str(out_path))
        assert_materials_equal(m1, m2)


def test_cif_default_material_export(tmp_path: Path) -> None:
    # Default Material (Ni)
    m1 = Material()
    out_path = tmp_path / 'default_out.cif'
    m1.write_cif(out_path)

    assert out_path.exists()

    m2 = Material(name='default', material_file=str(out_path))
    assert_materials_equal(m1, m2)
