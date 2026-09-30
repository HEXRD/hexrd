from pathlib import Path

import numpy as np
import pytest

from hexrd.core.material import Material
from hexrd.core.material.jcpds import JCPDS_extend

# thermal coefficients chosen to exercise every term of the JCPDS
# thermal equation of state
PT_JCPDS = """VERSION: 4
COMMENT: Pt test
K0:       266.0
K0P:      5.81
DK0DT:    -0.0287
DK0PDT:   0.0
SYMMETRY: CUBIC
A:        3.9231
ALPHAT:   2.66e-5
DALPHADT: 1.0e-8
"""

MGO_JCPDS = """VERSION: 4
COMMENT: MgO test
K0:       160.2
K0P:      3.99
DK0DT:    -0.0272
DK0PDT:   1.7e-4
SYMMETRY: CUBIC
A:        4.2112
ALPHAT:   3.12e-5
DALPHADT: 9.0e-9
"""

# V / V0 computed with the jcpds class of Dioptas 0.10.0
# (name, pressure in GPa, temperature in K, V / V0)
DIOPTAS_REFERENCE: list[tuple[str, float, float, float]] = [
    ('Pt', 300.0, 298.0, 0.6844975582721072),
    ('Pt', 10.0, 1000.0, 0.98384686924806),
    ('Pt', 0.1, 2500.0, 1.198082187494044),
    ('Pt', 100.0, 2500.0, 0.8103354978833354),
    ('MgO', 1.0, 2500.0, 1.1482739261935493),
    ('MgO', 300.0, 2500.0, 0.5162943364716202),
]


def load_jcpds_material(tmp_path: Path, name: str) -> Material:
    contents = {'Pt': PT_JCPDS, 'MgO': MGO_JCPDS}[name]
    path = tmp_path / f'{name}.jcpds'
    path.write_text(contents)

    mat = Material()
    jcpds = JCPDS_extend(str(path))
    jcpds.write_lattice_params_to_material(mat)
    jcpds.write_pt_params_to_material(mat)
    return mat


@pytest.mark.parametrize('name,pressure,temperature,ratio', DIOPTAS_REFERENCE)
def test_eos_matches_dioptas(
    tmp_path: Path, name: str, pressure: float, temperature: float, ratio: float
) -> None:
    mat = load_jcpds_material(tmp_path, name)
    v = mat.calc_volume(pressure=pressure, temperature=temperature)
    assert np.isclose(v / mat.v0, ratio, rtol=1e-10, atol=0)
    # calc_pressure is the inverse of calc_volume
    p = mat.calc_pressure(volume=v, temperature=temperature)
    assert np.isclose(p, pressure, rtol=1e-10, atol=1e-10)


def test_eos_out_of_range_raises(tmp_path: Path) -> None:
    mat = load_jcpds_material(tmp_path, 'Pt')
    # beyond the maximum pressure of the isotherm when K' < 4
    mat.k0p = 3.0
    with pytest.raises(ValueError):
        mat.calc_volume(pressure=1e4, temperature=298.0)
    # thermal expansion beyond what the isotherm can describe
    mat = load_jcpds_material(tmp_path, 'Pt')
    mat.alpha_t = 1e-3
    with pytest.raises(ValueError):
        mat.calc_volume(pressure=0.1, temperature=2500.0)


def test_jcpds_thermal_expansion_not_converted(tmp_path: Path) -> None:
    mat = load_jcpds_material(tmp_path, 'Pt')
    assert mat.thermal_expansion == 2.66e-5
    assert mat.thermal_expansion_dt == 1.0e-8
