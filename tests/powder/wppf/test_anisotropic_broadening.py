from types import SimpleNamespace

import numpy as np
import pytest

from hexrd.powder.wppf.peakfunctions import _anisotropic_peak_broadening
from hexrd.powder.wppf.wppfsupport import (
    _fill_shkl,
    _required_shkl_names,
    _rqd_shkl,
)


@pytest.mark.parametrize(
    'lattice_type,shkl',
    [
        ('hexagonal', {'s400': 1.3, 's004': 0.7, 's202': 0.4}),
        ('trigonal', {'s400': 1.3, 's004': 0.7, 's202': 0.4, 's301': 0.25}),
    ],
)
def test_equivalent_reflections_broaden_equally(
    lattice_type: str, shkl: dict[str, float]
) -> None:
    shkl_values = _fill_shkl(shkl, _rqd_shkl[lattice_type][1])
    rng = np.random.default_rng(0)
    for h, k, l_val in rng.integers(-6, 7, (20, 3)).astype(float):
        # Rotate 120 degrees about c, in hexagonal axes
        rotated = np.array([k, -h - k, l_val])
        expected = _anisotropic_peak_broadening(shkl_values, np.array([h, k, l_val]))
        result = _anisotropic_peak_broadening(shkl_values, rotated)
        assert np.isclose(result, expected)


def test_primitive_trigonal_uses_hexagonal_terms() -> None:
    # P-3m1
    mat = SimpleNamespace(latticeType='trigonal', sgnum=164, name='test')
    names, _, _, trig_ptype = _required_shkl_names(mat)
    assert names == ['s400', 's004', 's202']
    assert trig_ptype
