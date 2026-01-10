from pathlib import Path

import numpy as np
import pytest

from hexrd.core.material import load_materials_hdf5
from hexrd.phase_transition import variants
from hexrd.valunits import _kev, _nm


@pytest.fixture
def test_variants_material_file(test_data_dir: Path) -> Path:
    return test_data_dir / 'materials/materials_variants.h5'


def test_variants(test_variants_material_file: Path):
    beamenergy = _kev(10)
    dmin = _nm(0.075)
    mats = load_materials_hdf5(
        test_variants_material_file, dmin=dmin, kev=beamenergy
    )

    # 4H-SiC phase crystal structure
    fourH = mats["4H_SiC_HP"]

    # B1 phase crystal structure
    B1 = mats["SiC_B1"]

    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, 1.0])
    d1 = np.array([2.0, 1.0, 0.0])
    d2 = np.array([1.0, 1.0, 0.0])
    parallel_planes = np.array([p1, p2])
    parallel_directions = np.array([d1, d2])

    rmat_variants, num_variants = variants.getPhaseTransformationVariants(
        fourH,
        B1,
        parallel_planes,
        parallel_directions,
        plot=False,
        verbose=False,
    )

    assert len(rmat_variants) == num_variants
    assert len(rmat_variants) == 3
