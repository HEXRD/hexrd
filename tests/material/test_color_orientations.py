from pathlib import Path

import numpy as np

from hexrd.material.material import Material


def test_color_orientations() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "data" / "materials"
    rotations_file = examples_dir / "rotations.npy"
    rmats = np.load(rotations_file)
    assert rmats.shape == (2, 3, 3)

    mat = Material()
    v1 = np.round(mat.unitcell.color_orientations(rmats), 6)
    v2 = np.vstack(
        (
            np.round(mat.unitcell.color_orientations(rmats[0]), 6),
            np.round(mat.unitcell.color_orientations(rmats[1]), 6),
        )
    )

    assert v1 == v2
